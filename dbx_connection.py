"""
dbx_connection.py — Databricks SQL connector (singleton).

Used when LLM_PROVIDER=dbrx.  A single connection is opened on first use and
reused for the lifetime of the process, avoiding per-call auth latency.

The connection is established at most ONCE per process:
  • A threading.Lock() prevents concurrent threads (e.g. the cache-warmup
    pool) from each opening their own connection.
  • After the first successful connect, subsequent calls return the cached
    object immediately without any health-check round-trip.
  • Only a genuine query failure triggers a single reconnect attempt.

Credentials are read from .env via pydantic-settings (config.Settings):
    DATABRICKS_SERVER_HOSTNAME
    DATABRICKS_HTTP_PATH
    DATABRICKS_ACCESS_TOKEN

Usage:
    from dbx_connection import run_dbx_query
    df = run_dbx_query("SELECT * FROM `catalog`.`schema`.`table` LIMIT 10")
"""

from __future__ import annotations

import threading

import pandas as pd
from databricks import sql as dbx_sql
from loguru import logger

# config.Settings reads .env via pydantic-settings — always use it instead of
# raw os.getenv() so credentials are sourced from .env even when the shell
# environment has not been pre-populated.
from config import settings

# ── Singleton state ───────────────────────────────────────────────────────────
_connection = None
_connect_lock = threading.Lock()   # ensures exactly one connection attempt
_connected = False                 # True after the first successful connect


def _open_connection():
    """Open a brand-new Databricks connection and store it in the singleton."""
    global _connection, _connected

    server_hostname = settings.databricks_server_hostname
    http_path       = settings.databricks_http_path
    access_token    = settings.databricks_access_token

    if not all([server_hostname, http_path, access_token]):
        raise RuntimeError(
            "Missing Databricks credentials. "
            "Set DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, and "
            "DATABRICKS_ACCESS_TOKEN in your .env file."
        )

    logger.info("[DBX] Opening Databricks connection …")
    _connection = dbx_sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=access_token,
    )
    _connected = True
    logger.info("[DBX] Connection established — will be reused for this session.")


def _get_connection():
    """Return the process-level singleton Databricks connection.

    Connect-once semantics:
      • First caller acquires the lock and opens the connection.
      • All subsequent callers (including concurrent threads) skip straight
        to the return statement — no health-check, no extra cursor open.
      • A reconnect only happens when run_dbx_query() catches a stale-
        connection error during actual query execution.
    """
    global _connected

    if _connected:
        return _connection          # fast path — already connected

    with _connect_lock:
        # Double-checked locking: another thread may have connected while we
        # were waiting for the lock.
        if not _connected:
            _open_connection()

    return _connection


# ── Connection-error classification ─────────────────────────────────────────
_CONNECTION_ERROR_KEYWORDS = (
    "connection", "socket", "broken pipe", "reset by peer",
    "timeout", "eof", "closed", "authentication", "ssl", "transport",
    "network", "unreachable", "refused", "thrift", "disconnected",
)


def _is_connection_error(err: Exception) -> bool:
    """Return True only for transport / auth failures, not for SQL errors.

    SQL errors (UNRESOLVED_COLUMN, SYNTAX_ERROR, TABLE_OR_VIEW_NOT_FOUND, …)
    must *not* trigger a reconnect — they would just fail again and waste the
    latency of opening a second connection.
    """
    msg = str(err).lower()
    return any(kw in msg for kw in _CONNECTION_ERROR_KEYWORDS)


def run_dbx_query(query: str) -> pd.DataFrame:
    """Execute *query* against Databricks and return a pandas DataFrame.

    Reuses the process-level singleton connection.  On genuine stale-connection
    errors (network reset, auth expiry, etc.) the connection is re-established
    once and the query is retried.  Pure SQL errors (UNRESOLVED_COLUMN,
    SYNTAX_ERROR, TABLE_OR_VIEW_NOT_FOUND, …) are propagated immediately —
    reconnecting would not fix them and would waste 30+ seconds.

    Args:
        query: A valid SQL SELECT / CTE statement targeting Databricks tables.

    Returns:
        A pandas DataFrame with the query results.

    Raises:
        RuntimeError: If any required credential is missing.
        Exception:    Propagates Databricks execution errors to the caller.
    """
    global _connection, _connected

    def _execute(conn):
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows    = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)

    conn = _get_connection()
    try:
        return _execute(conn)
    except Exception as first_err:
        # Only reconnect when the error is a genuine connection/transport issue.
        # SQL errors (UNRESOLVED_COLUMN, SYNTAX_ERROR, …) should propagate now.
        if not _is_connection_error(first_err):
            logger.debug(f"[DBX] SQL error (no reconnect): {first_err}")
            raise

        logger.warning(f"[DBX] Connection error detected, reconnecting once: {first_err}")
        try:
            _connection.close()
        except Exception:
            pass
        _connected = False
        with _connect_lock:
            if not _connected:
                _open_connection()
        try:
            return _execute(_connection)
        except Exception as second_err:
            logger.error(f"[DBX] Query failed after reconnect: {second_err}")
            raise


# ── Schema discovery ──────────────────────────────────────────────────────────

def get_table_columns() -> list[str]:
    """Return the actual column names of the configured Databricks table.

    Uses ``SELECT * … LIMIT 0`` so no rows are fetched — only metadata.  The
    result is NOT cached here; callers cache it at module level as needed.

    Returns:
        List of column names (lower-cased) as they exist in the table.
    Raises:
        RuntimeError / Exception: propagates any credential or SQL error.
    """
    sql = f"SELECT * FROM {settings.dbx_full_table} LIMIT 0"
    df = run_dbx_query(sql)
    return [c.lower() for c in df.columns]
