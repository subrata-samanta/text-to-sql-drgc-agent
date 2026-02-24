"""
dbx_connection.py — Databricks SQL connector (singleton).

Used when LLM_PROVIDER=dbrx.  A single connection is opened on first use and
reused for the lifetime of the process, avoiding per-call auth latency.
If the connection goes stale it is transparently re-established.

Credentials are read from .env / env vars:
    DATABRICKS_SERVER_HOSTNAME
    DATABRICKS_HTTP_PATH
    DATABRICKS_ACCESS_TOKEN

Usage:
    from dbx_connection import run_dbx_query
    df = run_dbx_query("SELECT * FROM `catalog`.`schema`.`table` LIMIT 10")
"""

from __future__ import annotations

import os

import pandas as pd
from databricks import sql as dbx_sql

# ── Singleton connection — created once, reused per process ──────────────────
_connection = None


def _get_connection():
    """Return the shared Databricks connection, creating it on first call.

    If the cached connection has been closed or gone stale, a new one is
    opened transparently so callers never manage the lifecycle.
    """
    global _connection

    if _connection is not None:
        try:
            cur = _connection.cursor()
            cur.close()
            return _connection          # still alive — reuse
        except Exception:
            try:
                _connection.close()
            except Exception:
                pass
            _connection = None

    server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME", "")
    http_path       = os.getenv("DATABRICKS_HTTP_PATH", "")
    access_token    = os.getenv("DATABRICKS_ACCESS_TOKEN", "")

    if not all([server_hostname, http_path, access_token]):
        raise RuntimeError(
            "Missing Databricks credentials. "
            "Set DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, and "
            "DATABRICKS_ACCESS_TOKEN in your .env file."
        )

    print("[DBX] Opening Databricks connection …", flush=True)
    _connection = dbx_sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=access_token,
    )
    print("[DBX] Connection established.", flush=True)
    return _connection


def run_dbx_query(query: str) -> pd.DataFrame:
    """Execute *query* against Databricks and return a pandas DataFrame.

    Reuses the process-level singleton connection so no authentication
    round-trip occurs after the first call in a session.

    Args:
        query: A valid SQL SELECT / CTE statement targeting Databricks tables.

    Returns:
        A pandas DataFrame with the query results.

    Raises:
        RuntimeError: If any required credential env-var is missing.
        Exception:    Propagates any Databricks execution error to the caller.
    """
    conn = _get_connection()
    with conn.cursor() as cursor:
        cursor.execute(query)
        rows    = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)
