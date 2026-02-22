"""
Database utilities for schema inspection and query execution.

Supports two backends, selected by settings.llm_provider:
  "groq"  → SQLite (local)  via SQLAlchemy
  "dbrx"  → Databricks Delta Tables via databricks-sql-connector
"""

from typing import List, Dict, Any, Optional
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
import sqlglot
from loguru import logger
from config import settings
from pathlib import Path
import re


# ─────────────────────────────────────────────────────────────────────────────
# Local SQLite backend (Groq provider)
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_uri: Optional[str] = None):
        self.database_uri = database_uri or settings.database_uri
        
        # Create data directory if using SQLite and directory doesn't exist
        self._ensure_database_directory()
        
        self.db = SQLDatabase.from_uri(self.database_uri)
        self.engine = create_engine(self.database_uri)
        self.inspector = inspect(self.engine)
        logger.info(f"Connected to database: {self.database_uri}")
    
    def _ensure_database_directory(self):
        """Create database directory if it doesn't exist (for SQLite)."""
        if self.database_uri.startswith('sqlite:///'):
            # Extract file path from SQLite URI
            # sqlite:///./data/database.db -> ./data/database.db
            db_path = self.database_uri.replace('sqlite:///', '')
            
            # Handle absolute paths on Windows (e.g., C:/...)
            if not db_path.startswith('/') and ':' not in db_path:
                db_path_obj = Path(db_path)
                db_dir = db_path_obj.parent
                
                # Create directory if it doesn't exist
                if not db_dir.exists():
                    db_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created database directory: {db_dir}")
                    
                # Create empty database file if it doesn't exist
                if not db_path_obj.exists():
                    db_path_obj.touch()
                    logger.info(f"Created empty database file: {db_path_obj}")
    
    def get_all_table_names(self) -> List[str]:
        """Get list of all table names in the database."""
        return self.db.get_usable_table_names()
    
    def get_schema_for_tables(self, table_names: List[str]) -> str:
        try:
            return self.db.get_table_info(table_names)
        except Exception as e:
            logger.error(f"Error retrieving schema: {e}")
            return ""
    
    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        try:
            columns = self.inspector.get_columns(table_name)
            pk = self.inspector.get_pk_constraint(table_name)
            fks = self.inspector.get_foreign_keys(table_name)
            indexes = self.inspector.get_indexes(table_name)
            return {
                "name": table_name,
                "columns": columns,
                "primary_key": pk,
                "foreign_keys": fks,
                "indexes": indexes
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {table_name}: {e}")
            return {}
    
    def validate_sql_syntax(self, sql: str) -> tuple[bool, Optional[str]]:
        try:
            parsed = sqlglot.parse_one(sql)
            if parsed:
                return True, None
            return False, "Failed to parse SQL"
        except Exception as e:
            return False, str(e)
    
    def execute_query(self, sql: str, timeout: Optional[int] = None) -> tuple[Any, Optional[str], Optional[float]]:
        import time
        timeout = timeout or settings.query_timeout_seconds
        start = time.time()
        try:
            is_valid, syntax_error = self.validate_sql_syntax(sql)
            if not is_valid:
                return None, f"Syntax Error: {syntax_error}", None
            
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                if result.returns_rows:
                    columns = list(result.keys())
                    rows = result.fetchall()
                    rows_as_dicts = [dict(zip(columns, row)) for row in rows]
                    execution_time = (time.time() - start) * 1000
                    return rows_as_dicts, None, execution_time
                else:
                    execution_time = (time.time() - start) * 1000
                    return f"Query executed successfully. Rows affected: {result.rowcount}", None, execution_time
                    
        except SQLAlchemyError as e:
            execution_time = (time.time() - start) * 1000
            error_msg = str(e.orig) if hasattr(e, 'orig') else str(e)
            logger.error(f"SQL execution error: {error_msg}")
            return None, error_msg, execution_time
        except Exception as e:
            execution_time = (time.time() - start) * 1000
            logger.error(f"Unexpected error executing query: {e}")
            return None, str(e), execution_time
    
    def close(self):
        self.engine.dispose()
        logger.info("Database connections closed")


# ─────────────────────────────────────────────────────────────────────────────
# Databricks Delta Tables backend (DBRX provider)
# ─────────────────────────────────────────────────────────────────────────────

class DatabricksDatabaseManager:
    """
    Executes SQL queries against Databricks Delta tables using the
    databricks-sql-connector.  Mirrors the DatabaseManager interface
    exactly so all agents can use it transparently.
    """

    def __init__(self):
        self._host       = settings.databricks_host
        self._token      = settings.databricks_token
        self._http_path  = settings.databricks_http_path
        self._catalog    = settings.databricks_catalog
        self._schema     = settings.databricks_schema
        self._table_cache: Optional[List[str]] = None
        self._col_cache:   Dict[str, List[Dict]] = {}
        logger.info(
            f"DatabricksDatabaseManager initialised | "
            f"host={self._host} | catalog={self._catalog} | schema={self._schema}"
        )

    # ── connection factory ────────────────────────────────────────────────────

    def _connect(self):
        """Open and return a databricks-sql-connector connection."""
        try:
            from databricks import sql as dbsql
        except ImportError:
            raise ImportError(
                "databricks-sql-connector is not installed. "
                "Run: pip install databricks-sql-connector"
            )
        return dbsql.connect(
            server_hostname=self._host,
            http_path=self._http_path,
            access_token=self._token,
            catalog=self._catalog,
            schema=self._schema,
        )

    # ── public interface ──────────────────────────────────────────────────────

    def get_all_table_names(self) -> List[str]:
        if self._table_cache is not None:
            return self._table_cache
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SHOW TABLES IN `{self._catalog}`.`{self._schema}`")
                    rows = cur.fetchall()
                    # Result columns vary — table name is typically 2nd column
                    names = []
                    for r in rows:
                        row = list(r)
                        names.append(str(row[1]) if len(row) > 1 else str(row[0]))
                    self._table_cache = names
                    return names
        except Exception as e:
            logger.error(f"DatabricksDB: get_all_table_names failed: {e}")
            return []

    def get_schema_for_tables(self, table_names: List[str]) -> str:
        parts = []
        for tbl in table_names:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        fqn = f"`{self._catalog}`.`{self._schema}`.`{tbl}`"
                        cur.execute(f"DESCRIBE TABLE {fqn}")
                        rows = cur.fetchall()
                        col_defs = []
                        for r in rows:
                            r = list(r)
                            col_name = r[0] if r else ""
                            col_type = r[1] if len(r) > 1 else ""
                            if col_name and not col_name.startswith("#"):
                                col_defs.append(f"  `{col_name}` {col_type}")
                        ddl = f"CREATE TABLE `{tbl}` (\n" + ",\n".join(col_defs) + "\n);"
                        parts.append(ddl)
            except Exception as e:
                logger.error(f"DatabricksDB: schema fetch failed for {tbl}: {e}")
        return "\n\n".join(parts)

    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    fqn = f"`{self._catalog}`.`{self._schema}`.`{table_name}`"
                    cur.execute(f"DESCRIBE TABLE {fqn}")
                    rows = cur.fetchall()
                    columns = [
                        {"name": list(r)[0], "type": list(r)[1]}
                        for r in rows
                        if list(r)[0] and not str(list(r)[0]).startswith("#")
                    ]
                    return {"name": table_name, "columns": columns}
        except Exception as e:
            logger.error(f"DatabricksDB: get_table_metadata failed for {table_name}: {e}")
            return {}

    def validate_sql_syntax(self, sql: str) -> tuple[bool, Optional[str]]:
        """Validate using sqlglot spark dialect."""
        try:
            parsed = sqlglot.parse_one(sql, dialect="spark")
            return (True, None) if parsed else (False, "Failed to parse SQL")
        except Exception as e:
            return False, str(e)

    def execute_query(
        self, sql: str, timeout: Optional[int] = None
    ) -> tuple[Any, Optional[str], Optional[float]]:
        import time
        timeout = timeout or settings.query_timeout_seconds
        start = time.time()

        # Qualify unqualified table references automatically
        sql = self._qualify_tables(sql)

        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    if timeout:
                        try:
                            cur.execute(f"SET statement_timeout = {timeout * 1000}")
                        except Exception:
                            pass  # some warehouses don't support this
                    cur.execute(sql)
                    rows = cur.fetchall()
                    exec_ms = (time.time() - start) * 1000

                    if rows is None:
                        return "Query executed successfully.", None, exec_ms

                    # Convert to list-of-dicts
                    desc = cur.description or []
                    col_names = [d[0] for d in desc]
                    result = [dict(zip(col_names, list(r))) for r in rows]
                    return result, None, exec_ms

        except Exception as e:
            exec_ms = (time.time() - start) * 1000
            logger.error(f"DatabricksDB execute_query error: {e}")
            return None, str(e), exec_ms

    def close(self):
        logger.info("DatabricksDatabaseManager: connections are stateless — nothing to close")

    # ── helpers ────────────────────────────────────────────────────────────────

    def _qualify_tables(self, sql: str) -> str:
        """
        Prepend catalog.schema to bare table names in FROM / JOIN clauses
        so queries work without a USE statement.
        (Best-effort; sqlglot transpile is the robust path for complex CTEs.)
        """
        if not self._catalog or not self._schema:
            return sql
        prefix = f"`{self._catalog}`.`{self._schema}`."
        # Only qualify names that aren't already qualified (no dots / backticks before them)
        def _replace(m: re.Match) -> str:
            kw, ws, name = m.group(1), m.group(2), m.group(3)
            # Already qualified?
            if "." in name or name.startswith("`"):
                return m.group(0)
            return f"{kw}{ws}`{self._catalog}`.`{self._schema}`.`{name}`"

        sql = re.sub(
            r'\b(FROM|JOIN)(\s+)([`"]?\w[`"]?)',
            _replace,
            sql,
            flags=re.IGNORECASE,
        )
        return sql


# ─────────────────────────────────────────────────────────────────────────────
# Factory — returns the right manager based on current provider
# ─────────────────────────────────────────────────────────────────────────────

_manager_cache: Dict[str, Any] = {}


def get_db_manager():
    """
    Return (and cache) the appropriate DatabaseManager for the current
    llm_provider setting.  Safe to call on every request — cheap once
    the backend is warmed up.
    """
    provider = settings.llm_provider
    if provider not in _manager_cache:
        if provider == "dbrx":
            _manager_cache[provider] = DatabricksDatabaseManager()
        else:
            _manager_cache[provider] = DatabaseManager()
    return _manager_cache[provider]


def invalidate_db_cache(provider: Optional[str] = None):
    """Force a new DB manager to be created on the next get_db_manager() call."""
    if provider:
        _manager_cache.pop(provider, None)
    else:
        _manager_cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Backwards-compatible global alias  (used by modules that import db_manager directly)
# ─────────────────────────────────────────────────────────────────────────────

class _LazyProxy:
    """
    Transparent proxy for the active db_manager.
    Delegates every attribute/method call to get_db_manager() dynamically,
    so provider switches mid-session are automatically honoured without
    requiring any module reload.
    """
    def __getattr__(self, name: str):
        return getattr(get_db_manager(), name)

    def __repr__(self) -> str:
        return repr(get_db_manager())


db_manager = _LazyProxy()
