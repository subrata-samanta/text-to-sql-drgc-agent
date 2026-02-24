"""
Database utilities for schema inspection and query execution.
Provider-aware: routes to SQLite (groq) or Databricks (dbrx) based on LLM_PROVIDER.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
from config import settings
from pathlib import Path
import re
import time


class DatabaseManager:
    """Manages database connections and query execution.

    When LLM_PROVIDER=groq (default): uses local SQLite via SQLAlchemy.
    When LLM_PROVIDER=dbrx:          uses Databricks SQL via dbx_connection.
    """

    def __init__(self, database_uri: Optional[str] = None):
        self._provider = settings.llm_provider.lower().strip()

        if self._provider == "dbrx":
            # Databricks — no local DB file needed; queries go to Databricks SQL.
            logger.info("DatabaseManager: using Databricks SQL (dbrx provider)")
            self.database_uri = None
            self.db = None
            self.engine = None
            self.inspector = None
        else:
            # groq / default — use local SQLite
            from langchain_community.utilities import SQLDatabase
            from sqlalchemy import create_engine, inspect
            self.database_uri = database_uri or settings.database_uri
            self._ensure_database_directory()
            self.db = SQLDatabase.from_uri(self.database_uri)
            self.engine = create_engine(self.database_uri)
            self.inspector = inspect(self.engine)
            logger.info(f"DatabaseManager: connected to SQLite — {self.database_uri}")

    def _ensure_database_directory(self):
        """Create database directory if it doesn't exist (SQLite only)."""
        if not self.database_uri or not self.database_uri.startswith('sqlite:///'):
            return
        db_path = self.database_uri.replace('sqlite:///', '')
        if not db_path.startswith('/') and ':' not in db_path:
            db_path_obj = Path(db_path)
            db_dir = db_path_obj.parent
            if not db_dir.exists():
                db_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created database directory: {db_dir}")
            if not db_path_obj.exists():
                db_path_obj.touch()
                logger.info(f"Created empty database file: {db_path_obj}")

    def get_all_table_names(self) -> List[str]:
        """Get list of all usable table names."""
        if self._provider == "dbrx":
            return [settings.dbx_table]
        return self.db.get_usable_table_names()

    def get_schema_for_tables(self, table_names: List[str]) -> str:
        """Get DDL schema information for specific tables (SQLite only)."""
        if self._provider == "dbrx":
            # Schema is provided by nielsen_schema.py — not from DB introspection.
            return ""
        try:
            return self.db.get_table_info(table_names)
        except Exception as e:
            logger.error(f"Error retrieving schema: {e}")
            return ""

    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Get detailed metadata about a table (SQLite only)."""
        if self._provider == "dbrx":
            return {"name": table_name}
        try:
            columns = self.inspector.get_columns(table_name)
            pk = self.inspector.get_pk_constraint(table_name)
            fks = self.inspector.get_foreign_keys(table_name)
            indexes = self.inspector.get_indexes(table_name)
            return {"name": table_name, "columns": columns,
                    "primary_key": pk, "foreign_keys": fks, "indexes": indexes}
        except Exception as e:
            logger.error(f"Error getting metadata for {table_name}: {e}")
            return {}

    def validate_sql_syntax(self, sql: str) -> tuple[bool, Optional[str]]:
        """Validate SQL syntax without executing (SQLite-mode only)."""
        import sqlglot
        try:
            parsed = sqlglot.parse_one(sql)
            if parsed:
                return True, None
            return False, "Failed to parse SQL"
        except Exception as e:
            return False, str(e)

    def execute_query(
        self,
        sql: str,
        timeout: Optional[int] = None,
    ) -> tuple[Any, Optional[str], Optional[float]]:
        """Execute SQL query with error handling and timing.

        Routes to Databricks when LLM_PROVIDER=dbrx, otherwise uses SQLite.

        Returns:
            (result, error_message, execution_time_ms)
            result is a list-of-dicts for SELECT queries.
        """
        start = time.time()

        if self._provider == "dbrx":
            return self._execute_dbrx(sql, start)
        else:
            return self._execute_sqlite(sql, timeout, start)

    def _execute_dbrx(self, sql: str, start: float) -> tuple:
        """Execute a query against Databricks SQL."""
        try:
            from dbx_connection import run_dbx_query
            df = run_dbx_query(sql)
            execution_time = (time.time() - start) * 1000
            rows_as_dicts = df.to_dict(orient="records")
            return rows_as_dicts, None, execution_time
        except Exception as e:
            execution_time = (time.time() - start) * 1000
            logger.error(f"Databricks query error: {e}")
            return None, str(e), execution_time

    def _execute_sqlite(self, sql: str, timeout: Optional[int], start: float) -> tuple:
        """Execute a query against local SQLite."""
        from sqlalchemy import text
        from sqlalchemy.exc import SQLAlchemyError
        import sqlglot

        timeout = timeout or settings.query_timeout_seconds
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
                    return (
                        f"Query executed successfully. Rows affected: {result.rowcount}",
                        None,
                        execution_time,
                    )
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
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
        logger.info("Database connections closed")


# Global database instance
db_manager = DatabaseManager()
