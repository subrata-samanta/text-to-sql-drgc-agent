"""
Database utilities for schema inspection and query execution.
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
        """
        Get DDL schema information for specific tables.
        
        Args:
            table_names: List of table names to retrieve schema for
            
        Returns:
            Formatted schema string with CREATE TABLE statements
        """
        try:
            return self.db.get_table_info(table_names)
        except Exception as e:
            logger.error(f"Error retrieving schema: {e}")
            return ""
    
    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """
        Get detailed metadata about a table including columns, keys, and relationships.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table metadata
        """
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
        """
        Validate SQL syntax without executing.
        
        Args:
            sql: SQL query string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Use sqlglot to parse and validate
            parsed = sqlglot.parse_one(sql)
            if parsed:
                return True, None
            return False, "Failed to parse SQL"
        except Exception as e:
            return False, str(e)
    
    def execute_query(self, sql: str, timeout: Optional[int] = None) -> tuple[Any, Optional[str], Optional[float]]:
        """
        Execute SQL query with error handling and timing.
        
        Args:
            sql: SQL query to execute
            timeout: Query timeout in seconds
            
        Returns:
            Tuple of (result, error_message, execution_time_ms)
        """
        import time
        
        timeout = timeout or settings.query_timeout_seconds
        start = time.time()
        
        try:
            # First validate syntax
            is_valid, syntax_error = self.validate_sql_syntax(sql)
            if not is_valid:
                return None, f"Syntax Error: {syntax_error}", None
            
            # Execute query
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                
                # Fetch results for SELECT queries
                if result.returns_rows:
                    columns = list(result.keys())
                    rows = result.fetchall()
                    # Convert to plain list of dicts so results are portable
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
        """Close database connections."""
        self.engine.dispose()
        logger.info("Database connections closed")


# Global database instance
db_manager = DatabaseManager()
