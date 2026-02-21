"""
Data loader utility for importing CSV/Excel files into SQLite database.
Automatically creates database schema and tables from file data.
"""

import pandas as pd
import sqlite3
from pathlib import Path
from typing import List, Union, Dict, Optional
from loguru import logger
import re


class DataLoader:
    """Loads CSV/Excel files into SQLite database."""
    
    def __init__(self, db_path: str = "./data/database.db"):
        """
        Initialize data loader.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._ensure_data_directory()
        logger.info(f"DataLoader initialized for database: {db_path}")
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist."""
        data_dir = Path(self.db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)
    
    def _sanitize_table_name(self, filename: str) -> str:
        """
        Convert filename to valid SQL table name.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized table name
        """
        # Remove extension and special characters
        name = Path(filename).stem
        # Replace spaces and special chars with underscores
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', '_', name)
        # Ensure it starts with a letter
        if name and not name[0].isalpha():
            name = 'table_' + name
        return name.lower()
    
    def _infer_sql_type(self, dtype) -> str:
        """
        Infer SQL data type from pandas dtype.
        
        Args:
            dtype: Pandas data type
            
        Returns:
            SQL data type string
        """
        dtype_str = str(dtype)
        
        if 'int' in dtype_str:
            return 'INTEGER'
        elif 'float' in dtype_str:
            return 'REAL'
        elif 'datetime' in dtype_str:
            return 'TIMESTAMP'
        elif 'bool' in dtype_str:
            return 'BOOLEAN'
        else:
            return 'TEXT'
    
    def load_file(
        self,
        file_path: str,
        table_name: Optional[str] = None,
        if_exists: str = 'replace'
    ) -> Dict[str, any]:
        """
        Load a CSV or Excel file into the database.
        
        Args:
            file_path: Path to CSV or Excel file
            table_name: Optional custom table name (auto-generated if None)
            if_exists: What to do if table exists ('fail', 'replace', 'append')
            
        Returns:
            Dictionary with load statistics
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and read
        logger.info(f"Loading file: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            # Generate table name if not provided
            if table_name is None:
                table_name = self._sanitize_table_name(file_path.name)
            
            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            
            # Write to database
            df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
            
            stats = {
                'file': str(file_path),
                'table_name': table_name,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            logger.info(f"✓ Loaded {stats['rows']} rows into table '{table_name}'")
            logger.debug(f"Columns: {', '.join(stats['column_names'])}")
            
            self.conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            if self.conn:
                self.conn.close()
            raise
    
    def load_directory(
        self,
        directory_path: str,
        pattern: str = "*",
        if_exists: str = 'replace'
    ) -> List[Dict[str, any]]:
        """
        Load all CSV/Excel files from a directory.
        
        Args:
            directory_path: Path to directory containing data files
            pattern: File pattern to match (e.g., "*.csv", "sales_*")
            if_exists: What to do if table exists ('fail', 'replace', 'append')
            
        Returns:
            List of load statistics for each file
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        results = []
        
        # Find all CSV files
        csv_files = list(directory.glob(f"{pattern}.csv"))
        # Find all Excel files
        excel_files = list(directory.glob(f"{pattern}.xlsx")) + list(directory.glob(f"{pattern}.xls"))
        
        all_files = csv_files + excel_files
        
        if not all_files:
            logger.warning(f"No CSV/Excel files found in {directory} matching pattern '{pattern}'")
            return results
        
        logger.info(f"Found {len(all_files)} file(s) to load")
        
        for file_path in all_files:
            try:
                stats = self.load_file(file_path, if_exists=if_exists)
                results.append(stats)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                results.append({
                    'file': str(file_path),
                    'error': str(e),
                    'status': 'failed'
                })
        
        successful = sum(1 for r in results if 'error' not in r)
        logger.info(f"✓ Successfully loaded {successful}/{len(results)} files")
        
        return results
    
    def load_excel_sheets(
        self,
        file_path: str,
        sheet_names: Optional[List[str]] = None,
        if_exists: str = 'replace'
    ) -> List[Dict[str, any]]:
        """
        Load multiple sheets from an Excel file.
        Each sheet becomes a separate table.
        
        Args:
            file_path: Path to Excel file
            sheet_names: List of sheet names to load (None = all sheets)
            if_exists: What to do if table exists ('fail', 'replace', 'append')
            
        Returns:
            List of load statistics for each sheet
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in ['.xlsx', '.xls']:
            raise ValueError(f"Not an Excel file: {file_path}")
        
        logger.info(f"Loading Excel file with multiple sheets: {file_path}")
        
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        
        if sheet_names is None:
            sheet_names = excel_file.sheet_names
        
        logger.info(f"Found {len(sheet_names)} sheet(s): {', '.join(sheet_names)}")
        
        results = []
        self.conn = sqlite3.connect(self.db_path)
        
        try:
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Generate table name from sheet name
                    table_name = self._sanitize_table_name(sheet_name)
                    
                    df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
                    
                    stats = {
                        'file': str(file_path),
                        'sheet_name': sheet_name,
                        'table_name': table_name,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': list(df.columns)
                    }
                    
                    results.append(stats)
                    logger.info(f"✓ Loaded sheet '{sheet_name}' as table '{table_name}' ({len(df)} rows)")
                    
                except Exception as e:
                    logger.error(f"Failed to load sheet '{sheet_name}': {e}")
                    results.append({
                        'sheet_name': sheet_name,
                        'error': str(e),
                        'status': 'failed'
                    })
        finally:
            self.conn.close()
        
        return results
    
    def get_database_info(self) -> Dict[str, any]:
        """
        Get information about the current database.
        
        Returns:
            Dictionary with database statistics
        """
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        table_info = []
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [{'name': row[1], 'type': row[2]} for row in cursor.fetchall()]
            
            table_info.append({
                'table_name': table,
                'row_count': row_count,
                'column_count': len(columns),
                'columns': columns
            })
        
        self.conn.close()
        
        return {
            'database_path': self.db_path,
            'table_count': len(tables),
            'tables': table_info
        }


def setup_database_from_files(
    source_path: str,
    db_path: str = "./data/database.db",
    if_exists: str = 'replace',
    table_name: Optional[str] = None
) -> Dict[str, any]:
    """
    Convenience function to set up database from files or directory.
    
    Args:
        source_path: Path to file or directory containing data files
        db_path: Path to SQLite database
        if_exists: What to do if tables exist ('fail', 'replace', 'append')
        table_name: Optional explicit table name (auto-generated from filename if None).
                    Only applied when loading a single file.
        
    Returns:
        Summary of loaded data
    """
    loader = DataLoader(db_path)
    source = Path(source_path)
    
    if source.is_file():
        # Single file
        if source.suffix.lower() in ['.xlsx', '.xls']:
            # Check if it has multiple sheets
            excel_file = pd.ExcelFile(source)
            if len(excel_file.sheet_names) > 1 and table_name is None:
                logger.info(f"Loading Excel file with {len(excel_file.sheet_names)} sheets")
                results = loader.load_excel_sheets(source, if_exists=if_exists)
            else:
                # Single sheet or explicit table name override
                results = [loader.load_file(source, table_name=table_name, if_exists=if_exists)]
        else:
            results = [loader.load_file(source, table_name=table_name, if_exists=if_exists)]
    elif source.is_dir():
        # Directory – table_name override is ignored for directories
        if table_name:
            logger.warning("table_name is ignored when loading from a directory")
        results = loader.load_directory(source, if_exists=if_exists)
    else:
        raise ValueError(f"Invalid source path: {source_path}")
    
    # Get database info
    db_info = loader.get_database_info()
    
    logger.info("="*60)
    logger.info("DATABASE SETUP COMPLETE")
    logger.info("="*60)
    logger.info(f"Database: {db_path}")
    logger.info(f"Tables created: {db_info['table_count']}")
    for table in db_info['tables']:
        logger.info(f"  • {table['table_name']}: {table['row_count']} rows, {table['column_count']} columns")
    
    return {
        'load_results': results,
        'database_info': db_info
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <path_to_file_or_directory>")
        print("\nExample:")
        print("  python data_loader.py ./data/sales.csv")
        print("  python data_loader.py ./data/excel_files/")
        sys.exit(1)
    
    source_path = sys.argv[1]
    setup_database_from_files(source_path)
