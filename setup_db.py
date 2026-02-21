"""
Setup script to initialize database from CSV/Excel files.
Can be run from command line or imported as a module.
"""

import argparse
from pathlib import Path
from core.data_loader import DataLoader, setup_database_from_files
from loguru import logger
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Set up SQLite database from CSV/Excel files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load a single CSV file
  python setup_db.py --source ./data/sales.csv
  
  # Load an Excel file with multiple sheets
  python setup_db.py --source ./data/company_data.xlsx
  
  # Load all CSV/Excel files from a directory
  python setup_db.py --source ./data/csv_files/
  
  # Specify custom database path
  python setup_db.py --source ./data/sales.csv --db ./my_database.db
  
  # Append to existing tables instead of replacing
  python setup_db.py --source ./data/new_sales.csv --mode append
        """
    )
    
    parser.add_argument(
        '--source',
        '-s',
        required=True,
        help='Path to CSV/Excel file or directory containing data files'
    )
    
    parser.add_argument(
        '--db',
        '-d',
        default='./data/database.db',
        help='Path to SQLite database file (default: ./data/database.db)'
    )
    
    parser.add_argument(
        '--mode',
        '-m',
        choices=['replace', 'append', 'fail'],
        default='replace',
        help='What to do if table exists (default: replace)'
    )
    
    parser.add_argument(
        '--table-name',
        '-t',
        default='nielsen_pos',
        help='Explicit table name for the loaded data (default: nielsen_pos)'
    )

    parser.add_argument(
        '--show-info',
        action='store_true',
        help='Show database information after loading'
    )
    
    args = parser.parse_args()
    
    try:
        # Setup database
        logger.info("Starting database setup...")
        result = setup_database_from_files(
            source_path=args.source,
            db_path=args.db,
            if_exists=args.mode,
            table_name=args.table_name
        )
        
        # Show summary
        print("\n" + "="*70)
        print("✓ DATABASE SETUP SUCCESSFUL")
        print("="*70)
        
        db_info = result['database_info']
        print(f"\nDatabase Location: {db_info['database_path']}")
        print(f"Total Tables: {db_info['table_count']}")
        
        print("\nTables Created:")
        for table in db_info['tables']:
            print(f"  • {table['table_name']:<30} {table['row_count']:>8} rows  {table['column_count']:>3} columns")
        
        if args.show_info:
            print("\nDetailed Table Information:")
            for table in db_info['tables']:
                print(f"\n  Table: {table['table_name']}")
                print(f"  Columns:")
                for col in table['columns']:
                    print(f"    - {col['name']:<30} {col['type']}")
        
        print("\n" + "="*70)
        print("Next Steps:")
        print("  1. Update your .env file:")
        print(f"     DATABASE_URI=sqlite:///{args.db}")
        print("\n  2. Run the application:")
        print("     streamlit run app.py")
        print("\n  3. Or test with:")
        print("     python examples.py")
        print("="*70)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n✗ Error: {e}")
        print("\nPlease check that the file or directory exists.")
        return 1
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
