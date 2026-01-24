
import sys
import sqlite3
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from core.db_manager import get_db_manager

def debug_database():
    print("Debug Database Content")
    db_path = Path("data/ppc_test.db")
    
    if not db_path.exists():
        print(f"File {db_path} does not exist.")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    tables = ['weekly_stats', 'target_stats', 'actions_log', 'accounts']
    
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"Table '{table}': {count} rows")
            
            if count > 0:
                cursor.execute(f"SELECT * FROM {table} LIMIT 1")
                cols = [description[0] for description in cursor.description]
                print(f"  Columns: {cols}")
                print(f"  Sample: {cursor.fetchone()}")

                if table == 'weekly_stats' or table == 'target_stats':
                    cursor.execute(f"SELECT DISTINCT client_id FROM {table} LIMIT 5")
                    clients = cursor.fetchall()
                    print(f"  Clients: {clients}")
                    
        except Exception as e:
            print(f"Error reading {table}: {e}")
            
    conn.close()

if __name__ == "__main__":
    debug_database()
