
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from core.db_manager import get_db_manager
from core.postgres_manager import PostgresManager

def check_duplicates():
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    db = get_db_manager(test_mode=False)
    
    if not isinstance(db, PostgresManager):
        print("Not using PostgresManager")
        return

    clients_to_check = ['saddle_demo2', 'digiaansh_test']
    
    print(f"Checking for clients: {clients_to_check}")
    
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            # Check existence
            cursor.execute("SELECT DISTINCT client_id FROM actions_log WHERE client_id IN %s", (tuple(clients_to_check),))
            found_clients = [row[0] for row in cursor.fetchall()]
            print(f"Found in actions_log: {found_clients}")
            
            # Compare record counts
            for client in clients_to_check:
                cursor.execute("SELECT COUNT(*) FROM actions_log WHERE client_id = %s", (client,))
                act_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM target_stats WHERE client_id = %s", (client,))
                target_count = cursor.fetchone()[0]
                
                print(f"Client {client}: Actions={act_count}, TargetStats={target_count}")
                
            # Compare data content similarity (hash comparison of sorted actions)
            if len(clients_to_check) == 2:
                 c1, c2 = clients_to_check
                 
                 # Compare Action Dates/Types
                 cursor.execute("SELECT action_date, action_type, old_value, new_value FROM actions_log WHERE client_id = %s ORDER BY action_date, target_text", (c1,))
                 data1 = cursor.fetchall()
                 
                 cursor.execute("SELECT action_date, action_type, old_value, new_value FROM actions_log WHERE client_id = %s ORDER BY action_date, target_text", (c2,))
                 data2 = cursor.fetchall()
                 
                 if data1 == data2 and len(data1) > 0:
                     print(f"MATCH: {c1} and {c2} have IDENTICAL actions log content.")
                 elif len(data1) == len(data2):
                     print(f"SIMILAR: {c1} and {c2} have same row count but different content.")
                 else:
                     print(f"DIFFERENT: {c1} and {c2} are different.")

if __name__ == "__main__":
    check_duplicates()
