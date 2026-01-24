import os
import psycopg2
import sys

def get_db_url():
    # Try reading .env file
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('DATABASE_URL='):
                    return line.strip().split('=', 1)[1].strip('"\'')
    return os.environ.get('DATABASE_URL')

def main():
    print("Checking last action date directly via psycopg2...")
    
    db_url = get_db_url()
    if not db_url:
        print("Error: Could not find DATABASE_URL in .env or environment variables")
        return

    print("Connected to DB URL (masked): " + db_url[:20] + "...")
    
    client_id = 's2c_uae_test'
    
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Check actions_log
        print(f"Querying actions_log for client: {client_id}")
        cursor.execute("SELECT MAX(action_date) FROM actions_log WHERE client_id = %s", (client_id,))
        res = cursor.fetchone()
        
        if res and res[0]:
            print(f"✅ MAX action_date: {res[0]}")
        else:
            print("❌ No actions found for this client.")
            
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
