
import os
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables
load_dotenv()
load_dotenv('.env')

# DB Connection
try:
    import psycopg2
except ImportError:
    try:
        import psycopg as psycopg2
    except ImportError:
        print("❌ Error: No postgres driver found. Install psycopg2-binary or psycopg[binary]")
        exit(1)

def run_migration():
    print("🚀 Starting Migration 003: Password Security...")
    
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ Error: DATABASE_URL not found in .env")
        return

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Read SQL file
        sql_path = Path("migrations/003_add_password_security.sql")
        print(f"📄 Reading {sql_path}...")
        with open(sql_path, "r") as f:
            sql = f.read()
            
        print("⚡ Executing SQL...")
        cur.execute(sql)
        conn.commit()
        
        print("✅ Migration 003 Complete: Added 'must_reset_password' and 'password_updated_at'.")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Migration Failed: {e}")

if __name__ == "__main__":
    run_migration()
