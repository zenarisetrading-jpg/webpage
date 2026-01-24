
# ==========================================
# Run Migration Script
# ==========================================
import os
import psycopg2
from core.postgres_manager import PostgresManager

from pathlib import Path
from dotenv import load_dotenv

env_path = Path(".env")
load_dotenv(dotenv_path=env_path)

migration_file = Path("migrations/002_org_users_schema.sql")
# Try DATABASE_URL first (canonical from .env), then SUPABASE_DB_URL
db_url = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_DB_URL")

if not db_url:
    print("❌ Error: SUPABASE_DB_URL not found in environment")
    exit(1)

print(f"🔄 Executing migration: {migration_file}")

try:
    with open(migration_file, 'r') as f:
        sql = f.read()
        
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    print("✅ Migration executed successfully.")
    
    # VERIFICATION
    print("\n🔍 Verifying Tables:")
    cur.execute("SELECT COUNT(*) FROM organizations;")
    org_count = cur.fetchone()[0]
    print(f"- Organizations: {org_count}")
    
    cur.execute("SELECT COUNT(*) FROM users;")
    user_count = cur.fetchone()[0]
    print(f"- Users: {user_count}")
    
    cur.execute("SELECT COUNT(*) FROM amazon_accounts;")
    account_count = cur.fetchone()[0]
    print(f"- Amazon Accounts: {account_count}")
    
    cur.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Migration failed: {e}")
    exit(1)
