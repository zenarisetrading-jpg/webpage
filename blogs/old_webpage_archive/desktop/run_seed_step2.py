
# ==========================================
# Run Seed Script
# ==========================================
import os
import psycopg2
from core.auth.service import AuthService
from core.auth.permissions import Role
from core.postgres_manager import PostgresManager

# Load env variables
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Setup Direct DB Connection since AuthService needs one for Manual Seed
db_url = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_DB_URL") 
if not db_url:
    print("❌ Error: DB URL not found")
    exit(1)

print("🌱 Seeding Database...")

try:
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    # 1. Create Organization
    org_id = '00000000-0000-0000-0000-000000000001' # Fixed UUID for reliability
    
    # Check existence
    cur.execute("SELECT id FROM organizations WHERE id = %s", (org_id,))
    if not cur.fetchone():
        print("Creating Organization...")
        cur.execute("""
            INSERT INTO organizations (id, name, type, amazon_account_limit)
            VALUES (%s, 'Demo Agency', 'AGENCY', 10)
        """, (org_id,))
    else:
        print("Organization exists.")

    conn.commit()
    conn.close()

    # 2. Create User via Service (Handles hashing)
    auth = AuthService()
    email = "admin@example.com"
    password = "password123"
    
    # Check if user exists
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = %s", (email,))
    
    if not cur.fetchone():
        print(f"Creating User {email}...")
        success = auth.create_user_manual(email, password, Role.OWNER, org_id)
        if success:
            print("✅ User created successfully.")
        else:
            print("❌ Failed to create user.")
    else:
        print("User exists.")
        
    cur.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Seed failed: {e}")
    exit(1)
