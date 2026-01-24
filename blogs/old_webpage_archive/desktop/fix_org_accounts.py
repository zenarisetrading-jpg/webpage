
import os
import sys
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

# Target User (from screenshot)
TARGET_EMAIL = "aslam.yousuf@gmail.com"

def get_connection():
    try:
        import psycopg
        return psycopg.connect(DB_URL)
    except ImportError:
        import psycopg2
        return psycopg2.connect(DB_URL)

def run_fix():
    if not DB_URL:
        print("DATABASE_URL missing")
        return

    try:
        conn = get_connection()
        cur = conn.cursor()

        # 1. Get Org ID
        print(f"Finding Org for {TARGET_EMAIL}...")
        cur.execute("SELECT organization_id FROM users WHERE email = %s", (TARGET_EMAIL,))
        row = cur.fetchone()
        if not row:
            print("User not found!")
            return
        
        org_id = row[0]
        print(f"Found Org ID: {org_id}")

        # 2. Get Legacy Accounts
        print("Fetching legacy accounts...")
        cur.execute("SELECT account_id, account_name, account_type FROM accounts")
        legacy_accts = cur.fetchall()
        
        if not legacy_accts:
            print("No legacy accounts found.")
            return

        # 3. Insert into amazon_accounts
        count = 0
        for acc in legacy_accts:
            # v3 tuples: (id, name, type)
            legacy_id = acc[0] # String "s2c_test"
            name = acc[1]
            marketplace = 'US' 
            
            # Check if *Display Name* exists (since that maps to legacy ID)
            cur.execute("SELECT id FROM amazon_accounts WHERE display_name = %s AND organization_id = %s", (name, org_id))
            if cur.fetchone():
                print(f"Skipping {name} (already exists)")
                continue

            print(f"Linking {name} (Legacy: {legacy_id}) to Org {org_id}...")
            # Let DB generate UUID for 'id'
            cur.execute("""
                INSERT INTO amazon_accounts (organization_id, display_name, marketplace, status)
                VALUES (%s, %s, %s, 'ACTIVE')
            """, (org_id, name, marketplace))
            count += 1
            
        conn.commit()
        print(f"Success! Linked {count} accounts.")
        conn.close()

    except Exception as e:
        print(f"Fix failed: {e}")

if __name__ == "__main__":
    run_fix()
