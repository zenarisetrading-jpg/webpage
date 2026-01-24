
import os
import sys
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

def get_connection():
    try:
        import psycopg
        # print("Using psycopg (v3)")
        return psycopg.connect(DB_URL)
    except ImportError:
        try:
            import psycopg2
            # print("Using psycopg2")
            return psycopg2.connect(DB_URL)
        except ImportError:
            print("No postgres driver found!")
            sys.exit(1)

def check_tables():
    if not DB_URL:
        print("DATABASE_URL missing")
        return

    try:
        conn = get_connection()
        cur = conn.cursor()

        print("=== ORGANIZATIONS ===")
        try:
            cur.execute("SELECT id, name FROM organizations")
            orgs = cur.fetchall()
            for o in orgs:
                # v3 returns tuples by default unless row_factory set
                print(f"{o[0]} | {o[1]}")
        except Exception as e:
            print(f"Error reading orgs: {e}")
            conn.rollback()

        print("\n=== LEGACY ACCOUNTS ===")
        try:
            cur.execute("SELECT account_id, account_name FROM accounts")
            accts = cur.fetchall()
            for a in accts:
                print(f"{a[0]} | {a[1]}")
        except Exception as e:
            print(f"Error reading accounts: {e}")
            conn.rollback()
            
        print("\n=== NEW AMAZON_ACCOUNTS ===")
        try:
            cur.execute("SELECT id, display_name, organization_id FROM amazon_accounts")
            new_accts = cur.fetchall()
            if not new_accts:
                print("(Empty)")
            for a in new_accts:
                print(f"{a[0]} | {a[1]} -> {a[2]}")
        except Exception as e:
            print(f"Error reading amazon_accounts: {e}")
            conn.rollback()
            
        conn.close()

    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    check_tables()
