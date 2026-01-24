import os
import uuid
from dotenv import load_dotenv

# DB Connection Shim
try:
    import psycopg2
except ImportError:
    try:
        import psycopg as psycopg2
    except ImportError:
        print("❌ Error: No postgres driver found. Install psycopg2-binary or psycopg[binary]")
        exit(1)
load_dotenv()
load_dotenv('.env')

def test_db_constraints():
    print("🧪 Testing Phase 3.5 DB Constraints (Downgrade Only)...")
    
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ Error: DATABASE_URL not found")
        return

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    try:
        # 1. Setup Test Data
        org_id = str(uuid.uuid4())
        account_id = str(uuid.uuid4())
        
        # Create Org
        cur.execute("INSERT INTO organizations (id, name, type, subscription_plan) VALUES (%s, 'Test Org', 'AGENCY', 'agaye_tier1')", (org_id,))
        
        # Create Account
        cur.execute("INSERT INTO amazon_accounts (id, organization_id, display_name, marketplace, status) VALUES (%s, %s, 'Test Acct', 'US', 'ACTIVE')", (account_id, org_id))
        
        # Create Users with different roles
        viewer_id = str(uuid.uuid4())
        operator_id = str(uuid.uuid4())
        admin_id = str(uuid.uuid4())
        
        print("  • Creating test users...")
        cur.execute("INSERT INTO users (id, organization_id, email, password_hash, role, billable, status) VALUES (%s, %s, 'viewer@test.com', 'hash', 'VIEWER', true, 'ACTIVE')", (viewer_id, org_id))
        cur.execute("INSERT INTO users (id, organization_id, email, password_hash, role, billable, status) VALUES (%s, %s, 'operator@test.com', 'hash', 'OPERATOR', true, 'ACTIVE')", (operator_id, org_id))
        cur.execute("INSERT INTO users (id, organization_id, email, password_hash, role, billable, status) VALUES (%s, %s, 'admin@test.com', 'hash', 'ADMIN', true, 'ACTIVE')", (admin_id, org_id))
        
        conn.commit()
        
        # 2. Test Valid Overrides (Downgrades)
        print("  • Testing valid overrides (Downgrades)...")
        
        # ADMIN -> OPERATOR (Valid)
        cur.execute("INSERT INTO user_account_overrides (user_id, amazon_account_id, role) VALUES (%s, %s, 'OPERATOR')", (admin_id, account_id))
        print("    ✅ ADMIN -> OPERATOR allowed")
        
        # OPERATOR -> VIEWER (Valid)
        cur.execute("INSERT INTO user_account_overrides (user_id, amazon_account_id, role) VALUES (%s, %s, 'VIEWER')", (operator_id, account_id))
        print("    ✅ OPERATOR -> VIEWER allowed")
        
        # 3. Test Invalid Overrides (Elevations)
        print("  • Testing invalid overrides (Elevations)...")
        
        # VIEWER -> OPERATOR (Invalid)
        try:
            cur.execute("INSERT INTO user_account_overrides (user_id, amazon_account_id, role) VALUES (%s, %s, 'OPERATOR')", (viewer_id, account_id))
            print("    ❌ FAILED: VIEWER -> OPERATOR was NOT blocked!")
        except psycopg2.DatabaseError as e:
            conn.rollback()
            print("    ✅ Blocked: VIEWER -> OPERATOR rejected by DB")
            
        # OPERATOR -> ADMIN (Invalid - check constraint handles valid values, ADMIN is not in check constraint list so fails immediately on value check)
        try:
            cur.execute("INSERT INTO user_account_overrides (user_id, amazon_account_id, role) VALUES (%s, %s, 'ADMIN')", (operator_id, account_id))
            print("    ❌ FAILED: OPERATOR -> ADMIN was NOT blocked!")
        except psycopg2.DatabaseError as e:
            conn.rollback()
            print("    ✅ Blocked: ADMIN role rejected (not in enum)")

        # 4. Cleanup
        print("  • Cleaning up...")
        cur.execute("DELETE FROM organizations WHERE id = %s CASCADE", (org_id,))
        conn.commit()
        
        print("\n🎉 ALL DB CONSTRAINT TESTS PASSED!")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    test_db_constraints()
