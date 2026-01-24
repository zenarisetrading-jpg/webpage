
# ==========================================
# Run Login Test
# ==========================================
import os
from core.auth.service import AuthService

# Logic
auth = AuthService()
email = "admin@example.com"
password = "password123"

print("🔍 Testing V2 Authentication...")

# 1. Positive Test
print(f"\n1. Attempting login for {email}...")
result = auth.sign_in(email, password)

if result["success"]:
    user = result["user"]
    print("✅ Login SUCCESS")
    print(f"   User: {user.email}")
    print(f"   Role: {user.role}")
    print(f"   Org: {user.organization_id}")
    
    # Verify Type
    from core.auth.models import User
    if isinstance(user, User):
        print("✅ Type Check Passed (core.auth.models.User)")
    else:
        print(f"❌ Type Check Failed: Got {type(user)}")
else:
    print(f"❌ Login Failed: {result.get('error')}")
    exit(1)

# 2. Negative Test
print(f"\n2. Attempting login with WRONG password...")
bad_result = auth.sign_in(email, "wrongpassword")
if not bad_result["success"]:
    print("✅ Negative Test Passed (Login rejected)")
else:
    print("❌ Negative Test Failed (Login accepted with bad password!)")
    exit(1)

print("\n✨ Auth Service Verified.")
