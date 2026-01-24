"""
Auth Service (V2)
=================
Handles user authentication against the V2 custom schema.
PRD Reference: ORG_USERS_ROLES_PRD.md §12

Replaces legacy Supabase Auth (auth/service.py).
"""

import os
import os
import re
try:
    import streamlit as st
except ImportError:
    class MockSt:
        session_state = {}
    st = MockSt()

from typing import Optional, Dict, Any
from core.auth.models import User, Role, PasswordChangeResult
from core.auth.hashing import verify_password, hash_password
from core.postgres_manager import PostgresManager

# DB Driver Shim: Try psycopg2, fall back to psycopg (v3)
try:
    import psycopg2
    import psycopg2.errors
except ImportError:
    try:
        import psycopg as psycopg2 # Alias v3 to v2 name
        # V3 doesn't have 'errors' submodule in same way, map it
        import psycopg.errors
        psycopg2.errors = psycopg.errors
    except ImportError:
        raise ImportError("No Postgres driver found. Install psycopg2-binary or psycopg[binary].")

# Load env variables (robustness)
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


class AuthService:
    """
    V2 Authentication Service.
    Uses 'users' table in core Postgres schema.
    """
    
    def __init__(self):
        # Use existing PostgresManager connection logic if possible, 
        # but for Auth we prefer a direct tight loop or reuse the pool.
        # For simplicity in Phase 2, we reuse PostgresManager's connection params.
        self.db_url = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_DB_URL")
        # Reuse existing manager if we want pool benefits (TODO for Phase 3 optimization)
        
    def _get_connection(self):
        return psycopg2.connect(self.db_url)

    def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user by email and password.
        Sets session state on success.
        Returns: {success: bool, user: User, error: str}
        """
        if not email or not password:
             return {"success": False, "error": "Email and password required"}
             
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Fetch user + password_hash
            query = """
                SELECT id, organization_id, email, password_hash, role, billable, status, must_reset_password, password_updated_at
                FROM users 
                WHERE email = %s AND status = 'ACTIVE';
            """
            cur.execute(query, (email.lower().strip(),))
            row = cur.fetchone()
            
            cur.close()
            conn.close()
            
            if not row:
                # Generic error for security
                return {"success": False, "error": "Invalid credentials"}
                
            (uid, org_id, db_email, db_hash, role_str, billable, status, must_reset, pwd_updated) = row
            
            # Verify Password
            if not verify_password(password, db_hash):
                return {"success": False, "error": "Invalid credentials"}
                
            # Construct User Model
            # Load Overrides first
            account_overrides = {}
            try:
                cur = conn.cursor() # Re-open cursor if needed or better reuse logic above
                cur.execute("""
                    SELECT amazon_account_id, role 
                    FROM user_account_overrides 
                    WHERE user_id = %s
                """, (uid,))
                for row_ov in cur.fetchall():
                    # Parse UUID and Role
                    acc_id = UUID(str(row_ov[0])) # Ensure string if driver returns uuid
                    ov_role = Role(row_ov[1])
                    account_overrides[acc_id] = ov_role
            except Exception as e:
                print(f"Warning: Failed to load overrides: {e}")
                # Non-critical, continue login
            
            user = User(
                id=uid,
                organization_id=org_id,
                email=db_email,
                password_hash="REDACTED", # Don't keep hash in memory object
                role=Role(role_str),
                billable=billable,
                status=status,
                created_at=None, # Not needed for session usually
                must_reset_password=must_reset,
                password_updated_at=pwd_updated,
                account_overrides=account_overrides
            )
            
            # SESSION STORAGE (Canonical)
            st.session_state["user"] = user
            
            return {"success": True, "user": user}
            
        except Exception as e:
            print(f"Auth Error: {e}")
            return {"success": False, "error": "System error during login"}

    def get_current_user(self) -> Optional[User]:
        """
        Get the currently authenticated user from session state.
        Type-safe accessor.
        """
        user = st.session_state.get("user")
        if user and isinstance(user, User):
            return user
        return None

    def sign_out(self):
        """Clear user session."""
        if "user" in st.session_state:
            del st.session_state["user"]

    def create_user_manual(self, email: str, password: str, role: Role, org_id: str) -> bool:
        """
        Helper for SEEDING only. Not for public registration.
        """
        try:
            hashed = hash_password(password)
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO users (email, password_hash, role, organization_id, billable)
                VALUES (%s, %s, %s, %s, %s)
            """, (email.lower(), hashed, role.value, org_id, True))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Create Error: {e}")
            return False

    def list_users(self, organization_id: str) -> list[Dict[str, Any]]:
        """
        List all users in an organization.
        Returns list of dicts: {id, email, role, status, billable}
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT id, email, role, status, billable 
                FROM users 
                WHERE organization_id = %s
                ORDER BY created_at DESC
            """, (str(organization_id),))
            
            rows = cur.fetchall()
            cur.close()
            conn.close()
            
            users = []
            for r in rows:
                users.append({
                    "id": r[0],
                    "email": r[1],
                    "role": r[2],
                    "status": r[3],
                    "billable": r[4]
                })
            return users
            
        except Exception as e:
            print(f"List Users Error: {e}")
            return []

    def create_user_invite(self, email: str, role: Role, org_id: str, temp_password: str = "Welcome123!") -> Dict[str, Any]:
        """
        Create a new user (Invite flow).
        Since we don't have email sending yet, we set a temp password.
        Returns: {success: bool, error: str, temp_password: str}
        """
        if not email or not role or not org_id:
            return {"success": False, "error": "Missing fields"}

        try:
            hashed = hash_password(temp_password)
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Check for existing
            cur.execute("SELECT id FROM users WHERE email = %s", (email.lower(),))
            if cur.fetchone():
                return {"success": False, "error": "User already exists"}
            
            # Insert
            # Determine billable? (Simplified: Use default from role)
            from core.auth.permissions import get_billable_default
            billable = get_billable_default(role.value)
            
            cur.execute("""
                INSERT INTO users (email, password_hash, role, organization_id, billable, status)
                VALUES (%s, %s, %s, %s, %s, 'ACTIVE')
            """, (email.lower(), hashed, role.value, org_id, billable))
            
            conn.commit()
            conn.close()
            
            # Return instructions for the admin
            return {"success": True, "temp_password": temp_password}
            
        except Exception as e:
            print(f"Invite Error: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # PHASE 3.5: ACCOUNT ACCESS OVERRIDES
    # =========================================================================

    def update_user_role(self, user_id: str, new_role: Role, updated_by_user_id: str) -> Dict[str, Any]:
        """
        Update user's global role and auto-cleanup invalid overrides.
        
        Logic:
        1. Update global role.
        2. Clean up any overrides that violate "Override <= Global" rule.
           (e.g. if downgraded to VIEWER, remove OPERATOR overrides).
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # 1. Update Global Role
            cur.execute("""
                UPDATE users 
                SET role = %s 
                WHERE id = %s
            """, (new_role.value, str(user_id)))
            
            # 2. Auto-cleanup: invalid overrides logic
            # If new role is VIEWER, they cannot have OPERATOR overrides.
            if new_role == Role.VIEWER:
                cur.execute("""
                    DELETE FROM user_account_overrides
                    WHERE user_id = %s AND role = 'OPERATOR'
                """, (str(user_id),))
            
            # Note: If new role is OPERATOR/ADMIN/OWNER, existing VIEWER overrides remain valid.
            
            conn.commit()
            conn.close()
            return {"success": True}
            
        except Exception as e:
            print(f"Update Role Error: {e}")
            return {"success": False, "error": str(e)}

    def set_account_override(self, user_id: str, account_id: str, override_role: Role, set_by_user_id: str) -> Dict[str, Any]:
        """
        Set or update an account access override.
        """
        # Validate allowed override roles (Phase 3.5: VIEWER/OPERATOR only)
        if override_role not in [Role.VIEWER, Role.OPERATOR]:
            return {"success": False, "error": "Overrides can only be VIEWER or OPERATOR"}

        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # 1. Fetch user's global role validation (Application level double-check)
            cur.execute("SELECT role FROM users WHERE id = %s", (str(user_id),))
            row = cur.fetchone()
            if not row:
                conn.close()
                return {"success": False, "error": "User not found"}
                
            global_role_str = row[0]
            from core.auth.permissions import ROLE_HIERARCHY_STR
            
            global_level = ROLE_HIERARCHY_STR.get(global_role_str, 0)
            override_level = ROLE_HIERARCHY_STR.get(override_role.value, 0)
            
            # Verify downgrade-only rule
            if override_level > global_level:
                conn.close()
                return {"success": False, "error": "Cannot set override higher than global role"}
            
            # 2. Upsert Override
            cur.execute("""
                INSERT INTO user_account_overrides (user_id, amazon_account_id, role, created_by)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, amazon_account_id)
                DO UPDATE SET role = EXCLUDED.role;
            """, (str(user_id), str(account_id), override_role.value, str(set_by_user_id)))
            
            conn.commit()
            conn.close()
            return {"success": True}
            
        except Exception as e:
            print(f"Set Override Error: {e}")
            # Catch DB constraint violations
            if "check_override_downgrade" in str(e):
                 return {"success": False, "error": "Database rejected: Override cannot exceed global role"}
            return {"success": False, "error": str(e)}

    def remove_account_override(self, user_id: str, account_id: str) -> Dict[str, Any]:
        """Remove an access override, reverting to global role."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                DELETE FROM user_account_overrides 
                WHERE user_id = %s AND amazon_account_id = %s
            """, (str(user_id), str(account_id)))
            
            conn.commit()
            conn.close()
            return {"success": True}
        except Exception as e:
            print(f"Remove Override Error: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # PHASE 3: SECURITY & PASSWORD MANAGEMENT
    # =========================================================================

    def validate_password_strength(self, password: str) -> bool:
        """
        Policy: Min 8 chars, at least 1 number or special char.
        """
        if len(password) < 8:
            return False
        # Regex: At least one digit OR one special char
        if not re.search(r'[0-9!@#$%^&*(),.?":{}|<>]', password):
            return False
        return True

    def change_password(self, user_id: str, old_password: str, new_password: str) -> PasswordChangeResult:
        """
        Self-service password change.
        Verifies old password, validates new strength, and updates DB.
        """
        if not self.validate_password_strength(new_password):
            return PasswordChangeResult(False, "Password too weak. Must be 8+ chars with a number or symbol.")
            
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # 1. Verify Old Password
            cur.execute("SELECT password_hash FROM users WHERE id = %s", (str(user_id),))
            row = cur.fetchone()
            if not row:
                conn.close()
                return PasswordChangeResult(False, "User not found.")
                
            db_hash = row[0]
            if not verify_password(old_password, db_hash):
                conn.close()
                return PasswordChangeResult(False, "Current password incorrect.")
                
            # 2. Update to New Password
            new_hash = hash_password(new_password)
            cur.execute("""
                UPDATE users 
                SET password_hash = %s, 
                    password_updated_at = NOW(), 
                    must_reset_password = FALSE 
                WHERE id = %s
            """, (new_hash, str(user_id)))
            
            conn.commit()
            conn.close()
            
            return PasswordChangeResult(True, "Password updated successfully.")
            
        except Exception as e:
            print(f"Change Password Error: {e}")
            return PasswordChangeResult(False, "System error.")

    def admin_reset_password(self, admin_user: User, target_user_id: str) -> PasswordChangeResult:
        """
        Admin-assisted recovery.
        Generates a temp password and forces reset on next login.
        """
        # Security: Prevent Admin from resetting Owner
        if admin_user.role == Role.ADMIN:
            # Check target role
            # We need to fetch target role first
            pass # TODO: Optimize query to check role inside SQL or fetch first
            
        temp_password = "PPC-" + os.urandom(4).hex() + "!" # e.g. PPC-a1b2c3d4!
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # 1. Fetch Target Role (for protection)
            cur.execute("SELECT role FROM users WHERE id = %s", (str(target_user_id),))
            row = cur.fetchone()
            if not row:
                conn.close()
                return PasswordChangeResult(False, "Target user not found.")
                
            target_role = row[0]
            
            # RULE: STRICT HIERARCHY CHECK
            from core.auth.permissions import can_manage_role
            
            # handle enum vs string mismatch if any
            manager_role_str = admin_user.role.value if hasattr(admin_user.role, 'value') else str(admin_user.role)
            
            if not can_manage_role(manager_role_str, target_role):
                conn.close()
                return PasswordChangeResult(False, f"Insufficient privileges. {manager_role_str} cannot manage {target_role}.")
            
            # 2. Update
            new_hash = hash_password(temp_password)
            cur.execute("""
                UPDATE users 
                SET password_hash = %s, 
                    must_reset_password = TRUE 
                WHERE id = %s
            """, (new_hash, str(target_user_id)))
            
            conn.commit()
            conn.close()
            
            return PasswordChangeResult(True, temp_password)
            
        except Exception as e:
            print(f"Admin Reset Error: {e}")
            return PasswordChangeResult(False, str(e))

    def request_password_reset(self, email: str) -> bool:
        """
        Public endpoint for 'Forgot Password'.
        In a real system, this would generate a token and email it.
        Here, we just verify existence (silently) and return True to prevent enumeration (or simulated success).
        """
        if not email:
            return False
            
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE email = %s", (email.lower().strip(),))
            exists = cur.fetchone() is not None
            if exists:
                # 1. Generate Secure Temp Password
                # e.g. PPC-a1b2c3d4!
                temp_password = "PPC-" + os.urandom(4).hex() + "!"
                
                # 2. Update DB with new hash + forced reset flag
                new_hash = hash_password(temp_password)
                
                # Re-connect to update
                # (Optimization: We could have done SELECT FOR UPDATE above, but keeping it simple)
                conn = self._get_connection()
                cur = conn.cursor()
                cur.execute("""
                    UPDATE users 
                    SET password_hash = %s, 
                        must_reset_password = TRUE,
                        password_updated_at = NOW()
                    WHERE email = %s
                """, (new_hash, email.lower().strip()))
                conn.commit()
                conn.close()
                
                # 3. Send Email
                from utils.email_sender import EmailSender
                sender = EmailSender()
                
                subject = "Password Reset - Saddle"
                html = f"""
                <div style="font-family: sans-serif; color: #333;">
                    <h2>Password Reset</h2>
                    <p>Your temporary password is:</p>
                    <p style="font-size: 18px; font-weight: bold; background: #f4f4f5; padding: 10px; border-radius: 6px; display: inline-block;">
                        {temp_password}
                    </p>
                    <p>Please log in with this password. You will be asked to create a new one immediately.</p>
                    <br>
                    <small>If you did not request this, please contact support.</small>
                </div>
                """
                
                success = sender.send_email(email, subject, html)
                if success:
                    print(f"Password reset email sent to {email}")
                else:
                    print(f"Failed to send email to {email}")
                
            return True
            
        except Exception as e:
            print(f"Reset Request Error: {e}")
            return False
