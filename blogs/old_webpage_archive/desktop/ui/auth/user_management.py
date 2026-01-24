"""
User Management UI
==================
Screen for managing users, roles, and invites.
PRD Reference: ORG_USERS_ROLES_PRD.md §11, §13

Features:
- List users
- Invite user (with billing warning)
- Role assignment
"""

import streamlit as st
import streamlit as st
from core.auth.permissions import Role, PERMISSION_MATRIX, get_billable_default, can_manage_role, has_permission
from core.auth.middleware import require_permission

# Example price (could come from config/DB)
SEAT_PRICE = 49.00

def render_user_management():
    st.header("User Management")
    
    # NUCLEAR CSS OVERRIDE - Dark Vine Theme
    st.markdown("""
    <style>
    /* Force Dark Vine on ALL primary buttons */
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    div[data-testid="stForm"] button[kind="primary"] {
        background: linear-gradient(135deg, #464156 0%, #2E2A36 100%) !important;
        color: #E9EAF0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #5B5670 0%, #464156 100%) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Dark Vine Alert/Warning Ribbons */
    div[data-testid="stAlert"] {
        background-color: rgba(46, 42, 54, 0.95) !important;
        border: 1px solid rgba(154, 154, 170, 0.2) !important;
        color: #E9EAF0 !important;
    }
    div[data-testid="stAlert"] * {
        color: #E9EAF0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # V2 Backend Wiring
    from core.auth.service import AuthService
    from core.auth.service import AuthService
    auth = AuthService()
    current_user = auth.get_current_user()
    
    # 0. Page Access Guard
    if not current_user or not has_permission(current_user.role, 'manage_users'):
        st.error("⛔ Access Denied: Administrator privileges required.")
        return
    
    # 1. User List (Real Data)
    st.subheader("Team Members")
    
    
    # helper for Phase 3.5 UI
    def render_account_override_editor(user):
        """Show per-account access restrictions for a user."""
        st.write("---")
        st.subheader(f"Account Permissions: {user['email']}")
        st.caption(f"Global Role: **{user['role']}** · Overrides can only reduce access, never increase it")
        
        # Get all org accounts (Need a helper or direct query)
        db = st.session_state.get('db_manager')
        if not db:
             st.error("Database connection missing.")
             return

        accounts_query = "SELECT id, display_name, marketplace FROM amazon_accounts WHERE organization_id = %s ORDER BY display_name"
        
        try:
            # PostgresManager doesn't have fetch_all, use raw connection
            with db._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(accounts_query, (str(current_user.organization_id),))
                    # Manual dict mapping
                    columns = [desc[0] for desc in cur.description]
                    accounts = [dict(zip(columns, row)) for row in cur.fetchall()]
        except Exception as e:
            st.error(f"Error fetching accounts: {e}")
            accounts = []

        if not accounts:
            st.info("No accounts in this organization.")
            return

        for acc in accounts:
            col_a, col_b, col_c = st.columns([3, 2, 1])
            with col_a:
                st.write(f"**{acc['display_name']}** ({acc['marketplace']})")
            
            with col_b:
                # Get current override role
                # Since auth.list_users doesn't return overrides, we fetch it or pass full user obj
                # Optimization: We should probably fetch overrides for specific user on demand
                # For now, let's just query it directly to be safe and simple
                current_ov_role = "DEFAULT"
                try: 
                    # This is N+1 query pattern but OK for admin UI with < 50 accounts
                    ov_res = db.fetch_one(
                        "SELECT role FROM user_account_overrides WHERE user_id = %s AND amazon_account_id = %s",
                        (str(user['id']), str(acc['id']))
                    )
                    if ov_res:
                        current_ov_role = ov_res['role']
                except Exception:
                    pass

                # Available options (Phase 3.5: Downgrade Only)
                # If Global is ADMIN/OPERATOR -> Can enable VIEWER
                # If Global is ADMIN -> Can enable OPERATOR
                
                # Logic:
                # 1. Start with DEFAULT
                options = ["DEFAULT"]
                
                user_role_str = user['role']
                
                # If global is > VIEWER, can restrict to VIEWER
                if user_role_str in ['OWNER', 'ADMIN', 'OPERATOR']:
                    options.append("VIEWER")
                    
                # If global is > OPERATOR, can restrict to OPERATOR
                if user_role_str in ['OWNER', 'ADMIN']:
                     options.append("OPERATOR")
                     
                safe_index = 0
                if current_ov_role in options:
                    safe_index = options.index(current_ov_role)
                
                selected_role = st.selectbox(
                    "Access Level",
                    options=options,
                    index=safe_index,
                    key=f"ov_{user['id']}_{acc['id']}",
                    label_visibility="collapsed"
                )

            with col_c:
                if st.button("Save", key=f"save_{user['id']}_{acc['id']}", type="secondary"):
                    if selected_role == "DEFAULT":
                        res = auth.remove_account_override(str(user['id']), str(acc['id']))
                    else:
                        res = auth.set_account_override(
                            str(user['id']), 
                            str(acc['id']), 
                            Role(selected_role),
                            str(current_user.id)
                        )
                    
                    if res["success"]:
                        st.success("Saved")
                        st.rerun()
                    else:
                        st.error(res.get("error"))

    if current_user:
        users = auth.list_users(current_user.organization_id)
        if not users:
             st.info("No users found (except you?)")
             
        for user in users:
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.write(user["email"])
                if user["id"] == current_user.id:
                    st.caption("(You)")
            with col2:
                st.code(user["role"])
            with col3:
                st.caption(user["status"])
                
                # Check strict hierarchy: Can I manage this user?
                # User Role usually comes as string from list_users
                can_edit = can_manage_role(current_user.role.value, user["role"])
                
                # Admin Reset Action
                if user["id"] != current_user.id and can_edit:
                    # Unique key needed for button inside loop
                    if st.button("Reset Pwd", key=f"reset_{user['id']}", help="Force reset user password", type="primary"):
                        res = auth.admin_reset_password(current_user, str(user["id"]))
                        if res.success:
                            st.warning(f"⚠️ Temp Password for {user['email']}: {res.reason}")
                            st.info("User will be forced to change this on next login.")
                        else:
                            st.error(res.reason)
                        
                # Phase 3.5: Account Access Overrides (Expandable)
                # Only show if I can manage them
                if user['id'] != current_user.id and can_edit:
                    with st.expander("Manage Account Access"):
                        render_account_override_editor(user)
    else:
        st.error("Session error.")
            
    st.divider()
    
    # 2. Invite User
    st.subheader("Invite New User")
    
    with st.form("invite_user_form"):
        new_email = st.text_input("Email Address")
        new_role_str = st.selectbox(
            "Role", 
            options=[r.value for r in Role], 
            index=2 # Default to OPERATOR
        )
        
        # Billing Warning Logic (Soft Enforcement)
        is_billable = get_billable_default(new_role_str)
        if is_billable:
            st.warning(
                f"⚠️ Adding this user will add ${SEAT_PRICE}/month to your bill "
                "(pro-rated for the rest of this billing cycle)."
            )
        else:
            st.info("ℹ️ This role is non-billable (free).")
            
        submitted = st.form_submit_button("Create User", type="primary")
        
        if submitted:
            if not new_email:
                st.error("Email required")
            else:
                if current_user:
                    res = auth.create_user_invite(new_email, Role(new_role_str), current_user.organization_id)
                    if res["success"]:
                        st.success(f"User created: {new_email}")
                        st.info(f"🔑 Temporary Password: **{res.get('temp_password', 'Welcome123!')}**")
                        st.balloons()
                    else:
                        st.error(f"Failed: {res.get('error')}")
                else:
                     st.error("You must be logged in.")
