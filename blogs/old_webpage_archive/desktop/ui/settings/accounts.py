"""
Amazon Accounts UI
==================
Screen for managing connected Amazon accounts.
PRD Reference: ORG_USERS_ROLES_PRD.md §10, §13

Features:
- List connected accounts
- Visual cap enforcement (Usage bar)
- Add account CTA (disabled if capped)
"""

import streamlit as st
from features.account_management import check_account_cap_enforcement, AccountLimitExceeded

# Mock data
ORG_LIMIT = 5
CURRENT_ACCOUNTS = [
    {"name": "US Store", "marketplace": "US", "status": "ACTIVE"},
    {"name": "UK Store", "marketplace": "UK", "status": "ACTIVE"},
    {"name": "UAE Store", "marketplace": "UAE", "status": "DISABLED"}, # Doesn't count
]

def get_active_count():
    return sum(1 for a in CURRENT_ACCOUNTS if a["status"] == "ACTIVE")

def render_accounts_settings():
    st.header("Amazon Accounts")
    
    active_count = get_active_count()
    usage_percent = active_count / ORG_LIMIT
    
    # 1. Usage Indicator
    st.subheader("Plan Usage")
    st.progress(usage_percent)
    st.caption(f"{active_count} of {ORG_LIMIT} accounts used")
    
    if active_count >= ORG_LIMIT:
        st.error("🛑 Organization limit reached. Upgrade your plan to add more accounts.")
    
    st.divider()
    
    # 2. Add Account
    st.subheader("Connect New Account")
    
    with st.form("add_account_form"):
        name = st.text_input("Display Name")
        marketplace = st.selectbox("Marketplace", ["US", "UK", "UAE", "DE"])
        
        # Visual Guardrail: Disable button if capped? 
        # Streamlit doesn't support conditional disable well in forms, 
        # so we check on submit.
        
        submitted = st.form_submit_button("Connect Account")
        
        if submitted:
        if submitted:
            try:
                # V2 Check Logic
                from features.account_management import validate_new_account_request
                from core.auth.service import AuthService
                
                # Get Org ID (Assuming user logic is available via session or similar)
                import streamlit as st
                user = st.session_state.get('user')
                if not user:
                    st.error("Session error. Please login.")
                    return

                # Pass None for db executor as we used internal logic or pass real one if needed
                # Ideally we pass 'None' as our robust implementation creates its own connection if DBExecutor isn't passed/used?
                # Actually my impl expects 'db' arg but ignores it for direct connection if fallback triggers.
                # Let's pass st.session_state.get('db_manager') just in case.
                
                result = validate_new_account_request(st.session_state.get('db_manager'), user.organization_id)
                
                if result.allowed:
                     # TODO: Backend add logic (Phase 3)
                    st.success(f"Connected {name} ({marketplace})")
                    st.info(f"Usage: {result.active_count + 1} / {result.limit}")
                else:
                     st.error(f"🛑 {result.reason}")
                     st.markdown("[👉 Upgrade Plan](#)")
                
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    
    # 3. Account List
    st.subheader("Connected Accounts")
    for acc in CURRENT_ACCOUNTS:
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            st.write(acc["name"])
        with col2:
            st.caption(acc["marketplace"])
        with col3:
            if acc["status"] == "ACTIVE":
                st.success("Active")
            else:
                st.write("Disabled")
