"""
Account Manager UI Component

Provides universal account selector for both agencies and individual sellers.
Supports single-account mode (auto-hides selector) and multi-account management.
"""

import streamlit as st


def render_account_selector():
    """
    Universal account selector - works for agencies AND individual sellers.
    Shows account info even if only 1 account.
    """
    db = st.session_state.get('db_manager')
    
    if not db:
        st.sidebar.warning("⚠️ Database not initialized")
        return
    
    # Get all accounts
    accounts = db.get_all_accounts()  # [(id, name, type), ...]
    
    # SAFETY CHECK: If active_account_id is set but not in DB (e.g. after DB wipe), clear it.
    if 'active_account_id' in st.session_state:
        valid_ids = [a[0] for a in accounts]
        if st.session_state['active_account_id'] not in valid_ids:
            st.warning(f"⚠️ Account '{st.session_state['active_account_id']}' not found in DB. Resetting selection.")
            del st.session_state['active_account_id']
            if 'active_account_name' in st.session_state:
                del st.session_state['active_account_name']
            st.rerun()

    if not accounts:
        # DO NOT CREATE DEFAULT. Force user to create one.
        st.sidebar.warning("⚠️ No accounts found. Please create one.")
        _show_account_creation_form()
        return
    
    # Single account mode - show compact display with add option
    if len(accounts) == 1:
        account_id, account_name, account_type = accounts[0]
        
        # Check if we need to initialize/load data
        # Load data if: 1) Different account, OR 2) Same account but no data loaded
        data_exists = st.session_state.get('unified_data', {}).get('search_term_report') is not None
        
        if st.session_state.get('active_account_id') != account_id or not data_exists:
            print(f"[DEBUG] Loading data for {account_id}: account_changed={st.session_state.get('active_account_id') != account_id}, data_exists={data_exists}")
            st.session_state['active_account_id'] = account_id
            st.session_state['active_account_name'] = account_name
            
            # Load from database on initial setup
            from core.data_hub import DataHub
            hub = DataHub()
            result = hub.load_from_database(account_id)
            print(f"[DEBUG] load_from_database result: {result}")
        
        
        st.session_state['single_account_mode'] = True
        
        # Show current account with option to add more
        st.sidebar.markdown(f"**Account:** {account_name}")
        if st.sidebar.button("➕ Add Account", use_container_width=True, key="add_account_single"):
            st.session_state['show_account_form'] = True
        
        # Show form if requested
        if st.session_state.get('show_account_form'):
            _show_account_creation_form()
        
        st.sidebar.markdown("---")
        return
    
    # Multi-account mode - full selector
    st.session_state['single_account_mode'] = False
    
    # Phase 3.5: Decorate with Effective Role
    from core.auth.service import AuthService
    from core.auth.permissions import get_effective_role
    
    auth = AuthService()
    current_user = auth.get_current_user()
    
    options = {}
    for idx, (id, name, _) in enumerate(accounts):
        label = f"{name} ({id})"
        
        # Calculate role badge
        if current_user:
             # Check for overrides
             override_role = None
             if hasattr(current_user, 'account_overrides'):
                 # Ensure UUID match
                 import uuid
                 try:
                     u_id = uuid.UUID(str(id))
                     if u_id in current_user.account_overrides:
                         override_role = current_user.account_overrides[u_id].value
                 except:
                     pass
             
             effective = get_effective_role(current_user.role.value, override_role)
             
             # Decoration
             badge = f"[{effective}]"
             if override_role:
                 badge = f"🔒 {badge}" # Lock icon for restricted access
                 
             label = f"{name} {badge}"
             
        options[label] = idx

    options["➕ Add New Account"] = "NEW"
    
    # Get current selection
    current_idx = 0
    if 'active_account_id' in st.session_state:
        for idx, (id, name, _) in enumerate(accounts):
            if id == st.session_state['active_account_id']:
                current_idx = idx
                break
    
    key_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><circle cx="7.5" cy="15.5" r="5.5"/><path d="m21 2-9.6 9.6"/><path d="m15.5 7.5 3 3L22 7l-3-3z"/></svg>'
    st.markdown(f'<div style="display: flex; align-items: center; justify-content: center; margin-bottom: 8px;">{key_svg}<span style="color: #94a3b8; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;">Active Account</span></div>', unsafe_allow_html=True)
    selected = st.selectbox(
        "Active Account",
        options=options.keys(),
        index=current_idx,
        key="account_selector",
        label_visibility="collapsed"
    )
    
    selected_value = options[selected]
    
    if selected_value == "NEW":
        _show_account_creation_form()
    else:
        # Set active account
        account_id, account_name, account_type = accounts[selected_value]
        
        # DETECT ACCOUNT SWITCH - Clear session data if account changed
        previous_account = st.session_state.get('active_account_id')
        if previous_account and previous_account != account_id:
            st.toast(f"✅ Switched to account: {account_name}", icon="🔄")
            # Account switched! Clear uploaded data first
            if 'unified_data' in st.session_state:
                st.session_state.unified_data = {
                    'search_term_report': None,
                    'advertised_product_report': None,
                    'bulk_id_mapping': None,
                    'category_mapping': None,
                    'enriched_data': None,
                    'upload_status': {
                        'search_term_report': False,
                        'advertised_product_report': False,
                        'bulk_id_mapping': False,
                        'category_mapping': False
                    },
                    'upload_timestamps': {}
                }
            
            # Clear cached optimizer/simulator results
            keys_to_clear = [
                'latest_optimizer_run', 
                'optimizer_results', 
                'optimization_run',
                'impact_analysis_cache',
                'run_optimizer'  # Important: prevent auto-run on new account
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Now load data from database for new account
            from core.data_hub import DataHub
            hub = DataHub()
            loaded = hub.load_from_database(account_id)
            
            if loaded:
                st.toast(f"🔄 Switched to {account_name} (Loaded {hub.get_summary().get('search_terms', 0):,} rows)", icon="🔄")
            else:
                st.toast(f"🔄 Switched to {account_name} (No data)", icon="🔄")
        elif not previous_account:
            # First time loading - initialize data from database
            from core.data_hub import DataHub
            hub = DataHub()
            hub.load_from_database(account_id)
        
        st.session_state['active_account_id'] = account_id
        st.session_state['active_account_name'] = account_name
    
    
    # No trailing line here to avoid double lines in callers
    pass


# Helper for account overrides logic if needed
def get_current_account_id():
    """Helper to get currently selected account ID safely."""
    return st.session_state.get('active_account_id')


def _show_account_creation_form():
    """Show form to create new account."""
    with st.form("new_account"):
        st.subheader("Create New Account")
        
        name = st.text_input(
            "Account Name", 
            placeholder="MyBrand Premium or Acme Corp"
        )
        
        account_type = st.selectbox(
            "Account Type",
            ["brand", "client", "marketplace", "test"],
            help="Choose the type that best describes this account"
        )
        
        # Optional metadata
        with st.expander("Additional Info (Optional)"):
            marketplace = st.text_input("Marketplace", placeholder="Amazon US")
            currency = st.selectbox("Currency", ["USD", "AED", "SAR", "GBP", "EUR", "INR"])
            notes = st.text_area("Notes")
        
        if st.form_submit_button("Create Account"):
            db = st.session_state.get('db_manager')
            if db and name:
                # AUTO-GENERATE ID from name
                account_id = name.lower().replace(' ', '_').replace('-', '_')
                # Remove special characters
                account_id = ''.join(c for c in account_id if c.isalnum() or c == '_')
                
                metadata = {
                    "marketplace": marketplace,
                    "currency": currency,
                    "notes": notes
                }
                success = db.create_account(account_id, name, account_type, metadata)
                if success:
                    st.success(f"✅ Created: {name}")
                    st.caption(f"Account ID: `{account_id}`")
                    
                    # CLEAR CACHE FOR NEW ACCOUNT
                    if 'unified_data' in st.session_state:
                         st.session_state.unified_data = {
                            'search_term_report': None,
                            'advertised_product_report': None,
                            'bulk_id_mapping': None,
                            'category_mapping': None,
                            'enriched_data': None,
                            'upload_status': {
                                'search_term_report': False,
                                'advertised_product_report': False,
                                'bulk_id_mapping': False,
                                'category_mapping': False
                            },
                            'upload_timestamps': {}
                        }
                    
                    keys_to_clear = [
                        'latest_optimizer_run', 'optimizer_results', 'optimization_run',
                        'impact_analysis_cache', 'run_optimizer'
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]

                    st.session_state['active_account_id'] = account_id
                    st.session_state['active_account_name'] = name
                    st.session_state.pop('show_account_form', None)  # Hide form
                    st.rerun()
                else:
                    st.error(f"❌ Account already exists. Try a different name.")
            else:
                st.error("Please enter an Account Name")
