
"""
Account Settings Module (Consolidated)
======================================
Unified "Profile & Settings" center.
Combines User Profile, Password Security, and Ad Account Management.

Tabs:
1. My Profile (Everyone)
2. Ad Accounts (Admin Only)
3. Ingestion (Admin Only - Placeholder)
"""

import streamlit as st
from core.auth.service import AuthService
from core.auth.models import User
from core.auth.permissions import has_permission

def run_account_settings():
    """Main entry for Settings."""
    
    # --- Header ---
    icon_color = "#9A9AAA"
    settings_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 10px;"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>'
    st.markdown(f'<h1 style="margin-bottom: 2rem; color: #E9EAF0;">{settings_icon}SETTINGS</h1>', unsafe_allow_html=True)
    
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
    
    # --- Auth Context ---
    auth = AuthService()
    user = auth.get_current_user()
    if not user:
        st.error("Session invalid")
        return

    # --- Tabs ---
    # Determine which tabs to show based on Role
    
    # We use Streamlit tabs for cleaner UI than buttons
    tabs = ["My Profile"]
    
    if has_permission(user.role, 'manage_accounts'):
        tabs.append("Ad Accounts")
        
    if has_permission(user.role, 'configure_ingestion'):
        tabs.append("Ingestion")
        
    selected_tab = st.radio("Section", tabs, horizontal=True, label_visibility="collapsed")
    st.divider()
    
    # --- Render ---
    if selected_tab == "My Profile":
        _render_profile(user, auth)
    elif selected_tab == "Ad Accounts":
        _render_ad_accounts(user)
    elif selected_tab == "Ingestion":
        st.info("Ingestion Settings (Coming Soon)")


def _render_profile(user: User, auth: AuthService):
    """User Profile & Security."""
    
    # 1. Identity Card
    st.subheader("Identity")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.caption("EMAIL")
        st.markdown(f"**{user.email}**")
        if user.billable:
            st.caption("Billable Seat")
    with c2:
        st.caption("ROLE")
        st.code(user.role.value)
    with c3:
        st.caption("STATUS")
        st.caption("STATUS")
        # Brand compliant status (Neutral Light)
        icon = "⚪" if user.status == 'ACTIVE' else "⚫"
        st.write(f"{icon} {user.status}")

    # 2. Security Section
    st.divider()
    st.subheader("Security")
    
    with st.expander("Change Password", expanded=True):
        with st.form("pwd_change_form"):
            current_pwd = st.text_input("Current Password", type="password")
            new_pwd = st.text_input("New Password", type="password", help="Min 8 chars, 1 number/symbol")
            confirm_pwd = st.text_input("Confirm New Password", type="password")
            
            if st.form_submit_button("Update Password", type="primary"):
                if new_pwd != confirm_pwd:
                    st.error("New passwords do not match.")
                elif not current_pwd:
                    st.error("Current password required.")
                else:
                    res = auth.change_password(str(user.id), current_pwd, new_pwd)
                    if res.success:
                        st.success(res.reason)
                        st.balloons()
                    else:
                        st.error(f"Failed: {res.reason}")

    # 3. Session Info (Audit)
    st.divider()
    st.caption(f"Organization ID: `{user.organization_id}`")
    st.caption(f"User ID: `{user.id}`")


def _render_ad_accounts(user: User):
    """Ad Account Management (Admin Only)."""
    db = st.session_state.get('db_manager')
    if not db:
        st.error("Database connection missing.")
        return

    accounts = db.get_all_accounts()
    
    # List Existing
    st.subheader(f"Connected Accounts ({len(accounts)})")
    
    if accounts:
        for acc_id, acc_name, acc_type in accounts:
            with st.container():
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{acc_name}** `({acc_type})`")
                c1.caption(f"ID: {acc_id}")
                # Placeholder for disconnect
                # c2.button("Manage", key=f"manage_{acc_id}")
                st.divider()
    else:
        st.info("No accounts connected.")
        
    # Create New
    st.subheader("Connect New Account")
    with st.form("new_account_form"):
        name = st.text_input("Account Display Name")
        
        # Comprehensive Marketplace List
        marketplaces = [
            "Amazon US (United States)", "Amazon CA (Canada)", "Amazon MX (Mexico)", "Amazon BR (Brazil)",
            "Amazon UK (United Kingdom)", "Amazon DE (Germany)", "Amazon FR (France)", "Amazon IT (Italy)", "Amazon ES (Spain)", 
            "Amazon NL (Netherlands)", "Amazon SE (Sweden)", "Amazon PL (Poland)", "Amazon TR (Turkey)", "Amazon BE (Belgium)",
            "Amazon JP (Japan)", "Amazon IN (India)", "Amazon AU (Australia)", "Amazon SG (Singapore)",
            "Amazon AE (UAE)", "Amazon SA (Saudi Arabia)", "Amazon EG (Egypt)"
        ]
        marketplace = st.selectbox("Marketplace", marketplaces)
        
        if st.form_submit_button("Connect Account", type="primary"):
             # Mock creation for now, strictly speaking this should call a service
             # But keeping it consistent with previous file logic request
             if name:
                 # Logic would go here
                 st.warning("Account creation logic is migrating to V2. Please use SQL/Backend for now.")
             else:
                 st.error("Name required.")
