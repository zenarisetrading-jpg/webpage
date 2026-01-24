"""
Authentication middleware for protecting pages.
"""
import streamlit as st
from auth.service import AuthService
from auth.ui import render_auth_page


def init_auth_state() -> None:
    """
    Initialize authentication-related session state keys.
    Call this once at app startup.
    """
    defaults = {
        "user": None,
        "session": None,
        "access_token": None,
        "auth_redirect": None,
        "forgot_password_mode": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def require_authentication() -> dict:
    """
    Require authentication to access the page.
    
    Call this at the top of any page/app that requires login.
    If user is not authenticated, shows the login page and stops execution.
    If authenticated, returns the current user dict.
    
    Returns:
        dict: The current authenticated user object
        
    Usage:
        user = require_authentication()
        # Code below only runs if user is logged in
    """
    # Initialize state if needed
    init_auth_state()
    
    # DEBUG: Print to console to verify this function is being called
    print("[AUTH DEBUG] require_authentication() called")
    print(f"[AUTH DEBUG] Session state 'user': {st.session_state.get('user')}")
    
    auth = AuthService()
    
    is_authed = auth.is_authenticated()
    print(f"[AUTH DEBUG] is_authenticated: {is_authed}")
    
    if not is_authed:
        print("[AUTH DEBUG] Not authenticated - showing auth page and stopping")
        render_auth_page()
        st.stop()
    
    return auth.get_current_user()


def optional_authentication() -> dict | None:
    """
    Check authentication without requiring it.
    
    Returns:
        dict | None: The current user if authenticated, None otherwise
    """
    init_auth_state()
    auth = AuthService()
    return auth.get_current_user() if auth.is_authenticated() else None
