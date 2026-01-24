"""
Authentication module for SADDLE AdPulse.

Usage:
    from auth import require_authentication, render_user_menu
    
    # At the top of your main app:
    user = require_authentication()
    
    # In sidebar:
    render_user_menu()
"""
from auth.middleware import require_authentication, init_auth_state, optional_authentication
from auth.service import AuthService
from auth.ui import render_auth_page, render_user_menu

__all__ = [
    "require_authentication",
    "init_auth_state",
    "optional_authentication",
    "AuthService",
    "render_auth_page",
    "render_user_menu"
]
