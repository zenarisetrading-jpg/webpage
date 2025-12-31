"""
Account Utilities

Centralized functions for account ID resolution and session state management.
Consolidates duplicated account lookup logic across multiple feature modules.
"""

import streamlit as st
from typing import Optional


def get_active_account_id() -> Optional[str]:
    """
    Get the active account ID from session state with fallback chain.

    Priority order:
    1. active_account_id (direct selection)
    2. active_account_name (from account selector)
    3. last_stats_save.client_id (from last saved data)

    Returns:
        Account ID string or None if no account is active

    Usage:
        Used by impact_dashboard.py, report_card.py, and other modules
        that need to load account-specific data from the database.
    """
    return (
        st.session_state.get('active_account_id') or
        st.session_state.get('active_account_name') or
        st.session_state.get('last_stats_save', {}).get('client_id')
    )


def get_active_account_name() -> str:
    """
    Get the display name for the active account.

    Returns:
        Account display name or 'Unknown Account' if not found
    """
    account_id = get_active_account_id()
    return st.session_state.get('active_account_name', account_id or 'Unknown Account')


def require_active_account(error_message: str = None) -> Optional[str]:
    """
    Get active account ID or display error and return None.

    Args:
        error_message: Custom error message to display

    Returns:
        Account ID if found, None otherwise (with error displayed)

    Usage:
        client_id = require_active_account()
        if not client_id:
            return  # Error already displayed
    """
    client_id = get_active_account_id()

    if not client_id:
        if error_message is None:
            error_message = "⚠️ No account selected! Please select an account in the sidebar."
        st.error(error_message)
        return None

    return client_id


def get_test_mode() -> bool:
    """
    Get test mode flag from session state.

    Returns:
        True if test_mode is active, False otherwise
    """
    return st.session_state.get('test_mode', False)
