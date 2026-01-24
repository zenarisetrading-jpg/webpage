"""
Action Confirmation UI Components

Handles confirmation popup when leaving optimizer with pending actions,
undo toast functionality, and action management utilities.
"""

import streamlit as st
import time
from typing import Optional


@st.dialog("Unsaved Optimization Actions", width="large")
def show_confirmation_dialog():
    """
    Show confirmation dialog when leaving optimizer with pending actions.
    Uses Streamlit's native dialog for proper modal behavior.
    """
    pending = st.session_state.get('pending_actions')
    if not pending:
        st.session_state['_show_action_confirmation'] = False
        st.rerun()
        return
    
    action_count = len(pending.get('actions', []))
    target = st.session_state.get('_pending_navigation_target', 'home')
    
    # Action breakdown
    actions = pending.get('actions', [])
    action_types = {}
    for action in actions:
        atype = action.get('action_type', 'UNKNOWN')
        action_types[atype] = action_types.get(atype, 0) + 1
    
    st.markdown(f"""
    You have **{action_count} pending actions** that haven't been saved to your action history.
    
    These actions will be used to track your optimization impact over time.
    """)
    
    if action_types:
        breakdown = " • ".join([f"{v} {k.replace('_', ' ').title()}" for k, v in action_types.items()])
        st.caption(f"📊 {breakdown}")
    
    st.markdown("---")
    
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        if st.button("🗑️ Discard", use_container_width=True, type="secondary"):
            # Clear pending actions without saving
            st.session_state['pending_actions'] = None
            st.session_state['_show_action_confirmation'] = False
            # Navigate to intended target
            st.session_state['current_module'] = target
            st.session_state['_pending_navigation_target'] = None
            st.rerun()
    
    with btn_col2:
        if st.button("💾 Save to History", use_container_width=True, type="primary"):
            # Save actions to database
            _save_pending_actions_and_navigate(target)
    
    with btn_col3:
        if st.button("← Back", use_container_width=True):
            st.session_state['_show_action_confirmation'] = False
            st.session_state['_pending_navigation_target'] = None
            st.rerun()


def render_action_confirmation_modal():
    """
    Entry point - triggers the dialog if confirmation is needed.
    """
    if st.session_state.get('_show_action_confirmation'):
        show_confirmation_dialog()


def _save_pending_actions_and_navigate(target_module: str):
    """Save pending actions to database, set up undo, and navigate to target."""
    from core.db_manager import get_db_manager
    
    pending = st.session_state.get('pending_actions')
    if not pending:
        return
    
    db = get_db_manager(st.session_state.get('test_mode', False))
    
    try:
        actions = pending.get('actions', [])
        client_id = pending.get('client_id')
        batch_id = pending.get('batch_id')
        report_date = pending.get('report_date')
        
        if actions and client_id:
            db.log_action_batch(actions, client_id, batch_id, report_date)
            
            # Set up undo capability
            st.session_state['_last_saved_batch_id'] = batch_id
            st.session_state['_last_saved_client_id'] = client_id
            st.session_state['_undo_window_start'] = time.time()
            
            # Mark as accepted for auto-add on subsequent runs
            st.session_state['optimizer_actions_accepted'] = True
            
            # Clear pending and confirmation state
            st.session_state['pending_actions'] = None
            st.session_state['_show_action_confirmation'] = False
            st.session_state['_pending_navigation_target'] = None
            
            # Navigate to the intended target
            st.session_state['current_module'] = target_module
            
            st.toast(f"✅ {len(actions)} actions saved to history", icon="💾")
            st.rerun()
    except Exception as e:
        st.error(f"Failed to save actions: {str(e)}")


def show_undo_toast():
    """
    Check if undo is available and show toast with undo button.
    Call this from the main app to show undo option after save.
    """
    batch_id = st.session_state.get('_last_saved_batch_id')
    undo_start = st.session_state.get('_undo_window_start', 0)
    
    if not batch_id or not undo_start:
        return
    
    # 5 minute undo window
    UNDO_WINDOW_SECONDS = 300
    elapsed = time.time() - undo_start
    
    if elapsed > UNDO_WINDOW_SECONDS:
        # Undo window expired, clear the state
        st.session_state['_last_saved_batch_id'] = None
        st.session_state['_undo_window_start'] = None
        return
    
    remaining_mins = int((UNDO_WINDOW_SECONDS - elapsed) / 60)
    
    # Show undo option in sidebar
    st.markdown("---")
    st.caption(f"↩️ Undo available ({remaining_mins}m left)")
    if st.button("Undo Last Save", key="undo_actions_btn"):
        undo_last_batch()


def undo_last_batch():
    """Delete the last saved batch of actions."""
    from core.db_manager import get_db_manager
    
    batch_id = st.session_state.get('_last_saved_batch_id')
    client_id = st.session_state.get('_last_saved_client_id')
    
    if not batch_id or not client_id:
        st.warning("No recent actions to undo")
        return
    
    db = get_db_manager(st.session_state.get('test_mode', False))
    
    try:
        deleted = db.delete_action_batch(client_id, batch_id)
        
        # Clear undo state
        st.session_state['_last_saved_batch_id'] = None
        st.session_state['_last_saved_client_id'] = None
        st.session_state['_undo_window_start'] = None
        
        st.toast(f"↩️ Undone: {deleted} actions removed", icon="↩️")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to undo: {str(e)}")


def clear_todays_actions(client_id: str) -> int:
    """Clear all actions logged today for the given client."""
    from core.db_manager import get_db_manager
    
    db = get_db_manager(st.session_state.get('test_mode', False))
    
    try:
        deleted = db.clear_todays_actions(client_id)
        
        # Also clear any undo state since we're clearing everything
        st.session_state['_last_saved_batch_id'] = None
        st.session_state['_last_saved_client_id'] = None
        st.session_state['_undo_window_start'] = None
        st.session_state['optimizer_actions_accepted'] = False
        
        return deleted
    except Exception as e:
        st.error(f"Failed to clear actions: {str(e)}")
        return 0
