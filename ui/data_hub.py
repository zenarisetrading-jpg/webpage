"""
Data Hub UI Page

Upload all files in one place, use everywhere.
"""

import streamlit as st
from core.data_hub import DataHub
from datetime import datetime, timedelta

def render_data_hub():
    """Render the data hub upload interface."""
    
    # Theme-aware icon color
    icon_color = "#94a3b8"
    folder_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"></path></svg>'
    
    st.markdown(f"""
    <h1 style="font-family: Inter, sans-serif; font-weight: 700; display: flex; align-items: center; gap: 12px;">
        {folder_icon}
        <span>Data Hub</span>
    </h1>
    """, unsafe_allow_html=True)
    st.caption("Manage your data sources here.")
    
    # ===========================================
    # ACCOUNT CONTEXT BANNER
    # ===========================================
    active_account_id = st.session_state.get('active_account_id')
    active_account_name = st.session_state.get('active_account_name', 'No account selected')
    
    if not active_account_id:
        st.error("⚠️ **No account selected!** Please select or create an account in the sidebar.")
        return
    
    # Compact account indicator
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
        <span style="width: 8px; height: 8px; background: #22c55e; border-radius: 50%;"></span>
        <span style="color: #94a3b8; font-size: 0.9rem;">Uploading to Account: <strong style="color: #e2e8f0;">{active_account_name}</strong></span>
        <span style="color: #22c55e;">✓</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize data hub
    hub = DataHub()
    status = hub.get_upload_status()
    summary = hub.get_summary()
    timestamps = st.session_state.unified_data.get('upload_timestamps', {})
    
    # ===========================================
    # DATA SOURCES SECTION
    # ===========================================
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
        <span style="font-size: 1rem; font-weight: 700; color: #e2e8f0; letter-spacing: 0.02em;">Data Sources</span>
    </div>
    """, unsafe_allow_html=True)
    
    def _get_staleness_indicator(upload_time):
        """Return staleness badge HTML."""
        if not upload_time:
            return ""
        days_ago = (datetime.now() - upload_time).days
        if days_ago > 21:
            return f'<span style="color: #f59e0b; font-size: 0.75rem; margin-left: 8px;">⚠️ {days_ago}d old</span>'
        return f'<span style="color: #64748b; font-size: 0.75rem; margin-left: 8px;">{days_ago}d ago</span>'
    
    def _render_data_source_row(name, is_loaded, metric, is_required=False, expander_key=None, upload_time=None):
        """Render a data source row with checkbox, name, and metric."""
        check_color = "#22c55e" if is_loaded else "#475569"
        check_icon = "✓" if is_loaded else ""
        req_label = " — Required" if is_required else ""
        staleness = _get_staleness_indicator(upload_time)
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; 
                    border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 20px; height: 20px; border: 2px solid {check_color}; border-radius: 4px; 
                            display: flex; align-items: center; justify-content: center; 
                            background: {'rgba(34, 197, 94, 0.15)' if is_loaded else 'transparent'};
                            color: {check_color}; font-size: 12px; font-weight: bold;">{check_icon}</div>
                <span style="font-weight: 600; color: #e2e8f0;">{name}</span>
                <span style="color: #64748b; font-size: 0.8rem;">{req_label}</span>
                {staleness}
            </div>
            <span style="color: #94a3b8; font-size: 0.85rem;">{metric}</span>
        </div>
        """, unsafe_allow_html=True)
    
    
    # ===========================================
    # DATA SOURCES - 2x2 Layout
    # ===========================================
    
    # Row 1: Search Terms | Advertised Products
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        str_metric = f"{summary.get('search_terms', 0):,} rows" if status['search_term_report'] else "—"
        _render_data_source_row("Search Terms", status['search_term_report'], str_metric, True, "str", timestamps.get('search_term_report'))
        
        with st.expander("", expanded=not status['search_term_report']):
            if status['search_term_report']:
                if st.button("🔄 Replace", key="replace_str"):
                    st.session_state.unified_data['upload_status']['search_term_report'] = False
                    st.rerun()
            else:
                str_file = st.file_uploader("Upload Search Term Report", type=['csv', 'xlsx', 'xls'], key='str_upload', label_visibility="collapsed")
                if str_file:
                    st.markdown(f"<small style='color: #f59e0b;'>⚠️ Uploading to: <strong>{active_account_name}</strong></small>", unsafe_allow_html=True)
                    confirm = st.checkbox(f"I confirm this data belongs to {active_account_name}", key="confirm_str")
                    if confirm:
                        with st.spinner("Processing..."):
                            success, message = hub.upload_search_term_report(str_file)
                            # Store result in session state so it persists across rerun
                            st.session_state['last_upload_result'] = {'success': success, 'message': message, 'time': datetime.now()}
                            if success:
                                st.success(f"✅ {message}")
                                # Add small delay so user sees the message before rerun
                                import time
                                time.sleep(1.5)
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")

    
    with row1_col2:
        adv_metric = f"{summary.get('unique_asins', 0):,} ASINs" if status['advertised_product_report'] else "—"
        _render_data_source_row("Advertised Products", status['advertised_product_report'], adv_metric, False, "adv", timestamps.get('advertised_product_report'))
        
        with st.expander("", expanded=False):
            if status['advertised_product_report']:
                if st.button("🔄 Replace", key="replace_adv"):
                    st.session_state.unified_data['upload_status']['advertised_product_report'] = False
                    st.rerun()
            else:
                adv_file = st.file_uploader("Upload Advertised Product Report", type=['csv', 'xlsx', 'xls'], key='adv_upload', label_visibility="collapsed")
                if adv_file:
                    with st.spinner("Processing..."):
                        success, message = hub.upload_advertised_product_report(adv_file)
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
    
    # Row 2: Bulk ID Map | Category Map
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        bulk_metric = f"{summary.get('mapped_campaigns', 0):,} campaigns" if status['bulk_id_mapping'] else "—"
        _render_data_source_row("Bulk ID Map", status['bulk_id_mapping'], bulk_metric, False, "bulk", timestamps.get('bulk_id_mapping'))
        
        with st.expander("", expanded=False):
            if status['bulk_id_mapping']:
                if st.button("🔄 Replace", key="replace_bulk"):
                    st.session_state.unified_data['upload_status']['bulk_id_mapping'] = False
                    st.rerun()
            else:
                bulk_file = st.file_uploader("Upload Bulk File", type=['csv', 'xlsx', 'xls'], key='bulk_upload', label_visibility="collapsed")
                if bulk_file:
                    with st.spinner("Processing..."):
                        success, message = hub.upload_bulk_id_mapping(bulk_file)
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
    
    with row2_col2:
        cat_metric = f"{summary.get('categorized_skus', 0):,} SKUs" if status['category_mapping'] else "—"
        _render_data_source_row("Category Map", status['category_mapping'], cat_metric, False, "cat", timestamps.get('category_mapping'))
        
        with st.expander("", expanded=False):
            if status['category_mapping']:
                if st.button("🔄 Replace", key="replace_cat"):
                    st.session_state.unified_data['upload_status']['category_mapping'] = False
                    st.rerun()
            else:
                cat_file = st.file_uploader("Upload Category Mapping", type=['csv', 'xlsx', 'xls'], key='cat_upload', label_visibility="collapsed")
                if cat_file:
                    with st.spinner("Processing..."):
                        success, message = hub.upload_category_mapping(cat_file)
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ===========================================
    # DATASET SUMMARY SECTION
    # ===========================================
    with st.expander("**Dataset Summary** (for reference)", expanded=status['search_term_report']):
        if status['search_term_report']:
            # System Ready Box
            st.markdown("""
            <div style="background: rgba(34, 197, 94, 0.08); border: 1px solid rgba(34, 197, 94, 0.3); 
                        border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <div style="width: 24px; height: 24px; background: #22c55e; border-radius: 4px; 
                                display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">✓</div>
                    <span style="font-size: 1.1rem; font-weight: 700; color: #e2e8f0;">System Ready</span>
                </div>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
                    ✓ Data processed. All features available.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # 3 CTAs - All primary style with SVG icons
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Account Overview", use_container_width=True, type="primary", key="cta_overview"):
                    st.session_state['current_module'] = 'performance'
                    st.rerun()
            with c2:
                if st.button("Run Optimizer", use_container_width=True, type="primary", key="cta_optimizer"):
                    st.session_state['current_module'] = 'optimizer'
                    st.rerun()
            with c3:
                if st.button("Ask AI Strategist", use_container_width=True, type="primary", key="cta_ai"):
                    st.session_state['current_module'] = 'assistant'
                    st.rerun()
        else:
            st.info("👋 Upload a **Search Term Report** above to unlock the dashboard.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ===========================================
    # ADVANCED / ADMIN SECTION
    # ===========================================
    with st.expander("**› Advanced / Admin**", expanded=False):
        st.markdown("### Data Reassignment")
        st.warning("⚠️ **Use with caution!** This permanently moves data from one account to another.")
        
        db = st.session_state.get('db_manager')
        if db:
            # Get accounts
            registered_accounts = db.get_all_accounts()
            account_options = {name: acc_id for acc_id, name, _ in registered_accounts}
            
            # Historical/Ghost Accounts
            try:
                with db._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT client_id FROM target_stats")
                    stats_ids = {row[0] for row in cursor.fetchall()}
                    cursor.execute("SELECT DISTINCT client_id FROM actions_log")
                    log_ids = {row[0] for row in cursor.fetchall()}
                    ghost_ids = (stats_ids | log_ids) - set(account_options.values())
                    for gid in ghost_ids:
                        account_options[f"{gid} (Legacy)"] = gid
            except:
                pass
            
            col1, col2 = st.columns(2)
            with col1:
                from_name = st.selectbox("From Account", list(account_options.keys()), key="reassign_from")
            with col2:
                to_name = st.selectbox("To Account", list(account_options.keys()), key="reassign_to")
            
            from_id = account_options[from_name]
            to_id = account_options[to_name]
            
            st.markdown("**Date Range:**")
            col3, col4 = st.columns(2)
            with col3:
                start_date = st.date_input("Start", key="reassign_start")
            with col4:
                end_date = st.date_input("End", key="reassign_end")
            
            if st.button("Preview Data to Move", key="preview_reassign"):
                st.session_state['reassign_preview_active'] = True
                
            if st.session_state.get('reassign_preview_active', False):
                try:
                    with db._get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            SELECT COUNT(*), SUM(spend), SUM(sales)
                            FROM target_stats
                            WHERE client_id = ? AND start_date BETWEEN ? AND ?
                        ''', (from_id, str(start_date), str(end_date)))
                        count, spend, sales = cursor.fetchone()
                        
                        cursor.execute('''
                            SELECT COUNT(*)
                            FROM actions_log
                            WHERE client_id = ? AND DATE(action_date) BETWEEN ? AND ?
                        ''', (from_id, str(start_date), str(end_date)))
                        actions_count = cursor.fetchone()[0]
                        
                    if (count and count > 0) or (actions_count and actions_count > 0):
                        st.info(f"**{count:,} rows** | **{actions_count:,} actions** | AED {spend or 0:,.0f} spend")
                        
                        confirm_move = st.checkbox(f"✅ Confirm move to {to_name}", key="final_reassign_confirm")
                        if confirm_move and st.button("Execute Move", type="primary", key="execute_reassign"):
                            success = db.reassign_data(from_id, to_id, str(start_date), str(end_date))
                            if success:
                                st.success(f"✅ Moved!")
                                del st.session_state['reassign_preview_active']
                            else:
                                st.error("❌ Failed")
                    else:
                        st.warning("No data found in range")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("---")
        
        # Clear all
        if any(status.values()):
            if st.button("🗑️ Reset & Clear All Data", type="secondary"):
                hub.clear_all()
                st.success("Data cleared.")
                st.rerun()


def _validate_campaigns(hub: DataHub, account_id: str) -> dict:
    """
    Validate uploaded campaigns against historical data for this account.
    Returns dict with validation results.
    """
    from core.db_manager import get_db_manager
    
    uploaded_data = hub.get_data('search_term_report')
    if uploaded_data is None or 'Campaign Name' not in uploaded_data.columns:
        return {'needs_review': False, 'overlap_pct': 100}
    
    uploaded_campaigns = set(uploaded_data['Campaign Name'].dropna().unique())
    
    try:
        db = get_db_manager(st.session_state.get('test_mode', False))
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT campaign_name 
                FROM target_stats 
                WHERE client_id = ?
            ''', (account_id,))
            historical_campaigns = set(row[0] for row in cursor.fetchall())
    except:
        return {'needs_review': False, 'overlap_pct': 100}
    
    if not historical_campaigns:
        return {'needs_review': False, 'overlap_pct': 100, 'first_upload': True}
    
    overlap = uploaded_campaigns & historical_campaigns
    new_campaigns = uploaded_campaigns - historical_campaigns
    missing_campaigns = historical_campaigns - uploaded_campaigns
    
    overlap_pct = (len(overlap) / len(historical_campaigns) * 100) if historical_campaigns else 100
    needs_review = overlap_pct < 30
    
    return {
        'needs_review': needs_review,
        'overlap_pct': overlap_pct,
        'new_count': len(new_campaigns),
        'missing_count': len(missing_campaigns),
        'total_uploaded': len(uploaded_campaigns),
        'total_historical': len(historical_campaigns),
        'overlap_campaigns': overlap,
        'new_campaigns': new_campaigns,
        'missing_campaigns': missing_campaigns
    }
