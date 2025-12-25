"""
Impact Dashboard Module

Sleek before/after analysis dashboard showing the ROI of optimization actions.
Features:
- Hero tiles with key metrics
- Waterfall chart by action type
- Winners/Losers bar chart
- Detailed drill-down table
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from core.db_manager import get_db_manager

@st.cache_data(ttl=3600, show_spinner=False)  # Restored production TTL
def _fetch_impact_data(client_id: str, test_mode: bool, window_days: int = 7, cache_version: str = "v4_dedup") -> Tuple[pd.DataFrame, Dict[str, Any]]:

    """
    Cached data fetcher for impact analysis.
    Prevents re-querying the DB on every rerun or tab switch.
    
    Args:
        client_id: Account ID
        test_mode: Whether using test database
        window_days: Number of days for before/after comparison window
        cache_version: Version string that changes when data is uploaded (invalidates cache)
    """
    try:
        db = get_db_manager(test_mode)
        impact_df = db.get_action_impact(client_id, window_days=window_days)
        full_summary = db.get_impact_summary(client_id, window_days=window_days)
        return impact_df, full_summary
    except Exception as e:
        # Return empty structures on failure to prevent UI crash
        print(f"Cache miss error: {e}")
        return pd.DataFrame(), {
            'total_actions': 0, 
            'roas_before': 0, 'roas_after': 0, 'roas_lift_pct': 0,
            'incremental_revenue': 0,
            'p_value': 1.0, 'is_significant': False, 'confidence_pct': 0,
            'implementation_rate': 0, 'confirmed_impact': 0, 'pending': 0,
            'win_rate': 0, 'winners': 0, 'losers': 0,
            'by_action_type': {}
        }





def render_impact_dashboard():
    """Main render function for Impact Dashboard."""
    
    # Header Layout with Toggle
    col_header, col_toggle = st.columns([3, 1])
    
    with col_header:
        st.markdown("## :material/monitoring: Impact & Results")
        st.caption("Measured impact of executed optimization actions")

    with col_toggle:
        st.write("") # Spacer
        # Use radio buttons with horizontal layout for time frame selection
        time_frame = st.radio(
            "Time Frame",
            options=["7D", "14D", "30D", "60D", "90D"],
            index=2,  # Default to 30D (index 2)
            horizontal=True,
            label_visibility="collapsed",
            key="impact_time_frame"
        )
        if time_frame is None:
            time_frame = "30D"

    
    # Dark theme compatible CSS
    st.markdown("""
    <style>
    /* Dark theme buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        transform: translateY(-1px);
    }
    /* Data table dark theme compatibility */
    .stDataFrame {
        background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check for database manager
    db_manager = st.session_state.get('db_manager')
    if db_manager is None:
        st.warning("⚠️ Database not initialized. Please ensure you're in the main app.")
        return
    
    # USE ACTIVE ACCOUNT from session state
    selected_client = st.session_state.get('active_account_id', 'default_client')
    
    if not selected_client:
        st.error("⚠️ No account selected! Please select an account in the sidebar.")
        return
    
    # Get available dates for selected account
    available_dates = db_manager.get_available_dates(selected_client)
    
    if not available_dates:
        st.warning(f"⚠️ No action data found for account '{st.session_state.get('active_account_name', selected_client)}'. "\
                   "Run the optimizer to log actions.")
        return
    
    # Sidebar info - show active account
    with st.sidebar:
        # Just show account info, removed comparison settings
        st.info(f"**Account:** {st.session_state.get('active_account_name', selected_client)}")
        st.caption(f"📅 Data available: {len(available_dates)} weeks")
    
    # Get impact data using auto time-lag matching (no date params needed)
    # Get impact data using auto time-lag matching (cached)
    with st.spinner("Calculating impact..."):
        # Use cached fetcher
        test_mode = st.session_state.get('test_mode', False)
        # Force cache bust with version + timestamp
        cache_version = "v4_roas_" + str(st.session_state.get('data_upload_timestamp', 'init'))
        # Parse time frame to days for the before/after comparison window
        time_frame_days = {"7D": 7, "14D": 14, "30D": 30, "60D": 60, "90D": 90}
        window_days = time_frame_days.get(time_frame, 7)
        impact_df, full_summary = _fetch_impact_data(selected_client, test_mode, window_days, cache_version)


    
    # Fixed KeyError: Use 'all' summary for initial check
    if full_summary.get('all', {}).get('total_actions', 0) == 0:
        st.info("No actions with matching 'next week' performance data found. This means either:\n"
                "- Actions were logged but no performance data for the following week exists yet.\n"
                "- Upload next week's Search Term Report and run the optimizer to see impact.")
        return
        
    # Period Header Preparation
    compare_text = ""
    p = full_summary.get('period_info', {})
    if p.get('before_start'):
        try:
            def fmt(d):
                if isinstance(d, str):
                    return datetime.strptime(d[:10], "%Y-%m-%d").strftime("%b %d")
                return d.strftime("%b %d")
            
            b_range = f"{fmt(p['before_start'])} - {fmt(p['before_end'])}"
            a_range = f"{fmt(p['after_start'])} - {fmt(p['after_end'])}"
            compare_text = f"Comparing <code>{b_range}</code> (Before) vs. <code>{a_range}</code> (After)"
        except Exception as e:
            print(f"Header date error: {e}")
        
    
    # Time frame (7D, 14D, 30D) now controls the before/after comparison window in the SQL query
    # No need to filter actions here - all actions with measurable impact are returned
    time_frame_days = {
        "7D": 7,
        "14D": 14,
        "30D": 30,
        "60D": 60,
        "90D": 90
    }
    days = time_frame_days.get(time_frame, 30)
    filter_label = f"{days}-Day Window"
    
    # Get latest available date for reference
    available_dates = db_manager.get_available_dates(selected_client)
    ref_date = pd.to_datetime(available_dates[0]) if available_dates else pd.Timestamp.now()
    
    # NO ADDITIONAL FILTERING - get_action_impact already handles:
    # 1. Fixed windows based on selected days
    # 2. Only eligible actions
    # The UI uses full_summary directly from the backend for statistical rigor.

    
    # Redundant date range callout removed (merged into top header)
    
    # ==========================================
    # CONSOLIDATED PREMIUM HEADER
    # ==========================================
    if not impact_df.empty:
        theme_mode = st.session_state.get('theme_mode', 'dark')
        cal_color = "#60a5fa" if theme_mode == 'dark' else "#3b82f6"
        calendar_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{cal_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>'
        
        action_count = len(impact_df)
        unique_weeks = impact_df['action_date'].nunique() if 'action_date' in impact_df.columns else 1
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%);
                    border: 1px solid rgba(59, 130, 246, 0.3);
                    border-radius: 12px; padding: 16px; margin-bottom: 24px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                <div style="display: flex; align-items: center;">
                    {calendar_icon}
                    <span style="font-weight: 600; font-size: 1.1rem; color: #60a5fa; margin-right: 12px;">{time_frame} Impact Summary</span>
                    <span style="color: #94a3b8; font-size: 0.95rem;">{compare_text}</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.85rem; opacity: 0.8;">
                    {action_count} actions total across {unique_weeks} runs
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # UNIVERSAL VALIDATION TOGGLE
    # ==========================================
    toggle_col1, toggle_col2 = st.columns([1, 5])
    with toggle_col1:
        show_validated_only = st.toggle(
            "Validated Only", 
            value=True, 
            help="Show only actions confirmed by actual CPC/Bid data"
        )
    with toggle_col2:
        if show_validated_only:
            st.caption("✓ Showing **validated actions only** — filtering all cards and charts.")
        else:
            st.caption("📊 Showing **all actions** — including pending and unverified.")
            
    # ==========================================
    # DATA PREPARATION: ACTIVE vs DORMANT
    # ==========================================
    # Filter based on toggle BEFORE splitting
    v_mask = impact_df['validation_status'].str.contains('✓|CPC Validated|CPC Match|Directional|Confirmed|Normalized', na=False, regex=True)
    display_df = impact_df[v_mask].copy() if show_validated_only else impact_df.copy()
    
    # Measured (Active) vs Pending (Dormant)
    # Measured if there was spend in BEFORE or AFTER window
    measured_mask = (display_df['before_spend'].fillna(0) + display_df['observed_after_spend'].fillna(0)) > 0
    active_df = display_df[measured_mask].copy()
    dormant_df = display_df[~measured_mask].copy()
    
    # Use pre-calculated summary from backend for the tiles
    display_summary = full_summary.get('validated' if show_validated_only else 'all', {})
    
    # HERO TILES (Now synchronized)
    _render_hero_tiles(display_summary, len(active_df), len(dormant_df))
    
    st.divider()

    with st.expander("🔍 View supporting evidence", expanded=True):
        # ==========================================
        # MEASURED vs PENDING IMPACT TABS
        # ==========================================
        tab_measured, tab_pending = st.tabs([
            "▸ Measured Impact", 
            "▸ Pending Impact"
        ])
        
        with tab_measured:
            # Filter active_df based on the universal toggle
            active_display = display_df[(display_df['before_spend'].fillna(0) + display_df['after_spend'].fillna(0)) > 0] if not display_df.empty else display_df
            
            if active_display.empty:
                st.info("No measured impact data for the selected filter")
            else:
                # IMPACT ANALYTICS: Attribution Waterfall + Stacked Revenue Bar
                _render_new_impact_analytics(display_summary, active_display, show_validated_only)
                
                st.divider()
                
                # Drill-down table with migration badges
                _render_drill_down_table(active_display, show_migration_badge=True)
        
        with tab_pending:
            if dormant_df.empty:
                st.success("✨ All executed optimizations have measured activity!")
            else:
                st.info("💤 **Pending Impact** — These actions were applied to keywords/targets "
                    "with $0 spend in both periods. The baseline is established and impact is pending traffic.")
                
                # Simple table for dormant
                _render_dormant_table(dormant_df)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "This view presents measured outcomes of executed actions over the selected period. "
        "Detailed diagnostics are available for deeper investigation when required."
    )


def _render_empty_state():
    """Render empty state when no data exists."""
    # Theme-aware chart icon
    icon_color = "#8F8CA3"
    empty_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-opacity="0.2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>'
    st.markdown(f"""
    <div style="text-align: center; padding: 60px 20px;">
        <div style="margin-bottom: 20px;">{empty_icon}</div>
        <h2 style="color: #8F8CA3; opacity: 0.5;">No Impact Data Yet</h2>
        <p style="color: #8F8CA3; opacity: 0.35; max-width: 400px; margin: 0 auto;">
            Run the optimizer and download the report to start tracking actions. 
            Then upload next week's data to see the impact.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### How to use Impact Analysis:
    
    1. **Week 1**: Upload Search Term Report → Run Optimizer → Download Full Report
    2. **Week 2**: Upload new Search Term Report → Come here to see before/after comparison
    """)


def _render_hero_tiles(summary: Dict[str, Any], active_count: int = 0, dormant_count: int = 0):
    """Render the hero metric tiles with ROAS analytics."""
    
    # Theme-aware colors
    theme_mode = st.session_state.get('theme_mode', 'dark')
    
    if theme_mode == 'dark':
        positive_text = "#4ade80"  # Green-400
        negative_text = "#f87171"  # Red-400
        neutral_text = "#cbd5e1"  # Slate-300
        muted_text = "#8F8CA3"
    else:
        positive_text = "#16a34a"  # Green-600
        negative_text = "#dc2626"  # Red-600
        neutral_text = "#475569"  # Slate-600
        muted_text = "#64748b"
    
    # SVG Icons
    icon_color = "#8F8CA3"
    
    # Chart bars icon (Actions)
    actions_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>'
    
    # Trending up icon (ROAS Lift)
    roas_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>'
    
    # Dollar sign icon (Revenue)
    revenue_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>'
    
    # Check circle icon (Implementation)
    impl_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'
    
    # Stat sig indicators
    sig_positive = f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="#22c55e" stroke="#22c55e" stroke-width="2"><circle cx="12" cy="12" r="10"></circle></svg>'
    sig_negative = f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2"><circle cx="12" cy="12" r="10"></circle></svg>'
    
    # Extract metrics
    total_actions = summary.get('total_actions', 0)
    roas_lift = summary.get('roas_lift_pct', 0)
    incr_revenue = summary.get('incremental_revenue', 0)
    impl_rate = summary.get('implementation_rate', 0)
    is_sig = summary.get('is_significant', False)
    p_value = summary.get('p_value', 1.0)
    confidence = summary.get('confidence_pct', 0)
    

    
    # By action type for callout
    by_type = summary.get('by_action_type', {})
    cost_saved = sum(v['net_sales'] for k, v in by_type.items() if 'NEGATIVE' in k.upper())
    harvest_gains = sum(v['net_sales'] for k, v in by_type.items() if 'HARVEST' in k.upper())
    bid_changes = sum(v['net_sales'] for k, v in by_type.items() if 'BID' in k.upper())
    
    # CSS
    st.markdown("""
    <style>
    .hero-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(143, 140, 163, 0.15);
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
    }
    .hero-label {
        font-size: 0.7rem;
        color: #8F8CA3;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
    }
    .hero-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .hero-sub {
        font-size: 0.75rem;
        color: #8F8CA3;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 4 Hero Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="hero-card">
            <div class="hero-label">{actions_icon} Actions</div>
            <div class="hero-value" style="color: {neutral_text};">{total_actions:,}</div>
            <div class="hero-sub">optimization runs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        roas_color = positive_text if roas_lift > 0 else negative_text if roas_lift < 0 else neutral_text
        prefix = '+' if roas_lift > 0 else ''
        sig_icon = sig_positive if is_sig else sig_negative
        st.markdown(f"""
        <div class="hero-card">
            <div class="hero-label">{roas_icon} ROAS Lift</div>
            <div class="hero-value" style="color: {roas_color};">{prefix}{roas_lift:.1f}%</div>
            <div class="hero-sub">{sig_icon} {'significant' if is_sig else 'not significant'} (N={active_count})</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        from utils.formatters import get_account_currency
        currency = get_account_currency()
        rev_color = positive_text if incr_revenue > 0 else negative_text if incr_revenue < 0 else neutral_text
        prefix = '+' if incr_revenue > 0 else ''
        st.markdown(f"""
        <div class="hero-card">
            <div class="hero-label">{revenue_icon} Revenue Impact</div>
            <div class="hero-value" style="color: {rev_color};">{prefix}{currency}{incr_revenue:,.0f}</div>
            <div class="hero-sub">incremental change</div>
        </div>
        """, unsafe_allow_html=True)

    
    with col4:
        impl_color = positive_text if impl_rate >= 70 else negative_text if impl_rate < 40 else neutral_text
        st.markdown(f"""
        <div class="hero-card">
            <div class="hero-label">{impl_icon} Implementation</div>
            <div class="hero-value" style="color: {impl_color};">{impl_rate:.0f}%</div>
            <div class="hero-sub">confirmed applied</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Insightful callout row based on ROAS metrics
    roas_lift = summary.get('roas_lift_pct', 0)
    is_significant = summary.get('is_significant', False)
    before_roas = summary.get('roas_before', 0)
    after_roas = summary.get('roas_after', 0)
    
    if incr_revenue > 0 and roas_lift > 0:
        if is_significant:
            callout_text = f"<strong>{roas_lift:+.1f}%</strong> ROAS improvement is statistically significant — validated across {total_actions} optimizations"
        else:
            callout_text = f"ROAS improved from <strong>{before_roas:.2f}x → {after_roas:.2f}x</strong> ({roas_lift:+.1f}%) on validated bid changes"
        callout_color = positive_text
    elif incr_revenue < 0:
        callout_text = f"ROAS declined <strong>{roas_lift:.1f}%</strong> — review bid strategies or wait for more data"
        callout_color = negative_text
    else:
        callout_text = f"Impact is break-even — continue monitoring validated actions"
        callout_color = neutral_text
    
    win_rate = summary.get('win_rate', 0)
    confirmed = summary.get('confirmed_impact', 0)
    pending = summary.get('pending', 0)
    
    wr_color = positive_text if win_rate >= 60 else negative_text if win_rate < 40 else neutral_text
    
    st.markdown(f"""
    <div style="background: rgba(143, 140, 163, 0.05); border: 1px solid rgba(143, 140, 163, 0.1); border-radius: 8px; 
                padding: 12px 20px; margin-top: 16px; display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="color: {callout_color}; font-weight: 500;">{callout_text}</span>
        </div>
        <div style="display: flex; gap: 24px; color: #8F8CA3; font-size: 0.85rem;">
            <span>Win Rate: <strong style="color: {wr_color};">{win_rate:.0f}%</strong></span>
            <span>Measured: <strong>{confirmed}</strong> | Pending: <strong>{pending}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)




def _render_new_impact_analytics(summary: Dict[str, Any], impact_df: pd.DataFrame, validated_only: bool = True):
    """Render new impact analytics: Attribution Waterfall + Stacked Revenue Bar."""
    
    from utils.formatters import get_account_currency
    currency = get_account_currency()
    
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        _render_attribution_waterfall(summary, impact_df, currency, validated_only)
    
    with col2:
        _render_stacked_revenue_bar(summary, currency, validated_only)


def _render_attribution_waterfall(summary: Dict[str, Any], impact_df: pd.DataFrame, currency: str, validated_only: bool):
    """Render attribution-based waterfall showing ROAS contribution by action type."""
    
    label = "📊 ROAS Contribution by Type" if validated_only else "📊 Sales Change by Type"
    st.markdown(f"#### {label}")
    
    if impact_df.empty:
        st.info("No data to display")
        return
    
    # Break down by MATCH TYPE for more granular attribution (instead of action type)
    # This shows AUTO, BROAD, EXACT, etc. contributions like the Account Overview donut
    match_type_col = 'match_type' if 'match_type' in impact_df.columns else None
    
    contributions = {}
    
    if match_type_col and impact_df[match_type_col].notna().any():
        # Group by match type for richer breakdown
        for match_type in impact_df[match_type_col].dropna().unique():
            type_df = impact_df[impact_df[match_type_col] == match_type]
            type_df = type_df[(type_df['before_spend'] > 0) & (type_df['observed_after_spend'] > 0)]
            
            if len(type_df) == 0:
                continue
            
            # Calculate this type's ROAS contribution
            before_spend = type_df['before_spend'].sum()
            before_sales = type_df['before_sales'].sum()
            after_spend = type_df['observed_after_spend'].sum()
            after_sales = type_df['observed_after_sales'].sum()
            
            roas_before = before_sales / before_spend if before_spend > 0 else 0
            roas_after = after_sales / after_spend if after_spend > 0 else 0
            
            contribution = before_spend * (roas_after - roas_before)
            
            # Clean match type name
            name = str(match_type).upper() if match_type else 'OTHER'
            contributions[name] = contributions.get(name, 0) + contribution
    else:
        # Fallback to action type if no match type
        display_names = {
            'BID_CHANGE': 'Bid Optim.',
            'NEGATIVE': 'Cost Saved',
            'HARVEST': 'Harvest Gains',
            'BID_ADJUSTMENT': 'Bid Optim.'
        }
        
        for action_type in impact_df['action_type'].unique():
            type_df = impact_df[impact_df['action_type'] == action_type]
            type_df = type_df[(type_df['before_spend'] > 0) & (type_df['observed_after_spend'] > 0)]
            
            if len(type_df) == 0:
                continue
            
            before_spend = type_df['before_spend'].sum()
            before_sales = type_df['before_sales'].sum()
            after_spend = type_df['observed_after_spend'].sum()
            after_sales = type_df['observed_after_sales'].sum()
            
            roas_before = before_sales / before_spend if before_spend > 0 else 0
            roas_after = after_sales / after_spend if after_spend > 0 else 0
            
            contribution = before_spend * (roas_after - roas_before)
            
            name = display_names.get(action_type, action_type.replace('_', ' ').title())
            contributions[name] = contributions.get(name, 0) + contribution
    
    if not contributions:
        st.info("Insufficient data for attribution")
        return
    
    # Get the authoritative total from summary (must match hero tile)
    target_total = summary.get('incremental_revenue', 0)
    calculated_total = sum(contributions.values())
    
    # Scale contributions proportionally so they sum to the hero tile's incremental_revenue
    if calculated_total != 0 and target_total != 0:
        scale_factor = target_total / calculated_total
        contributions = {k: v * scale_factor for k, v in contributions.items()}
    
    # Sort and create chart
    sorted_data = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_data]
    impacts = [x[1] for x in sorted_data]
    
    # Color palette matching donut chart (purple-slate-gray scale, cyan only for total)
    bar_colors = ['#5B556F', '#8F8CA3', '#475569', '#334155', '#64748b']  # Purple to slate
    colors = [bar_colors[i % len(bar_colors)] for i in range(len(impacts))]
    colors.append('#22d3ee')  # Cyan for total only
    
    # Total must match hero tile exactly
    final_total = target_total if target_total != 0 else sum(impacts)
    
    # Brand colors from Account Overview: Purple (#5B556F), Cyan (#22d3ee)
    fig = go.Figure(go.Waterfall(
        name="Contribution",
        orientation="v",
        measure=["relative"] * len(impacts) + ["total"],
        x=names + ['Total'],
        y=impacts + [final_total],
        connector={"line": {"color": "rgba(143, 140, 163, 0.3)"}},  # #8F8CA3
        decreasing={"marker": {"color": "#8F8CA3"}},   # Neutral slate (for negatives)
        increasing={"marker": {"color": "#5B556F"}},   # Brand Purple (for positives)
        totals={"marker": {"color": "#22d3ee"}},       # Accent Cyan
        textposition="outside",
        textfont=dict(size=14, color="#e2e8f0"),
        text=[f"{currency}{v:+,.0f}" for v in impacts] + [f"{currency}{final_total:+,.0f}"]
    ))
    
    fig.update_layout(
        showlegend=False,
        height=380,
        margin=dict(t=60, b=40, l=30, r=30),  # Much more space for labels
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)', tickformat=',.0f', tickfont=dict(color='#94a3b8', size=12)),
        xaxis=dict(showgrid=False, tickfont=dict(color='#cbd5e1', size=12))
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_stacked_revenue_bar(summary: Dict[str, Any], currency: str, validated_only: bool = True):
    """Render stacked bar showing Before Revenue vs After (Baseline + Incremental)."""
    
    title = "#### 📈 Baseline vs. Incremental Sales" if validated_only else "#### 📈 Revenue Comparison"
    st.markdown(title)
    
    # Get actual values from summary
    before_sales = summary.get('before_sales', 0)
    after_sales = summary.get('after_sales', 0)
    incremental = summary.get('incremental_revenue', 0)
    roas_before = summary.get('roas_before', 0)
    roas_after = summary.get('roas_after', 0)
    
    # If we have actual sales values, use them
    if before_sales > 0 and after_sales > 0:
        fig = go.Figure()
        
        # Before bar - Brand Purple
        fig.add_trace(go.Bar(
            name='Sales (Before)',
            x=['Before'],
            y=[before_sales],
            marker_color='#5B556F',  # Brand Purple
            text=[f"{currency}{before_sales:,.0f}"],
            textposition='auto',
            textfont=dict(color='#e2e8f0', size=13),
        ))
        
        # After bar with incremental highlight
        fig.add_trace(go.Bar(
            name='Baseline (Expected)',
            x=['After'],
            y=[before_sales],  # Same as before (baseline)
            marker_color='#5B556F',  # Brand Purple
            showlegend=True,
        ))
        
        # Use ROAS-based incremental from summary (matches waterfall and hero tile)
        # This is: before_spend × (roas_after - roas_before)
        lift = incremental  # Use the calculated incremental, not raw sales delta
        lift_color = '#22d3ee'  # Accent Cyan for incremental
        fig.add_trace(go.Bar(
            name='Incremental (Lift)',
            x=['After'],
            y=[lift],
            marker_color=lift_color,
            text=[f"{'+' if lift >= 0 else ''}{currency}{lift:,.0f}"],
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=14),
        ))
        
        fig.update_layout(
            barmode='stack',
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(color='#94a3b8', size=11)),
            height=380,
            margin=dict(t=60, b=40, l=30, r=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)', tickfont=dict(color='#94a3b8', size=12)),
            xaxis=dict(showgrid=False, tickfont=dict(color='#cbd5e1', size=12))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif roas_before > 0 or roas_after > 0:
        # Fallback: Show ROAS comparison bars with brand colors
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Before', 'After'],
            y=[roas_before, roas_after],
            marker_color=['#5B556F', '#22d3ee'],  # Brand Purple to Cyan
            text=[f"{roas_before:.2f}x", f"{roas_after:.2f}x"],
            textposition='auto',
            textfont=dict(color='#e2e8f0', size=14),
        ))
        fig.update_layout(
            showlegend=False,
            height=380,
            margin=dict(t=40, b=40, l=30, r=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)', title="ROAS", tickfont=dict(color='#94a3b8', size=12)),
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No comparative data")


def _render_impact_analytics(summary: Dict[str, Any], impact_df: pd.DataFrame):
    """Render the dual-chart impact analytics section."""
    
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        _render_waterfall_chart(summary)
    
    with col2:
        _render_roas_comparison(summary)


def _render_waterfall_chart(summary: Dict[str, Any]):
    """Render waterfall chart showing incremental revenue by action type."""
    
    # Target icon for action type
    icon_color = "#8F8CA3"
    target_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>'
    st.markdown(f"#### {target_icon}Revenue Impact by Type", unsafe_allow_html=True)
    
    by_type = summary.get('by_action_type', {})
    if not by_type:
        st.info("No action type breakdown available")
        return
    
    # Map raw types to display names
    display_names = {
        'BID_CHANGE': 'Bid Optim.',
        'NEGATIVE': 'Cost Saved',
        'HARVEST': 'Harvest Gains',
        'BID_ADJUSTMENT': 'Bid Optim.'
    }
    
    # Aggregate data
    agg_data = {}
    for t, data in by_type.items():
        name = display_names.get(t, t.replace('_', ' ').title())
        agg_data[name] = agg_data.get(name, 0) + data['net_sales']
    
    # Sort
    sorted_data = sorted(agg_data.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_data]
    impacts = [x[1] for x in sorted_data]
    
    from utils.formatters import get_account_currency
    chart_currency = get_account_currency()
    
    fig = go.Figure(go.Waterfall(
        name="Impact",
        orientation="v",
        measure=["relative"] * len(impacts) + ["total"],
        x=names + ['Total'],
        y=impacts + [sum(impacts)],
        connector={"line": {"color": "rgba(148, 163, 184, 0.2)"}},
        decreasing={"marker": {"color": "rgba(248, 113, 113, 0.5)"}}, 
        increasing={"marker": {"color": "rgba(74, 222, 128, 0.6)"}}, 
        totals={"marker": {"color": "rgba(143, 140, 163, 0.6)"}},
        textposition="outside",
        text=[f"{chart_currency}{v:+,.0f}" for v in impacts] + [f"{chart_currency}{sum(impacts):+,.0f}"]
    ))
    
    fig.update_layout(
        showlegend=False,
        height=320,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)', tickformat=',.0f'),
        xaxis=dict(showgrid=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_roas_comparison(summary: Dict[str, Any]):
    """Render side-by-side ROAS before/after comparison."""
    
    icon_color = "#8F8CA3"
    trend_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>'
    st.markdown(f"#### {trend_icon}Account ROAS Shift", unsafe_allow_html=True)
    
    r_before = summary.get('roas_before', 0)
    r_after = summary.get('roas_after', 0)
    
    if r_before == 0 and r_after == 0:
        st.info("No comparative ROAS data")
        return
        
    fig = go.Figure()
    
    # Before Bar
    fig.add_trace(go.Bar(
        x=['Before Optim.'],
        y=[r_before],
        name="Before",
        marker_color="rgba(148, 163, 184, 0.4)",
        text=[f"{r_before:.2f}"],
        textposition='auto',
    ))
    
    # After Bar
    color = "rgba(74, 222, 128, 0.6)" if r_after >= r_before else "rgba(248, 113, 113, 0.6)"
    fig.add_trace(go.Bar(
        x=['After Optim.'],
        y=[r_after],
        name="After",
        marker_color=color,
        text=[f"{r_after:.2f}"],
        textposition='auto',
    ))
    
    fig.update_layout(
        showlegend=False,
        height=320,
        margin=dict(t=10, b=10, l=40, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)', title="Account ROAS"),
        xaxis=dict(showgrid=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_winners_losers_chart(impact_df: pd.DataFrame):
    """Render top contributors by incremental revenue."""
    
    # Chart icon 
    icon_color = "#8F8CA3"
    chart_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>'
    st.markdown(f"#### {chart_icon}Top Revenue Contributors", unsafe_allow_html=True)
    
    if impact_df.empty:
        st.info("No targeting data available")
        return
    
    # AGGREGATE BY CAMPAIGN > AD GROUP > TARGET
    agg_cols = {
        'impact_score': 'sum',
        'before_spend': 'sum',
        'after_spend': 'sum'
    }
    # Include campaign and ad group to avoid merging "close-match" etc account-wide
    group_cols = ['campaign_name', 'ad_group_name', 'target_text']
    target_perf = impact_df.groupby(group_cols).agg(agg_cols).reset_index()
    
    # Filter to targets that actually had activity
    target_perf = target_perf[(target_perf['before_spend'] > 0) | (target_perf['after_spend'] > 0)]
    
    if target_perf.empty:
        st.info("No matched targets with performance data found")
        return
    
    # Get top 5 winners and bottom 5 losers by impact_score
    winners = target_perf.sort_values('impact_score', ascending=False).head(5)
    losers = target_perf.sort_values('impact_score', ascending=True).head(5)
    
    # Combine for chart
    chart_df = pd.concat([winners, losers]).drop_duplicates().sort_values('impact_score', ascending=False)
    
    # Create descriptive labels
    def create_label(row):
        target = row['target_text']
        cam = row['campaign_name'][:15] + '..' if len(row['campaign_name']) > 15 else row['campaign_name']
        adg = row['ad_group_name'][:10] + '..' if len(row['ad_group_name']) > 10 else row['ad_group_name']
        
        # If it's an auto-type, emphasize the type but show campaign
        if target.lower() in ['close-match', 'loose-match', 'substitutes', 'complements']:
            return f"{target} ({cam})"
        return f"{target[:20]}.. ({cam})"

    chart_df['display_label'] = chart_df.apply(create_label, axis=1)
    chart_df['full_context'] = chart_df.apply(lambda r: f"Cam: {r['campaign_name']}<br>Ad Group: {r['ad_group_name']}<br>Target: {r['target_text']}", axis=1)
    
    # Rename for the chart library to use
    chart_df['raw_perf'] = chart_df['impact_score']
    
    # Brand-aligned palette: Muted violet for positive, muted wine for negative
    chart_df['color'] = chart_df['raw_perf'].apply(
        lambda x: "rgba(91, 85, 111, 0.6)" if x > 0 else "rgba(136, 19, 55, 0.5)"
    )
    
    from utils.formatters import get_account_currency
    bar_currency = get_account_currency()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=chart_df['display_label'],
        x=chart_df['raw_perf'],
        orientation='h',
        marker_color=chart_df['color'],
        text=[f"{bar_currency}{v:+,.0f}" for v in chart_df['raw_perf']],
        textposition='outside',
        hovertext=chart_df['full_context'],
        hoverinfo='text+x'
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(t=20, b=20, l=20, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', zeroline=True, zerolinecolor='rgba(128,128,128,0.5)'),
        yaxis=dict(showgrid=False, autorange='reversed')
    )

    
    st.plotly_chart(fig, use_container_width=True)


def _render_drill_down_table(impact_df: pd.DataFrame, show_migration_badge: bool = False):
    """Render detailed drill-down table with optional migration badges."""
    
    with st.expander("📋 Detailed Action Log", expanded=False):
        if impact_df.empty:
            st.info("No actions to display")
            return
        
        # Create display dataframe
        display_df = impact_df.copy()
        
        # Add migration badge for HARVEST with before_spend > 0
        if show_migration_badge and 'is_migration' in display_df.columns:
            display_df['action_display'] = display_df.apply(
                lambda r: f"🔄 {r['action_type']}" if r.get('is_migration', False) else r['action_type'],
                axis=1
            )
        else:
            display_df['action_display'] = display_df['action_type']
        
        # Define display columns based on available data
        display_cols = [
            'action_display', 'target_text', 'reason',
            'before_spend', 'observed_after_spend', 'delta_spend',
            'before_sales', 'observed_after_sales', 'delta_sales',
            'impact_score', 'validation_status'
        ]
        
        # Filter to columns that actually exist
        cols_to_use = [c for c in display_cols if c in display_df.columns]
        display_df = display_df[cols_to_use].copy()
        
        # Format Result label for display (internal logic)
        if 'impact_score' in display_df.columns:
            def get_impact_status(score):
                if score > 0: return "📈 Positive"
                if score < 0: return "📉 Negative"
                return "➖ Neutral"
            display_df['Result Status'] = display_df['impact_score'].apply(get_impact_status)

        # Rename for user-friendly display
        final_rename = {
            'action_display': 'Action Taken',
            'target_text': 'Target',
            'reason': 'Logic Basis',
            'before_spend': 'Before Spend',
            'observed_after_spend': 'After Spend',
            'delta_spend': 'Spend Change',
            'before_sales': 'Before Sales',
            'observed_after_sales': 'After Sales',
            'delta_sales': 'Sales Change',
            'impact_score': 'Revenue Impact',
            'validation_status': 'Validation Status'
        }
        display_df = display_df.rename(columns=final_rename)
        
        # Format currency columns
        from utils.formatters import get_account_currency
        df_currency = get_account_currency()
        currency_cols = ['Before Spend', 'After Spend', 'Spend Change', 'Before Sales', 'After Sales', 'Sales Change', 'Revenue Impact']
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{df_currency}{x:,.2f}" if pd.notna(x) else "-")
        
        # Show migration legend if applicable
        if show_migration_badge and 'is_migration' in impact_df.columns and impact_df['is_migration'].any():
            st.caption("🔄 = **Migration Tracking**: Efficiency gain from harvesting search term to exact match.")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Revenue Impact": st.column_config.TextColumn(
                    "Revenue Impact",
                    help="Incremental revenue calculated using before_spend * (after_roas - before_roas)"
                ),
                "Validation Status": st.column_config.TextColumn(
                    "Validation Status",
                    help="Verification that the action was actually applied based on subsequent spend reporting"
                )
            }
        )

        
        # Download button
        csv = impact_df.to_csv(index=False)
        st.download_button(
            "📥 Download Full Data (CSV)",
            csv,
            "impact_analysis.csv",
            "text/csv"
        )


def _render_dormant_table(dormant_df: pd.DataFrame):
    """Render simple table for dormant actions ($0 spend in both periods)."""
    
    if dormant_df.empty:
        return
    
    # Simplified view for dormant
    display_cols = ['action_type', 'target_text', 'old_value', 'new_value', 'reason']
    available_cols = [c for c in display_cols if c in dormant_df.columns]
    display_df = dormant_df[available_cols].copy()
    
    display_df = display_df.rename(columns={
        'action_type': 'Action',
        'target_text': 'Target',
        'old_value': 'Old Value',
        'new_value': 'New Value',
        'reason': 'Reason'
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.caption(f"💡 These {len(dormant_df)} optimizations have an established baseline but are pending traffic. "
              "They will appear in Measured Impact once the targets receive impressions.")


def render_reference_data_badge():
    """Render reference data status badge for sidebar."""
    
    db_manager = st.session_state.get('db_manager')
    if db_manager is None:
        return
    
    try:
        status = db_manager.get_reference_data_status()
        
        if not status['exists']:
            st.markdown("""
            <div style="padding: 8px 12px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; border-left: 3px solid #EF4444;">
                <span style="font-size: 0.85rem;">❌ <strong>No Reference Data</strong></span>
            </div>
            """, unsafe_allow_html=True)
        elif status['is_stale']:
            days = status['days_ago']
            st.markdown(f"""
            <div style="padding: 8px 12px; background: rgba(245, 158, 11, 0.1); border-radius: 8px; border-left: 3px solid #F59E0B;">
                <span style="font-size: 0.85rem;">⚠️ <strong>Data Stale</strong> ({days} days ago)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            days = status['days_ago']
            count = status['record_count']
            st.markdown(f"""
            <div style="padding: 8px 12px; background: rgba(16, 185, 129, 0.1); border-radius: 8px; border-left: 3px solid #10B981;">
                <span style="font-size: 0.85rem;">✅ <strong>Data Loaded</strong> ({days}d ago, {count:,} records)</span>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        pass  # Silently handle errors

def get_recent_impact_summary() -> Optional[dict]:
    """
    Helper for Home Page cockpit.
    Returns impact summary metrics from DB for the last 30 days.
    Simple and reliable - matches get_account_health_score pattern.
    """
    from core.db_manager import get_db_manager
    
    # Check for test mode
    test_mode = st.session_state.get('test_mode', False)
    db_manager = get_db_manager(test_mode)
    
    # Fallback chain for account ID (same as health score)
    selected_client = (
        st.session_state.get('active_account_id') or 
        st.session_state.get('active_account_name') or 
        st.session_state.get('last_stats_save', {}).get('client_id')
    )
    
    if not db_manager or not selected_client:
        return None
        
    try:
        # Direct DB query - simple and reliable
        summary = db_manager.get_impact_summary(selected_client, window_days=30)
        
        if not summary:
            return None
        
        # Handle dual-summary structure (all/validated)
        active_summary = summary.get('validated', summary.get('all', summary))
        
        if active_summary.get('total_actions', 0) == 0:
            return None
        
        # Extract key metrics
        incremental_revenue = active_summary.get('incremental_revenue', 0)
        win_rate = active_summary.get('win_rate', 0)
        
        # Get top action type
        by_type = active_summary.get('by_action_type', {})
        top_action_type = None
        if by_type:
            top_action_type = max(by_type, key=lambda k: by_type[k].get('count', 0))
        
        return {
            'sales': incremental_revenue,
            'win_rate': win_rate,
            'top_action_type': top_action_type,
            'roi': active_summary.get('roas_lift_pct', 0)
        }
        
    except Exception as e:
        print(f"[Impact Summary] Error: {e}")
        return None

