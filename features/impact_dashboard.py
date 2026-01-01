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

# ==========================================
# MULTI-HORIZON IMPACT MEASUREMENT CONFIG
# ==========================================
IMPACT_WINDOWS = {
    "before_window_days": 14,       # Fixed 14-day baseline for all horizons
    "maturity_buffer_days": 3,      # Days after window for attribution to settle
    
    "horizons": {
        "14D": {
            "days": 14,
            "maturity": 17,  # 14 + 3
            "label": "14-Day Impact",
            "description": "Early signal — did the action have an effect?",
        },
        "30D": {
            "days": 30,
            "maturity": 33,  # 30 + 3
            "label": "30-Day Impact",
            "description": "Confirmed — is the impact sustained?",
        },
        "60D": {
            "days": 60,
            "maturity": 63,  # 60 + 3
            "label": "60-Day Impact",
            "description": "Long-term — did the gains hold?",
        },
    },
    
    "default_horizon": "14D",
    "available_horizons": ["14D", "30D", "60D"],
}

# ==========================================
# CONFIDENCE CLASSIFICATION (High/Medium/Low)
# ==========================================
from math import sqrt
from typing import List, Literal

def compute_confidence(
    actions_df: pd.DataFrame,
    min_validated_actions: int = 30
) -> Dict[str, Any]:
    """
    Compute confidence classification for aggregated Decision Impact.
    
    Confidence is a classification layer only — does NOT alter impact values.
    Based on signal-to-noise ratio derived from data sufficiency, market conditions, and variance.
    
    Args:
        actions_df: DataFrame with columns: decision_impact, confidence_weight, market_tag, is_validated
        min_validated_actions: Minimum validated actions needed for "High" confidence
        
    Returns:
        dict with: confidence ("High"/"Medium"/"Low"), signalRatio, totalSigma
    """
    if actions_df.empty:
        return {"confidence": "Low", "signalRatio": 0.0, "totalSigma": 0.0}
    
    # Filter to validated actions
    validated = actions_df[actions_df.get('is_validated', False) == True].copy()
    
    if len(validated) == 0:
        return {"confidence": "Low", "signalRatio": 0.0, "totalSigma": 0.0}
    
    total_impact = 0.0
    variance_sum = 0.0
    downshift_impact = 0.0
    
    for _, row in validated.iterrows():
        # Use direct column access for pandas Series (not dict .get())
        impact = row['decision_impact'] if pd.notna(row['decision_impact']) else 0
        total_impact += impact
        
        # Per-action variance: sigma_i = abs(impact) * (1 - confidence_weight)
        cw = row['confidence_weight'] if 'confidence_weight' in row.index and pd.notna(row['confidence_weight']) else 0.5
        sigma = abs(impact) * (1 - cw)
        
        # Apply market multiplier for downshift
        market_tag = row['market_tag'] if 'market_tag' in row.index else 'Normal'
        if market_tag == "Market Downshift":
            sigma *= 1.3
            downshift_impact += abs(impact)
        
        variance_sum += sigma ** 2
    
    # Aggregate variance
    total_sigma = sqrt(variance_sum) if variance_sum > 0 else 0
    
    # Signal-to-noise ratio
    signal_ratio = abs(total_impact) / total_sigma if total_sigma > 0 else 0
    
    # Confidence classification
    if signal_ratio >= 1.5 and len(validated) >= min_validated_actions:
        confidence = "High"
    elif signal_ratio >= 0.8:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Optional downgrade: if >40% of impact from Market Downshift
    downshift_ratio = downshift_impact / abs(total_impact) if total_impact != 0 else 0
    if downshift_ratio > 0.4:
        if confidence == "High":
            confidence = "Medium"
        elif confidence == "Medium":
            confidence = "Low"
    
    return {
        "confidence": confidence,
        "signalRatio": round(signal_ratio, 2),
        "totalSigma": round(total_sigma, 2)
    }

def compute_spend_avoided_confidence(
    actions_df: pd.DataFrame,
    min_validated_actions: int = 10
) -> Dict[str, Any]:
    """
    Compute confidence classification for Spend Avoided summary.
    
    Uses auction variance (not revenue CW) - reflects auction stability.
    Variance factors: Normal = 0.15, Market Downshift = 0.25
    
    Args:
        actions_df: DataFrame with columns: before_spend, observed_after_spend, market_tag, is_validated
        min_validated_actions: Minimum validated actions for "High" confidence
        
    Returns:
        dict with: confidence, signalRatio, totalSigma, totalSpendAvoided
    """
    if actions_df.empty:
        return {"confidence": "Low", "signalRatio": 0.0, "totalSigma": 0.0, "totalSpendAvoided": 0.0}
    
    # Filter to validated actions only
    validated = actions_df[actions_df.get('is_validated', False) == True].copy()
    
    if len(validated) == 0:
        return {"confidence": "Low", "signalRatio": 0.0, "totalSigma": 0.0, "totalSpendAvoided": 0.0}
    
    total_spend_avoided = 0.0
    variance_sum = 0.0
    downshift_spend_avoided = 0.0
    valid_action_count = 0
    
    # Auction variance factors
    VARIANCE_NORMAL = 0.15
    VARIANCE_DOWNSHIFT = 0.25
    
    for _, row in validated.iterrows():
        before_spend = row['before_spend'] if pd.notna(row['before_spend']) else 0
        after_spend = row['observed_after_spend'] if 'observed_after_spend' in row.index and pd.notna(row['observed_after_spend']) else 0
        
        # Spend avoided = max(0, before - after)
        spend_avoided = max(0, before_spend - after_spend)
        
        # Skip rows with zero spend avoided
        if spend_avoided == 0:
            continue
        
        total_spend_avoided += spend_avoided
        valid_action_count += 1
        
        # Determine auction variance factor
        market_tag = row['market_tag'] if 'market_tag' in row.index else 'Normal'
        variance_factor = VARIANCE_DOWNSHIFT if market_tag == "Market Downshift" else VARIANCE_NORMAL
        
        # Per-action variance: sigma_i = spend_avoided * variance_factor
        sigma = spend_avoided * variance_factor
        
        if market_tag == "Market Downshift":
            downshift_spend_avoided += spend_avoided
        
        variance_sum += sigma ** 2
    
    # Aggregate variance
    total_sigma = sqrt(variance_sum) if variance_sum > 0 else 0
    
    # Signal-to-noise ratio
    signal_ratio = total_spend_avoided / total_sigma if total_sigma > 0 else 0
    
    # Confidence classification (stricter thresholds than Decision Impact)
    if signal_ratio >= 2.0 and valid_action_count >= min_validated_actions:
        confidence = "High"
    elif signal_ratio >= 1.0:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Optional downgrade: if >30% of spend avoided from Market Downshift
    downshift_ratio = downshift_spend_avoided / total_spend_avoided if total_spend_avoided > 0 else 0
    if downshift_ratio > 0.3:
        if confidence == "High":
            confidence = "Medium"
        elif confidence == "Medium":
            confidence = "Low"
    
    return {
        "confidence": confidence,
        "signalRatio": round(signal_ratio, 2),
        "totalSigma": round(total_sigma, 2),
        "totalSpendAvoided": round(total_spend_avoided, 2)
    }

def get_maturity_status(action_date, latest_data_date, horizon: str = "14D") -> dict:
    """
    Check if action has matured enough for impact calculation at a specific horizon.
    
    Maturity formula: action_date + horizon_days + buffer_days ≤ latest_data_date
    
    Args:
        action_date: Date the action was logged (T0)
        latest_data_date: The most recent date in the data (from DB)
        horizon: Measurement horizon - "14D", "30D", or "60D"
        
    Returns:
        dict with is_mature, maturity_date, days_until_mature, status, horizon
    """
    # Get horizon config
    if horizon not in IMPACT_WINDOWS["horizons"]:
        horizon = IMPACT_WINDOWS["default_horizon"]
    horizon_config = IMPACT_WINDOWS["horizons"][horizon]
    after_window_days = horizon_config["days"]
    maturity_buffer_days = IMPACT_WINDOWS["maturity_buffer_days"]
    
    # Parse action_date to date object
    if isinstance(action_date, str):
        action_date = datetime.strptime(action_date[:10], "%Y-%m-%d").date()
    elif isinstance(action_date, datetime):
        action_date = action_date.date()
    elif hasattr(action_date, 'date'):  # pd.Timestamp
        action_date = action_date.date()
    
    # Parse latest_data_date
    if isinstance(latest_data_date, str):
        latest_data_date = datetime.strptime(latest_data_date[:10], "%Y-%m-%d").date()
    elif isinstance(latest_data_date, datetime):
        latest_data_date = latest_data_date.date()
    elif hasattr(latest_data_date, 'date'):
        latest_data_date = latest_data_date.date()
    
    # Calculate when this action will be mature
    after_window_end = action_date + timedelta(days=after_window_days)
    maturity_date = after_window_end + timedelta(days=maturity_buffer_days)
    
    # Check against latest data date, not today
    days_until_mature = (maturity_date - latest_data_date).days
    is_mature = latest_data_date >= maturity_date
    
    if is_mature:
        status = "Measured"
    elif latest_data_date >= after_window_end:
        status = f"Pending ({days_until_mature}d)"
    else:
        days_in = (latest_data_date - action_date).days
        status = f"In Window ({max(0, days_in)}/{after_window_days}d)"
    
    return {
        "is_mature": is_mature,
        "maturity_date": maturity_date,
        "days_until_mature": max(0, days_until_mature),
        "status": status,
        "horizon": horizon,
        "horizon_config": horizon_config,
    }

@st.cache_data(ttl=3600, show_spinner=False)  # Restored production TTL
def _fetch_impact_data(client_id: str, test_mode: bool, before_days: int = 14, after_days: int = 14, cache_version: str = "v6_data_filter") -> Tuple[pd.DataFrame, Dict[str, Any]]:

    """
    Cached data fetcher for impact analysis.
    Prevents re-querying the DB on every rerun or tab switch.
    
    Args:
        client_id: Account ID
        test_mode: Whether using test database
        before_days: Number of days for before comparison window (fixed at 14)
        after_days: Number of days for after comparison window (14, 30, or 60)
        cache_version: Version string that changes when data is uploaded (invalidates cache)
    """
    try:
        db = get_db_manager(test_mode)
        impact_df = db.get_action_impact(client_id, before_days=before_days, after_days=after_days)
        full_summary = db.get_impact_summary(client_id, before_days=before_days, after_days=after_days)
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
        # Horizon selector - measurement period after action
        horizon = st.radio(
            "Measurement Horizon",
            options=IMPACT_WINDOWS["available_horizons"],  # ["14D", "30D", "60D"]
            index=0,  # Default to 14D
            horizontal=True,
            label_visibility="collapsed",
            key="impact_horizon",
            help="How long after the action to measure impact"
        )
        if horizon is None:
            horizon = IMPACT_WINDOWS["default_horizon"]

    
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
        cache_version = "v13_indexed_lateral_" + str(st.session_state.get('data_upload_timestamp', 'init'))
        
        # Get horizon config
        horizon_config = IMPACT_WINDOWS["horizons"].get(horizon, IMPACT_WINDOWS["horizons"]["14D"])
        before_days = IMPACT_WINDOWS["before_window_days"]  # Fixed 14 days
        after_days = horizon_config["days"]  # 14, 30, or 60 based on selection
        buffer_days = IMPACT_WINDOWS["maturity_buffer_days"]  # 3 days
        
        impact_df, full_summary = _fetch_impact_data(selected_client, test_mode, before_days, after_days, cache_version)
        
        # === MATURITY GATE ===
        # Add maturity status to each action - determines if impact can be calculated
        # Maturity is based on whether the DATA covers enough time after the action
        # for the after-window + attribution buffer to have settled
        
        # Get the latest date in the data from full_summary
        period_info = full_summary.get('period_info', {})
        latest_data_date = period_info.get('after_end') or period_info.get('latest_date')
        
        if not impact_df.empty and 'action_date' in impact_df.columns and latest_data_date:
            impact_df['is_mature'] = impact_df['action_date'].apply(
                lambda d: get_maturity_status(d, latest_data_date, horizon)['is_mature']
            )
            impact_df['maturity_status'] = impact_df['action_date'].apply(
                lambda d: get_maturity_status(d, latest_data_date, horizon)['status']
            )
            
            mature_count = int(impact_df['is_mature'].sum())
            pending_attr_count = len(impact_df) - mature_count
            
            # Debug: Show the cutoff date
            maturity_days = after_days + buffer_days
            cutoff_date = pd.to_datetime(latest_data_date) - pd.Timedelta(days=maturity_days)
            print(f"Maturity cutoff ({horizon}): Actions from {cutoff_date.strftime('%b %d')} or earlier are mature (data through {pd.to_datetime(latest_data_date).strftime('%b %d')})")
            
            # === EMPTY HORIZON CALLOUT ===
            # If no actions are mature for this horizon, show a helpful message
            if mature_count == 0 and len(impact_df) > 0:
                st.warning(f"""
                    **No actions mature for {horizon} measurement yet.**
                    
                    The {horizon_config['label']} requires {maturity_days} days of data after each action.
                    Your most recent data is from {pd.to_datetime(latest_data_date).strftime('%b %d')}.
                    
                    **Options:**
                    - Select **14D** horizon for earlier insights
                    - Wait for more data to accumulate
                    - {pending_attr_count} actions pending for this horizon
                """)
        else:
            # Fallback if no action_date column or no latest date
            impact_df['is_mature'] = True
            impact_df['maturity_status'] = 'Measured'
            mature_count = len(impact_df)
            pending_attr_count = 0
        
        # Terminal debug: Show Decision Impact metrics
        print(f"\n=== DECISION IMPACT DEBUG ({selected_client}) ===")
        print(f"Maturity Gate: {mature_count} measured, {pending_attr_count} pending attribution")
        for w in [7, 14, 30]:
            try:
                db = get_db_manager(test_mode)
                s = db.get_impact_summary(selected_client, window_days=w)
                val = s.get('validated', {})
                print(f"{w}D: ROAS {val.get('roas_before',0):.2f}x -> {val.get('roas_after',0):.2f}x | Lift: {val.get('roas_lift_pct',0):.1f}% | N={val.get('total_actions',0)}")
                print(f"    Decision Impact: {val.get('decision_impact',0):.0f} | Spend Avoided: {val.get('spend_avoided',0):.0f}")
            except Exception as e:
                print(f"{w}D: Error - {e}")
        print("===================================\n")


    
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
        
    
    
    # Horizon-based measurement: 14D before (fixed) vs horizon after window
    # No need to filter actions here - all actions with measurable impact are returned
    filter_label = f"{horizon} Impact Window"
    
    # Get latest available date for reference
    available_dates = db_manager.get_available_dates(selected_client)
    ref_date = pd.to_datetime(available_dates[0]) if available_dates else pd.Timestamp.now()
    
    # NO ADDITIONAL FILTERING - get_action_impact already handles:
    # 1. Fixed windows based on selected days
    # 2. Only eligible actions
    # The UI uses full_summary directly from the backend for statistical rigor.

    
    # Redundant date range callout removed (merged into top header)
    
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
    # DATA PREPARATION: MATURE + VALIDATED
    # ==========================================
    # Step 1: Filter by validation toggle
    v_mask = impact_df['validation_status'].str.contains('✓|CPC Validated|CPC Match|Directional|Confirmed|Normalized|Volume', na=False, regex=True)
    display_df = impact_df[v_mask].copy() if show_validated_only else impact_df.copy()
    
    # Step 2: MATURITY GATE - Split mature vs pending attribution
    mature_mask = display_df['is_mature'] == True
    mature_df = display_df[mature_mask].copy()  # ONLY mature actions for aggregates
    pending_attr_df = display_df[~mature_mask].copy()  # Pending attribution
    
    # Step 3: Within mature, split Active vs Dormant (by spend)
    spend_mask = (mature_df['before_spend'].fillna(0) + mature_df['observed_after_spend'].fillna(0)) > 0
    active_df = mature_df[spend_mask].copy()
    dormant_df = mature_df[~spend_mask].copy()
    
    # ==========================================
    # CONFIDENCE CLASSIFICATION
    # ==========================================
    # Add required columns for confidence calculation
    import numpy as np
    
    # confidence_weight: based on data quality (clicks, spend, validation)
    if not active_df.empty:
        active_df['confidence_weight'] = (
            np.clip(np.log1p(active_df['before_clicks'].fillna(0)) / 5, 0, 0.3) +  # Max 0.3 from clicks
            np.clip(np.log1p(active_df['before_spend'].fillna(0)) / 10, 0, 0.3) +  # Max 0.3 from spend
            np.clip(np.log1p(active_df['after_clicks'].fillna(0)) / 5, 0, 0.2) +   # Max 0.2 from after data
            (active_df['validation_status'].str.contains('✓', na=False).astype(float) * 0.2)  # 0.2 if validated
        ).clip(0, 1)
        
        # market_tag: detect market downshift
        def get_market_tag(row):
            if row.get('before_clicks', 0) == 0:
                return "Low Data"
            if row.get('market_downshift', False) == True:
                return "Market Downshift"
            return "Normal"
        active_df['market_tag'] = active_df.apply(get_market_tag, axis=1)
        
        # is_validated: already used for filtering
        active_df['is_validated'] = True  # All in active_df are validated if toggle is on
        
        # Compute confidence for Decision Impact
        confidence_result = compute_confidence(active_df, min_validated_actions=30)
        
        # Compute confidence for Spend Avoided
        spend_avoided_result = compute_spend_avoided_confidence(active_df, min_validated_actions=10)
    else:
        confidence_result = {"confidence": "Low", "signalRatio": 0.0, "totalSigma": 0.0}
        spend_avoided_result = {"confidence": "Low", "signalRatio": 0.0, "totalSigma": 0.0, "totalSpendAvoided": 0.0}
    
    # Use pre-calculated summary from backend for the tiles
    display_summary = full_summary.get('validated' if show_validated_only else 'all', {})
    display_summary['confidence'] = confidence_result['confidence']
    display_summary['signal_ratio'] = confidence_result['signalRatio']
    display_summary['decision_impact_sigma'] = confidence_result.get('totalSigma', 0.0)
    display_summary['spend_avoided_confidence'] = spend_avoided_result['confidence']
    display_summary['spend_avoided_sigma'] = spend_avoided_result['totalSigma']
    
    # HERO TILES (Now synchronized with FILTERED maturity counts)
    # Use len(mature_df) and len(pending_attr_df) which respect the Validated Only toggle
    from utils.formatters import get_account_currency
    currency = get_account_currency()
    
    # 4. Calculate explicit counts for display consistency
    measured_count = len(active_df)
    pending_display_count = len(pending_attr_df) + len(dormant_df)
    
    # Update summary to reflect the filtered visual counts in cards
    display_summary['total_actions'] = measured_count

    # Pass the summary's decision_impact directly to ensure matches the card
    total_verified_impact = display_summary.get('decision_impact', 0)

    # ==========================================
    # CONSOLIDATED PREMIUM HEADER
    # ==========================================
    if not impact_df.empty:
        theme_mode = st.session_state.get('theme_mode', 'dark')
        cal_color = "#E9EAF0" if theme_mode == 'dark' else "#5B5670" # Soft White / Saddle Purple
        calendar_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{cal_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>'
        
        # Prepare Analysis Period text
        p = full_summary.get('period_info', {})
        analysis_range = ""
        try:
            def fmt(d):
                if not d: return ""
                if isinstance(d, str): d = pd.to_datetime(d)
                return d.strftime("%b %d")
            
            # Use strict before_start (baseline start) to after_end (impact end)
            start = p.get('before_start')
            end = p.get('after_end')
            if start and end:
                analysis_range = f"{fmt(start)} - {fmt(end)}"
            else:
                # Fallback to action dates
                analysis_range = f"{fmt(impact_df['action_date'].min())} - {fmt(impact_df['action_date'].max())}"
        except:
            analysis_range = "Current Period"

        compare_text = f"Analysis Period: <code>{analysis_range}</code>"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(91, 86, 112, 0.4) 0%, rgba(91, 86, 112, 0.1) 100%); 
                    border: 1px solid rgba(91, 86, 112, 0.5); 
                    border-radius: 12px; padding: 16px; margin-bottom: 24px; display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 8px;">
                {calendar_icon}
                <span style="font-weight: 600; font-size: 1.1rem; color: #E9EAF0; margin-right: 12px;">{horizon_config['label']}</span>
                <span style="color: #94a3b8; font-size: 0.95rem;">{compare_text}</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.85rem; font-family: monospace;">
                Measured: <span style="color: #e2e8f0; font-weight: 600;">{measured_count}</span> | Pending: <span style="color: #e2e8f0; font-weight: 600;">{pending_display_count}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Logic moved to top to fix UnboundLocalError
    pass
    _render_hero_banner(
        active_df, currency, filter_label, 
        total_verified_impact=total_verified_impact,
        summary=display_summary,
        mature_count=measured_count,
        pending_count=pending_display_count
    )
    
    st.divider()

    with st.expander("📊 Impact Summary (Modeled vs Baseline)", expanded=True):
        # Panel header subtext
        st.markdown("""
        <div style="color: #8F8CA3; font-size: 0.8rem; margin-bottom: 16px; font-style: italic;">
            Market-adjusted • based on validated actions only
        </div>
        """, unsafe_allow_html=True)
        
        # ==========================================
        # MEASURED vs PENDING IMPACT TABS
        # ==========================================
        pending_tab_label = f"▸ Pending Impact ({len(pending_attr_df) + len(dormant_df)})" if (len(pending_attr_df) + len(dormant_df)) > 0 else "▸ Pending Impact"
        tab_measured, tab_pending = st.tabs([
            "▸ Measured Impact", 
            pending_tab_label
        ])
        
        with tab_measured:
            # Show only MATURE actions with activity
            if active_df.empty:
                st.info("No measured impact data for the selected filter")
            else:
                # IMPACT ANALYTICS: Attribution Waterfall + Stacked Revenue Bar
                _render_new_impact_analytics(display_summary, active_df, show_validated_only, mature_count=measured_count, pending_count=pending_display_count, raw_impact_df=impact_df)
                
                st.divider()
                
                # Drill-down table with migration badges
                _render_drill_down_table(active_df, show_migration_badge=True)
        
        with tab_pending:
            # Section 1: Pending Attribution (immature actions)
            if not pending_attr_df.empty:
                st.markdown("### ⏳ Pending Attribution")
                st.caption("These actions are waiting for Amazon attribution data to settle. Impact will be calculated once mature.")
                
                pending_display = pending_attr_df[['action_date', 'action_type', 'target_text', 'maturity_status']].copy()
                pending_display.columns = ['Action Date', 'Type', 'Target', 'Status']
                st.dataframe(pending_display, use_container_width=True, hide_index=True)
                st.divider()
            
            # Section 2: Dormant (zero spend)
            if not dormant_df.empty:
                st.markdown("### 💤 Waiting for Traffic")
                st.caption("These mature actions have $0 spend in both periods. Impact is pending traffic.")
                _render_dormant_table(dormant_df)
            
            # Success message if nothing pending
            if pending_attr_df.empty and dormant_df.empty:
                st.success("✨ All executed optimizations have measured activity!")

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


def _render_hero_banner(impact_df: pd.DataFrame, currency: str, horizon_label: str = "30D", total_verified_impact: float = None, summary: Dict[str, Any] = {}, mature_count: int = 0, pending_count: int = 0):
    """Render the new primary Hero Banner."""
    
    # 1. Use passed impact if available, else derive from dataframe (fallback)
    if total_verified_impact is not None:
        verified_impact = total_verified_impact
    elif impact_df.empty:
        verified_impact = 0
    else:
        # Fallback filter for verified actions
        verified_mask = impact_df['validation_status'].str.contains('✓|Confirmed|Validated|Directional', na=False, regex=True)
        verified_impact = impact_df.loc[verified_mask, 'decision_impact'].sum()
        
    # Mocking removed: Real previous period data requires a separate query.
    # For now, we only show current period impact.
    impact_prefix = '+' if verified_impact > 0 else ''

    # Styling
    # Styling
    theme_mode = st.session_state.get('theme_mode', 'dark')
    
    is_positive = verified_impact >= 0
    if is_positive:
        banner_bg = "linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.02) 100%)" if theme_mode == 'dark' else "linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%)"
        border_color = "rgba(16, 185, 129, 0.2)"
        text_color = "#34d399" if theme_mode == 'dark' else "#059669" # Emerald-400 / 600
        icon_color = "#34d399"
    else:
        banner_bg = "linear-gradient(135deg, rgba(244, 63, 94, 0.1) 0%, rgba(244, 63, 94, 0.02) 100%)" if theme_mode == 'dark' else "linear-gradient(135deg, rgba(244, 63, 94, 0.15) 0%, rgba(244, 63, 94, 0.05) 100%)"
        border_color = "rgba(244, 63, 94, 0.2)"
        text_color = "#fb7185" if theme_mode == 'dark' else "#e11d48" # Rose-400 / 600
        icon_color = "#fb7185"
    
    st.markdown(f"""
    <div style="background: {banner_bg}; border: 1px solid {border_color}; border-radius: 16px; padding: 32px; text-align: center; margin-bottom: 32px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
        <div style="font-size: 2.5rem; font-weight: 800; color: {text_color}; margin-bottom: 8px;">
            {impact_prefix}{currency}{verified_impact:,.0f} Verified Impact vs Baseline <span style="font-size: 1.2rem; opacity: 0.7; font-weight: 600;">({horizon_label})</span>
        </div>
        <div style="font-size: 1.1rem; color: #94a3b8; font-weight: 500; display: flex; align-items: center; justify-content: center; gap: 8px;">
            Measured revenue preserved from optimization decisions
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline><polyline points="16 7 22 7 22 13"></polyline></svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Ribbon Logic
    if summary:
        win_rate = summary.get('win_rate', 0)
        market_downshift = summary.get('market_downshift_count', 0)
        market_context = f" ({market_downshift} market shifts detected)" if market_downshift > 0 else ""
        
        # Theme colors for ribbon
        if theme_mode == 'dark':
            positive_text = "#4ade80"
            negative_text = "#f87171"
            neutral_text = "#cbd5e1"
        else:
            positive_text = "#16a34a"
            negative_text = "#dc2626"
            neutral_text = "#475569"
            
        wr_color = positive_text if win_rate >= 60 else negative_text if win_rate < 40 else neutral_text
        confirmed = mature_count if mature_count > 0 else summary.get('confirmed_impact', 0)
        pending = pending_count

        st.markdown(f"""
        <div style="background: rgba(143, 140, 163, 0.08); border: 1px solid rgba(143, 140, 163, 0.15); border-radius: 8px; 
                    padding: 12px 20px; margin-top: 16px; display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.1rem;">💡</span>
                <span style="color: #8F8CA3; font-size: 0.85rem;">
                    Impact metrics are calculated deltas vs a no-optimization baseline. They are not additive to actual performance.{market_context}
                </span>
            </div>
            <div style="display: flex; gap: 24px; color: #8F8CA3; font-size: 0.85rem;">
                <span>Win Rate: <strong style="color: {wr_color};">{win_rate:.0f}%</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_validation_rate_chart(impact_df: pd.DataFrame):
    """Render the Validation Rate chart (Vertical Bar)."""
    import plotly.graph_objects as go
    
    st.markdown("#### 🛡️ Validation Rate")
    st.caption("Proportion of decisions by status")
    
    if impact_df.empty:
        st.info("No data")
        return
        
    def get_status_category(row):
        s = str(row.get('validation_status', ''))
        m = str(row.get('maturity_status', ''))
        
        is_verified = '✓' in s or 'Confirmed' in s or 'Validated' in s or 'Directional' in s
        
        if not is_verified:
            return 'Unverified'
            
        # Robust check for pending status (Immature OR No Spend)
        # 1. Maturity Check
        # Try explicit status first, then boolean flag
        is_pending_maturity = 'Pending' in m or 'Immature' in s
        if not is_pending_maturity and 'is_mature' in row.index:
            is_pending_maturity = not bool(row['is_mature'])
            
        # 2. Spend Check (Treat NaN as 0)
        import pandas as pd
        b_spend = row.get('before_spend', 0)
        a_spend = row.get('observed_after_spend', 0)
        
        # Safe float conversion
        try:
            b_val = 0.0 if pd.isna(b_spend) else float(b_spend)
            a_val = 0.0 if pd.isna(a_spend) else float(a_spend)
        except:
            b_val, a_val = 0.0, 0.0
            
        has_spend = (b_val + a_val) > 0
        
        # If either immature OR no spend -> Pending
        if is_pending_maturity or not has_spend:
            return 'Pending'
        else:
            return 'Measured'
            
    impact_df['status_cat'] = impact_df.apply(get_status_category, axis=1)
    counts = impact_df['status_cat'].value_counts()
    
    measured = counts.get('Measured', 0)
    pending = counts.get('Pending', 0)
    unverified = counts.get('Unverified', 0)
    
    total = len(impact_df)
    measured_pct = measured / total * 100 if total > 0 else 0
    pending_pct = pending / total * 100 if total > 0 else 0
    unverified_pct = unverified / total * 100 if total > 0 else 0
    
    fig = go.Figure()
    
    categories = ['Measured', 'Pending', 'Unverified']
    values = [measured, pending, unverified]
    text_labels = [f"{measured_pct:.0f}%", f"{pending_pct:.0f}%", f"{unverified_pct:.0f}%"]
    colors = ['#5B5670', '#9A9AAA', '#D6D7DE'] # Saddle Purple, Slate Grey, Light Grey
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=text_labels,
        textposition='auto',
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    total = len(impact_df)
    if total > 0:
        conf_pct = measured / total * 100
        st.caption(f"**{conf_pct:.0f}%** of optimization decisions have verified impact.")
        st.markdown("""
        <div style="font-size: 0.75rem; color: #8F8CA3; margin-top: 8px; line-height: 1.4;">
            Verified = statistically measurable impact<br>
            Pending = awaiting sufficient data<br>
            Unverified = insufficient signal
        </div>
        """, unsafe_allow_html=True)


def _render_cumulative_impact_chart(impact_df: pd.DataFrame, currency: str):
    """Render Cumulative Impact Over Time (Line Chart with Area)."""
    import plotly.graph_objects as go
    
    st.markdown("#### 📈 Cumulative Verified Impact")
    st.caption("Impact accumulation over the analysis period")
    
    if impact_df.empty:
        st.info("No data")
        return
        
    df = impact_df[impact_df['validation_status'].str.contains('✓|Confirmed|Validated|Directional', na=False, regex=True)].copy()
    
    if df.empty:
        st.info("No verified impact data to plot")
        return
        
    if 'action_date' not in df.columns:
        return
        
    df['action_date'] = pd.to_datetime(df['action_date'])
    df = df.sort_values('action_date')
    
    daily = df.groupby('action_date')['decision_impact'].sum().reset_index()
    daily['cumulative_impact'] = daily['decision_impact'].cumsum()
    
    total_val = daily['cumulative_impact'].iloc[-1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily['action_date'],
        y=daily['cumulative_impact'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#5B5670', width=3),
        fillcolor='rgba(91, 86, 112, 0.2)',
        name='Cumulative Impact'
    ))
    
    fig.add_trace(go.Scatter(
        x=[daily['action_date'].iloc[-1]],
        y=[daily['cumulative_impact'].iloc[-1]],
        mode='markers+text',
        marker=dict(color='#5B5670', size=10),
        text=[f"+{currency}{total_val:,.0f}"],
        textposition="top center",
        textfont=dict(color='#E9EAF0', size=12, weight='bold'),
        showlegend=False
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, gridcolor='rgba(128,128,128,0.1)', tickfont=dict(color='#94a3b8')),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)', tickfont=dict(color='#94a3b8')),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_new_impact_analytics(summary: Dict[str, Any], impact_df: pd.DataFrame, validated_only: bool = True, mature_count: int = 0, pending_count: int = 0, raw_impact_df: pd.DataFrame = None):
    """
    Render new impact analytics layout.
    Structure:
    - Hero Tiles (Estimated Impact, Capital Protected, Decision Quality, Implementation)
    - Row 1: Decision Quality | Impact by Action Type | Validation Rate
    - Row 2: Decision Outcome Matrix | Cumulative Impact
    """
    
    from utils.formatters import get_account_currency
    currency = get_account_currency()
    
    # Theme-aware colors
    theme_mode = st.session_state.get('theme_mode', 'dark')
    
    if theme_mode == 'dark':
        positive_text = "#E9EAF0"  # Soft White (Purple background makes purple text hard)
        negative_text = "#D6D7DE"  # Light Grey (Warnings/Negatives)
        neutral_text = "#9A9AAA"   # Slate Grey
        muted_text = "#8F8CA3"
        section_header_color = "#e2e8f0"
    else:
        positive_text = "#5B5670"  # Saddle Purple (Dark enough for light mode)
        negative_text = "#5E6475"  # Secondary Dark
        neutral_text = "#9A9AAA"   # Slate Grey
        muted_text = "#64748b"
        section_header_color = "#1e293b"
    
    # SVG Icons
    icon_color = "#8F8CA3"
    
    # Spend icon
    spend_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>'
    
    # Revenue icon
    revenue_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>'
    
    # ROAS trending icon
    roas_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>'
    
    # Target icon (for Estimated Revenue Impact)
    target_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>'
    
    # Shield icon (for Capital Protected)
    shield_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
    
    # Score icon
    score_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2"><path d="M12 20V10"></path><path d="M18 20V4"></path><path d="M6 20v-4"></path></svg>'
    
    # Implementation icon
    impl_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'
    
    # Info icon for tooltips
    info_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="cursor: help; margin-left: 4px;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'
    
    # Extract metrics
    total_actions = summary.get('total_actions', 0)
    impl_rate = summary.get('implementation_rate', 0)
    
    # Actual performance metrics
    total_before_spend = summary.get('total_before_spend', 0)
    total_after_spend = summary.get('total_after_spend', 0)
    total_before_sales = summary.get('total_before_sales', 0)
    total_after_sales = summary.get('total_after_sales', 0)
    roas_before = summary.get('roas_before', 0)
    roas_after = summary.get('roas_after', 0)
    
    # Estimated impact metrics (modeled)
    decision_impact = summary.get('decision_impact', 0)
    spend_avoided = summary.get('spend_avoided', 0)
    pct_good = summary.get('pct_good', 0)
    pct_neutral = summary.get('pct_neutral', 0)
    pct_bad = summary.get('pct_bad', 0)
    market_downshift = summary.get('market_downshift_count', 0)
    
    # NPS-style Decision Quality Score = Good% - Bad%
    decision_quality_score = pct_good - pct_bad
    
    # CSS for tiles and sections
    st.markdown("""
    <style>
    .hero-card {
        background: linear-gradient(135deg, rgba(91, 86, 112, 0.25) 0%, rgba(91, 86, 112, 0.05) 100%);
        border: 1px solid rgba(91, 86, 112, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
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
    .section-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        padding-left: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ==========================================
    # VERIFIED IMPACT VS BASELINE
    # ==========================================
    st.markdown(f"""
    <div class="section-header">📈 Verified Impact Analysis (vs Baseline)</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="medium")
    
    # Confidence classification styling
    confidence = summary.get('confidence', 'Low')
    conf_colors = {'High': '#5B5670', 'Medium': '#f59e0b', 'Low': '#D6D7DE'} # Brand Colors
    conf_color = conf_colors.get(confidence, '#D6D7DE')
    
    with col1:
        # Estimated Revenue Impact tile (renamed from Decision Impact)
        di_color = positive_text if decision_impact > 0 else negative_text if decision_impact < 0 else neutral_text
        di_prefix = '+' if decision_impact > 0 else ''
        decision_sigma = summary.get('decision_impact_sigma', 0)
        
        # Calculate 80% CI (Z=1.28)
        upper_bound = decision_impact + (1.28 * decision_sigma)
        lower_bound = decision_impact - (1.28 * decision_sigma)
        
        tooltip_text = "Measured revenue difference relative to a no-optimization baseline. This is a counterfactual estimate, not additional realized revenue."
        
        st.markdown(f"""
        <div class="hero-card" style="text-align: left;">
            <div>
                <div class="hero-label" style="justify-content: flex-start;">
                    {target_icon} VERIFIED REVENUE IMPACT
                    <span title="{tooltip_text}">{info_icon}</span>
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {di_color}; margin-top: 12px; margin-bottom: 8px;">
                    Measured Impact:<br>{di_prefix}{currency}{decision_impact:,.0f}
                </div>
                <div style="font-size: 0.9rem; color: #94a3b8; font-weight: 500; margin-bottom: 12px;">
                    Confidence: <span style="color: {conf_color}; font-weight: 600;">{confidence}</span>
                </div>
                <div style="font-size: 0.8rem; color: #8F8CA3; margin-bottom: 4px;">Confidence Range (80%)</div>
                <div style="font-size: 0.9rem; color: {muted_text}; font-family: monospace; margin-bottom: 16px;">
                    {di_prefix}{currency}{lower_bound:,.0f} ➜ {di_prefix}{currency}{upper_bound:,.0f}
                </div>
            </div>
            <div style="border-top: 1px solid rgba(143, 140, 163, 0.15); padding-top: 12px; margin-top: 12px;">
                <div class="hero-sub" style="justify-content: start; color: {muted_text};">Validated Actions: <span style="color: {neutral_text}; margin-left: 4px;">{total_actions}</span></div>
                <div class="hero-sub" style="justify-content: start; color: {muted_text}; margin-top: 4px;">Baseline: No-optimization baseline</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Capital Protected tile (renamed from Spend Avoided)
        sa_color = positive_text if spend_avoided > 0 else neutral_text
        sa_confidence = summary.get('spend_avoided_confidence', 'Low')
        sa_conf_color = conf_colors.get(sa_confidence, '#D6D7DE')
        sa_sigma = summary.get('spend_avoided_sigma', 0)
        
        # Calculate 80% CI (Z=1.28)
        sa_upper = spend_avoided + (1.28 * sa_sigma)
        sa_lower = spend_avoided - (1.28 * sa_sigma)
        
        tooltip_text = "Estimated reduction in ad spend relative to a no-optimization baseline. Represents reduced capital exposure, not incremental profit."
        
        st.markdown(f"""
        <div class="hero-card" style="text-align: left;">
            <div>
                <div class="hero-label" style="justify-content: flex-start;">
                    {shield_icon} CAPITAL PROTECTED
                    <span title="{tooltip_text}">{info_icon}</span>
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {sa_color}; margin-top: 12px; margin-bottom: 8px;">
                    Measured Impact:<br>{currency}{spend_avoided:,.0f}
                </div>
                <div style="font-size: 0.9rem; color: #94a3b8; font-weight: 500; margin-bottom: 12px;">
                    Protection Confidence: <span style="color: {sa_conf_color}; font-weight: 600;">{sa_confidence}</span>
                </div>
                <div style="font-size: 0.8rem; color: #8F8CA3; margin-bottom: 4px;">Confidence Range (80%)</div>
                <div style="font-size: 0.9rem; color: {muted_text}; font-family: monospace; margin-bottom: 16px;">
                    {currency}{sa_lower:,.0f} ➜ {currency}{sa_upper:,.0f}
                </div>
            </div>
            <div style="border-top: 1px solid rgba(143, 140, 163, 0.15); padding-top: 12px; margin-top: 12px;">
                <div class="hero-sub" style="justify-content: start; color: {muted_text};">Validated Actions: <span style="color: {neutral_text}; margin-left: 4px;">{total_actions}</span></div>
                <div class="hero-sub" style="justify-content: start; color: {muted_text}; margin-top: 4px;">Baseline: No-optimization baseline</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Removed duplicate Decision Quality tile per request
    pass
    

    


    # Validated DF for immune charts
    validation_df = raw_impact_df if raw_impact_df is not None else impact_df

    # Spacer to separate Hero Cards from Charts
    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

    # 2. Row 1: Two Columns (Matrix + Cumulative)
    r1c1, r1c2 = st.columns(2, gap="medium")
    
    with r1c1:
        _render_decision_outcome_matrix(impact_df, summary)
        
    with r1c2:
        _render_cumulative_impact_chart(impact_df, currency)

    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

    # 3. Row 2: Three Columns (Quality, Type, Validation)
    r2c1, r2c2, r2c3 = st.columns(3)
    
    with r2c1:
        _render_decision_quality_distribution(summary)
        
    with r2c2:
        st.markdown("#### ⚡ Impact by Action Type")
        st.caption("Revenue preserved by type")
        if not impact_df.empty:
            type_impact = impact_df.groupby('action_type')['decision_impact'].sum().sort_values(ascending=False)
            top_types = type_impact.head(3)
            
            for atype, val in top_types.items():
                clean_type = str(atype).replace('_', ' ').title()
                val_color = "#2A8EC9" if val > 0 else "#9A9AAA"
                st.markdown(f"""
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #cbd5e1; margin-bottom: 4px;">
                        <span>{clean_type}</span>
                        <span style="color: {val_color}; font-weight: 600;">{currency}{val:,.0f}</span>
                    </div>
                    <div style="width: 100%; background: rgba(255,255,255,0.05); height: 6px; border-radius: 3px;">
                        <div style="width: {min(100, abs(val)/type_impact.abs().max()*100)}%; background: {val_color}; height: 100%; border-radius: 3px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No data")

    with r2c3:
        _render_validation_rate_chart(validation_df)
def _render_decision_outcome_matrix(impact_df: pd.DataFrame, summary: Dict[str, Any]):
    """Chart 1: Decision Outcome Matrix - CPC Change vs Decision Impact."""
    
    import plotly.graph_objects as go
    import numpy as np
    
    st.markdown("#### 🎯 Decision Outcome Matrix")
    st.caption("Were decisions correct given market conditions?")
    
    if impact_df.empty:
        st.info("No data to display")
        return
    
    # Filter to confirmed actions with valid data
    df = impact_df.copy()
    df = df[df['before_spend'] > 0]
    
    # Calculate CPC before and after
    df['cpc_before'] = df['before_spend'] / df['before_clicks'].replace(0, np.nan)
    df['cpc_after'] = df['observed_after_spend'] / df['after_clicks'].replace(0, np.nan)
    df['cpc_change_pct'] = ((df['cpc_after'] - df['cpc_before']) / df['cpc_before'] * 100).fillna(0)
    
    # Calculate Decision Impact (market-adjusted)
    df['spc_before'] = df['before_sales'] / df['before_clicks'].replace(0, np.nan)
    df['expected_clicks'] = df['observed_after_spend'] / df['cpc_before']
    df['expected_sales'] = df['expected_clicks'] * df['spc_before']
    df['decision_impact'] = df['observed_after_sales'] - df['expected_sales']
    
    # Clean infinite/nan values
    df = df[np.isfinite(df['cpc_change_pct']) & np.isfinite(df['decision_impact'])]
    df = df[df['cpc_change_pct'].abs() < 300]  # Filter extreme CPC outliers
    
    # CLIP OUTLIERS: Bound Y-axis to 5th-95th percentile to prevent chart compression
    lower_bound = df['decision_impact'].quantile(0.05)
    upper_bound = df['decision_impact'].quantile(0.95)
    df['impact_display'] = df['decision_impact'].clip(lower_bound, upper_bound)
    
    if len(df) < 3:
        st.info("Insufficient data for matrix")
        return
    
    # Color by action type
    action_colors = {
        'BID': '#2A8EC9',         # Blue
        'BID_CHANGE': '#2A8EC9',  # Blue
        'NEGATIVE': '#9A9AAA',    # Slate (Negative)
        'NEGATIVE_ADD': '#9A9AAA',
        'HARVEST': '#8FC9D6',     # Muted Cyan (Secondary)
    }
    
    # Normalize action types for display
    df['action_clean'] = df['action_type'].str.upper().str.replace('_CHANGE', '').str.replace('_ADD', '')
    df['action_clean'] = df['action_clean'].replace({'BID': 'Bid', 'NEGATIVE': 'Negative', 'HARVEST': 'Harvest'})
    
    fig = go.Figure()
    
    # Add dots for each action type
    for action_type, color in [('Bid', '#2A8EC9'), ('Negative', '#9A9AAA'), ('Harvest', '#8FC9D6')]:
        type_df = df[df['action_clean'] == action_type]
        if type_df.empty:
            continue
        
        fig.add_trace(go.Scatter(
            x=type_df['cpc_change_pct'],
            y=type_df['impact_display'],  # Use clipped values
            mode='markers',
            name=action_type,
            marker=dict(size=10, color=color, opacity=0.8),
            customdata=type_df['decision_impact'],  # Actual value for hover
            hovertemplate='CPC: %{x:.1f}%<br>Impact: %{customdata:,.0f}<extra></extra>',
        ))
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    
    # Add quadrant labels in corners
    x_range = max(abs(df['cpc_change_pct'].min()), abs(df['cpc_change_pct'].max()), 50)
    y_range = df['impact_display'].abs().max()
    
    annotations = [
        dict(x=-x_range*0.7, y=y_range*0.85, text="🟢 Good Defense", showarrow=False, font=dict(color='#2A8EC9', size=11)),
        dict(x=x_range*0.7, y=y_range*0.85, text="🟢 Good Offense", showarrow=False, font=dict(color='#2A8EC9', size=11)),
        dict(x=-x_range*0.7, y=-y_range*0.85, text="🟡 Market-Driven Loss", showarrow=False, font=dict(color='#9A9AAA', size=11)),
        dict(x=x_range*0.7, y=-y_range*0.85, text="🔴 Decision Error", showarrow=False, font=dict(color='#D6D7DE', size=11)),
    ]
    
    fig.update_layout(
        height=400,
        margin=dict(t=30, b=50, l=50, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=dict(text="CPC Change %", font=dict(color='#94a3b8')),
            showgrid=True, gridcolor='rgba(128,128,128,0.1)',
            tickfont=dict(color='#94a3b8'),
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Decision Impact", font=dict(color='#94a3b8')),
            showgrid=True, gridcolor='rgba(128,128,128,0.1)',
            tickfont=dict(color='#94a3b8'),
            zeroline=False
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5,
            font=dict(color='#94a3b8', size=11)
        ),
        annotations=annotations
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_decision_quality_distribution(summary: Dict[str, Any]):
    """Chart 2: Decision Quality Distribution (NPS-Style Donut)."""
    
    import plotly.graph_objects as go
    
    st.markdown("#### 📊 Decision Quality Distribution")
    
    pct_good = summary.get('pct_good', 0)
    pct_neutral = summary.get('pct_neutral', 0)
    pct_bad = summary.get('pct_bad', 0)
    
    # NPS-style score
    decision_quality_score = pct_good - pct_bad
    
    if pct_good + pct_neutral + pct_bad == 0:
        st.info("No outcome data")
        return
    
    # Donut chart
    fig = go.Figure(data=[go.Pie(
        values=[pct_good, pct_neutral, pct_bad],
        labels=['Good', 'Neutral', 'Bad'],
        hole=0.6,
        marker=dict(colors=['#2A8EC9', '#9A9AAA', '#D6D7DE']),
        textinfo='label+percent',
        textfont=dict(size=12, color='#e2e8f0'),
        hovertemplate='%{label}: %{value:.1f}%<extra></extra>',
        sort=False
    )])
    
    # Add score in center
    score_color = '#2A8EC9' if decision_quality_score > 0 else '#D6D7DE' if decision_quality_score < 0 else '#9A9AAA'
    score_prefix = '+' if decision_quality_score > 0 else ''
    
    fig.add_annotation(
        text=f"<b style='font-size:28px; color:{score_color};'>{score_prefix}{decision_quality_score:.0f}</b><br><span style='font-size:11px; color:#8F8CA3;'>Quality Score</span>",
        showarrow=False,
        font=dict(size=14, color='#e2e8f0')
    )
    
    fig.update_layout(
        height=350,
        margin=dict(t=30, b=60, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Important copy
    st.caption("*Neutrals excluded to focus on signal, not noise.*")


def _render_capital_allocation_flow(impact_df: pd.DataFrame, currency: str):
    """Chart 3: Capital Allocation Flow - Before vs After Spend Distribution."""
    
    import plotly.graph_objects as go
    import numpy as np
    
    st.markdown("#### 💰 Spend Flow: Before → After")
    st.caption("How your spend shifted between periods")
    
    if impact_df.empty:
        st.info("No data to display")
        return
    
    df = impact_df.copy()
    
    # Total spend in each period
    total_before = df['before_spend'].sum()
    total_after = df['observed_after_spend'].sum()
    
    if total_before == 0 and total_after == 0:
        st.info("No spend data")
        return
    
    # Categorize each action by spend change direction
    df['spend_change'] = df['observed_after_spend'] - df['before_spend']
    
    # Segment by action type and direction
    # Reduced: Spend decreased (negatives, bid downs)
    reduced_mask = df['spend_change'] < 0
    reduced_before = df.loc[reduced_mask, 'before_spend'].sum()
    reduced_after = df.loc[reduced_mask, 'observed_after_spend'].sum()
    
    # Maintained: Spend roughly same (within 10%)
    maintained_mask = (df['spend_change'].abs() / df['before_spend'].replace(0, np.nan)).fillna(0) <= 0.10
    maintained_before = df.loc[maintained_mask, 'before_spend'].sum()
    maintained_after = df.loc[maintained_mask, 'observed_after_spend'].sum()
    
    # Increased: Spend increased (bid ups)
    increased_mask = df['spend_change'] > 0
    increased_before = df.loc[increased_mask, 'before_spend'].sum()
    increased_after = df.loc[increased_mask, 'observed_after_spend'].sum()
    
    # Build Sankey: Before (left) → Categories (middle) → After (right)
    # Nodes: 0=Before Total, 1=Reduced, 2=Maintained, 3=Increased, 4=After Total
    
    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            pad=30,
            thickness=30,
            line=dict(color='rgba(0,0,0,0)', width=0),
            label=[
                f"Before<br>{currency}{total_before:,.0f}",
                f"Reduced<br>{currency}{reduced_after:,.0f}",
                f"Maintained<br>{currency}{maintained_after:,.0f}",
                f"Increased<br>{currency}{increased_after:,.0f}",
                f"After<br>{currency}{total_after:,.0f}"
            ],
            color=['#5B556F', '#22c55e', '#64748b', '#3b82f6', '#5B556F'],
            x=[0, 0.5, 0.5, 0.5, 1],
            y=[0.5, 0.15, 0.5, 0.85, 0.5]
        ),
        link=dict(
            source=[0, 0, 0, 1, 2, 3],
            target=[1, 2, 3, 4, 4, 4],
            value=[
                reduced_before,      # Before → Reduced
                maintained_before,   # Before → Maintained
                increased_before,    # Before → Increased
                reduced_after,       # Reduced → After
                maintained_after,    # Maintained → After
                increased_after      # Increased → After
            ],
            color=[
                'rgba(34,197,94,0.3)',   # Green - money saved
                'rgba(100,116,139,0.3)', # Gray - maintained
                'rgba(59,130,246,0.3)',  # Blue - invested
                'rgba(34,197,94,0.3)',
                'rgba(100,116,139,0.3)',
                'rgba(59,130,246,0.3)'
            ]
        )
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(t=30, b=30, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', size=11)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary stats
    spend_delta = total_after - total_before
    spend_delta_pct = (spend_delta / total_before * 100) if total_before > 0 else 0
    delta_color = "#22c55e" if spend_delta < 0 else "#3b82f6" if spend_delta > 0 else "#64748b"
    delta_prefix = "+" if spend_delta > 0 else ""
    
    st.markdown(f"""
    <div style="display: flex; justify-content: space-around; text-align: center; margin-top: 8px;">
        <div>
            <span style="color: #22c55e; font-weight: 600;">🟢 Reduced:</span>
            <span style="color: #94a3b8;"> {currency}{reduced_before - reduced_after:,.0f} freed</span>
        </div>
        <div>
            <span style="color: #3b82f6; font-weight: 600;">🔵 Increased:</span>
            <span style="color: #94a3b8;"> {currency}{increased_after - increased_before:,.0f} invested</span>
        </div>
        <div>
            <span style="color: {delta_color}; font-weight: 600;">Net:</span>
            <span style="color: #94a3b8;"> {delta_prefix}{currency}{spend_delta:,.0f} ({delta_prefix}{spend_delta_pct:.1f}%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Legacy chart functions (kept for backward compatibility but not called)
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
    """Render detailed drill-down table with decision-adjusted metrics."""
    
    import numpy as np
    
    with st.expander("📋 Detailed Action Log", expanded=False):
        if impact_df.empty:
            st.info("No actions to display")
            return
        
        # Create display dataframe with all decision-adjusted calculations
        display_df = impact_df.copy()
        
        # Add migration badge for HARVEST with before_spend > 0
        if show_migration_badge and 'is_migration' in display_df.columns:
            display_df['action_display'] = display_df.apply(
                lambda r: f"🔄 {r['action_type']}" if r.get('is_migration', False) else r['action_type'],
                axis=1
            )
        else:
            display_df['action_display'] = display_df['action_type']
        
        # ==========================================
        # USE BACKEND-CALCULATED METRICS (Single Source of Truth)
        # All decision metrics are pre-calculated in get_action_impact()
        # Frontend only displays - no recalculation
        # ==========================================
        
        # Ensure columns exist (defensive - backend should provide these)
        for col in ['decision_impact', 'spend_avoided', 'cpc_before', 'cpc_after', 
                    'cpc_change_pct', 'expected_sales', 'spc_before', 'market_downshift']:
            if col not in display_df.columns:
                display_df[col] = np.nan
        
        # Market Tag logic - uses backend-calculated market_downshift
        def get_market_tag(row):
            if row.get('before_clicks', 0) == 0:
                return "Low Data"
            if row.get('market_downshift', False) == True:
                return "Market Downshift"
            return "Normal"
        display_df['market_tag'] = display_df.apply(get_market_tag, axis=1)
        
        # Decision Outcome logic
        def get_decision_outcome(row):
            action = str(row['action_type']).upper()
            di = row['decision_impact'] if pd.notna(row['decision_impact']) else 0
            sa = row['spend_avoided'] if pd.notna(row['spend_avoided']) else 0
            bs = row['before_spend'] if pd.notna(row['before_spend']) else 0
            market_tag = row['market_tag']
            
            # Low Data → Neutral
            if market_tag == "Low Data":
                return "🟡 Neutral"
            
            # Good: DI > 0 OR (defensive + significant spend avoided + market downshift)
            if di > 0:
                return "🟢 Good"
            if action in ['BID_DOWN', 'PAUSE', 'NEGATIVE'] and bs > 0 and sa >= 0.1 * bs:
                return "🟢 Good"
            
            # Neutral: small impact
            before_sales = row['before_sales'] if pd.notna(row['before_sales']) else 0
            threshold = max(0.05 * before_sales, 10)  # 5% of before_sales or $10
            if abs(di) < threshold:
                return "🟡 Neutral"
            
            # Bad: negative impact in normal market
            if di < 0 and market_tag == "Normal":
                return "🔴 Bad"
            
            # Default to Neutral for edge cases
            return "🟡 Neutral"
        
        display_df['decision_outcome'] = display_df.apply(get_decision_outcome, axis=1)
        
        # ==========================================
        # SELECT FINAL COLUMNS (per spec)
        # ==========================================
        display_cols = [
            'action_display', 'target_text', 'reason',
            'before_spend', 'observed_after_spend', 'spend_avoided',
            'before_sales', 'observed_after_sales',
            'cpc_before', 'cpc_after', 'cpc_change_pct',
            'expected_sales', 'decision_impact',
            'market_tag', 'decision_outcome', 'validation_status'
        ]
        
        # Filter to columns that actually exist
        cols_to_use = [c for c in display_cols if c in display_df.columns]
        display_df = display_df[cols_to_use].copy()
        
        # Rename for user-friendly display
        final_rename = {
            'action_display': 'Action Taken',
            'target_text': 'Target',
            'reason': 'Logic Basis',
            'before_spend': 'Before Spend',
            'observed_after_spend': 'After Spend',
            'spend_avoided': 'Spend Avoided',
            'before_sales': 'Before Sales',
            'observed_after_sales': 'After Sales',
            'cpc_before': 'CPC Before',
            'cpc_after': 'CPC After',
            'cpc_change_pct': 'CPC Change %',
            'expected_sales': 'Expected Sales',
            'decision_impact': 'Decision Impact',
            'market_tag': 'Market Tag',
            'decision_outcome': 'Decision Outcome',
            'validation_status': 'Validation Status'
        }
        display_df = display_df.rename(columns=final_rename)
        
        # Format currency columns
        from utils.formatters import get_account_currency
        df_currency = get_account_currency()
        currency_cols = ['Before Spend', 'After Spend', 'Spend Avoided', 'Before Sales', 'After Sales', 'Expected Sales', 'Decision Impact']
        for col in currency_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{df_currency}{x:,.2f}" if pd.notna(x) else "-")
        
        # Format CPC columns
        cpc_cols = ['CPC Before', 'CPC After']
        for col in cpc_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{df_currency}{x:.2f}" if pd.notna(x) else "-")
        
        # Format CPC Change %
        if 'CPC Change %' in display_df.columns:
            display_df['CPC Change %'] = display_df['CPC Change %'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
        
        # Show migration legend if applicable
        if show_migration_badge and 'is_migration' in impact_df.columns and impact_df['is_migration'].any():
            st.caption("🔄 = **Migration Tracking**: Efficiency gain from harvesting search term to exact match.")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Decision Impact": st.column_config.TextColumn(
                    "Decision Impact",
                    help="Market-adjusted: After_Sales - Expected_Sales (what would have happened without change)"
                ),
                "Expected Sales": st.column_config.TextColumn(
                    "Expected Sales",
                    help="Counterfactual: (After_Spend / CPC_Before) × CVR_Before × AOV_Before"
                ),
                "Market Tag": st.column_config.TextColumn(
                    "Market Tag",
                    help="Normal | Market Downshift (CPC dropped >25%) | Low Data (no baseline clicks)"
                ),
                "Decision Outcome": st.column_config.TextColumn(
                    "Decision Outcome",
                    help="Good: positive impact or successful defense | Neutral: small/ambiguous | Bad: negative impact in stable market"
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
    Returns impact summary metrics from DB for the last 14 days (Decision Impact focus).
    Matches new 'Recent Impact' tile definition.
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
        # Direct DB query - 14 DAY WINDOW for "Recent" impact
        summary = db_manager.get_impact_summary(selected_client, after_days=14)
        
        if not summary:
            return None
        
        # Handle dual-summary structure (all/validated)
        active_summary = summary.get('validated', summary.get('all', summary))
        
        if active_summary.get('total_actions', 0) == 0:
            return None
        
        # Extract key metrics - DECISION IMPACT FOCUS
        decision_impact = active_summary.get('decision_impact', 0)
        win_rate = active_summary.get('win_rate', 0)
        
        # Get top action type
        by_type = active_summary.get('by_action_type', {})
        top_action_type = None
        if by_type:
            top_action_type = max(by_type, key=lambda k: by_type[k].get('decision_impact', 0)) # Sort by impact
        
        pct_good = active_summary.get('pct_good', 0)
        pct_bad = active_summary.get('pct_bad', 0)
        quality_score = pct_good - pct_bad

        return {
            'sales': decision_impact, # Mapped to 'sales' key for compatibility but represents impact
            'label': 'Decision Impact',
            'win_rate': win_rate,
            'top_action_type': top_action_type,
            'quality_score': quality_score,
            'roi': active_summary.get('roas_lift_pct', 0)
        }
        
    except Exception as e:
        print(f"[Impact Summary] Error: {e}")
        return None

