import streamlit as st

# ==========================================
# PAGE CONFIGURATION (Must be very first ST command)
# ==========================================
st.set_page_config(
    page_title="Saddle AdPulse", 
    layout="wide", 
    page_icon="🚀"
)

import pandas as pd
from datetime import datetime
import os

# BRIDGE: Load Streamlit Secrets into OS Environment for Core Modules
try:
    if "DATABASE_URL" in st.secrets:
        os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
except FileNotFoundError:
    pass 


# Delay heavy feature imports by moving them into routing/main logic
from ui.layout import setup_page, render_sidebar, render_home
from core.data_hub import DataHub
from core.db_manager import DatabaseManager, get_db_manager
from utils.matchers import ExactMatcher
from utils.formatters import format_currency
from core.data_loader import safe_numeric
from pathlib import Path

# === AUTHENTICATION ===
from auth import require_authentication, render_user_menu

# Global dark theme CSS for sidebar buttons
st.markdown("""
<style>
/* Fix sidebar buttons in dark mode */
[data-testid="stSidebar"] .stButton > button {
    background-color: rgba(30, 41, 59, 0.8) !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(51, 65, 85, 0.9) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}
/* Download buttons */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
}

/* Dark mode/Test mode toggle overrides */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #F5F5F7 !important;
    font-weight: 500 !important;
}
/* Toggle Switch Background when Checked */
[data-testid="stSidebar"] div[data-testid="stCheckbox"] > label > div[role="switch"][aria-checked="true"] {
    background-color: #5B556F !important;
}
/* Radio Button Outer Circle when active */
[data-testid="stSidebar"] div[data-testid="stRadio"] label div:first-child[data-baseweb="radio"] > div:first-child {
    border-color: #5B556F !important;
}
/* Radio Button Inner Dot when checked */
[data-testid="stSidebar"] div[data-testid="stRadio"] label div:first-child[data-baseweb="radio"] > div:first-child > div {
    background-color: #5B556F !important;
}

/* Print Mode: Hide sidebar and UI elements when printing */
@media print {
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    .stDeployButton { display: none !important; }
    .stDownloadButton { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    header { display: none !important; }
    .main .block-container { padding: 1rem !important; max-width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_module' not in st.session_state:
    st.session_state['current_module'] = 'home'

if 'data' not in st.session_state:
    st.session_state['data'] = {}

if 'test_mode' not in st.session_state:
    st.session_state['test_mode'] = False

if 'db_manager' not in st.session_state:
    st.session_state['db_manager'] = None


# ==========================================
# PERFORMANCE HUB (Snapshot + Report Card)
# ==========================================
# ==========================================
# PERFORMANCE HUB (Snapshot + Report Card)
# ==========================================
def run_performance_hub():
    """Consolidated Account Overview + Report Card."""
    # === TAB NAVIGATION (Premium Button Style) ===
    st.markdown("""
    <style>
    /* Premium Tab Buttons */
    div[data-testid="stHorizontalBlock"] div.stButton > button {
        background: rgba(143, 140, 163, 0.05) !important;
        border: 1px solid rgba(143, 140, 163, 0.15) !important;
        color: #8F8CA3 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        padding: 8px 16px !important;
    }
    div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
        background: rgba(143, 140, 163, 0.1) !important;
        border-color: rgba(91, 85, 111, 0.3) !important;
        color: #F5F5F7 !important;
    }
    /* Active Tab Styling - Using Primary kind */
    div[data-testid="stHorizontalBlock"] div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #5B556F 0%, #464156 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #F5F5F7 !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if 'active_perf_tab' not in st.session_state:
        st.session_state['active_perf_tab'] = "Account Health"
        
    c1, c2 = st.columns(2)
    with c1:
        is_active = st.session_state['active_perf_tab'] == "Account Health"
        if st.button("🛡️ ACCOUNT HEALTH", key="btn_tab_report", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state['active_perf_tab'] = "Account Health"
            st.rerun()
    with c2:
        is_active = st.session_state['active_perf_tab'] == "Performance Overview"
        if st.button("🧭 PERFORMANCE OVERVIEW", key="btn_tab_perf", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state['active_perf_tab'] = "Performance Overview"
            st.rerun()
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state['active_perf_tab'] == "Account Health":
        from features.report_card import ReportCardModule
        ReportCardModule().run()
    else:
        from features.performance_snapshot import PerformanceSnapshotModule
        PerformanceSnapshotModule().run()

# ==========================================
# CONSOLIDATED V4 OPTIMIZER
# ==========================================
def run_consolidated_optimizer():
    """Execution logic: Optimizer + ASIN Mapper + AI Insights all in one view."""
    # Lazy imports for Optimizer
    from features.optimizer import (
        OptimizerModule, 
        prepare_data, 
        identify_harvest_candidates, 
        identify_negative_candidates, 
        calculate_bid_optimizations, 
        create_heatmap, 
        run_simulation,
        calculate_account_benchmarks
    )
    from utils.matchers import ExactMatcher
    
    # Theme-aware optimization icon (sliders/tune icon)
    theme_mode = st.session_state.get('theme_mode', 'dark')
    opt_icon_color = "#94a3b8" if theme_mode == 'dark' else "#64748b"
    opt_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{opt_icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 10px;"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>'
    
    st.markdown(f'<h1 style="margin-bottom: 2rem; font-size: 2.2rem; font-weight: 800; text-transform: uppercase; letter-spacing: -0.02em; color: #F5F5F7;">{opt_icon}OPTIMIZATION ENGINE</h1>', unsafe_allow_html=True)
    
    # Check for data - try session first, then database
    hub = DataHub()
    
    # If no data in session, try loading from database
    if not hub.is_loaded("search_term_report"):
        account_id = st.session_state.get('active_account_id')
        if account_id:
            print(f"[Optimizer] No session data found. Attempting to load from database for account: {account_id}")
            loaded = hub.load_from_database(account_id)
            if loaded:
                print(f"[Optimizer] Successfully loaded {hub.get_summary().get('search_terms', 0):,} rows from database")
            else:
                print(f"[Optimizer] Database load returned False")
        
        # Check again after database load attempt
        if not hub.is_loaded("search_term_report"):
            st.warning("⚠️ Please upload a Search Term Report in the Data Hub first.")
            st.info("Go to **Data Hub** → Upload files → Return here")
            return
    
    # Get data (now either from session upload OR database)
    df_raw = hub.get_data("search_term_report")
    if df_raw is None or df_raw.empty:
        st.error("❌ No Search Term Report data found.")
        return
    
    # Work with a copy to avoid modifying Hub data in-place
    df = df_raw.copy()
    
    # Show which data we're using
    upload_ts = st.session_state.unified_data.get('upload_timestamps', {}).get('search_term_report')
    # (Callout content moved below to consolidate with date range context)
    
    # Apply enrichment (IDs, SKUs) to the fresh data WITHOUT mixing with DB historical data
    # This adds CampaignId, AdGroupId, SKU, etc. from bulk/APR files
    enriched = hub.get_enriched_data()
    if enriched is not None and len(enriched) == len(df):
        # Use enriched version if it matches the upload size (no extra rows from DB)
        df = enriched.copy()

    # =====================================================
    # DB INTEGRATION: Allow extending window with historical data
    # =====================================================
    # Retrieve client ID (support both new primitive key and old dict legacy)
    client_id = st.session_state.get('active_account_id')
    if not client_id:
        # Fallback to legacy dictionary if present
        client_id = st.session_state.get('active_account', {}).get('account_id')
    db_manager = st.session_state.get('db_manager')
    
    # DEBUG: Check why DB merge is skipped
    if not client_id:
        st.warning(f"⚠️ No Client ID detected. Session Active Account: {st.session_state.get('active_account')}")
    # End Debug
    
    include_db = False
    if client_id and db_manager:
        with st.expander("🛠️ Data Controls", expanded=True):
            # Default to TRUE so users see full history (e.g. for Visibility Boost)
            include_db = st.checkbox("Include Historical Data (from Database)", value=True, help="Extend analysis window by pulling previous records from the database.")
            if include_db:
                # DEBUG: Show what client_id we're using
                st.write(f"🔍 DEBUG: Fetching data for client_id = '{client_id}'")
                
                with st.spinner("Fetching historical data..."):
                    db_df = db_manager.get_target_stats_df(client_id)
                    
                    # DEBUG: Show DB fetch results
                    st.write(f"🔍 DEBUG: DB returned {len(db_df)} rows")
                    if not db_df.empty:
                        st.write(f"🔍 DEBUG: DB date range: {db_df['Date'].min()} to {db_df['Date'].max()}")
                    
                    if not db_df.empty:
                        # Rename columns for consistency if needed (DB already returns standard names)
                        # Fix column names to match STR upload headers
                        db_df = db_df.rename(columns={
                            'Date': 'Date',
                            'Targeting': 'Targeting',
                            'Match Type': 'Match Type',
                            'Campaign Name': 'Campaign Name',
                            'Ad Group Name': 'Ad Group Name'
                        })
                        
                        # CONSISTENCY FIX: Use DB data directly like Performance Overview
                        # No merge, no dedup - just use what's in the database
                        df = db_df.copy()
                        st.success(f"✅ Using complete database history: {len(df):,} rows from {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # =====================================================
    # DATE RANGE FILTER: Default to last 30 days
    # =====================================================
    date_cols = ["Date", "Start Date", "date", "Report Date", "start_date"]
    date_col = None
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        from datetime import datetime, timedelta
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        if include_db:
            # If DB included, allow full range selection
            default_start = min_date
        else:
            # Default: last 30 days from max date in data
            default_start = max(min_date, max_date - timedelta(days=30))
        
        # Theme-aware calendar icon inline in expander
        theme_mode = st.session_state.get('theme_mode', 'dark')
        with st.expander("▸ Date Range Selection", expanded=True):
            col_a, col_b = st.columns(2)
            # Use wider bounds (1 year back) if date filter needs more room
            abs_min = min(min_date, datetime.now() - timedelta(days=365))
            
            # --- CLAMP SESSION STATE TO VALID RANGE ---
            # This prevents StreamlitAPIException if the previous file had a wider range
            s_min, s_max = abs_min.date(), max_date.date()
            if "opt_date_start" in st.session_state:
                if st.session_state["opt_date_start"] < s_min: st.session_state["opt_date_start"] = s_min
                if st.session_state["opt_date_start"] > s_max: st.session_state["opt_date_start"] = s_max
            if "opt_date_end" in st.session_state:
                if st.session_state["opt_date_end"] < s_min: st.session_state["opt_date_end"] = s_min
                if st.session_state["opt_date_end"] > s_max: st.session_state["opt_date_end"] = s_max
            # ------------------------------------------

            with col_a:
                start_date = st.date_input("Start Date", 
                                            value=st.session_state.get("opt_date_start", default_start.date()), 
                                            min_value=s_min, 
                                            max_value=s_max,
                                            key="opt_date_start")
            with col_b:
                end_date = st.date_input("End Date", 
                                          value=st.session_state.get("opt_date_end", max_date.date()),
                                          min_value=s_min, 
                                          max_value=s_max,
                                          key="opt_date_end")
        
        # Filter data to selected range
        if date_col:
            # DEBUG: Check date range application
            st.write(f"DEBUG: Filtering on {date_col} from {start_date} to {end_date}")
            st.write(f"DEBUG: Before filter: {len(df)} rows. Min: {df[date_col].min()}, Max: {df[date_col].max()}")
            
            mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
            df = df[mask].copy()
            
            st.write(f"DEBUG: After filter: {len(df)} rows")
        else:
            st.error("❌ Could not identify Date column for filtering.")
            st.write("Columns found:", df.columns.tolist())
            
        days_selected = (end_date - start_date).days + 1
        
        # Baseline metrics calculation
        total_rows = len(df)
        total_spend = df["Spend"].sum()
        total_sales = df["Sales"].sum()
        roas = total_sales / total_spend if total_spend > 0 else 0
        acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
        
        # === SECTION 1 — DATASET CONTEXT (BASELINE) ===
        if not st.session_state.get("run_optimizer"):
            ts_info = f" • STR Upload: {upload_ts.strftime('%H:%M')}" if upload_ts else ""
            
            # Consistent icons for baseline metrics
            icon_color = "#8F8CA3"
            rows_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>'
            spend_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>'
            roas_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>'
            acos_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>'
            
            # Shared styling for Section 1 (Flatter, Background feel)
            context_tile_style = """
                background: rgba(143, 140, 163, 0.05);
                border: 1px solid rgba(143, 140, 163, 0.1);
                border-radius: 8px;
                padding: 12px;
                text-align: center;
            """
            context_label_style = "color: #8F8CA3; font-size: 0.7rem; text-transform: uppercase; font-weight: 500; letter-spacing: 0.5px; margin-bottom: 4px;"
            context_value_style = "color: #B6B4C2; font-size: 1.1rem; font-weight: 600;"

            st.markdown(f"""
            <div style="background: rgba(143, 140, 163, 0.02); border: 1px solid rgba(143, 140, 163, 0.05); border-radius: 12px; padding: 20px; margin-bottom: 40px; margin-top: 10px;">
                <div style="color: #8F8CA3; font-size: 0.8rem; margin-bottom: 16px; opacity: 0.8; padding-left: 4px; text-transform: uppercase; letter-spacing: 1px;">
                    Analyzing <strong>{start_date.strftime('%b %d')} – {end_date.strftime('%b %d, %Y')}</strong> ({days_selected} days){ts_info}
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
                    <div style="{context_tile_style}">
                        <div style="{context_label_style}">{rows_icon}Rows</div>
                        <div style="{context_value_style}">{total_rows:,}</div>
                    </div>
                    <div style="{context_tile_style}">
                        <div style="{context_label_style}">{spend_icon}Total Spend</div>
                        <div style="{context_value_style}">{format_currency(total_spend)}</div>
                    </div>
                    <div style="{context_tile_style}">
                        <div style="{context_label_style}">{roas_icon}ROAS</div>
                        <div style="{context_value_style}">{roas:.2f}x</div>
                    </div>
                    <div style="{context_tile_style}">
                        <div style="{context_label_style}">{acos_icon}ACoS</div>
                        <div style="{context_value_style}">{acos:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
    # 1. Configuration - render in main panel BEFORE run, sidebar AFTER run
    opt = OptimizerModule()
    
    if not st.session_state.get("run_optimizer"):
        # === PRE-RUN STATE: PRIMARY ACTION PANEL ===
        st.subheader("Ready to optimize")
        
        st.markdown(
            "The system will adjust bids, add negatives, and harvest high-performing terms "
            "based on current account performance."
        )
        
        # Brand purple/wine palette: #5B556F (Wine/Slate Purple)
        st.markdown("""
        <style>
        /* Primary CTA Button - Brand Wine */
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #5B556F 0%, #464156 100%) !important;
            border: none !important;
            font-size: 1.1rem !important;
            padding: 0.75rem 2rem !important;
            color: #F5F5F7 !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #6A6382 0%, #5B556F 100%) !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25) !important;
            color: #ffffff !important;
        }

        /* Slider Styling - Brand Wine */
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div:nth-child(2) {
            background: #5B556F !important;
        }
        div[data-testid="stSlider"] div[role="slider"] {
            background-color: #5B556F !important;
            border: 2px solid #F5F5F7 !important;
        }
        div[data-testid="stSlider"] span[data-baseweb="typography"] {
            color: #5B556F !important;
            font-weight: 700 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # === QUICK PRESETS ===
        st.markdown("**Quick Presets**")
        preset_options = ["Conservative", "Balanced", "Aggressive"]
        active_preset = st.session_state.get("last_preset", "Balanced")
        preset_idx = preset_options.index(active_preset) if active_preset in preset_options else 1
        
        preset = st.radio(
            "Choose optimization style",
            preset_options,
            index=preset_idx,
            horizontal=True,
            label_visibility="collapsed",
            key="opt_preset_main"
        )
        
        # Define preset values (no currency thresholds)
        preset_configs = {
            "Conservative": {
                "harvest_clicks": 15, "harvest_orders": 4, "harvest_roas": 90,
                "alpha_exact": 0.15, "alpha_broad": 0.12, "max_change": 0.15, "target_roas": 2.5,
                "neg_clicks": 15,
                "min_clicks_exact": 8, "min_clicks_pt": 8, "min_clicks_broad": 12, "min_clicks_auto": 12
            },
            "Balanced": {
                "harvest_clicks": 10, "harvest_orders": 3, "harvest_roas": 80,
                "alpha_exact": 0.20, "alpha_broad": 0.16, "max_change": 0.20, "target_roas": 2.5,
                "neg_clicks": 10,
                "min_clicks_exact": 5, "min_clicks_pt": 5, "min_clicks_broad": 10, "min_clicks_auto": 10
            },
            "Aggressive": {
                "harvest_clicks": 8, "harvest_orders": 2, "harvest_roas": 70,
                "alpha_exact": 0.25, "alpha_broad": 0.20, "max_change": 0.25, "target_roas": 2.5,
                "neg_clicks": 8,
                "min_clicks_exact": 3, "min_clicks_pt": 3, "min_clicks_broad": 8, "min_clicks_auto": 8
            }
        }
        
        # Apply preset to config if changed
        if st.session_state.get("last_preset") != preset:
            st.session_state["last_preset"] = preset
            config = preset_configs[preset]
            
            # Update opt.config
            opt.config["HARVEST_CLICKS"] = config["harvest_clicks"]
            opt.config["HARVEST_ORDERS"] = config["harvest_orders"]
            opt.config["HARVEST_ROAS_MULT"] = config["harvest_roas"] / 100
            opt.config["ALPHA_EXACT"] = config["alpha_exact"]
            opt.config["ALPHA_BROAD"] = config["alpha_broad"]
            opt.config["MAX_BID_CHANGE"] = config["max_change"]
            opt.config["TARGET_ROAS"] = config["target_roas"]
            opt.config["NEGATIVE_CLICKS_THRESHOLD"] = config["neg_clicks"]
            opt.config["MIN_CLICKS_EXACT"] = config["min_clicks_exact"]
            opt.config["MIN_CLICKS_PT"] = config["min_clicks_pt"]
            opt.config["MIN_CLICKS_BROAD"] = config["min_clicks_broad"]
            opt.config["MIN_CLICKS_AUTO"] = config["min_clicks_auto"]
            
            # CRITICAL: Also update session state so widgets pick up preset values on next rerun
            # This prevents widgets from resetting to default values
            st.session_state["main_h_clicks"] = config["harvest_clicks"]
            st.session_state["main_h_orders"] = config["harvest_orders"]
            st.session_state["main_alpha_exact"] = config["alpha_exact"]
            st.session_state["main_max_bid"] = config["max_change"]
            st.session_state["main_target_roas"] = config["target_roas"]
            st.session_state["main_neg_clicks"] = config["neg_clicks"]
            st.session_state["min_clicks_exact"] = config["min_clicks_exact"]
            st.session_state["min_clicks_pt"] = config["min_clicks_pt"]
            st.session_state["min_clicks_broad"] = config["min_clicks_broad"]
            st.session_state["min_clicks_auto"] = config["min_clicks_auto"]
        
        st.caption("*Conservative = slower, safer changes • Aggressive = faster, bigger changes*")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Simulation toggle ABOVE button for state initialization
        run_sim = st.checkbox("Include Simulation & Forecasting", value=True, key="run_simulation_main")
        
        # PRIMARY CTA (commit first)
        if st.button("Start optimization", type="primary", use_container_width=True):
            st.session_state["optimizer_config"] = opt.config.copy()
            st.session_state["run_optimizer"] = True
            st.session_state["should_log_actions"] = True  # Only log on explicit button click
            # Read current checkbox state
            st.session_state["run_simulation"] = st.session_state.get("run_simulation_main", True)
            st.rerun()
        
        st.divider()
        
        # SVG Icons for Pre-run
        icon_color = "#8F8CA3"
        settings_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>'
        
        # Chiclet Header Helper for Main Panel
        def main_chiclet_header(label, icon_html):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); 
                        border: 1px solid rgba(124, 58, 237, 0.2); 
                        border-radius: 8px; 
                        padding: 10px 15px; 
                        margin-bottom: -15px;
                        display: flex; 
                        align-items: center; 
                        gap: 10px;">
                {icon_html}
                <span style="color: #F5F5F7; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">{label}</span>
            </div>
            """, unsafe_allow_html=True)

        main_chiclet_header("Advanced Settings", settings_icon)
        with st.expander("Expand configuration", expanded=False):
            st.caption("Fine-tune optimization behavior — defaults work well for most accounts")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Harvest Thresholds**")
                opt.config["HARVEST_CLICKS"] = st.number_input(
                    "Min Clicks", value=opt.config["HARVEST_CLICKS"], min_value=1, key="main_h_clicks"
                )
                opt.config["HARVEST_ORDERS"] = st.number_input(
                    "Min Orders", value=opt.config["HARVEST_ORDERS"], min_value=1, key="main_h_orders"
                )
                # HARVEST_SALES removed - currency threshold doesn't work across geos
            
            with col2:
                st.markdown("**Bid Optimization**")
                opt.config["ALPHA_EXACT"] = st.slider(
                    "Alpha (Exact)", min_value=0.05, max_value=0.50, 
                    value=opt.config["ALPHA_EXACT"], step=0.05, key="main_alpha_exact"
                )
                opt.config["MAX_BID_CHANGE"] = st.slider(
                    "Max Change %", min_value=0.05, max_value=0.50, 
                    value=opt.config["MAX_BID_CHANGE"], step=0.05, key="main_max_bid"
                )
                opt.config["TARGET_ROAS"] = st.number_input(
                    "Target ROAS", value=opt.config["TARGET_ROAS"], 
                    min_value=0.5, max_value=10.0, step=0.1, key="main_target_roas"
                )
            
            with col3:
                st.markdown("**Negative Thresholds**")
                opt.config["NEGATIVE_CLICKS_THRESHOLD"] = st.number_input(
                    "Min Clicks (0 Sales)", value=opt.config["NEGATIVE_CLICKS_THRESHOLD"], min_value=5, key="main_neg_clicks"
                )
                # NEGATIVE_SPEND_THRESHOLD removed for currency-neutrality
                # All negative logic is now clicks-based
            
            st.divider()
            st.markdown("**Min Clicks per Bucket**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                opt.config["MIN_CLICKS_EXACT"] = st.number_input(
                    "Exact KW", value=opt.config.get("MIN_CLICKS_EXACT", 5), min_value=1, max_value=50, key="min_clicks_exact"
                )
            with c2:
                opt.config["MIN_CLICKS_PT"] = st.number_input(
                    "Prod Target", value=opt.config.get("MIN_CLICKS_PT", 5), min_value=1, max_value=50, key="min_clicks_pt"
                )
            with c3:
                opt.config["MIN_CLICKS_BROAD"] = st.number_input(
                    "Broad/Phrase", value=opt.config.get("MIN_CLICKS_BROAD", 10), min_value=1, max_value=50, key="min_clicks_broad"
                )
            with c4:
                opt.config["MIN_CLICKS_AUTO"] = st.number_input(
                    "Auto/Category", value=opt.config.get("MIN_CLICKS_AUTO", 10), min_value=1, max_value=50, key="min_clicks_auto"
                )
        
        return  # Exit early, don't render results yet
    
    # === POST-RUN STATE: Settings move to sidebar ===
    with st.sidebar:
        st.divider()
        if st.button("← Stop / Edit Settings", use_container_width=True):
            st.session_state["run_optimizer"] = False
            st.rerun()
    
    opt._render_sidebar()
    
    # 2. Main Logic Trigger
    with st.spinner("Running Optimization Engine..."):
        # A. Prepare Data & Run Core Logic
        df_prep, date_info = prepare_data(df, opt.config)
        
        # Consolidation Fix: Calculate benchmarks ONCE and pass them down
        benchmarks = calculate_account_benchmarks(df_prep, opt.config)
        universal_median = benchmarks.get('universal_median_roas', opt.config.get("TARGET_ROAS", 2.5))
        
        matcher = ExactMatcher(df_prep)
        
        harvest_df = identify_harvest_candidates(df_prep, opt.config, matcher, benchmarks)
        # Build harvested terms set from BOTH Harvest_Term (Targeting-based) and Customer Search Term
        corrected_term_set = set()
        if not harvest_df.empty:
            if "Harvest_Term" in harvest_df.columns:
                corrected_term_set.update(harvest_df["Harvest_Term"].str.lower().tolist())
            if "Customer Search Term" in harvest_df.columns:
                corrected_term_set.update(harvest_df["Customer Search Term"].str.lower().tolist())
        
        neg_kw, neg_pt, your_products_review = identify_negative_candidates(df_prep, opt.config, harvest_df, benchmarks)
        
        # Build negative_terms set for exclusion from bid optimization
        negative_terms_set = set()
        if not neg_kw.empty:
            for _, row in neg_kw.iterrows():
                key = (str(row.get("Campaign Name", "")).strip(), 
                       str(row.get("Ad Group Name", "")).strip(), 
                       str(row.get("Term", "")).strip().lower())
                negative_terms_set.add(key)
        if not neg_pt.empty:
            for _, row in neg_pt.iterrows():
                key = (str(row.get("Campaign Name", "")).strip(), 
                       str(row.get("Ad Group Name", "")).strip(), 
                       str(row.get("Term", "")).strip().lower())
                negative_terms_set.add(key)
        
        data_days = date_info.get("days", 7)
        bids_exact, bids_pt, bids_agg, bids_auto = calculate_bid_optimizations(
            df_prep, opt.config, corrected_term_set, negative_terms_set, universal_median,
            data_days=data_days
        )
        # Combine for backward compatibility with simulation/heatmap
        direct_bids = pd.concat([bids_exact, bids_pt], ignore_index=True)
        agg_bids = pd.concat([bids_agg, bids_auto], ignore_index=True)
        heatmap_df = create_heatmap(df_prep, opt.config, harvest_df, neg_kw, neg_pt, direct_bids, agg_bids)
        
        simulation = None
        if st.session_state.get("run_simulation", True):
            simulation = run_simulation(df_prep, direct_bids, agg_bids, harvest_df, opt.config, date_info)
        
        # Calculate health for Home Cockpit sync
        health = opt._calculate_account_health(df_prep, {"direct_bids": direct_bids, "agg_bids": agg_bids, "harvest": harvest_df})
        
        # Persist results for AI Assistant and Home Cockpit sync
        r = {
            "bids_exact": bids_exact,
            "bids_pt": bids_pt,
            "bids_agg": bids_agg,
            "bids_auto": bids_auto,
            "direct_bids": direct_bids,
            "agg_bids": agg_bids,
            "harvest": harvest_df,
            "neg_kw": neg_kw,
            "neg_pt": neg_pt,
            "heatmap": heatmap_df,
            "simulation": simulation,
            "date_info": date_info,
            "df": df_prep,
            "health": health  # ADDED for Home Cockpit sync
        }
        st.session_state['latest_optimizer_run'] = r
    
    # ==========================================
    # LOG ACTIONS FOR IMPACT ANALYSIS
    # ==========================================
    from features.optimizer import _log_optimization_events
    
    # 1. Determine active client
    active_client = (
        st.session_state.get('active_account_id') or 
        st.session_state.get('last_stats_save', {}).get('client_id') or 
        'default_client'
    )
    
    # 2. Determine report date (use END date of data range, not start)
    # This ensures actions are logged at the most recent data point, not the oldest
    action_log_date = date_info.get('end_date') or date_info.get('start_date')
    if action_log_date and isinstance(action_log_date, datetime):
        action_log_date = action_log_date.strftime('%Y-%m-%d')
    elif not action_log_date:
        # Fallback to current date
        action_log_date = datetime.now().strftime('%Y-%m-%d')
    
    # Toast removed per user request
    # Only log actions on EXPLICIT button click, not on every rerender
    if st.session_state.get("should_log_actions", False):
        logged_count = _log_optimization_events(r, active_client, action_log_date)
        st.session_state["should_log_actions"] = False  # Clear flag to prevent re-logging

    # === POST-OPTIMIZATION INSIGHT LAYER ===
    @st.fragment
    def render_optimizer_results(
        bids_exact, bids_pt, bids_agg, bids_auto, 
        neg_kw, neg_pt, harvest_df, simulation, 
        heatmap_df, df_prep, r, date_info
    ):
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 1. Metric Calculations
        total_touched = len(set(
            (bids_exact['Ad Group Name'].tolist() if not bids_exact.empty else []) +
            (bids_pt['Ad Group Name'].tolist() if not bids_pt.empty else []) +
            (bids_agg['Ad Group Name'].tolist() if not bids_agg.empty else []) +
            (bids_auto['Ad Group Name'].tolist() if not bids_auto.empty else []) +
            (pd.concat([neg_kw, neg_pt])['Ad Group Name'].tolist() if not pd.concat([neg_kw, neg_pt]).empty else []) +
            (harvest_df['Ad Group Name'].tolist() if not harvest_df.empty else [])
        ))

        total_bid_changes = len(bids_exact) + len(bids_pt) + len(bids_agg) + len(bids_auto)
        total_negatives = len(neg_kw) + len(neg_pt)
        total_harvests = len(harvest_df)
        
        # Icons & Styles
        icon_color = "#8F8CA3"
        layers_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>'
        sliders_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>'
        shield_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
        leaf_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8a8 8 0 0 1-8 8Z"></path><path d="M11 20c0-2.5 2-5.5 2-5.5"></path></svg>'
        search_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>'

        sim_badge = ""
        if simulation:
            sim_badge = '<span style="background: rgba(154, 219, 232, 0.08); color: #9ADBE8; padding: 4px 12px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; border: 1px solid rgba(154, 219, 232, 0.2); letter-spacing: 0.5px; text-transform: uppercase; float: right;">Simulation included</span>'

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h3 style="margin: 0; font-weight: 700;">Optimization Summary</h3>
            {sim_badge}
        </div>
        """, unsafe_allow_html=True)

        # Summary Tiles
        c1, c2, c3, c4 = st.columns(4)
        tile_style = "background: linear-gradient(135deg, rgba(91, 85, 111, 0.15) 0%, rgba(91, 85, 111, 0.08) 100%); border: 1px solid rgba(91, 85, 111, 0.3); border-radius: 12px; padding: 18px; text-align: center; backdrop-filter: blur(10px); box-shadow: 0 4px 24px rgba(0,0,0,0.06); transition: all 0.3s ease;"
        label_style = "color: #8F8CA3; font-size: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.7px; margin-bottom: 8px;"
        value_style = "color: #F5F5F7; font-size: 1.25rem; font-weight: 700;"

        with c1: st.markdown(f'<div style="{tile_style}"><div style="{label_style}">{layers_icon}Touched</div><div style="{value_style}">{total_touched}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div style="{tile_style}"><div style="{label_style}">{sliders_icon}Bids</div><div style="{value_style}">{total_bid_changes}</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div style="{tile_style}"><div style="{label_style}">{shield_icon}Negatives</div><div style="{value_style}">{total_negatives}</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div style="{tile_style}"><div style="{label_style}">{leaf_icon}Harvest</div><div style="{value_style}">{total_harvests}</div></div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

        # Tab Navigation
        tabs_list = [
            {"name": "Overview", "icon": layers_icon},
            {"name": "Defence", "icon": shield_icon},
            {"name": "Bids", "icon": sliders_icon},
            {"name": "Harvest", "icon": leaf_icon},
            {"name": "Audit", "icon": search_icon},
            {"name": "Bulk Export", "icon": search_icon}
        ]
        active_tab = st.session_state.get('active_opt_tab', 'Overview')
        tab_cols = st.columns(len(tabs_list))
        
        for i, tab in enumerate(tabs_list):
            if tab_cols[i].button(tab["name"], key=f"tab_{tab['name']}", use_container_width=True, type="primary" if active_tab == tab["name"] else "secondary"):
                st.session_state['active_opt_tab'] = tab["name"]
                st.rerun()

        st.markdown("<div style='margin-bottom: 32px;'></div>", unsafe_allow_html=True)
        active_tab = st.session_state.get('active_opt_tab', 'Overview')
        
        if active_tab == "Overview":
            opt._display_dashboard_v2({"direct_bids": direct_bids, "agg_bids": agg_bids, "harvest": harvest_df, "simulation": simulation, "df": df_prep, "neg_kw": neg_kw})
        elif active_tab == "Defence":
            opt._display_negatives(neg_kw, neg_pt)
        elif active_tab == "Bids":
            opt._display_bids(bids_exact=bids_exact, bids_pt=bids_pt, bids_agg=bids_agg, bids_auto=bids_auto)
        elif active_tab == "Harvest":
            opt._display_harvest(harvest_df)
        elif active_tab == "Audit":
            opt._display_heatmap(heatmap_df)
        elif active_tab == "Bulk Export":
            opt._display_downloads(r)

    # Invoke Fragment
    render_optimizer_results(
        bids_exact, bids_pt, bids_agg, bids_auto, 
        neg_kw, neg_pt, harvest_df, simulation, 
        heatmap_df, df_prep, r, date_info
    )

    # Original logic followed here - we can delete the redundant parts below
    



    # Bid Adjustments
    total_bid_changes = len(bids_exact) + len(bids_pt) + len(bids_agg) + len(bids_auto)
    
    # Negatives Identified
    total_negatives = len(neg_kw) + len(neg_pt)
    
    # Harvest Opportunities
    total_harvests = len(harvest_df)
    
    # Icons
    icon_color = "#8F8CA3"
    layers_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>'
    search_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>'
    sliders_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>'
    shield_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
    leaf_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 3.5 2 7a7 7 0 0 1-10 11z"></path></svg>'

    sim_badge = ""
    if simulation:
        sim_badge = '<span style="background: rgba(154, 219, 232, 0.08); color: #9ADBE8; padding: 4px 12px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; border: 1px solid rgba(154, 219, 232, 0.2); letter-spacing: 0.5px; text-transform: uppercase; float: right;">Simulation included</span>'










# ==========================================
# MAIN ROUTER
# ==========================================
def main():
    setup_page()
    
    # === AUTHENTICATION GATE ===
    # Shows login page if not authenticated, blocks access to main app
    user = require_authentication()
    
    # === DATABASE INITIALIZATION ===
    # Initialize db_manager right after auth, before any UI that needs it
    if st.session_state.get('db_manager') is None:
        st.session_state['db_manager'] = get_db_manager(st.session_state.get('test_mode', False))
    
    # === TOP-RIGHT HEADER (Profile, Account, Logout) ===
    # This renders a fixed-position header component
    render_user_menu()
    
    # Helper: Safe navigation (checks for pending actions when leaving optimizer)
    def safe_navigate(target_module):
        current = st.session_state.get('current_module', 'home')
        
        # Check if leaving optimizer with pending actions that haven't been accepted
        if current == 'optimizer' and target_module != 'optimizer':
            pending = st.session_state.get('pending_actions')
            accepted = st.session_state.get('optimizer_actions_accepted', False)
            
            if pending and not accepted:
                # Store the target and show confirmation
                st.session_state['_pending_navigation_target'] = target_module
                st.session_state['_show_action_confirmation'] = True
                st.rerun()
                return
        
        st.session_state['current_module'] = target_module
        st.rerun()
    
    # Simplified V4 Sidebar
    with st.sidebar:
        # Sidebar Logo at TOP (theme-aware, prominent)
        import base64
        from pathlib import Path
        theme_mode = st.session_state.get('theme_mode', 'dark')
        logo_filename = "saddle_logo.png" if theme_mode == 'dark' else "saddle_logo_light.png"
        logo_path = Path(__file__).parent / "static" / logo_filename
        
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                logo_data = base64.b64encode(f.read()).decode()
            st.markdown(
                f'<div style="text-align: center; padding: 15px 0 20px 0;"><img src="data:image/png;base64,{logo_data}" style="width: 200px;" /></div>',
                unsafe_allow_html=True
            )
        
        # Account selector (right after logo)
        from ui.account_manager import render_account_selector
        render_account_selector()
        
        # Logout button (compact)
        from auth.service import AuthService
        auth = AuthService()
        if st.button("⏻ Logout", key="sidebar_logout", use_container_width=True, help="Sign out"):
            auth.sign_out()
            st.rerun()
        
        st.divider()
        
        # =========================
        # PRIMARY NAVIGATION
        # =========================
        # Side Navigation Icons
        nav_icon_color = "#8F8CA3"
        home_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{nav_icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>'
        performance_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{nav_icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 12v5"/><path d="M12 9v8"/><path d="M17 11v6"/></svg>'
        report_card_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{nav_icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>'
        impact_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{nav_icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>'
        check_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{nav_icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="m9 11 3 3L22 4"></path><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path></svg>'
        sim_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{nav_icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>'
        rocket_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{nav_icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"></path><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"></path><path d="m9 12 2.5 2.5"></path></svg>'
        storage_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{nav_icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M21 20V4a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v16"></path><rect x="3" y="4" width="18" height="4" rx="2"></rect><rect x="3" y="12" width="18" height="4" rx="2"></rect></svg>'
        help_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{nav_icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
        settings_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{nav_icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>'

        # Stylized nav button with CSS injection for hover effects and integrated SVG
        st.markdown(f"""
        <style>
        .nav-chiclet {{
            background: rgba(143, 140, 163, 0.05);
            border: 1px solid rgba(143, 140, 163, 0.1);
            border-radius: 10px;
            padding: 10px 15px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
            color: #B6B4C2;
            text-decoration: none;
        }}
        .nav-chiclet:hover {{
            background: rgba(143, 140, 163, 0.1);
            border-color: rgba(124, 58, 237, 0.4);
            color: #F5F5F7;
            transform: translateX(4px);
        }}
        .nav-chiclet.active {{
            background: linear-gradient(135deg, rgba(124, 58, 237, 0.15) 0%, rgba(124, 58, 237, 0.08) 100%);
            border-color: rgba(124, 58, 237, 0.5);
            color: #F5F5F7;
        }}
        </style>
        """, unsafe_allow_html=True)

        def nav_chiclet_link(label, icon_html, module_key):
            is_active = st.session_state.get('current_module') == module_key
            active_class = "active" if is_active else ""
            
            # Use a transparent button over the chiclet for interactivity
            if st.button(label, key=f"nav_{module_key}", use_container_width=True):
                # Check if leaving optimizer with pending actions
                if st.session_state.get('current_module') == 'optimize' and st.session_state.get('pending_actions'):
                    # Trigger confirmation dialog instead of navigating
                    st.session_state['_show_action_confirmation'] = True
                    st.session_state['_pending_navigation_target'] = module_key
                    st.rerun()
                else:
                    # Navigate directly
                    st.session_state['current_module'] = module_key
                    st.rerun()

        # Re-using the nav_button logic but with the chiclet feel properly integrated
        # We'll use Streamlit's native buttons but style them to look like the chiclets
        st.markdown("""
        <style>
        /* Base Sidebar Button Styling - Targets all buttons in our custom wrappers */
        [data-testid="stSidebar"] .nav-item-wrapper div.stButton > button,
        [data-testid="stSidebar"] .sub-nav-wrapper div.stButton > button {
            background: rgba(143, 140, 163, 0.05) !important;
            border: 1px solid rgba(143, 140, 163, 0.1) !important;
            border-radius: 10px !important;
            color: #B6B4C2 !important;
            text-align: left !important;
            padding: 8px 12px !important;
            margin-bottom: 0px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        
        /* Balanced vertical spacing between sidebar elements */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.5rem !important;
        }
        
        /* Fix the alignment and gap of the icon column */
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
            gap: 0.5rem !important;
        }

        /* Balanced Dividers */
        [data-testid="stSidebar"] hr {
            margin: 1rem 0 !important;
            opacity: 0.15 !important;
        }
        
        [data-testid="stSidebar"] .nav-item-wrapper div.stButton > button:hover {
            background: rgba(143, 140, 163, 0.1) !important;
            border-color: rgba(91, 85, 111, 0.4) !important;
            color: #F5F5F7 !important;
            transform: translateX(4px) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Helper for stylized nav buttons with inline SVGs
        def nav_button_chiclet(label, icon_html, key):
            is_active = st.session_state.get('current_module') == key
            active_bg = "linear-gradient(135deg, rgba(91, 85, 111, 0.2) 0%, rgba(91, 85, 111, 0.1) 100%)" if is_active else "rgba(143, 140, 163, 0.05)"
            active_border = "rgba(91, 85, 111, 0.5)" if is_active else "rgba(143, 140, 163, 0.1)"
            
            # Use specific CSS for active button targeting the wrapper
            st.markdown(f"""
            <style>
            .nav-wrapper-{key} div.stButton > button {{
                background: {active_bg} !important;
                border-color: {active_border} !important;
                color: {"#F5F5F7" if is_active else "#B6B4C2"} !important;
            }}
            </style>
            """, unsafe_allow_html=True)

            st.markdown(f'<div class="nav-item-wrapper nav-wrapper-{key}">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 6])
            with col1:
                st.markdown(f'<div style="margin-top: 5px; margin-left: 5px; opacity: {"1.0" if is_active else "0.6"};">{icon_html}</div>', unsafe_allow_html=True)
            with col2:
                if st.button(label, use_container_width=True, key=f"nav_btn_v6_{key}"):
                    safe_navigate(key)
            st.markdown('</div>', unsafe_allow_html=True)

        nav_button_chiclet("Home", home_icon, "home")
        nav_button_chiclet("Account Overview", performance_icon, "performance")
        
        st.divider()
        st.markdown("##### ANALYZE")
        nav_button_chiclet("Actions Review", check_icon, "optimizer")
        nav_button_chiclet("What If (Forecast)", sim_icon, "simulator")
        nav_button_chiclet("Impact & Results", impact_icon, "impact")
        nav_button_chiclet("Launch", rocket_icon, "creator")

        st.divider()

        # =========================
        # SECONDARY / SYSTEM
        # =========================
        nav_button_chiclet("Data Setup", storage_icon, "data_hub")
        nav_button_chiclet("Account Settings", settings_icon, "account_settings")
        nav_button_chiclet("Help", help_icon, "readme")
        
        # Show undo toast if available
        from ui.action_confirmation import show_undo_toast
        show_undo_toast()
        
        # Theme Toggle (logout moved to top header)
        st.divider()
        from ui.theme import ThemeManager
        ThemeManager.render_toggle()
        
        # Database Mode Toggle (below Help)
        st.divider()
        test_mode = st.toggle("Test Mode", value=st.session_state.get('test_mode', False))
        if test_mode != st.session_state.get('test_mode', False):
            st.session_state['test_mode'] = test_mode
            st.session_state['db_manager'] = get_db_manager(test_mode)
            st.rerun()
        if st.session_state['db_manager'] is None:
            st.session_state['db_manager'] = get_db_manager(st.session_state['test_mode'])
        if st.session_state['test_mode']:
            st.caption("Using: `ppc_test.db`")
        else:
            # Show actual database type
            db = st.session_state.get('db_manager')
            if db and type(db).__name__ == 'PostgresManager':
                st.caption("Using: `Supabase (Postgres)`")
            else:
                st.caption("Using: `ppc_live.db`")
            
    # Routing
    current = st.session_state.get('current_module', 'home')
    
    # Check for pending actions confirmation dialog
    from ui.action_confirmation import render_action_confirmation_modal
    render_action_confirmation_modal()
    
    # Show test mode warning banner
    if st.session_state.get('test_mode', False):
        st.warning("⚠️ **TEST MODE ACTIVE** — All data is being saved to `ppc_test.db`. Switch off to use production database.")
    
    if current == 'home':
        render_home()
    
    elif current == 'data_hub':
        from ui.data_hub import render_data_hub
        render_data_hub()
    
    elif current == 'account_settings':
        from features.account_settings import run_account_settings
        run_account_settings()
        
    elif current == 'readme':
        from ui.readme import render_readme
        render_readme()
    
    elif current == 'optimizer':
        run_consolidated_optimizer()
        
    elif current == 'simulator':
        from features.simulator import SimulatorModule
        SimulatorModule().run()
        
    elif current == 'performance':
        run_performance_hub()
    
    elif current == 'creator':
        from features.creator import CreatorModule
        creator = CreatorModule()
        creator.run()
    
    elif current == 'assistant':
        from features.assistant import AssistantModule
        AssistantModule().render_interface()
        
    # ASIN/AI modules are now inside Optimizer, but we keep routing valid just in case
    elif current == 'asin_mapper':
        from features.asin_mapper import ASINMapperModule
        ASINMapperModule().run()
    elif current == 'ai_insights':
        from features.kw_cluster import AIInsightsModule
        AIInsightsModule().run()
    elif current == 'impact':
        from features.impact_dashboard import render_impact_dashboard
        render_impact_dashboard()

    # Render Floating Chat Bubble (unless already on assistant page)
    if current != 'assistant':
        from features.assistant import AssistantModule
        assistant = AssistantModule()
        assistant.render_floating_interface()
        assistant.render_interface()

if __name__ == "__main__":
    main()

