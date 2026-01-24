"""
Saddle AdPulse - V4 (Consolidated)
Main Application Entry Point

Features:
- Optimizer now acts as the central hub.
- ASIN Intent Mapper and AI Insights (Clusters) are integrated as tabs within Optimizer.
- Clean Home Page.
- Consolidated navigation.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os

# BRIDGE: Load Streamlit Secrets into OS Environment for Core Modules
# This allows db_manager to find 'DATABASE_URL' without importing streamlit directly
try:
    if "DATABASE_URL" in st.secrets:
        os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
except FileNotFoundError:
    pass # No secrets file found, rely on standard env vars or local DB


# Import Core Modules
from ui.layout import setup_page, render_sidebar, render_home
from core.data_hub import DataHub
from features.optimizer import (
    OptimizerModule, 
    prepare_data, 
    identify_harvest_candidates, 
    identify_negative_candidates, 
    calculate_bid_optimizations, 
    create_heatmap, 
    run_simulation,
    calculate_account_benchmarks,
    DEFAULT_CONFIG
)
from features.creator import CreatorModule
from features.asin_mapper import ASINMapperModule
from features.kw_cluster import AIInsightsModule
from features.simulator import SimulatorModule
from features.assistant import AssistantModule
from features.performance_snapshot import PerformanceSnapshotModule
from features.report_card import ReportCardModule
from utils.matchers import ExactMatcher
from utils.formatters import format_currency
from core.data_loader import safe_numeric
from core.db_manager import DatabaseManager, get_db_manager
from features.impact_dashboard import render_impact_dashboard, render_reference_data_badge
from pathlib import Path

# === AUTHENTICATION ===
from auth import require_authentication, render_user_menu

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Saddle AdPulse", 
    layout="wide", 
    page_icon="🚀"
)

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
        ReportCardModule().run()
    else:
        PerformanceSnapshotModule().run()

# ==========================================
# CONSOLIDATED V4 OPTIMIZER
# ==========================================
def run_consolidated_optimizer():
    """Execution logic: Optimizer + ASIN Mapper + AI Insights all in one view."""
    
    st.title("📊 Optimization Engine")
    
    # Check for data
    hub = DataHub()
    if not hub.is_loaded("search_term_report"):
        st.warning("⚠️ Please upload a Search Term Report in the Data Hub first.")
        st.info("Go to **Data Hub** → Upload files → Return here")
        return
    
    # FIXED: Use ONLY the freshly uploaded STR from current session
    # This ensures we optimize the specific time period uploaded, not historical DB data
    df_raw = hub.get_data("search_term_report")
    if df_raw is None or df_raw.empty:
        st.error("❌ No Search Term Report data found in session.")
        return
    
    # Work with a copy to avoid modifying Hub data in-place
    df = df_raw.copy()
    
    
    # Apply enrichment (IDs, SKUs) to the fresh data WITHOUT mixing with DB historical data
    # This adds CampaignId, AdGroupId, SKU, etc. from bulk/APR files
    enriched = hub.get_enriched_data()
    if enriched is not None and len(enriched) == len(df):
        # Use enriched version if it matches the upload size (no extra rows from DB)
        df = enriched.copy()

    # =====================================================
    # DB INTEGRATION: Allow extending window with historical data
    # =====================================================
    client_id = st.session_state.get('active_account', {}).get('account_id')
    db_manager = st.session_state.get('db_manager')
    
    include_db = False
    if client_id and db_manager:
        with st.expander("🛠️ Data Controls", expanded=False):
            include_db = st.checkbox("Include Historical Data (from Database)", value=False, help="Extend analysis window by pulling previous records from the database.")
            if include_db:
                with st.spinner("Fetching historical data..."):
                    db_df = db_manager.get_target_stats_df(client_id)
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
                        
                        # Merge current upload with DB data
                        # We use Date, Campaign, Ad Group, and Targeting as keys to avoid duplicates
                        combined = pd.concat([df, db_df], ignore_index=True)
                        
                        # Fix types for deduplication
                        combined['Date'] = pd.to_datetime(combined['Date'])
                        combined['Campaign Name'] = combined['Campaign Name'].astype(str).str.strip()
                        combined['Ad Group Name'] = combined['Ad Group Name'].astype(str).str.strip()
                        combined['Targeting'] = combined['Targeting'].astype(str).str.strip()
                        
                        # Drop duplicates (keep newest/session data which might have more recent metrics)
                        df = combined.drop_duplicates(subset=['Date', 'Campaign Name', 'Ad Group Name', 'Targeting'], keep='first')
                        st.success(f"✅ Merged {len(db_df)} historical records from database.")
    
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
        
        with st.expander("📅 Date Range Selection", expanded=True):
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
        mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
        df = df[mask].copy()
        days_selected = (end_date - start_date).days + 1
        st.caption(f"📆 Analyzing **{days_selected} days** ({start_date.strftime('%b %d')} - {end_date.strftime('%b %d')}) | {len(df):,} rows")
        
    # Helper to calculate summary metrics early for Header
    if "Spend" in df.columns and "Sales" in df.columns:
        # CLEANUP: Ensure numeric types for calculation
        df["Spend"] = safe_numeric(df["Spend"])
        df["Sales"] = safe_numeric(df["Sales"])
        
        from ui.components import metric_card
        
        c1, c2, c3, c4 = st.columns(4)
        total_spend = df["Spend"].sum()
        total_sales = df["Sales"].sum()
        roas = total_sales / total_spend if total_spend > 0 else 0
        acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
        
        with c1: metric_card("Total Spend", format_currency(total_spend))
        with c2: metric_card("Total Sales", format_currency(total_sales))
        with c3: metric_card("ROAS", f"{roas:.2f}x")
        with c4: metric_card("ACoS", f"{acos:.1f}%")
        st.divider()

    # 1. Configuration - render in main panel BEFORE run, sidebar AFTER run
    opt = OptimizerModule()
    
    if not st.session_state.get("run_optimizer"):
        # === PRE-RUN STATE: Show settings in MAIN panel ===
        st.subheader("Configure Optimization")
        
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
            opt.config["NEGATIVE_SPEND_THRESHOLD"] = st.number_input(
                "Min Spend (0 Sales)", value=opt.config["NEGATIVE_SPEND_THRESHOLD"], min_value=1.0, key="main_neg_spend"
            )
        
        # Min Clicks per Bucket (collapsible)
        with st.expander("⚙️ Min Clicks per Bucket", expanded=False):
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
        
        st.divider()
        
        run_sim = st.checkbox("Include Simulation", value=True, key="run_simulation_main")
        
        # Teal button styling
        st.markdown("""
        <style>
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%) !important;
            border: none !important;
            font-size: 1.1rem !important;
            padding: 0.75rem 2rem !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("Run Optimization", type="primary", use_container_width=True):
            st.session_state["optimizer_config"] = opt.config.copy()  # SAVE OH panel values to session state!
            st.session_state["run_optimizer"] = True
            st.session_state["run_simulation"] = run_sim
            st.rerun()
        
        return  # Exit early, don't render results yet
    
    # === POST-RUN STATE: Settings move to sidebar ===
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
        
        bids_exact, bids_pt, bids_agg, bids_auto = calculate_bid_optimizations(
            df_prep, opt.config, corrected_term_set, negative_terms_set, universal_median
        )
        # Combine for backward compatibility with simulation/heatmap
        direct_bids = pd.concat([bids_exact, bids_pt], ignore_index=True)
        agg_bids = pd.concat([bids_agg, bids_auto], ignore_index=True)
        heatmap_df = create_heatmap(df_prep, opt.config, harvest_df, neg_kw, neg_pt, direct_bids, agg_bids)
        
        simulation = None
        if st.session_state.get("run_simulation", True):
            simulation = run_simulation(df_prep, direct_bids, agg_bids, harvest_df, opt.config, date_info)
        
        # Persist results for AI Assistant
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
            "df": df_prep
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
    
    # 2. Determine report date (defaults to max date in data if missing)
    action_log_date = date_info.get('start_date')
    if action_log_date and isinstance(action_log_date, datetime):
        action_log_date = action_log_date.strftime('%Y-%m-%d')
    elif not action_log_date:
        # Fallback to current date or last stats save
        action_log_date = st.session_state.get('last_stats_save', {}).get('start_date') or datetime.now().strftime('%Y-%m-%d')
    
    # 3. Trigger Logging
    logged_count = _log_optimization_events(r, active_client, action_log_date)
    if logged_count > 0:
        st.toast(f"✅ Logged {logged_count} actions for impact analysis.", icon="📊")

    # B. Render Tabs (Integrated)
    # Order: Overview | Negatives | Competitor Shield | Bids | Harvest | Audit | Download
    tabs = st.tabs([
        "Overview",   
        "Defence",
        "Bids",
        "Harvest",
        "Audit",
        "Bulk Export"
    ])
    
    # Core Tabs
    with tabs[0]:
        opt._display_dashboard_v2({
             "direct_bids": direct_bids, "agg_bids": agg_bids,
             "harvest": harvest_df, "simulation": simulation,
             "df": df_prep, "neg_kw": neg_kw
        })
        
    with tabs[1]:
        defence_tabs = st.tabs(["Keyword Defence", "ASIN Defence"])
        with defence_tabs[0]:
            opt._display_negatives(neg_kw, neg_pt)
        with defence_tabs[1]:
            st.subheader("ASIN Defence")
            asin_module = ASINMapperModule()
            asin_module.run()
        
    with tabs[2]:
        opt._display_bids(bids_exact=bids_exact, bids_pt=bids_pt, bids_agg=bids_agg, bids_auto=bids_auto)
        
    with tabs[3]:
        opt._display_harvest(harvest_df)
        
    with tabs[4]:
        opt._display_heatmap(heatmap_df)

    with tabs[5]:
        res = {
            "harvest": harvest_df,
            "neg_kw": neg_kw, "neg_pt": neg_pt,
            "direct_bids": direct_bids, "agg_bids": agg_bids,
            "date_info": date_info,
            "heatmap": heatmap_df,
            "simulation": simulation 
        }
        opt._display_downloads(res)


# ==========================================
# MAIN ROUTER
# ==========================================
def main():
    setup_page()
    
    # === CONFIRMATION DIALOG CHECK ===
    # If confirmation is needed, show popup dialog (overlays on current page)
    if st.session_state.get('_show_action_confirmation'):
        from ui.action_confirmation import render_action_confirmation_modal
        render_action_confirmation_modal()
        # Dialog shows as popup - continue rendering page underneath
    
    # === AUTHENTICATION GATE ===
    # Shows login page if not authenticated, blocks access to main app
    user = require_authentication()
    
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
    
    # Simplified V4 Sidebar - Remove Feature Breakdown since they are tabs now
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
        
        # Account Selector
        from ui.account_manager import render_account_selector
        render_account_selector()
        
        # Consolidation: render_account_selector used to have its own line, 
        # now callers handle it. 
        st.markdown("---")
        
        # User menu (logout, account settings)
        render_user_menu()
        
        if st.button("Home", use_container_width=True):
            safe_navigate('home')
        if st.button("Account Overview", use_container_width=True):
            safe_navigate('performance')
        if st.button("Report Card", use_container_width=True):
            safe_navigate('report_card')
        
        st.markdown("##### SYSTEM")
        if st.button("Data Hub", use_container_width=True):
            safe_navigate('data_hub')
        
        st.markdown("##### ANALYZE")
        if st.button("Impact Analyzer", use_container_width=True):
            safe_navigate('impact')
        if st.button("Optimization Hub", use_container_width=True):
            safe_navigate('optimizer')
        if st.button("Simulator", use_container_width=True):
            safe_navigate('simulator')
        
        
        st.markdown("##### ACTIONS")
        if st.button("Campaign Launcher", use_container_width=True):
             safe_navigate('creator')
        if st.button("Ask Zenny", use_container_width=True):
             safe_navigate('assistant')
        
        st.divider()
        if st.button("Help", use_container_width=True):
            safe_navigate('readme')
        
        # Show undo toast if available
        from ui.action_confirmation import show_undo_toast
        show_undo_toast()
        
        # Theme Toggle at BOTTOM
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
            st.caption("Using: `ppc_live.db`")
            
    # Routing
    current = st.session_state.get('current_module', 'home')
    
    # Show test mode warning banner
    if st.session_state.get('test_mode', False):
        st.warning("⚠️ **TEST MODE ACTIVE** — All data is being saved to `ppc_test.db`. Switch off to use production database.")
    
    if current == 'home':
        render_home()
    
    elif current == 'data_hub':
        from ui.data_hub import render_data_hub
        render_data_hub()
        
    elif current == 'readme':
        from ui.readme import render_readme
        render_readme()
    
    elif current == 'optimizer':
        run_consolidated_optimizer()
        
    elif current == 'simulator':
        SimulatorModule().run()
        
    elif current == 'performance':
        run_performance_hub()
    
    elif current == 'creator':
        creator = CreatorModule()
        creator.run()
    
    elif current == 'assistant':
        AssistantModule().render_interface()
        
    # ASIN/AI modules are now inside Optimizer, but we keep routing valid just in case
    elif current == 'asin_mapper':
        ASINMapperModule().run()
    elif current == 'ai_insights':
        AIInsightsModule().run()
    elif current == 'impact':
        render_impact_dashboard()

    # Render Floating Chat Bubble (unless already on assistant page)
    if current != 'assistant':
        assistant = AssistantModule()
        assistant.render_floating_interface()
        assistant.render_interface()

if __name__ == "__main__":
    main()

