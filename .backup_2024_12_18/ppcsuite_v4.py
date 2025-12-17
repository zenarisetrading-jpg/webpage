"""
S2C LaunchPad Suite - V4 (Consolidated)
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

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="S2C LaunchPad Suite V4", 
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
    df = hub.get_data("search_term_report")
    if df is None or df.empty:
        st.error("❌ No Search Term Report data found in session.")
        return
    
    # Show which data we're using
    upload_ts = st.session_state.unified_data.get('upload_timestamps', {}).get('search_term_report')
    if upload_ts:
        st.info(f"📊 Using Search Term Report uploaded at {upload_ts.strftime('%Y-%m-%d %H:%M')}")
    
    # Apply enrichment (IDs, SKUs) to the fresh data WITHOUT mixing with DB historical data
    # This adds CampaignId, AdGroupId, SKU, etc. from bulk/APR files
    enriched = hub.get_enriched_data()
    if enriched is not None and len(enriched) == len(df):
        # Use enriched version if it matches the upload size (no extra rows from DB)
        df = enriched
        
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
            opt.config["HARVEST_SALES"] = st.number_input(
                "Min Sales ($)", value=opt.config["HARVEST_SALES"], min_value=0.0, step=10.0, key="main_h_sales"
            )
        
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
        matcher = ExactMatcher(df_prep)
        
        harvest_df = identify_harvest_candidates(df_prep, opt.config, matcher)
        # Build harvested terms set from BOTH Harvest_Term (Targeting-based) and Customer Search Term
        corrected_term_set = set()
        if not harvest_df.empty:
            if "Harvest_Term" in harvest_df.columns:
                corrected_term_set.update(harvest_df["Harvest_Term"].str.lower().tolist())
            if "Customer Search Term" in harvest_df.columns:
                corrected_term_set.update(harvest_df["Customer Search Term"].str.lower().tolist())
        neg_kw, neg_pt, your_products_review = identify_negative_candidates(df_prep, opt.config, harvest_df)
        
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
            df_prep, opt.config, corrected_term_set, negative_terms_set
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
    # LOG ACTIONS FOR IMPACT ANALYSIS (Silent)
    # ==========================================
    from features.optimizer import _log_optimization_events
    try:
        client_id = st.session_state.get('last_stats_save', {}).get('client_id', 'default_client')
        report_date = date_info.get('start_date') or st.session_state.get('last_stats_save', {}).get('start_date')
        _log_optimization_events(r, client_id, report_date)
    except Exception:
        pass  # Fail silently

    # B. Render Tabs (Integrated)
    # Order: Overview | Negatives | Competitor Shield | Bids | Harvest | Audit | Download
    tabs = st.tabs([
        "Overview",   
        "Negatives",
        "Competitor Shield",
        "Bids",
        "Harvest",
        "Audit",
        "Download"
    ])
    
    # Core Tabs
    with tabs[0]:
        opt._display_dashboard_v2({
             "direct_bids": direct_bids, "agg_bids": agg_bids,
             "harvest": harvest_df, "simulation": simulation,
             "df": df_prep, "neg_kw": neg_kw
        })
        
    with tabs[1]:
        opt._display_negatives(neg_kw, neg_pt)
        
    with tabs[2]:
        st.subheader("Competitor Shield")
        asin_module = ASINMapperModule()
        asin_module.run()
        
    with tabs[3]:
        opt._display_bids(bids_exact=bids_exact, bids_pt=bids_pt, bids_agg=bids_agg, bids_auto=bids_auto)
        
    with tabs[4]:
        opt._display_harvest(harvest_df)
        
    with tabs[5]:
        opt._display_heatmap(heatmap_df)

    with tabs[6]:
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
    
    # Simplified V4 Sidebar - Remove Feature Breakdown since they are tabs now
    with st.sidebar:
        st.title("S2C LaunchPad")
        
        # Theme Toggle
        from ui.theme import ThemeManager
        ThemeManager.render_toggle()
        
        st.caption("v4.0 — PPC Intelligence Suite")
        
        # Account Selector
        from ui.account_manager import render_account_selector
        render_account_selector()
        
        if st.button("Home", use_container_width=True):
            st.session_state['current_module'] = 'home'
        
        st.markdown("##### SYSTEM")
        if st.button("Data Upload", use_container_width=True):
            st.session_state['current_module'] = 'data_hub'
        
        st.markdown("##### ANALYZE")
        if st.button("Account Overview", use_container_width=True):
            st.session_state['current_module'] = 'performance'
        if st.button("Impact Analyzer", use_container_width=True):
            st.session_state['current_module'] = 'impact'
        if st.button("📄 Report Card", use_container_width=True):
            st.session_state['current_module'] = 'report_card'
        if st.button("Optimization Hub", use_container_width=True):
            st.session_state['current_module'] = 'optimizer'
        if st.button("Simulator", use_container_width=True):
            st.session_state['current_module'] = 'simulator'
        
        
        st.markdown("##### ACTIONS")
        if st.button("Campaign Launcher", use_container_width=True):
             st.session_state['current_module'] = 'creator'
        if st.button("Ask Zenny", use_container_width=True):
             st.session_state['current_module'] = 'assistant'
        
        st.divider()
        if st.button("Help", use_container_width=True):
            st.session_state['current_module'] = 'readme'
        
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
        PerformanceSnapshotModule().run()
    
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
    elif current == 'report_card':
        ReportCardModule().run()

if __name__ == "__main__":
    main()


