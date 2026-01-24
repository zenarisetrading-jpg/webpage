"""
Saddle AdPulse - Main Application
Central entry point

This is the thin orchestrator that coordinates all modules.
All business logic lives in features/ directory.
"""

import streamlit as st
from ui.layout import setup_page, render_sidebar, render_home
from features.optimizer import OptimizerModule
from features.creator import CreatorModule
from features.asin_mapper import ASINMapperModule
from features.ai_insights import AIInsightsModule

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Saddle AdPulse", 
    layout="wide", 
    page_icon="🚀"
)

# Initialize session state
if 'current_module' not in st.session_state:
    st.session_state['current_module'] = 'home'

if 'data' not in st.session_state:
    st.session_state['data'] = {}

# ==========================================
# NAVIGATION
# ==========================================
def navigate_to(module: str):
    """Navigate between modules."""
    st.session_state['current_module'] = module

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    """Main application orchestrator."""
    
    # Setup page layout (CSS, etc.)
    setup_page()
    
    # === CONFIRMATION DIALOG CHECK ===
    # If confirmation is needed, show popup dialog (overlays on current page)
    if st.session_state.get('_show_action_confirmation'):
        from ui.action_confirmation import render_action_confirmation_modal
        render_action_confirmation_modal()
        # Dialog shows as popup - continue rendering page underneath
    
    # Render sidebar navigation
    selected_module = render_sidebar(navigate_to)
    
    # Route to selected module
    current = st.session_state['current_module']
    
    if current == 'home':
        render_home()
    
    elif current == 'data_hub':
        from ui.data_hub import render_data_hub
        render_data_hub()
    
    elif current == 'optimizer':
        optimizer = OptimizerModule()
        optimizer.run()
    
    elif current == 'creator':
        creator = CreatorModule()
        creator.run()
    
    elif current == 'asin_mapper':
        asin_mapper = ASINMapperModule()
        asin_mapper.run()
    
    elif current == 'ai_insights':
        ai_insights = AIInsightsModule()
        ai_insights.run()
    
    elif current == 'readme':
        from ui.readme import render_readme
        render_readme()

if __name__ == "__main__":
    main()
