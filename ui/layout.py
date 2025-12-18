"""
UI Layout Components

Page setup, sidebar navigation, and home page.
"""

import streamlit as st

from ui.theme import ThemeManager

def setup_page():
    """Setup page CSS and styling."""
    # Apply dynamic theme CSS
    ThemeManager.apply_css()

def render_sidebar(navigate_to):
    """
    Render sidebar navigation.
    
    Args:
        navigate_to: Function to navigate between modules
        
    Returns:
        Selected module name
    """
    # Sidebar Logo at TOP (theme-aware, prominent)
    import base64
    from pathlib import Path
    theme_mode = st.session_state.get('theme_mode', 'dark')
    logo_filename = "saddle_logo.png" if theme_mode == 'dark' else "saddle_logo_light.png"
    logo_path = Path(__file__).parent.parent / "static" / logo_filename
    
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        st.sidebar.markdown(
            f'<div style="text-align: center; padding: 15px 0 20px 0;"><img src="data:image/png;base64,{logo_data}" style="width: 200px;" /></div>',
            unsafe_allow_html=True
        )
    
    # Account selector
    from ui.account_manager import render_account_selector
    render_account_selector()
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Home", use_container_width=True):
        navigate_to('home')
    
    st.sidebar.markdown("##### SYSTEM")
    
    # Data Hub - central upload
    if st.sidebar.button("Data Hub", use_container_width=True):
        navigate_to('data_hub')
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### ANALYZE")
    
    # Core features
    if st.sidebar.button("Optimizer", use_container_width=True):
        navigate_to('optimizer')
    
    if st.sidebar.button("ASIN Shield", use_container_width=True):
        navigate_to('asin_mapper')
    
    if st.sidebar.button("Clusters", use_container_width=True):
        navigate_to('ai_insights')
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### ACTIONS")
    
    if st.sidebar.button("Launchpad", use_container_width=True):
        navigate_to('creator')
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Help", use_container_width=True):
        navigate_to('readme')
    
    # Theme Toggle at BOTTOM
    st.sidebar.markdown("---")
    ThemeManager.render_toggle()
    
    return st.session_state.get('current_module', 'home')

def render_home():
    """Render the Saddle AdPulse Dashboard."""
    
    # Hero Section with Logo
    import base64
    from pathlib import Path
    
    # Determine which logo to use based on theme
    theme_mode = st.session_state.get('theme_mode', 'dark')
    logo_filename = "saddle_logo.png" if theme_mode == 'dark' else "saddle_logo_light.png"
    
    # Load logo image
    logo_path = Path(__file__).parent.parent / "static" / logo_filename
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{logo_data}" style="width: 500px; margin-bottom: 0;" />'
    else:
        logo_html = ""
    
    # Dynamic text colors based on theme
    title_color = "#e2e8f0" if theme_mode == 'dark' else "#1e293b"
    subtitle_color = "#94a3b8" if theme_mode == 'dark' else "#475569"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 30px 0 20px 0; margin-top: -60px; position: relative; z-index: 10;">
        {logo_html}
        <p style="font-size: 1.8rem; color: {subtitle_color}; margin-top: -100px; font-weight: 400;">Decision Engine for Amazon PPC</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Active Context Badge
    if 'db_manager' in st.session_state and st.session_state['db_manager']:
        db_mode = "TEST (ppc_test.db)" if st.session_state.get('test_mode') else "LIVE (ppc_live.db)"
        st.caption(f"🟢 System Active | Database: {db_mode}")
        
        # Show active account (if multi-account mode)
        if not st.session_state.get('single_account_mode', False):
            active_account = st.session_state.get('active_account_name', 'No account selected')
            st.caption(f"📊 Active Account: **{active_account}**")
    else:
        st.caption("🔴 System Idle | Database Not Connected")

    st.markdown("<hr style='border: none; border-top: 1px solid #334155; margin: 20px 0;'>", unsafe_allow_html=True)
    
    # Workflow Cards (Ingest -> Analyze -> Optimize -> Execute)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 1. Ingest")
        st.info("**Data Hub**")
        st.caption("Upload Search Term Reports, Bulk Files, and Business Reports.")
        if st.button("📂 Go to Data Hub", use_container_width=True):
            st.session_state['current_module'] = 'data_hub'
            st.rerun()

    with col2:
        st.markdown("### 2. Analyze")
        st.info("**Impact Analyzer**")
        st.caption("View wasted spend, missed sales, and ROI opportunities.")
        if st.button("📉 View Impact", use_container_width=True):
            st.session_state['current_module'] = 'impact'
            st.rerun()

    with col3:
        st.markdown("### 3. Optimize")
        st.info("**Optimization Hub**")
        st.caption("Review and approve Bids, Negatives, and Harvests.")
        if st.button("⚡ Run Optimizer", use_container_width=True):
            st.session_state['current_module'] = 'optimizer'
            st.rerun()
            
    with col4:
        st.markdown("### 4. Execute")
        st.info("**Launcher**")
        st.caption("Launch optimized campaigns and push changes to Amazon.")
        if st.button("🚀 Launch Campaigns", use_container_width=True):
            st.session_state['current_module'] = 'creator'
            st.rerun()

    st.divider()
    
    # AI Teaser & Quick Actions
    c_ai, c_docs = st.columns([2, 1])
    
    with c_ai:
        st.markdown("""
        ### 🧠 AI Strategist is Ready
        Your data is automatically analyzed by the AI. Open the chat bubble in the bottom right 
        to ask questions like:
        - *"Where am I wasting the most money?"*
        - *"Draft a strategy to launch SKU X"*
        """)
        
    with c_docs:
        st.markdown("### 📚 Resources")
        if st.button("📖 Documentation", use_container_width=True):
            st.session_state['current_module'] = 'readme'
            st.rerun()
        st.markdown("[Report a Bug](mailto:support@s2c.com)")
    

