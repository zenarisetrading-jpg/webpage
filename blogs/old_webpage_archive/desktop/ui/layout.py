"""
UI Layout Components

Page setup, sidebar navigation, and home page.
"""

import streamlit as st

from ui.theme import ThemeManager
from features.impact_dashboard import get_recent_impact_summary
from features.report_card import get_account_health_score

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
    # Wrap navigate_to to check for pending actions when leaving optimizer
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
        
        navigate_to(target_module)
        navigate_to(target_module)
    # Sidebar Logo at TOP (theme-aware, prominent)
    theme_mode = st.session_state.get('theme_mode', 'dark')
    logo_data = ThemeManager.get_cached_logo(theme_mode)
    
    if logo_data:
        st.sidebar.markdown(
            f'<div style="text-align: center; padding: 15px 0 20px 0;"><img src="data:image/png;base64,{logo_data}" style="width: 200px;" /></div>',
            unsafe_allow_html=True
        )
        
    # Account selector
    from ui.account_manager import render_account_selector
    render_account_selector()
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Home", use_container_width=True):
        safe_navigate('home')
    
    if st.sidebar.button("Account Overview", use_container_width=True):
        safe_navigate('performance')
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### SYSTEM")
    
    # Data Hub - central upload
    if st.sidebar.button("Data Hub", use_container_width=True):
        safe_navigate('data_hub')
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### ANALYZE")
    
    # Core features
    if st.sidebar.button("Optimizer", use_container_width=True):
        safe_navigate('optimizer')
    
    if st.sidebar.button("ASIN Shield", use_container_width=True):
        safe_navigate('asin_mapper')
    
    if st.sidebar.button("Clusters", use_container_width=True):
        safe_navigate('ai_insights')
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### ACTIONS")
    
    if st.sidebar.button("Launchpad", use_container_width=True):
        safe_navigate('creator')
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Help", use_container_width=True):
        safe_navigate('readme')
    
    # Show undo toast if available
    from ui.action_confirmation import show_undo_toast
    show_undo_toast()
    
    # Theme Toggle at BOTTOM
    st.sidebar.markdown("---")
    ThemeManager.render_toggle()
    
    return st.session_state.get('current_module', 'home')

def render_home():
    st.markdown("""
        <style>
        /* Specific card targeting via markers */
        [data-testid="stColumn"]:has(.cockpit-marker) > div {
            background: var(--card-bg, #ffffff);
            border: 1px solid var(--border-color, #e2e8f0);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            min-height: 220px;
            display: flex;
            flex-direction: column;
        }
        [data-theme="dark"] [data-testid="stColumn"]:has(.cockpit-marker) > div {
            background: #1e293b;
            border-color: #334155;
        }
        
        .cockpit-label {
            font-size: 0.75rem;
            color: #64748b;
            font-weight: 700;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .cockpit-value {
            font-size: 2rem;
            font-weight: 800;
            color: var(--text-color);
            margin: 0;
            line-height: 1.2;
        }
        
        .cockpit-subtext {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 4px;
        }

        /* Insight Tiles - Horizontal and slimmer */
        .insight-tile {
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 10px;
            padding: 10px 14px;
            display: flex;
            align-items: center;
            gap: 12px;
            width: 100%;
        }
        .insight-icon {
            font-size: 1.1rem;
            min-width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: white;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        [data-theme="dark"] .insight-icon { background: #0f172a; }
        
        /* Fix for plotly height adjustment */
        .js-plotly-plot { margin-top: -30px; }
        
        /* Hide the marker itself */
        .cockpit-marker { display: none; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h2 style="font-family: Inter, sans-serif; font-weight: 700; letter-spacing: 0.02em;">DECISION COCKPIT</h2>', unsafe_allow_html=True)
    st.caption("Strategic overview of your account performance")
    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3)
    
    with t1:
        st.markdown('<div class="cockpit-marker"></div>', unsafe_allow_html=True)
        # Determine source badge
        source = st.session_state.get('_cockpit_data_source', 'db')
        sync_badge = '<span style="font-size: 0.55rem; background: rgba(34, 197, 94, 0.15); color: #22c55e; padding: 2px 6px; border-radius: 4px; font-weight: 800;">LIVE SYNC</span>' if source == 'live' else ''
        st.markdown(f'<div class="cockpit-label" style="text-align:center;"><span>Health Score</span>{sync_badge}</div>', unsafe_allow_html=True)
        
        health = get_account_health_score()
        if health is not None:
            health = round(health)
            import plotly.graph_objects as go
            
            # Status thresholds
            if health > 75:
                status_text = "HEALTHY"
                status_color = "#22c55e"
            elif health >= 40:
                status_text = "STABLE"
                status_color = "#f59e0b"
            else:
                status_text = "ATTENTION"
                status_color = "#ef4444"
            
            # Dashboard-consistent gauge (cyan bar, grey arc, clean ticks)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health,
                number={
                    'suffix': "%",
                    'font': {'size': 26, 'color': '#06b6d4', 'family': 'Inter, sans-serif'}
                },
                gauge={
                    'axis': {
                        'range': [0, 100],
                        'tickwidth': 1,
                        'tickcolor': '#64748b',
                        'ticklen': 8,
                        'tickvals': [0, 25, 50, 75, 100],
                        'ticktext': ['0', '25', '50', '75', '100'],
                        'tickfont': {'size': 9, 'color': '#64748b'}
                    },
                    'bar': {'color': '#06b6d4', 'thickness': 0.6},  # Thinner bar to avoid tick overlap
                    'bgcolor': '#374151',  # Grey background arc
                    'borderwidth': 0,
                }
            ))
            
            fig.update_layout(
                height=110,
                margin=dict(l=25, r=25, t=25, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'family': 'Inter, sans-serif'}
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown(f'<div style="text-align: center; color: {status_color}; font-weight: 700; margin-top: -10px; font-size: 0.7rem; letter-spacing: 0.05em;">{status_text}</div>', unsafe_allow_html=True)
            
            # Get actual scores from stored health data
            db_manager = st.session_state.get('db_manager')
            selected_client = st.session_state.get('active_account_id') or st.session_state.get('active_account_name')
            roas_score, efficiency_score, cvr_score = 0, 0, 0
            if db_manager and selected_client:
                try:
                    health_data = db_manager.get_account_health(selected_client)
                    if health_data:
                        roas_score = health_data.get('roas_score', 0)
                        efficiency_score = health_data.get('waste_score', 0)  # DB column is waste_score
                        cvr_score = health_data.get('cvr_score', 0)
                except:
                    pass
            
            # Display actual scores - bolder but not loud
            st.markdown(f'''<div style="display: flex; justify-content: space-around; margin-top: 10px; text-align: center;">
                <div><div style="font-size: 0.95rem; font-weight: 700; color: #94a3b8;">{roas_score:.0f}</div><div style="font-size: 0.6rem; color: #64748b;">ROAS</div></div>
                <div><div style="font-size: 0.95rem; font-weight: 700; color: #94a3b8;">{efficiency_score:.0f}</div><div style="font-size: 0.6rem; color: #64748b;">Efficiency</div></div>
                <div><div style="font-size: 0.95rem; font-weight: 700; color: #94a3b8;">{cvr_score:.0f}</div><div style="font-size: 0.6rem; color: #64748b;">CVR</div></div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('<div class="cockpit-value" style="margin-top:20px">—</div>', unsafe_allow_html=True)
            st.markdown('<div class="cockpit-subtext">Run optimizer to calculate</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="cockpit-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="cockpit-label" style="text-align:center;">14-Day Decision Impact</div>', unsafe_allow_html=True)
        impact_data = get_recent_impact_summary()
        st.markdown('<div style="flex-grow:1; display:flex; flex-direction:column; justify-content:space-between; text-align:center;">', unsafe_allow_html=True)
        if impact_data is not None:
            impact = impact_data.get('sales', 0)
            win_rate = impact_data.get('win_rate', 0)
            top_action = impact_data.get('top_action_type', None)
            
            # Center main content
            from utils.formatters import get_account_currency
            home_currency = get_account_currency()
            st.markdown('<div style="flex-grow:1; display:flex; flex-direction:column; justify-content:center;">', unsafe_allow_html=True)
            st.markdown(f'<div class="cockpit-value" style="text-align:center;">{f"+{home_currency}{impact:,.0f}" if impact >= 0 else f"-{home_currency}{abs(impact):,.0f}"}</div>', unsafe_allow_html=True)
            st.markdown('<div class="cockpit-subtext" style="text-align:center;">Net Change Last 14 Days</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Bottom row callouts - at absolute bottom
            action_display = ""
            if top_action:
                action_display = {"HARVEST": "Harvests", "NEGATIVE": "Keyword Defense", "BID_UPDATE": "Bid Changes", "BID_CHANGE": "Bid Changes"}.get(top_action, top_action.title())
            
            # Trend indicator with clearer labels and tooltip
            # Positive: Sales increased after optimizer actions - good!
            # Attention: Sales decreased - may need to review actions or wait for more data
            # Stable: No net change - actions had neutral effect
            if impact > 0:
                arrow_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2.5" style="vertical-align:middle"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline><polyline points="16 7 22 7 22 13"></polyline></svg>'
                trend_text = "Growing"
                trend_color = "#22c55e"
                trend_tooltip = "Decision Impact is positive over the last 14 days. Your optimization actions are driving value!"
            elif impact < 0:
                arrow_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2.5" style="vertical-align:middle"><polyline points="6 9 12 15 18 9"></polyline></svg>'
                trend_text = "Review Needed"
                trend_color = "#f59e0b"
                trend_tooltip = "Decision Impact is negative. This requires review to ensure actions are having the desired effect."
            else:
                arrow_svg = ''
                trend_text = "Stable"
                trend_color = "#64748b"
                trend_tooltip = "Decision Impact is neutral. Actions have stabilized performance."
            
            trend_html = f'{arrow_svg} <span style="color:{trend_color}">{trend_text}</span>'
            
            # CSS tooltip that works in Streamlit (title attribute doesn't work reliably)
            tooltip_css = '''
            <style>
            .tooltip-container { position: relative; display: inline-block; cursor: help; }
            .tooltip-container .tooltip-text {
                visibility: hidden;
                width: 220px;
                background-color: #1e293b;
                color: #e2e8f0;
                text-align: left;
                border-radius: 6px;
                padding: 8px 10px;
                position: absolute;
                z-index: 1000;
                bottom: 125%;
                left: 50%;
                margin-left: -110px;
                opacity: 0;
                transition: opacity 0.2s;
                font-size: 0.75rem;
                line-height: 1.4;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }
            .tooltip-container:hover .tooltip-text { visibility: visible; opacity: 1; }
            .info-icon { font-size: 0.7rem; color: #64748b; margin-left: 3px; }
            </style>
            '''
            
            st.markdown(f'''{tooltip_css}
            <div style="display: flex; justify-content: space-around; text-align: center; margin-top: auto;">
                <div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #94a3b8;">{action_display or "—"}</div>
                    <div style="font-size: 0.6rem; color: #64748b;">
                        Top Driver
                        <span class="tooltip-container"><span class="info-icon">ⓘ</span><span class="tooltip-text">The action type that contributed most to your recent decision impact.</span></span>
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.95rem; font-weight: 700;">{trend_html}</div>
                    <div style="font-size: 0.6rem; color: #64748b;">
                        14-Day Trend
                        <span class="tooltip-container"><span class="info-icon">ⓘ</span><span class="tooltip-text">{trend_tooltip}</span></span>
                    </div>
                </div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('<div class="cockpit-value" style="text-align:center;">—</div>', unsafe_allow_html=True)
            st.markdown('<div class="cockpit-subtext" style="text-align:center;">Run optimizer to track impact</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="cockpit-marker"></div>', unsafe_allow_html=True)
        st.markdown('<div class="cockpit-label">Next Step</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-weight: 700; font-size: 1rem; margin-bottom: 4px;">Optimization Ready</div>', unsafe_allow_html=True)
        st.markdown('<div class="cockpit-subtext" style="margin-bottom: 20px;">Review your optimization recommendations.</div>', unsafe_allow_html=True)
        if st.button("Review Actions", use_container_width=True, type="primary"):
            st.session_state['current_module'] = 'optimizer'
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### Key Insights")
    i1, i2, i3 = st.columns(3)
    
    # Theme-aware Icon Colors
    theme_mode = st.session_state.get('theme_mode', 'dark')
    
    # Get dynamic insights from knowledge graph (same data as AI assistant)
    # Show loading state to improve perceived performance
    from features.assistant import get_dynamic_key_insights
    with st.spinner(""):  # Empty spinner to avoid text, just shows loading indicator
        insights = get_dynamic_key_insights()
    
    # Icon definitions
    def get_insight_icon(icon_type):
        colors = {
            "success": "#22c55e" if theme_mode == 'dark' else "#16a34a",
            "info": "#60a5fa" if theme_mode == 'dark' else "#2563eb",
            "warning": "#fbbf24" if theme_mode == 'dark' else "#f59e0b",
            "note": "#94a3b8"
        }
        c = colors.get(icon_type, colors["info"])
        
        if icon_type == "success":
            return f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>'
        elif icon_type == "warning":
            return f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
        elif icon_type == "note":
            return f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'
        else:  # info
            return f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'
    
    with i1:
        icon = get_insight_icon(insights[0]["icon_type"])
        st.markdown(f'<div class="insight-tile"><div class="insight-icon">{icon}</div><div><div style="font-weight:700; font-size:1.1rem">{insights[0]["title"]}</div><div style="font-size:0.85rem; color:#94a3b8">{insights[0]["subtitle"]}</div></div></div>', unsafe_allow_html=True)
    with i2:
        icon = get_insight_icon(insights[1]["icon_type"])
        st.markdown(f'<div class="insight-tile"><div class="insight-icon">{icon}</div><div><div style="font-weight:700; font-size:1.1rem">{insights[1]["title"]}</div><div style="font-size:0.85rem; color:#94a3b8">{insights[1]["subtitle"]}</div></div></div>', unsafe_allow_html=True)
    with i3:
        icon = get_insight_icon(insights[2]["icon_type"])
        st.markdown(f'<div class="insight-tile"><div class="insight-icon">{icon}</div><div><div style="font-weight:700; font-size:1.1rem">{insights[2]["title"]}</div><div style="font-size:0.85rem; color:#94a3b8">{insights[2]["subtitle"]}</div></div></div>', unsafe_allow_html=True)


    
    



