import streamlit as st

class ThemeManager:
    """Manages dynamic theme switching (Dark/Light) via CSS injection."""
    
    @staticmethod
    def init_theme():
        """Initialize theme state if not present."""
        if 'theme_mode' not in st.session_state:
            st.session_state.theme_mode = 'dark' # Default

    @staticmethod
    def render_toggle():
        """Render the toggle in sidebar and apply styles."""
        ThemeManager.init_theme()
        
        # Toggle Switch
        is_dark = st.sidebar.toggle('🌙 Dark Mode', value=(st.session_state.theme_mode == 'dark'))
        
        # Update State
        new_mode = 'dark' if is_dark else 'light'
        if new_mode != st.session_state.theme_mode:
            st.session_state.theme_mode = new_mode
            st.rerun() # Rerun to apply changes instantly
            
        # Apply CSS
        ThemeManager.apply_css()
        
    @staticmethod
    def apply_css():
        """Inject CSS based on current mode - Saddle AdPulse Theme."""
        ThemeManager.init_theme() # Ensure state exists
        mode = st.session_state.theme_mode
        
        # Saddle AdPulse Color Palette (from logo)
        # Primary: Dark navy/slate blues
        # Secondary: Light silver/grays  
        # Accent: Cyan/teal
        
        if mode == 'dark':
            bg_color = "#151b26"           # Deep navy background
            sec_bg = "#1e2736"             # Slightly lighter navy for sidebar
            text_color = "#e2e8f0"         # Light silver text
            text_muted = "#94a3b8"         # Muted silver
            border_color = "#2d3a4f"       # Navy border
            card_bg = "#1e2736"            # Card background
            accent = "#22d3ee"             # Cyan accent (from logo dots)
            accent_hover = "#06b6d4"       # Darker cyan
            sidebar_text = "#e2e8f0"       # Light text for dark sidebar
        else:
            bg_color = "#f8fafc"           # Light background
            sec_bg = "#e2e8f0"             # Light silver sidebar
            text_color = "#1e293b"         # Dark navy text
            text_muted = "#475569"         # Darker muted for better contrast
            border_color = "#cbd5e1"       # Light border
            card_bg = "#ffffff"            # White cards
            accent = "#0891b2"             # Cyan accent
            accent_hover = "#0e7490"       # Darker cyan
            sidebar_text = "#334155"       # Dark slate for sidebar

        css = f"""
        <style>
            :root {{
                --bg-color: {bg_color};
                --secondary-bg: {sec_bg};
                --text-color: {text_color};
                --text-muted: {text_muted};
                --border-color: {border_color};
                --card-bg: {card_bg};
                --accent: {accent};
                --accent-hover: {accent_hover};
                --sidebar-text: {sidebar_text};
            }}
            
            /* Main App Background */
            .stApp {{
                background-color: var(--bg-color);
                color: var(--text-color);
            }}
            
            /* Sidebar Background */
            [data-testid="stSidebar"] {{
                background-color: var(--secondary-bg);
            }}
            
            /* Text Colors - Targeted Fixes */
            h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown li, .stText {{
                color: var(--text-color);
            }}

            /* Sidebar Specific Text Fix - Using sidebar_text for better contrast */
            [data-testid="stSidebar"] p, 
            [data-testid="stSidebar"] span, 
            [data-testid="stSidebar"] label, 
            [data-testid="stSidebar"] .stMarkdown,
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4,
            [data-testid="stSidebar"] h5,
            [data-testid="stSidebar"] em,
            [data-testid="stSidebar"] strong {{
                color: var(--sidebar-text) !important;
            }}
            
            /* Sidebar button text */
            [data-testid="stSidebar"] button {{
                color: var(--sidebar-text) !important;
            }}

            /* Metric Values Fix */
            [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
                color: var(--text-color) !important;
            }}
            
            /* Inputs / Selectboxes */
            .stSelectbox div[data-baseweb="select"] > div {{
                background-color: var(--secondary-bg);
                color: var(--text-color);
                border-color: var(--border-color);
            }}
            
            /* Streamlit Metrics (Built-in) */
            div[data-testid="metric-container"] {{
                background-color: var(--card-bg);
                border: 1px solid var(--border-color);
                padding: 10px;
                border-radius: 8px;
            }}
            
            /* Buttons - Accent Color */
            .stButton > button {{
                background-color: var(--accent) !important;
                color: #0f172a !important;
                border: none !important;
                font-weight: 500;
            }}
            .stButton > button:hover {{
                background-color: var(--accent-hover) !important;
            }}
            
            /* Primary button styling */
            .stButton > button[kind="primary"] {{
                background-color: var(--accent) !important;
            }}
            
            /* Tabs accent */
            .stTabs [data-baseweb="tab-highlight"] {{
                background-color: var(--accent) !important;
            }}
            
            /* Links */
            a {{
                color: var(--accent) !important;
            }}
            a:hover {{
                color: var(--accent-hover) !important;
            }}
            
            /* Expander headers */
            .streamlit-expanderHeader {{
                color: var(--text-color) !important;
            }}
            
            /* Info boxes - subtle cyan tint */
            .stAlert {{
                background-color: rgba(8, 145, 178, 0.08) !important;
                border-left-color: var(--accent) !important;
            }}
            
            /* Sidebar title styling */
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {{
                color: var(--text-color) !important;
            }}
            
            /* Card-like info boxes */
            .stInfo {{
                background-color: var(--card-bg) !important;
                border: 1px solid var(--border-color) !important;
            }}
            
            /* Download button - keep accent */
            .stDownloadButton > button {{
                background-color: var(--accent) !important;
                color: white !important;
            }}
            
            /* Caption text - muted */
            .stCaption, small {{
                color: var(--text-muted) !important;
            }}
            
            /* Data frames / tables */
            .stDataFrame {{
                background-color: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 8px;
            }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    @staticmethod
    def get_chart_template():
        """Return the Plotly template name."""
        return 'plotly_dark' if st.session_state.get('theme_mode', 'dark') == 'dark' else 'plotly_white'
