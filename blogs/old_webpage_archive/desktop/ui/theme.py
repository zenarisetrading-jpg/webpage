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
            [data-testid="stSidebar"] em,
            [data-testid="stSidebar"] strong {{
                color: var(--sidebar-text) !important;
            }}
            
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4,
            [data-testid="stSidebar"] h5 {{
                color: var(--sidebar-text) !important;
                text-transform: uppercase !important;
                letter-spacing: 0.1em !important;
            }}
            
            /* Sidebar button text */
            [data-testid="stSidebar"] button {{
                color: var(--sidebar-text) !important;
                text-transform: uppercase !important;
                letter-spacing: 0.05em !important;
                font-weight: 600 !important;
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
            
            /* Sidebar Specific Selectbox Styling */
            [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
                background-color: rgba(30, 41, 59, 0.8) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 8px !important;
            }}
            [data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {{
                border-color: rgba(255, 255, 255, 0.2) !important;
            }}
            [data-testid="stSidebar"] div[data-baseweb="select"] span {{
                color: white !important;
                font-family: inherit !important;
                font-weight: 500 !important;
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
            
            /* Primary button styling - Brand Purple Gradient */
            /* Primary button styling - Brand Purple (Dark Vine) - High Specificity */
            .stApp .stButton > button[kind="primary"],
            .stApp div[data-testid="stForm"] button[kind="primary"],
            .stApp button[data-testid="baseButton-primary"] {{
                background: linear-gradient(135deg, #464156 0%, #2E2A36 100%) !important;
                color: #E9EAF0 !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
            }}
            .stApp .stButton > button[kind="primary"]:hover,
            .stApp div[data-testid="stForm"] button[kind="primary"]:hover,
            .stApp button[data-testid="baseButton-primary"]:hover {{
                background: linear-gradient(135deg, #5B5670 0%, #464156 100%) !important;
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3) !important;
                transform: translateY(-1px);
            }}
            
            /* Download button - use Primary styling */
            .stApp .stDownloadButton > button {{
                background: linear-gradient(135deg, #464156 0%, #2E2A36 100%) !important;
                color: white !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
            }}
            .stApp .stDownloadButton > button:hover {{
                background: linear-gradient(135deg, #5B5670 0%, #464156 100%) !important;
            }}
            
            /* Tabs accent and Boldness */
            .stTabs [data-baseweb="tab"] {{
                font-weight: 800 !important;
                font-size: 1rem !important;
                letter-spacing: 0.5px !important;
                text-transform: uppercase !important;
            }}
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
            
            /* Info boxes - Brand Compliant (Dark Vine Transparent) */
            .stApp div[data-testid="stAlert"] {{
                background-color: rgba(46, 42, 54, 0.95) !important; /* Higher opacity for visibility */
                border: 1px solid rgba(154, 154, 170, 0.2) !important;
                border-left-color: var(--accent) !important;
                color: #E9EAF0 !important;
                border-radius: 8px !important;
            }}
            .stApp div[data-testid="stAlert"] > div, 
            .stApp div[data-testid="stAlert"] p {{
                color: #E9EAF0 !important;
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

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_cached_logo(mode: str) -> str:
        """Cache logo file reading to avoid disk I/O on every frame."""
        import base64
        from pathlib import Path
        
        filename = "saddle_logo.png" if mode == 'dark' else "saddle_logo_light.png"
        # Navigate up from ui/theme.py -> ui -> saddle -> static
        logo_path = Path(__file__).parent.parent / "static" / filename
        
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return ""
