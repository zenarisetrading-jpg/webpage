"""
Reusable UI Style Components

Centralized CSS gradients, card styles, and UI components used across features.
Reduces duplication of inline CSS across feature modules.
"""

# ==========================================
# GRADIENT STYLES
# ==========================================

# Brand purple/wine gradient (primary action buttons)
GRADIENT_PRIMARY = "linear-gradient(135deg, #5B556F 0%, #464156 100%)"

# Success gradient (green)
GRADIENT_SUCCESS = "linear-gradient(135deg, #10B981 0%, #059669 100%)"

# Info gradient (cyan/teal)
GRADIENT_INFO = "linear-gradient(135deg, #22d3ee 0%, #06b6d4 100%)"

# Premium card background gradient (subtle)
GRADIENT_CARD_PREMIUM = "linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%)"

# Hero card background (for dashboard)
GRADIENT_HERO_CARD = "linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%)"


# ==========================================
# BRAND COLORS
# ==========================================

class Colors:
    """Brand color palette from Saddle design system."""

    # Primary wine/purple palette
    WINE_PRIMARY = "#5B556F"
    WINE_SECONDARY = "#464156"
    WINE_LIGHT = "#8F8CA3"

    # Accent colors
    CYAN = "#22d3ee"
    CYAN_DARK = "#06b6d4"

    # Status colors
    GREEN = "#10B981"
    GREEN_DARK = "#059669"
    RED = "#ef4444"
    YELLOW = "#f59e0b"

    # Neutral colors
    TEXT_PRIMARY = "#F5F5F7"
    TEXT_MUTED = "#8F8CA3"
    BORDER = "rgba(143, 140, 163, 0.2)"


# ==========================================
# CSS COMPONENT GENERATORS
# ==========================================

def premium_card_css(include_style_tag: bool = True) -> str:
    """
    Generate CSS for premium card layout.

    Used by: report_card.py, creator.py

    Args:
        include_style_tag: If True, wraps in <style> tags

    Returns:
        CSS string
    """
    css = f"""
    .premium-card {{
        background: {GRADIENT_CARD_PREMIUM};
        border: 1px solid {Colors.BORDER};
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .premium-card-title {{
        color: {Colors.TEXT_PRIMARY};
        font-size: 1.5rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .premium-card-subtitle {{
        color: {Colors.TEXT_MUTED};
        font-size: 0.8rem;
        font-weight: 600;
    }}
    """

    if include_style_tag:
        return f"<style>{css}</style>"
    return css


def hero_card_css(include_style_tag: bool = True) -> str:
    """
    Generate CSS for hero metric cards.

    Used by: impact_dashboard.py

    Args:
        include_style_tag: If True, wraps in <style> tags

    Returns:
        CSS string
    """
    css = f"""
    .hero-card {{
        background: {GRADIENT_HERO_CARD};
        border: 1px solid rgba(143, 140, 163, 0.15);
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
    }}
    .hero-label {{
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: {Colors.TEXT_MUTED};
        font-weight: 600;
        margin-bottom: 8px;
    }}
    .hero-value {{
        font-size: 2rem;
        font-weight: 800;
        color: {Colors.TEXT_PRIMARY};
        margin-bottom: 4px;
    }}
    .hero-sub {{
        font-size: 0.75rem;
        color: {Colors.TEXT_MUTED};
    }}
    """

    if include_style_tag:
        return f"<style>{css}</style>"
    return css


def download_button_css(include_style_tag: bool = True) -> str:
    """
    Generate CSS for styled download buttons.

    Used by: impact_dashboard.py, multiple modules

    Args:
        include_style_tag: If True, wraps in <style> tags

    Returns:
        CSS string
    """
    css = f"""
    .stDownloadButton > button {{
        background: {GRADIENT_SUCCESS} !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }}
    .stDownloadButton > button:hover {{
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15) !important;
        transform: translateY(-1px);
    }}
    """

    if include_style_tag:
        return f"<style>{css}</style>"
    return css


def primary_button_css(include_style_tag: bool = True) -> str:
    """
    Generate CSS for primary action buttons.

    Used by: optimizer.py, creator.py, multiple modules

    Args:
        include_style_tag: If True, wraps in <style> tags

    Returns:
        CSS string
    """
    css = f"""
    [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {{
        background: {GRADIENT_PRIMARY} !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }}
    [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #6c6684 0%, #5b556f 100%) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15) !important;
        transform: translateY(-1px);
    }}
    """

    if include_style_tag:
        return f"<style>{css}</style>"
    return css


def section_header(title: str, subtitle: str = None, icon_svg: str = None) -> str:
    """
    Generate HTML for a styled section header.

    Args:
        title: Main title text
        subtitle: Optional subtitle text
        icon_svg: Optional SVG icon HTML

    Returns:
        HTML string ready for st.markdown()

    Example:
        st.markdown(section_header("PPC Optimizer", "Account: ABC Inc"), unsafe_allow_html=True)
    """
    icon_html = icon_svg if icon_svg else ""
    subtitle_html = f'<div class="premium-card-subtitle">{subtitle}</div>' if subtitle else ""

    return f"""
    <div class="premium-card">
        <div style="display: flex; align-items: center;">
            {icon_html}
            <span class="premium-card-title">{title}</span>
        </div>
        {subtitle_html}
    </div>
    {premium_card_css()}
    """


def metric_hero_card(label: str, value: str, subtitle: str = None) -> str:
    """
    Generate HTML for a hero metric card.

    Args:
        label: Metric label (e.g., "Total Actions")
        value: Metric value (e.g., "143")
        subtitle: Optional subtitle text

    Returns:
        HTML string ready for st.markdown()

    Example:
        col1.markdown(metric_hero_card("ROAS", "2.5x", "Target: 2.0x"), unsafe_allow_html=True)
    """
    subtitle_html = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ""

    return f"""
    <div class="hero-card">
        <div class="hero-label">{label}</div>
        <div class="hero-value">{value}</div>
        {subtitle_html}
    </div>
    {hero_card_css()}
    """


# ==========================================
# ICON SVGs
# ==========================================

class Icons:
    """Common SVG icons with consistent styling."""

    @staticmethod
    def settings(color: str = "#8F8CA3", size: int = 16) -> str:
        """Settings gear icon."""
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>'''

    @staticmethod
    def bolt(color: str = "#8F8CA3", size: int = 16) -> str:
        """Lightning bolt icon."""
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>'''

    @staticmethod
    def overview(color: str = "#8F8CA3", size: int = 24) -> str:
        """Overview/dashboard icon."""
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 12px;"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>'''
