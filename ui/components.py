"""
Shared UI Components

Reusable UI elements to ensure consistent styling across modules.
"""

import streamlit as st

def metric_card(label: str, value: str, icon_name: str = None, delta: str = None, color: str = None, border_color: str = "#333", subtitle: str = None):
    """
    Render a styled metric card using HTML.
    
    Args:
        label: The label text (e.g. "Spend")
        value: The value text (e.g. "AED 1,234")
        icon_name: Optional icon name (Feather icon)
        delta: Optional percentage change (e.g. "+12%")
        color: Text color for the value (default white)
        border_color: Border color for the card (default dark grey)
        subtitle: Optional subtitle text below the value
    """
    delta_html = ""
    if delta:
        delta_val = delta.strip().replace('%', '').replace('+', '')
        try:
            is_positive = float(delta_val) > 0
            text_color = "#4ade80" if is_positive else "#f87171"  # Green / Red
            arrow = "↑" if is_positive else "↓"
            delta_html = f'<span style="color: {text_color}; font-size: 14px; margin-left: 8px;">{arrow} {delta}</span>'
        except:
            delta_html = f'<span style="color: #888; font-size: 14px; margin-left: 8px;">{delta}</span>'

    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<p style="color: #64748b; font-size: 11px; margin: 4px 0 0 0;">{subtitle}</p>'

    icon_html = ""
    if icon_name:
        icon_color = "#8F8CA3"
        icons = {
            "spend": f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>',
            "revenue": f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>',
            "acos": f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>',
            "roas": f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline><polyline points="16 7 22 7 22 13"></polyline></svg>',
            "orders": f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><circle cx="9" cy="21" r="1"></circle><circle cx="20" cy="21" r="1"></circle><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"></path></svg>',
            "impressions": f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>',
            "clicks": f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><path d="m13 10 3.5 3.5M2 2l20 20M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>',
            "layers": f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>',
            "check": f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><polyline points="20 6 9 17 4 12"></polyline></svg>',
            "shield": f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>',
            "sliders": f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>',
            "leaf": f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 8px;"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8a8 8 0 0 1-10 10Z"></path><path d="M11 20s.2-4.5 4-7"></path></svg>'
        }
        icon_html = icons.get(icon_name, "")

    label_color = "#94a3b8"
    val_color = "#F5F5F7"
    if color: val_color = color
    
    # Premium background and border
    bg_color = "rgba(15, 23, 42, 0.6)" # Darker slate
    border_color = "rgba(255, 255, 255, 0.05)"
    
    # Force a single-line, clean HTML string with no newlines or leading spaces
    html = f'<div style="background-color: {bg_color}; padding: 20px 16px; border-radius: 12px; border: 1px solid {border_color}; margin-bottom: 10px; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;"><div style="color: {label_color}; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px; display: flex; align-items: center; justify-content: center; width: 100%;">{icon_html}{label}</div><div style="margin: 0; padding: 0; display: flex; align-items: baseline; gap: 8px; justify-content: center; width: 100%;"><span style="color: {val_color}; font-size: 1.6rem; font-weight: 800; line-height: 1;">{value}</span>{delta_html}</div>{subtitle_html}</div>'
    st.markdown(html, unsafe_allow_html=True)


def metric_card_with_tooltip(label: str, value: str, tooltip: str, icon_html: str = "", 
                              value_color: str = None, confidence_text: str = None, 
                              confidence_color: str = "#94a3b8"):
    """
    Render a styled metric card with an info icon tooltip.
    
    Args:
        label: The label text (e.g. "Estimated Revenue Impact")
        value: The value text (e.g. "+INR 1,234")
        tooltip: Tooltip text shown on hover (PRD-compliant copy)
        icon_html: Optional SVG icon HTML
        value_color: Text color for the value
        confidence_text: Optional confidence level text (e.g. "High", "Medium", "Low")
        confidence_color: Color for the confidence indicator
    """
    label_color = "#94a3b8"
    val_color = value_color or "#F5F5F7"
    bg_color = "rgba(15, 23, 42, 0.6)"
    border_color = "rgba(255, 255, 255, 0.05)"
    
    # Info icon SVG for tooltip trigger
    info_icon = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="cursor: help; margin-left: 4px;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'
    
    # Confidence line if provided
    confidence_html = ""
    if confidence_text:
        confidence_html = f'<div style="font-size: 0.75rem; color: #8F8CA3; margin-top: 8px;">Confidence: <span style="color: {confidence_color}; font-weight: 600;">{confidence_text}</span></div>'
    
    # Escape tooltip for HTML attribute (replace quotes)
    safe_tooltip = tooltip.replace('"', '&quot;').replace("'", "&#39;")
    
    html = f'''
    <div style="background-color: {bg_color}; padding: 20px 16px; border-radius: 12px; border: 1px solid {border_color}; margin-bottom: 10px; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
        <div style="color: {label_color}; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px; display: flex; align-items: center; justify-content: center; width: 100%;">
            {icon_html}{label}
            <span title="{safe_tooltip}" style="display: inline-flex; align-items: center;">{info_icon}</span>
        </div>
        <div style="margin: 0; padding: 0; display: flex; align-items: baseline; gap: 8px; justify-content: center; width: 100%;">
            <span style="color: {val_color}; font-size: 1.6rem; font-weight: 800; line-height: 1;">{value}</span>
        </div>
        {confidence_html}
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)
