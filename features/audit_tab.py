"""
Audit Tab Module - Wasted Spend Heatmap Display

Displays performance heatmap with priority classifications and recommended actions.
Separated from optimizer.py for cleaner maintenance and faster loading.
"""

import streamlit as st
import pandas as pd
from typing import Optional
from ui.components import metric_card


def render_audit_tab(heatmap_df: Optional[pd.DataFrame]) -> None:
    """
    Render the Audit Tab (Wasted Spend Heatmap).
    
    Args:
        heatmap_df: DataFrame with columns:
            - Priority (🔴 High / 🟡 Medium / 🟢 Good)
            - Campaign Name, Ad Group Name
            - Actions_Taken, Reason_Summary
            - Spend, Sales, ROAS, CVR
    """
    icon_color = "#8F8CA3"
    search_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>'
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); 
                border: 1px solid rgba(124, 58, 237, 0.2); 
                border-radius: 8px; 
                padding: 12px 16px; 
                margin-bottom: 20px;
                display: flex; 
                align-items: center; 
                gap: 10px;">
        {search_icon}
        <span style="color: #F5F5F7; font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Wasted Spend Heatmap</span>
    </div>
    """, unsafe_allow_html=True)

    if heatmap_df is not None and not heatmap_df.empty:
        st.markdown("""
        <div style="background: rgba(91, 85, 111, 0.05); border-left: 4px solid #5B556F; padding: 12px 20px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
            <p style="color: #B6B4C2; font-size: 0.9rem; margin: 0;">
                <strong>Visual Intelligence</strong>: Red indicates immediate fix required, Yellow requires monitoring, and Green shows efficient performance.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- 1. Top Cards ---
        p1, p2, p3 = st.columns(3)
        high_count = len(heatmap_df[heatmap_df["Priority"].str.contains("High")])
        med_count = len(heatmap_df[heatmap_df["Priority"].str.contains("Medium")])
        good_count = len(heatmap_df[heatmap_df["Priority"].str.contains("Good")])
        
        with p1: metric_card("High Priority", str(high_count), "shield", color="#f87171")
        with p2: metric_card("Medium Priority", str(med_count), "shield", color="#fbbf24")
        with p3: metric_card("Good Performance", str(good_count), "check", color="#4ade80")
        
        st.divider()
        
        # --- 2. Action Status Cards ---
        addressed_mask = ~heatmap_df["Actions_Taken"].str.contains("Hold|No action", na=False)
        addressed_count = addressed_mask.sum()
        needs_attn_count = len(heatmap_df) - addressed_count
        high_addressed = (heatmap_df[heatmap_df["Priority"].str.contains("High") & addressed_mask]).shape[0]
        coverage = (addressed_count / len(heatmap_df) * 100) if len(heatmap_df) > 0 else 0

        # --- PREMIUM HERO TILES (Impact Dashboard Style) - Neutralized ---
        theme_mode = st.session_state.get('theme_mode', 'dark')
        # Saddle brand colors extracted from logo (Neutralized version)
        brand_purple = "#5B556F"
        brand_muted = "#8F8CA3"
        brand_slate = "#444357"
        brand_text = "#F5F5F7"
        brand_muted_text = "#B6B4C2"
        
        # Surface and Glow
        surface_glow = "rgba(91, 85, 111, 0.08)"
        
        icon_color = brand_muted
        
        # Action Icons
        check_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><polyline points="20 6 9 17 4 12"></polyline></svg>'
        warning_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
        bolt_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>'
        target_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>'

        st.markdown("""
        <style>
        .hero-tile {
            background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            margin-bottom: 10px;
        }
        .hero-tile:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        }
        .hero-value {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 4px;
            margin-top: 8px;
        }
        .hero-label {
            font-size: 0.75rem;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f"""
            <div class="hero-tile" style="border-left: 4px solid {brand_purple}; background: linear-gradient(135deg, {surface_glow} 0%, rgba(255,255,255,0.02) 100%);">
                <div class="hero-label">{check_icon}Being Addressed</div>
                <div class="hero-value" style="color: {brand_text};">{addressed_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            st.markdown(f"""
            <div class="hero-tile" style="border-left: 4px solid {brand_slate}; background: linear-gradient(135deg, {surface_glow} 0%, rgba(255,255,255,0.02) 100%);">
                <div class="hero-label">{warning_icon}Needs Attention</div>
                <div class="hero-value" style="color: {brand_muted_text};">{needs_attn_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c3:
            st.markdown(f"""
            <div class="hero-tile" style="border-left: 4px solid {brand_purple}; background: linear-gradient(135deg, {surface_glow} 0%, rgba(255,255,255,0.02) 100%);">
                <div class="hero-label">{bolt_icon}High Priority Fixed</div>
                <div class="hero-value" style="color: {brand_text};">{high_addressed}/{high_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c4:
            st.markdown(f"""
            <div class="hero-tile" style="border-left: 4px solid {brand_slate}; background: linear-gradient(135deg, {surface_glow} 0%, rgba(255,255,255,0.02) 100%);">
                <div class="hero-label">{target_icon}Coverage</div>
                <div class="hero-value" style="color: {brand_muted_text};">{coverage:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # --- 3. Interactive Filters ---
        f1, f2 = st.columns(2)
        with f1:
            p_filter = st.multiselect("Filter by Priority", ["🔴 High", "🟡 Medium", "🟢 Good"], default=["🔴 High", "🟡 Medium"])
        with f2:
            has_action = st.selectbox("Filter by Actions", ["All", "Only with Actions", "Only Holds/No Action"])
        
        filtered_df = heatmap_df.copy()
        if p_filter:
            filtered_df = filtered_df[filtered_df["Priority"].isin(p_filter)]
        
        if has_action == "Only with Actions":
            filtered_df = filtered_df[~filtered_df["Actions_Taken"].str.contains("Hold|No action", na=False)]
        elif has_action == "Only Holds/No Action":
            filtered_df = filtered_df[filtered_df["Actions_Taken"].str.contains("Hold|No action", na=False)]
            
        # Heatmap icon
        icon_color = "#8F8CA3"
        heat_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>'
        st.markdown(f"#### {heat_icon}Performance Heatmap with Actions ({len(filtered_df)} items)", unsafe_allow_html=True)
        
        cols = ["Priority", "Campaign Name", "Ad Group Name", "Actions_Taken", "Reason_Summary", "Spend", "Sales", "ROAS", "CVR"]
        display_df = filtered_df[[c for c in cols if c in filtered_df.columns]].copy()
        
        # Rename for display
        display_df = display_df.rename(columns={"Reason_Summary": "Reason"})
        
        def style_priority(val):
            if "High" in val: return "color: #ef4444; font-weight: bold"
            if "Medium" in val: return "color: #eab308; font-weight: bold"
            if "Good" in val: return "color: #22c55e; font-weight: bold"
            return ""

        from utils.formatters import get_account_currency
        format_currency = get_account_currency()
        
        styled = display_df.style.map(style_priority, subset=["Priority"]) \
                                 .background_gradient(subset=["ROAS"], cmap="RdYlGn") \
                                 .background_gradient(subset=["CVR"], cmap="YlGn") \
                                 .format({"Spend": f"{format_currency}{{:,.2f}}", "Sales": f"{format_currency}{{:,.2f}}", "ROAS": "{:.2f}x", "CVR": "{:.2f}%"})
        
        st.dataframe(styled, use_container_width=True, height=500)
    else:
        st.info("No heatmap data available.")
