"""
Negatives Tab Module - Negative Keyword and Product Targeting Display

Displays negative keyword and product targeting recommendations.
Separated from optimizer.py for cleaner maintenance and faster loading.
"""

import streamlit as st
import pandas as pd
from typing import Callable, Optional


def render_negatives_tab(neg_kw: pd.DataFrame, neg_pt: pd.DataFrame, extract_validation_fn: Optional[Callable] = None) -> None:
    """
    Render the Negatives tab with keyword and product targeting recommendations.
    
    Args:
        neg_kw: DataFrame with negative keyword recommendations
        neg_pt: DataFrame with negative product targeting recommendations
        extract_validation_fn: Optional function to extract validation info from recommendations
    """
    # Icons
    icon_color = "#8F8CA3"
    shield_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
    
    def tab_header(label, icon_html):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); 
                    border: 1px solid rgba(124, 58, 237, 0.2); 
                    border-radius: 8px; 
                    padding: 12px 16px; 
                    margin-bottom: 20px;
                    display: flex; 
                    align-items: center; 
                    gap: 10px;">
            {icon_html}
            <span style="color: #F5F5F7; font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">{label}</span>
        </div>
        """, unsafe_allow_html=True)

    # sub-navigation for negatives
    if 'active_neg_tab' not in st.session_state:
        st.session_state['active_neg_tab'] = "Keyword Negatives"
    
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    active_neg = st.radio(
        "Select Negative Type",
        options=["🛑 Keyword Negatives", "🎯 Product Targeting Negatives"],
        label_visibility="collapsed",
        horizontal=True,
        key="neg_radio_nav"
    )
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    st.session_state['active_neg_tab'] = active_neg.split(" ", 1)[1]  # Strip emoji


    st.markdown("<br>", unsafe_allow_html=True)
    
    active_tab = st.session_state['active_neg_tab']
    if active_tab == "Keyword Negatives":
        tab_header("Negative Keywords Identified", shield_icon)
        if not neg_kw.empty:
            # Apply validation extraction if function provided
            neg_kw_ui = extract_validation_fn(neg_kw) if extract_validation_fn else neg_kw
            cols = list(neg_kw_ui.columns)
            if not st.session_state.get("opt_show_ids", False):
                cols = [c for c in cols if "Id" not in c and "Basis" not in c and "Validation Issues" not in c]
            st.data_editor(neg_kw_ui[cols], use_container_width=True, height=400, disabled=True, hide_index=True)
        else:
            st.info("No negative keywords found.")
    else:
        tab_header("Product Targeting Candidates", shield_icon)
        if not neg_pt.empty:
            # Apply validation extraction if function provided
            neg_pt_ui = extract_validation_fn(neg_pt) if extract_validation_fn else neg_pt
            cols = list(neg_pt_ui.columns)
            if not st.session_state.get("opt_show_ids", False):
                cols = [c for c in cols if "Id" not in c and "Basis" not in c and "Validation Issues" not in c]
            st.data_editor(neg_pt_ui[cols], use_container_width=True, height=400, disabled=True, hide_index=True)
        else:
            st.info("No product targeting negatives found.")
