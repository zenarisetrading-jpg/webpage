"""
Bids Tab Module - Bid Optimization Recommendations Display

Displays bid adjustment recommendations across different targeting types.
Separated from optimizer.py for cleaner maintenance and faster loading.
"""

import streamlit as st
import pandas as pd
from typing import Callable, Optional


def render_bids_tab(
    bids_exact: Optional[pd.DataFrame] = None,
    bids_pt: Optional[pd.DataFrame] = None,
    bids_agg: Optional[pd.DataFrame] = None,
    bids_auto: Optional[pd.DataFrame] = None,
    extract_validation_fn: Optional[Callable] = None
) -> None:
    """
    Render the Bids tab with bid adjustment recommendations.
    
    Args:
        bids_exact: Exact match keyword bids
        bids_pt: Product targeting bids
        bids_agg: Broad/Phrase match bids
        bids_auto: Auto/Category targeting bids
        extract_validation_fn: Optional function to extract validation info
    """
    icon_color = "#8F8CA3"
    sliders_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>'
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); 
                border: 1px solid rgba(124, 58, 237, 0.2); 
                border-radius: 8px; 
                padding: 12px 16px; 
                margin-bottom: 20px;
                display: flex; 
                align-items: center; 
                gap: 10px;">
        {sliders_icon}
        <span style="color: #F5F5F7; font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Bid Optimizations</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Define preferred column order (includes bid columns from bulk file)
    preferred_cols = ["Targeting", "Campaign Name", "Match Type", "Clicks", "Orders", "Sales", "ROAS", "Ad Group Default Bid", "Bid", "CPC", "New Bid", "Reason", "Decision_Basis", "Bucket"]
    
    if st.session_state.get("opt_show_ids", False):
        preferred_cols = ["CampaignId", "AdGroupId", "KeywordId", "TargetingId"] + preferred_cols
    
    # sub-navigation for bids
    if 'active_bid_tab' not in st.session_state:
        st.session_state['active_bid_tab'] = "Exact Keywords"
    
    # Use horizontal radio for clean tertiary navigation
    active_bid = st.radio(
        "Select Bid Category",
        options=["🎯 Exact Keywords", "📦 Product Targeting", "📈 Broad / Phrase", "⚡ Auto / Category"],
        label_visibility="collapsed",
        horizontal=True,
        key="bid_radio_nav"
    )
    st.session_state['active_bid_tab'] = active_bid.split(" ", 1)[1]  # Strip emoji


    st.markdown("<br>", unsafe_allow_html=True)
    
    def safe_display(df):
        if df is not None and not df.empty:
            # Apply validation extraction if function provided
            df_ui = extract_validation_fn(df) if extract_validation_fn else df
            display_cols = ["Status"] + [c for c in preferred_cols if c in df_ui.columns] if "Status" in df_ui.columns else [c for c in preferred_cols if c in df_ui.columns]
            if st.session_state.get("opt_show_ids", False) and "Validation Issues" in df_ui.columns:
                 display_cols.append("Validation Issues")
            st.data_editor(df_ui[display_cols], use_container_width=True, height=400, disabled=True, hide_index=True)
        else:
            st.info("No bid adjustments needed for this bucket.")

    active_tab = st.session_state['active_bid_tab']
    if active_tab == "Exact Keywords": safe_display(bids_exact)
    elif active_tab == "Product Targeting": safe_display(bids_pt)
    elif active_tab == "Broad / Phrase": safe_display(bids_agg)
    elif active_tab == "Auto / Category": safe_display(bids_auto)
