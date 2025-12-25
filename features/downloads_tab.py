"""
Downloads Tab Module - Bulk File Export Interface

Displays download buttons for Amazon Bulk Files with validation summaries.
Separated from optimizer.py for cleaner maintenance and faster loading.
"""

import streamlit as st
import pandas as pd
from typing import Dict
from features.bulk_export import generate_negatives_bulk, generate_bids_bulk
from utils.formatters import dataframe_to_excel


def render_downloads_tab(results: Dict, group_issues_fn=None) -> None:
    """
    Render the Downloads Tab with bulk file export interface.
    
    Args:
        results: Dictionary containing optimization results
        group_issues_fn: Optional function to group validation issues
    """
    icon_color = "#8F8CA3"
    dl_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>'
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); 
                border: 1px solid rgba(124, 58, 237, 0.2); 
                border-radius: 8px; 
                padding: 12px 16px; 
                margin-bottom: 20px;
                display: flex; 
                align-items: center; 
                gap: 10px;">
        {dl_icon}
        <span style="color: #F5F5F7; font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Export optimizations</span>
    </div>
    """, unsafe_allow_html=True)
    
    
    # 1. Negative Keywords + PT (combined)
    neg_kw = results.get("neg_kw", pd.DataFrame())
    neg_pt = results.get("neg_pt", pd.DataFrame())
    
    # Combine both KW and PT negatives into one file
    if not neg_kw.empty or not neg_pt.empty:
        shield_icon_sub = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
        st.markdown(f"<div style='color: #F5F5F7; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center;'>{shield_icon_sub}Negative Keywords Bulk</div>", unsafe_allow_html=True)
        kw_bulk, kw_issues = generate_negatives_bulk(neg_kw, neg_pt)
        
        # Calculate counts for display
        total_rows = len(kw_bulk)
        if total_rows > 0:
            st.markdown(f"<div style='color: #30D158; font-size: 14px; margin-bottom: 8px;'>✅ <b>{total_rows}</b> valid negative records ready for export</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color: #FF453A; font-size: 14px; margin-bottom: 8px;'>❌ <b>0</b> valid negative records available</div>", unsafe_allow_html=True)
        
        # Display validation warnings (grouped)
        if kw_issues and group_issues_fn:
            grouped_kw = group_issues_fn(kw_issues)
            with st.expander(f"⚠️ {len(kw_issues)} Validation Issues ({len(grouped_kw)} types)", expanded=False):
                for issue in grouped_kw:
                    rule = issue.get('rule') or issue.get('code', 'UNKNOWN')
                    msg = issue.get('msg') or issue.get('message', '')
                    severity = issue.get('severity', 'warning')
                    if severity == 'error':
                        st.error(f"**[{rule}]** {msg}")
                    else:
                        st.warning(f"**[{rule}]** {msg}")
        
        with st.expander("👁️ Preview File Content", expanded=False):
            st.dataframe(kw_bulk, use_container_width=True, height=300)
        
        buf = dataframe_to_excel(kw_bulk)
        st.download_button(
            label="📥 Download Negative Keywords (.xlsx)", 
            data=buf, 
            file_name="negative_keywords.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_neg_btn",
            type="primary",
            use_container_width=True,
            disabled=(total_rows == 0)
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # 2. Bids
    all_bids = pd.concat([results.get("direct_bids", pd.DataFrame()), results.get("agg_bids", pd.DataFrame())], ignore_index=True)
    if not all_bids.empty:
        sliders_icon_sub = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>'
        st.markdown(f"<div style='color: #F5F5F7; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center;'>{sliders_icon_sub}Bid Optimizations Bulk</div>", unsafe_allow_html=True)
        bid_bulk, bid_issues = generate_bids_bulk(all_bids)
        
        # Calculate counts
        total_rows = len(bid_bulk)
        if total_rows > 0:
            st.markdown(f"<div style='color: #30D158; font-size: 14px; margin-bottom: 8px;'>✅ <b>{total_rows}</b> valid bid updates ready for export</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color: #FF453A; font-size: 14px; margin-bottom: 8px;'>❌ <b>0</b> valid bid updates available</div>", unsafe_allow_html=True)
        
        # Display bid validation warnings (grouped)
        if bid_issues and group_issues_fn:
            grouped_bid = group_issues_fn(bid_issues)
            with st.expander(f"⚠️ {len(bid_issues)} Validation Issues ({len(grouped_bid)} types)", expanded=False):
                for issue in grouped_bid:
                    rule = issue.get('rule') or issue.get('code', 'UNKNOWN')
                    msg = issue.get('msg') or issue.get('message', '')
                    severity = issue.get('severity', 'warning')
                    if severity == 'error':
                        st.error(f"**[{rule}]** {msg}")
                    else:
                        st.warning(f"**[{rule}]** {msg}")
        
        with st.expander("👁️ Preview File Content", expanded=False):
            st.dataframe(bid_bulk, use_container_width=True, height=300)
        
        buf = dataframe_to_excel(bid_bulk)
        st.download_button(
            label="📥 Download Bid Adjustments (.xlsx)", 
            data=buf, 
            file_name="bid_optimizations.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_bid_btn",
            type="primary",
            use_container_width=True,
            disabled=(total_rows == 0)
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # 3. Harvest
    harvest = results.get("harvest", pd.DataFrame())
    if not harvest.empty:
        leaf_icon_sub = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8a8 8 0 0 1-8 8Z"></path><path d="M11 20c0-2.5 2-5.5 2-5.5"></path></svg>'
        st.markdown(f"<div style='color: #F5F5F7; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center;'>{leaf_icon_sub}Harvest Candidates</div>", unsafe_allow_html=True)
        with st.expander("👁️ Preview Candidate List", expanded=False):
            st.dataframe(harvest.head(5), use_container_width=True)
        
        buf = dataframe_to_excel(harvest)
        st.download_button(
            label="📥 Download Harvest List (.xlsx)", 
            data=buf, 
            file_name="harvest_candidates.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_harvest_btn",
            type="primary",
            use_container_width=True
        )
