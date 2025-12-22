"""
Harvest Tab Module - High-Performing Search Terms Display

Displays harvest candidates ready for promotion to Exact Match campaigns.
Integrates seamlessly with Campaign Creator via session state.
Separated from optimizer.py for cleaner maintenance and faster loading.
"""

import streamlit as st
import pandas as pd
from typing import Optional


def render_harvest_tab(harvest_df: Optional[pd.DataFrame]) -> None:
    """
    Render the Harvest Tab showing high-performing search terms.
    
    Includes integration button to pass data to Campaign Creator via:
    - st.session_state['harvest_payload']
    - st.session_state['active_creator_tab']
    - st.session_state['current_module']
    
    Args:
        harvest_df: DataFrame with harvest candidates
    """
    icon_color = "#8F8CA3"
    leaf_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8a8 8 0 0 1-8 8Z"></path><path d="M11 20c0-2.5 2-5.5 2-5.5"></path></svg>'
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); 
                border: 1px solid rgba(124, 58, 237, 0.2); 
                border-radius: 8px; 
                padding: 12px 16px; 
                margin-bottom: 20px;
                display: flex; 
                align-items: center; 
                gap: 10px;">
        {leaf_icon}
        <span style="color: #F5F5F7; font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Harvest Candidates</span>
    </div>
    """, unsafe_allow_html=True)

    if harvest_df is not None and not harvest_df.empty:
        st.markdown("""
        <div style="background: rgba(91, 85, 111, 0.05); border-left: 4px solid #5B556F; padding: 12px 20px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
            <p style="color: #B6B4C2; font-size: 0.9rem; margin: 0;">
                <strong>Success Strategy</strong>: These high-performing search terms have been identified for promotion to Exact Match campaigns to secure placement and improve ROI.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Bridge to Campaign Creator with Streamlit button (no page reload)
        cta_left, cta_right = st.columns([3, 1])
        with cta_left:
            st.markdown("""
            <div style="background: rgba(124, 58, 237, 0.08); border: 1px solid rgba(124, 58, 237, 0.2); padding: 15px; border-radius: 12px; display: flex; align-items: center;">
                <div style="color: #F5F5F7; font-size: 0.95rem;">
                    💡 <strong>Ready to Scale?</strong> Export these terms directly to the Campaign Creator.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with cta_right:
            # Store harvest data for Creator and navigate
            if st.button("OPEN CAMPAIGN CREATOR", type="primary", use_container_width=True):
                st.session_state['harvest_payload'] = harvest_df
                st.session_state['active_creator_tab'] = "Harvest Winners"
                st.session_state['current_module'] = 'creator'
                st.rerun()
        
        st.data_editor(harvest_df, use_container_width=True, height=400, disabled=True, hide_index=True)
    else:
        st.info("No harvest candidates met the performance criteria for this period.")
