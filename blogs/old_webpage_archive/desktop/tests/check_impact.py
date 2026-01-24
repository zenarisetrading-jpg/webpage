#!/usr/bin/env python3
"""
Quick diagnostic script to check 30-day impact values for demo_client.
Run this with: streamlit run check_impact.py
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

st.title("Impact Diagnostic for demo_client")

try:
    from core.postgres_manager import PostgresManager
    
    db = PostgresManager()
    
    # Get impact data
    impact_df = db.get_action_impact('demo_client', window_days=30)
    summary = db.get_impact_summary('demo_client', window_days=30)
    
    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Actions", summary['total_actions'])
        st.metric("ROAS Before", f"{summary['roas_before']:.2f}x")
        st.metric("ROAS After", f"{summary['roas_after']:.2f}x")
    with col2:
        st.metric("ROAS Lift", f"{summary['roas_lift_pct']:.1f}%")
        st.metric("Incremental Revenue", f"${summary['incremental_revenue']:,.0f}")
        st.metric("Significant?", "✅ Yes" if summary['is_significant'] else "❌ No")
    with col3:
        st.metric("Confirmed", summary['confirmed_impact'])
        st.metric("Pending", summary['pending'])
        st.metric("P-Value", f"{summary['p_value']:.4f}")
    
    st.subheader("Validated BID Actions Check")
    if not impact_df.empty:
        # Filter same way as significance calc
        validated_mask = impact_df['validation_status'].str.contains('✓|CPC Validated|CPC Match|Directional|Confirmed', na=False, regex=True)
        bid_mask = impact_df['action_type'].str.contains('BID', na=False)
        validated_bid_df = impact_df[validated_mask & bid_mask].copy()
        validated_bid_df = validated_bid_df[(validated_bid_df['before_spend'] > 0) & (validated_bid_df['before_sales'] > 0)]
        validated_bid_df = validated_bid_df[validated_bid_df['observed_after_spend'] > 0]
        
        st.write(f"**Validated BID actions for significance calc:** {len(validated_bid_df)}")
        st.write(f"**Total BID actions:** {bid_mask.sum()}")
        st.write(f"**Total validated actions:** {validated_mask.sum()}")
        
        if len(validated_bid_df) > 0:
            st.dataframe(validated_bid_df[['target_text', 'before_spend', 'before_sales', 'observed_after_spend', 'observed_after_sales', 'validation_status']].head(20))
    
    st.subheader("By Action Type")
    for action_type, data in summary.get('by_action_type', {}).items():
        st.write(f"**{action_type}**: {data}")
        
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())
