
import os
import sys
import pandas as pd

# Add the parent directory to the path so we can import core modules
# Add the parent directory to the path so we can import core modules
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.db_manager import get_db_manager

def check_raw_metrics(client_id):
    db = get_db_manager()
    windows = [7, 14, 30]
    
    results = []
    
    for w in windows:
        print(f"\n--- Checking {w}D Window for {client_id} ---")
        summary = db.get_impact_summary(client_id, window_days=w)
        
        # Get the 'validated' summary specifically
        v = summary.get('validated', {})
        p = summary.get('period_info', {})
        
        results.append({
            'Window': f"{w}D",
            'Period': f"{p.get('after_start')} to {p.get('after_end')}",
            'Before Period': f"{p.get('before_start')} to {p.get('before_end')}",
            'Validated Actions': v.get('total_actions'),
            'Before Sales': v.get('roas_before') * v.get('total_actions'), # Rough estimate for display
            'Before ROAS': v.get('roas_before'),
            'After ROAS': v.get('roas_after'),
            'Lift %': v.get('roas_lift_pct'),
            'Significant': v.get('is_significant'),
            'P-Value': v.get('p_value')
        })
        
        # Also let's look at the raw sums from _calculate_metrics_from_df logic
        impact_df = db.get_action_impact(client_id, window_days=w)
        validated_mask = impact_df['validation_status'].str.contains('✓|CPC Validated|CPC Match|Directional|Confirmed|Normalized', na=False, regex=True)
        bid_mask = impact_df['action_type'].str.contains('BID', na=False)
        v_df = impact_df[validated_mask & bid_mask].copy()
        v_df = v_df[(v_df['before_spend'] > 0) | (v_df['observed_after_spend'] > 0)]
        
        b_spend = v_df['before_spend'].sum()
        b_sales = v_df['before_sales'].sum()
        a_spend = v_df['observed_after_spend'].sum()
        a_sales = v_df['observed_after_sales'].sum()
        
        print(f"RAW SUMS (Validated BID):")
        print(f"  Before: Sales {b_sales:,.2f} / Spend {b_spend:,.2f} = {(b_sales/b_spend if b_spend>0 else 0):.2f} ROAS")
        print(f"  After: Sales {a_sales:,.2f} / Spend {a_spend:,.2f} = {(a_sales/a_spend if a_spend>0 else 0):.2f} ROAS")
        
    df_results = pd.DataFrame(results)
    print("\n--- Summary Table ---")
    print(df_results.to_string())

if __name__ == "__main__":
    check_raw_metrics("demo_client")
