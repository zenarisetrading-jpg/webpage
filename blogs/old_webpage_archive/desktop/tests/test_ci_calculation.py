"""
Confidence Interval Validation Script
=====================================
Tests the proposed CI calculation on real data from digiaansh_test before implementation.

Usage:
    python tests/test_ci_calculation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from core.db_manager import get_db_manager

def compute_confidence_intervals(impact_df: pd.DataFrame, z: float = 1.96) -> dict:
    """
    Compute confidence intervals for aggregate impact metrics.
    
    Method: Cross-action standard error
    - Each action is treated as an independent observation
    - Variance is computed across action-level deltas
    - SE_total = std(deltas) * sqrt(N)
    """
    if impact_df.empty:
        return {"error": "No data"}
    
    # Filter to validated actions only (those with actual measurement)
    validated = impact_df[impact_df['validation_status'].str.contains('✓|Confirmed|Harvested', case=False, na=False)].copy()
    
    print(f"\n📊 DATA SUMMARY")
    print(f"   Total actions: {len(impact_df)}")
    print(f"   Validated actions: {len(validated)}")
    
    if validated.empty:
        print("   ⚠️ No validated actions - using all actions with before_spend > 0")
        validated = impact_df[impact_df['before_spend'] > 0].copy()
    
    n = len(validated)
    if n < 2:
        return {"error": f"Need at least 2 actions, got {n}"}
    
    # Compute per-action deltas
    # For NEGATIVE: impact = before_spend (cost avoided)
    # For HARVEST: impact = (after_sales - before_sales) for exact match
    # For BID_CHANGE: impact = (delta_sales - delta_spend)
    
    neg_mask = validated['action_type'].isin(['NEGATIVE', 'NEGATIVE_ADD'])
    harvest_mask = validated['action_type'].str.contains('HARVEST', case=False, na=False)
    bid_mask = validated['action_type'].str.contains('BID', case=False, na=False)
    
    # === SPEND AVOIDED (Negatives) ===
    neg_data = validated[neg_mask].copy()
    if len(neg_data) > 1:
        spend_avoided_per_action = neg_data['before_spend'].values  # Each negative saves its before_spend
        total_spend_avoided = spend_avoided_per_action.sum()
        std_spend_avoided = np.std(spend_avoided_per_action, ddof=1)
        se_spend_avoided = std_spend_avoided * np.sqrt(len(neg_data))
        ci_spend_avoided = (total_spend_avoided - z * se_spend_avoided, 
                           total_spend_avoided + z * se_spend_avoided)
    else:
        total_spend_avoided = neg_data['before_spend'].sum() if len(neg_data) > 0 else 0
        ci_spend_avoided = (total_spend_avoided, total_spend_avoided)  # No CI with n<2
        se_spend_avoided = 0
    
    # === REVENUE IMPACT (All actions) ===
    # Revenue delta per action = observed_after_sales - before_sales
    validated['revenue_delta'] = validated['observed_after_sales'] - validated['before_sales']
    revenue_deltas = validated['revenue_delta'].values
    
    total_revenue_impact = revenue_deltas.sum()
    std_revenue = np.std(revenue_deltas, ddof=1)
    se_revenue = std_revenue * np.sqrt(n)
    ci_revenue = (total_revenue_impact - z * se_revenue, 
                  total_revenue_impact + z * se_revenue)
    
    # === PROFIT IMPACT (Sales delta - Spend delta) ===
    validated['spend_delta'] = validated['observed_after_spend'] - validated['before_spend']
    validated['profit_delta'] = validated['revenue_delta'] - validated['spend_delta']
    profit_deltas = validated['profit_delta'].values
    
    total_profit_impact = profit_deltas.sum()
    std_profit = np.std(profit_deltas, ddof=1)
    se_profit = std_profit * np.sqrt(n)
    ci_profit = (total_profit_impact - z * se_profit, 
                 total_profit_impact + z * se_profit)
    
    # === STATISTICAL SIGNIFICANCE ===
    is_significant_revenue = (ci_revenue[0] > 0) or (ci_revenue[1] < 0)
    is_significant_spend = (ci_spend_avoided[0] > 0)  # Spend avoided is always positive
    is_significant_profit = (ci_profit[0] > 0) or (ci_profit[1] < 0)
    
    return {
        'n_actions': n,
        'n_negatives': len(neg_data),
        
        'spend_avoided': {
            'point': total_spend_avoided,
            'ci_lower': ci_spend_avoided[0],
            'ci_upper': ci_spend_avoided[1],
            'se': se_spend_avoided,
            'is_significant': is_significant_spend
        },
        'revenue_impact': {
            'point': total_revenue_impact,
            'ci_lower': ci_revenue[0],
            'ci_upper': ci_revenue[1],
            'se': se_revenue,
            'is_significant': is_significant_revenue
        },
        'profit_impact': {
            'point': total_profit_impact,
            'ci_lower': ci_profit[0],
            'ci_upper': ci_profit[1],
            'se': se_profit,
            'is_significant': is_significant_profit
        },
        
        # Diagnostics
        'revenue_std_per_action': std_revenue,
        'profit_std_per_action': std_profit,
        'action_breakdown': {
            'negatives': len(neg_data),
            'harvests': len(validated[harvest_mask]),
            'bids': len(validated[bid_mask])
        }
    }


def format_currency(val, currency='INR'):
    """Format currency with sign."""
    sign = '+' if val > 0 else ''
    return f"{sign}{currency}{val:,.0f}"


def format_ci(result: dict, currency='INR') -> str:
    """Format confidence interval for display."""
    point = result['point']
    lower = result['ci_lower']
    upper = result['ci_upper']
    sig = "✓ Significant" if result['is_significant'] else "⚠️ Not Significant"
    
    return f"{format_currency(point, currency)} ({format_currency(lower, currency)} to {format_currency(upper, currency)}) — {sig}"


def main():
    print("=" * 60)
    print("CONFIDENCE INTERVAL VALIDATION TEST")
    print("=" * 60)
    
    # Connect to database
    db = get_db_manager(test_mode=False)
    if not db:
        print("❌ Failed to connect to database")
        return
    
    client_id = 'digiaansh_test'
    print(f"\n📡 Fetching impact data for: {client_id}")
    
    # Get impact data
    try:
        impact_df = db.get_action_impact(client_id, before_days=14, after_days=14)
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return
    
    if impact_df.empty:
        print("❌ No impact data found")
        return
    
    print(f"✅ Loaded {len(impact_df)} actions")
    
    # Show sample of data
    print("\n📋 SAMPLE DATA (first 5 rows):")
    cols_to_show = ['action_type', 'before_spend', 'before_sales', 'observed_after_spend', 'observed_after_sales', 'validation_status']
    available_cols = [c for c in cols_to_show if c in impact_df.columns]
    print(impact_df[available_cols].head().to_string())
    
    # Compute CIs
    print("\n" + "=" * 60)
    print("CONFIDENCE INTERVAL RESULTS")
    print("=" * 60)
    
    results = compute_confidence_intervals(impact_df)
    
    if 'error' in results:
        print(f"❌ {results['error']}")
        return
    
    print(f"\n📊 ACTIONS ANALYZED: {results['n_actions']}")
    print(f"   - Negatives: {results['action_breakdown']['negatives']}")
    print(f"   - Harvests: {results['action_breakdown']['harvests']}")
    print(f"   - Bids: {results['action_breakdown']['bids']}")
    
    print(f"\n💰 SPEND AVOIDED (Negatives):")
    print(f"   {format_ci(results['spend_avoided'])}")
    
    print(f"\n📈 REVENUE IMPACT (All Actions):")
    print(f"   {format_ci(results['revenue_impact'])}")
    
    print(f"\n🎯 PROFIT IMPACT (Revenue - Spend):")
    print(f"   {format_ci(results['profit_impact'])}")
    
    print(f"\n📉 VARIANCE DIAGNOSTICS:")
    print(f"   Std Dev per action (Revenue): INR{results['revenue_std_per_action']:,.0f}")
    print(f"   Std Dev per action (Profit): INR{results['profit_std_per_action']:,.0f}")
    
    # Interpretation guide
    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)
    print("""
    ✓ Significant: CI does NOT include zero — confident in the direction
    ⚠️ Not Significant: CI includes zero — need more data
    
    Wide CI = high variance across actions (some helped, some hurt)
    Narrow CI = consistent effect across actions (reliable signal)
    """)


if __name__ == "__main__":
    main()
