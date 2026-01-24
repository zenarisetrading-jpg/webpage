"""
Diagnostic: Match Type Baseline Impact Analysis

Compares current account-wide baseline vs. match-type-specific baselines
to see how many targets would be reclassified.

Run: python analyze_match_type_baseline.py
"""

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.db_manager import get_db_manager
import pandas as pd
import numpy as np

# Configuration
CLIENT_ID = "demo_client"  # Change to your account
WINDOW_DAYS = 7

def get_historical_spc_by_matchtype(db, client_id):
    """Get full historical SPC by match type from target_stats (all time, not just 7D window)."""
    query = """
        SELECT 
            match_type,
            SUM(sales) as total_sales,
            SUM(clicks) as total_clicks,
            SUM(spend) as total_spend
        FROM target_stats
        WHERE client_id = %s
        GROUP BY match_type
    """
    try:
        with db._get_connection() as conn:
            result = pd.read_sql(query, conn, params=(client_id,))
        result['spc'] = result['total_sales'] / result['total_clicks'].replace(0, np.nan)
        result['cpc'] = result['total_spend'] / result['total_clicks'].replace(0, np.nan)
        return dict(zip(result['match_type'], result['spc']))
    except Exception as e:
        print(f"Error fetching historical SPC: {e}")
        return {}

def analyze_baseline_impact():
    db = get_db_manager(test_mode=False)
    
    # Get current impact data
    impact_df = db.get_action_impact(CLIENT_ID, window_days=WINDOW_DAYS)
    
    if impact_df.empty:
        print("No impact data found for", CLIENT_ID)
        return
    
    print(f"\n{'='*60}")
    print(f"BASELINE ANALYSIS: {CLIENT_ID}")
    print(f"Window: {WINDOW_DAYS}D")
    print(f"Total Actions: {len(impact_df)}")
    print(f"{'='*60}\n")
    
    # ===== CURRENT ACCOUNT-WIDE BASELINE =====
    total_before_spend = impact_df['before_spend'].sum()
    total_after_spend = impact_df['observed_after_spend'].sum()
    total_before_sales = impact_df['before_sales'].sum()
    total_after_sales = impact_df['observed_after_sales'].sum()
    
    account_roas_before = total_before_sales / total_before_spend if total_before_spend > 0 else 0
    account_roas_after = total_after_sales / total_after_spend if total_after_spend > 0 else 0
    account_roas_change_pct = (account_roas_after / account_roas_before - 1) * 100 if account_roas_before > 0 else 0
    
    print(f"ACCOUNT-WIDE BASELINE:")
    print(f"  Before ROAS: {account_roas_before:.2f}x")
    print(f"  After ROAS:  {account_roas_after:.2f}x")
    print(f"  Account ROAS Change: {account_roas_change_pct:+.1f}%")
    print()
    
    # ===== MATCH TYPE BASELINES =====
    print(f"MATCH TYPE BASELINES:")
    print(f"{'Match Type':<15} {'Before ROAS':>12} {'After ROAS':>12} {'Change %':>10} {'Count':>8}")
    print("-" * 60)
    
    match_type_baselines = {}
    for match_type in impact_df['match_type'].unique():
        mt_df = impact_df[impact_df['match_type'] == match_type]
        mt_before_spend = mt_df['before_spend'].sum()
        mt_after_spend = mt_df['observed_after_spend'].sum()
        mt_before_sales = mt_df['before_sales'].sum()
        mt_after_sales = mt_df['observed_after_sales'].sum()
        
        mt_roas_before = mt_before_sales / mt_before_spend if mt_before_spend > 0 else 0
        mt_roas_after = mt_after_sales / mt_after_spend if mt_after_spend > 0 else 0
        mt_roas_change = (mt_roas_after / mt_roas_before - 1) * 100 if mt_roas_before > 0 else 0
        
        match_type_baselines[match_type] = mt_roas_change
        
        print(f"{match_type or 'N/A':<15} {mt_roas_before:>12.2f}x {mt_roas_after:>12.2f}x {mt_roas_change:>+9.1f}% {len(mt_df):>8}")
    
    print()
    
    # ===== IMPACT ON CLASSIFICATIONS =====
    # For each target, calculate its individual ROAS change and compare to baselines
    
    impact_df['individual_roas_before'] = impact_df['before_sales'] / impact_df['before_spend'].replace(0, np.nan)
    impact_df['individual_roas_after'] = impact_df['observed_after_sales'] / impact_df['observed_after_spend'].replace(0, np.nan)
    impact_df['individual_roas_change_pct'] = (impact_df['individual_roas_after'] / impact_df['individual_roas_before'].replace(0, np.nan) - 1) * 100
    
    # Compare to account-wide baseline
    impact_df['vs_account_baseline'] = impact_df['individual_roas_change_pct'] - account_roas_change_pct
    
    # Compare to match-type-specific baseline
    impact_df['match_type_baseline'] = impact_df['match_type'].map(match_type_baselines).fillna(0)
    impact_df['vs_matchtype_baseline'] = impact_df['individual_roas_change_pct'] - impact_df['match_type_baseline']
    
    # Classify outcomes
    def classify_account(row):
        delta = row['vs_account_baseline']
        if pd.isna(delta): return 'Neutral'
        if delta > 5: return 'Good'
        if delta < -5: return 'Bad'
        return 'Neutral'
    
    def classify_matchtype(row):
        delta = row['vs_matchtype_baseline']
        if pd.isna(delta): return 'Neutral'
        if delta > 5: return 'Good'
        if delta < -5: return 'Bad'
        return 'Neutral'
    
    impact_df['outcome_account'] = impact_df.apply(classify_account, axis=1)
    impact_df['outcome_matchtype'] = impact_df.apply(classify_matchtype, axis=1)
    
    # Count classifications
    print("CLASSIFICATION COMPARISON:")
    print()
    
    account_counts = impact_df['outcome_account'].value_counts()
    matchtype_counts = impact_df['outcome_matchtype'].value_counts()
    
    print(f"{'Outcome':<12} {'Account Baseline':>18} {'MatchType Baseline':>20}")
    print("-" * 52)
    for outcome in ['Good', 'Neutral', 'Bad']:
        acct_cnt = account_counts.get(outcome, 0)
        mt_cnt = matchtype_counts.get(outcome, 0)
        acct_pct = acct_cnt / len(impact_df) * 100
        mt_pct = mt_cnt / len(impact_df) * 100
        diff = mt_cnt - acct_cnt
        print(f"{outcome:<12} {acct_cnt:>8} ({acct_pct:>5.1f}%) {mt_cnt:>10} ({mt_pct:>5.1f}%)  [{diff:+d}]")
    
    print()
    
    # Show reclassifications
    reclassified = impact_df[impact_df['outcome_account'] != impact_df['outcome_matchtype']]
    print(f"TARGETS RECLASSIFIED: {len(reclassified)} / {len(impact_df)} ({len(reclassified)/len(impact_df)*100:.1f}%)")
    print()
    
    if not reclassified.empty:
        print("Sample reclassifications:")
        print(f"{'Match Type':<10} {'Target':<30} {'Account':<10} {'→ MatchType':<10}")
        print("-" * 65)
        for _, row in reclassified.head(10).iterrows():
            target = str(row.get('target_text', ''))[:28]
            mt = str(row.get('match_type', ''))[:8]
            print(f"{mt:<10} {target:<30} {row['outcome_account']:<10} → {row['outcome_matchtype']:<10}")
    
    print()
    print("=" * 60)
    print("INTERPRETATION:")
    print("- [+] More 'Good' = More targets credited for beating their peer group")
    print("- [-] Less 'Bad' = Fewer false negatives from market-wide downturns")
    print("- Reclassified % = How much impact this change would have")
    print("=" * 60)
    
    # ===== DECISION IMPACT COMPARISON =====
    print()
    print("=" * 60)
    print("DECISION IMPACT COMPARISON ($ Value)")
    print("=" * 60)
    print()
    
    # Current method: Account-wide counterfactual
    # Expected_Sales = After_Spend / Before_CPC * Before_SPC
    # Decision_Impact = Actual_Sales - Expected_Sales
    
    impact_df['cpc_before'] = impact_df['before_spend'] / impact_df['before_clicks'].replace(0, np.nan)
    impact_df['spc_before'] = impact_df['before_sales'] / impact_df['before_clicks'].replace(0, np.nan)
    impact_df['expected_sales_account'] = (impact_df['observed_after_spend'] / impact_df['cpc_before']) * impact_df['spc_before']
    impact_df['decision_impact_account'] = impact_df['observed_after_sales'] - impact_df['expected_sales_account']
    
    # Match-type method: Use HISTORICAL SPC (full target_stats, not just 7D window)
    # This avoids skew from small sample sizes in the before window
    print("Using HISTORICAL SPC (full data) for match-type baseline...")
    historical_spc = get_historical_spc_by_matchtype(db, CLIENT_ID)
    
    if historical_spc:
        print("Historical SPC by Match Type:")
        for mt, spc in sorted(historical_spc.items(), key=lambda x: str(x[0])):
            print(f"  {mt or 'N/A'}: AED {spc:.2f}/click")
        print()
        match_type_spc = historical_spc
    else:
        print("⚠️ Could not fetch historical SPC, falling back to before-window SPC")
        match_type_spc = {}
        for match_type in impact_df['match_type'].unique():
            mt_df = impact_df[impact_df['match_type'] == match_type]
            mt_sales = mt_df['before_sales'].sum()
            mt_clicks = mt_df['before_clicks'].sum()
            match_type_spc[match_type] = mt_sales / mt_clicks if mt_clicks > 0 else 0
    
    impact_df['spc_matchtype'] = impact_df['match_type'].map(match_type_spc).fillna(0)
    impact_df['expected_sales_matchtype'] = (impact_df['observed_after_spend'] / impact_df['cpc_before']) * impact_df['spc_matchtype']
    impact_df['decision_impact_matchtype'] = impact_df['observed_after_sales'] - impact_df['expected_sales_matchtype']
    
    # Aggregate
    total_di_account = impact_df['decision_impact_account'].dropna().sum()
    total_di_matchtype = impact_df['decision_impact_matchtype'].dropna().sum()
    
    diff = total_di_matchtype - total_di_account
    diff_pct = (diff / abs(total_di_account) * 100) if total_di_account != 0 else 0
    
    print(f"Decision Impact (Account Baseline):    AED {total_di_account:>12,.0f}")
    print(f"Decision Impact (MatchType Baseline):  AED {total_di_matchtype:>12,.0f}")
    print(f"                                       {'─' * 15}")
    print(f"Difference:                            AED {diff:>+12,.0f} ({diff_pct:+.1f}%)")
    print()
    
    if diff > 0:
        print("✅ Match-type baseline IMPROVES Decision Impact (less negative / more positive)")
    elif diff < 0:
        print("⚠️ Match-type baseline WORSENS Decision Impact (more negative)")
    else:
        print("➖ No change in Decision Impact")
    
    print()
    
    # Breakdown by match type
    print("Decision Impact by Match Type:")
    print(f"{'Match Type':<15} {'Account DI':>12} {'MatchType DI':>14} {'Diff':>12}")
    print("-" * 55)
    for match_type in impact_df['match_type'].unique():
        mt_df = impact_df[impact_df['match_type'] == match_type]
        di_acct = mt_df['decision_impact_account'].dropna().sum()
        di_mt = mt_df['decision_impact_matchtype'].dropna().sum()
        di_diff = di_mt - di_acct
        print(f"{(match_type or 'N/A')[:15]:<15} {di_acct:>12,.0f} {di_mt:>14,.0f} {di_diff:>+12,.0f}")
    
    # ===== KEY DRIVER TABLE =====
    print()
    print("=" * 80)
    print("KEY DRIVER: Sales Per Click (SPC) Baseline Comparison")
    print("=" * 80)
    print()
    print("DI = Actual_Sales - Expected_Sales")
    print("Expected_Sales = (After_Spend / Before_CPC) × SPC_Baseline")
    print()
    print("If SPC_Baseline is HIGHER → Expected_Sales is HIGHER → DI is LOWER (worse)")
    print()
    
    # Calculate account-wide SPC
    total_sales = impact_df['before_sales'].sum()
    total_clicks = impact_df['before_clicks'].sum()
    account_spc = total_sales / total_clicks if total_clicks > 0 else 0
    
    print(f"Account-wide SPC: AED {account_spc:.2f} per click")
    print()
    
    print(f"{'Match Type':<15} {'SPC (Acct)':>12} {'SPC (MT)':>12} {'Diff':>10} {'After Spend':>14} {'Exp Sales Diff':>16}")
    print("-" * 80)
    
    for match_type in sorted(impact_df['match_type'].unique(), key=lambda x: str(x)):
        mt_df = impact_df[impact_df['match_type'] == match_type]
        mt_spc = match_type_spc.get(match_type, 0)
        spc_diff = mt_spc - account_spc
        after_spend = mt_df['observed_after_spend'].sum()
        avg_cpc = mt_df['cpc_before'].dropna().mean()
        
        if avg_cpc and avg_cpc > 0:
            expected_clicks = after_spend / avg_cpc
            expected_sales_diff = expected_clicks * spc_diff
        else:
            expected_sales_diff = 0
        
        indicator = "👆" if mt_spc > account_spc else "👇" if mt_spc < account_spc else "➖"
        
        print(f"{(match_type or 'N/A')[:15]:<15} {account_spc:>12.2f} {mt_spc:>12.2f} {spc_diff:>+9.2f} {indicator} {after_spend:>13,.0f} {expected_sales_diff:>+15,.0f}")
    
    print()
    print("INTERPRETATION:")
    print("- SPC (MT) > SPC (Acct) → Match type has higher expectations → DI worsens")  
    print("- The 'Exp Sales Diff' shows HOW MUCH the expected sales changes by using MT baseline")
    print("- A large negative 'Exp Sales Diff' for EXACT means: we expect MORE from Exact → DI drops")


if __name__ == "__main__":
    analyze_baseline_impact()
