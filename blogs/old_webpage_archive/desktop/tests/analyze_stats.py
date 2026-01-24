
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.postgres_manager import PostgresManager

# DB_URL = "postgresql://postgres.wuakeiwxkjvhsnmkzywz:Zen%40rise%40123%21@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres"

# Use the Env var if available, else hardcode
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres.wuakeiwxkjvhsnmkzywz:Zen%40rise%40123%21@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres")

def main():
    try:
        pm = PostgresManager(DB_URL)
        
        # 1. Find profile
        target_name = 'digiaansh_test'
        client_id = None
        with pm._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT account_id FROM accounts WHERE account_name ILIKE %s", (f'%{target_name}%',))
                res = cur.fetchone()
                if res: client_id = res[0]
        
        if not client_id:
            print("Account not found.")
            return

        # 2. Fetch Data
        df = pm.get_action_impact(client_id, before_days=14, after_days=14)
        if df.empty:
            print("No actions.")
            return

        # 3. Filter Validated Only (as per UI)
        validated_mask = df['validation_status'].fillna('').str.contains('✓|CPC Validated|CPC Match|Directional|Confirmed|Normalized|Volume', regex=True)
        val_df = df[validated_mask].copy()
        
        # 4. Filter for Bids/Visibility (Decision Quality context)
        bid_mask = val_df['action_type'].astype(str).str.contains('BID|VISIBILITY', regex=True)
        bid_df = val_df[bid_mask].copy()
        
        # 5. Calculate Row-Level Metrics (Same logic as UI)
        # Using the same calculations as PostgresManager._calculate_metrics_from_df
        bid_df['cpc_before'] = pd.to_numeric(bid_df['old_value'], errors='coerce').fillna(
            bid_df['before_spend'] / bid_df['before_clicks'].replace(0, np.nan)
        )
        # SPC Before (using window as fallback)
        bid_df['spc_window'] = bid_df['before_sales'] / bid_df['before_clicks'].replace(0, np.nan)
        bid_df['spc_before'] = bid_df['rolling_30d_spc'].fillna(bid_df['spc_window'])
        
        # Counterfactual
        bid_df['expected_clicks'] = bid_df['observed_after_spend'] / bid_df['cpc_before'].replace(0, np.nan)
        bid_df['expected_sales'] = bid_df['expected_clicks'] * bid_df['spc_before']
        
        # Metrics
        bid_df['decision_impact'] = bid_df['observed_after_sales'] - bid_df['expected_sales']
        bid_df['spend_avoided'] = (bid_df['before_spend'] - bid_df['observed_after_spend']).clip(lower=0)
        
        # Classification
        def classify(row):
            impact = row.get('decision_impact', 0)
            if pd.isna(impact): return 'Neutral'
            # Simple thresholds for checking
            if impact > 0: return 'Good'
            if impact < -25: return 'Bad' 
            return 'Neutral'
        bid_df['outcome'] = bid_df.apply(classify, axis=1)

        # 6. OUTPUT COMPARISON
        print(f"\n--- DATA CONSISTENCY CHECK ---")
        print(f"Rows Analyzed: {len(bid_df)}")
        
        sum_impact = bid_df['decision_impact'].sum()
        sum_avoided = bid_df['spend_avoided'].sum()
        
        print(f"\n[Aggregate Sums]")
        print(f"Total Decision Impact: {sum_impact:,.2f}")
        print(f"Total Spend Avoided:   {sum_avoided:,.2f}")
        
        print("\n[Detail Rows Sample]")
        print(f"{'Action':<15} | {'Term':<20} | {'Impact':>12} | {'Avoided':>12} | {'Outcome':<8}")
        print("-" * 80)
        for i, row in bid_df.head(10).iterrows():
            term = str(row['target_text'])[:18]
            imp = row['decision_impact']
            avo = row['spend_avoided']
            out = row['outcome']
            print(f"{row['action_type']:<15} | {term:<20} | {imp:12.2f} | {avo:12.2f} | {out:<8}")
            
        print("\n[Outcomes]")
        print(bid_df['outcome'].value_counts())

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
