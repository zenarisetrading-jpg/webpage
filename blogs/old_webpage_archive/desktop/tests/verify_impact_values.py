"""
Quick script to verify impact values for s2c_uae_test
Run: export $(cat .env | xargs) && .venv/bin/python tests/verify_impact_values.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np

db_url = os.environ.get("DATABASE_URL")
if not db_url:
    print("ERROR: DATABASE_URL not set")
    sys.exit(1)

print(f"Connecting to DB...")


def get_impact_data(client_id, before_days, after_days):
    """Simplified impact query using correct column names"""
    query = """
    WITH action_data AS (
        SELECT 
            a.id, a.client_id, a.target_text, a.action_type, a.action_date,
            t.start_date as report_date, t.spend, t.sales, t.clicks
        FROM actions_log a
        LEFT JOIN target_stats t ON 
            LOWER(a.target_text) = LOWER(t.target_text)
            AND a.client_id = t.client_id
        WHERE a.client_id = %s
    )
    SELECT 
        target_text, action_type, action_date,
        SUM(CASE WHEN report_date < action_date 
            AND report_date >= action_date - INTERVAL '%s days' 
            THEN clicks ELSE 0 END)::float as before_clicks,
        SUM(CASE WHEN report_date < action_date 
            AND report_date >= action_date - INTERVAL '%s days' 
            THEN spend ELSE 0 END)::float as before_spend,
        SUM(CASE WHEN report_date < action_date 
            AND report_date >= action_date - INTERVAL '%s days' 
            THEN sales ELSE 0 END)::float as before_sales,
        SUM(CASE WHEN report_date >= action_date 
            AND report_date < action_date + INTERVAL '%s days' 
            THEN clicks ELSE 0 END)::float as after_clicks,
        SUM(CASE WHEN report_date >= action_date 
            AND report_date < action_date + INTERVAL '%s days' 
            THEN spend ELSE 0 END)::float as observed_after_spend,
        SUM(CASE WHEN report_date >= action_date 
            AND report_date < action_date + INTERVAL '%s days' 
            THEN sales ELSE 0 END)::float as observed_after_sales
    FROM action_data
    GROUP BY target_text, action_type, action_date
    HAVING SUM(CASE WHEN report_date < action_date THEN clicks ELSE 0 END) > 0
        OR SUM(CASE WHEN report_date >= action_date THEN clicks ELSE 0 END) > 0
    """
    
    with psycopg2.connect(db_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (client_id, before_days, before_days, before_days, after_days, after_days, after_days))
            rows = cursor.fetchall()
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    
    # Calculate metrics (replicating postgres_manager logic)
    df['spc_before'] = df['before_sales'] / df['before_clicks'].replace(0, np.nan)
    df['cpc_before'] = df['before_spend'] / df['before_clicks'].replace(0, np.nan)
    df['cpc_after'] = df['observed_after_spend'] / df['after_clicks'].replace(0, np.nan)
    
    # Expected clicks
    df['expected_clicks'] = df['observed_after_spend'] / df['cpc_before']
    
    # Expected sales (counterfactual)
    df['expected_sales'] = df['expected_clicks'] * df['spc_before']
    
    # Raw decision impact
    df['decision_impact'] = df['observed_after_sales'] - df['expected_sales']
    
    # GUARDRAIL 1: Zero out low-sample
    MIN_CLICKS = 5
    low_sample = df['before_clicks'] < MIN_CLICKS
    df.loc[low_sample, 'decision_impact'] = 0
    
    # ENHANCEMENT 1: Confidence Weight
    df['confidence_weight'] = (df['before_clicks'] / 15.0).clip(upper=1.0)
    df['final_decision_impact'] = df['decision_impact'] * df['confidence_weight']
    
    # ENHANCEMENT 2: Impact Tier
    conditions = [
        (df['decision_impact'] == 0),
        (df['decision_impact'] != 0) & (df['before_clicks'] < 15),
        (df['decision_impact'] != 0) & (df['before_clicks'] >= 15)
    ]
    choices = ['Excluded', 'Directional', 'Validated']
    df['impact_tier'] = np.select(conditions, choices, default='Excluded')
    
    return df


def main():
    client_id = "s2c_uae_test"
    
    print("=" * 60)
    print(f"Impact Values for: {client_id}")
    print("=" * 60)
    
    for after_days in [14, 30]:
        print(f"\n--- {after_days}D Horizon ---")
        
        impact_df = get_impact_data(client_id, before_days=14, after_days=after_days)
        
        if impact_df.empty:
            print("  No data found")
            continue
        
        print(f"  Total Rows: {len(impact_df)}")
        
        raw_total = impact_df['decision_impact'].sum()
        final_total = impact_df['final_decision_impact'].sum()
        
        print(f"\n  RAW Decision Impact Total: {raw_total:,.2f}")
        print(f"  FINAL (Weighted) Impact Total: {final_total:,.2f}")
        
        dampening = ((raw_total - final_total) / raw_total * 100) if raw_total != 0 else 0
        print(f"  Dampening Effect: {dampening:.1f}%")
        
        tier_counts = impact_df['impact_tier'].value_counts()
        print(f"\n  Impact Tier Breakdown:")
        for tier, count in tier_counts.items():
            print(f"    {tier}: {count}")
        
        avg_conf = impact_df['confidence_weight'].mean()
        print(f"\n  Avg Confidence Weight: {avg_conf:.2f}")


if __name__ == "__main__":
    main()
