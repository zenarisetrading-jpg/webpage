"""
Decompose Market Impact into structural components.
Goal: Validate if "Market Impact" on ROAS is genuine or confounded.

Checks:
1. Spend Scale Effect (spending more → ROAS dilution)
2. Campaign Structure Changes (portfolio shifts)
3. CPC/CVR Changes (genuine market forces)
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from core.db_manager import get_db_manager
from core.postgres_manager import PostgresManager


def decompose_market_impact(client_id: str, days: int = 30):
    """Run diagnostic queries to decompose market impact."""
    
    db = get_db_manager(test_mode=False)
    if not isinstance(db, PostgresManager):
        print("ERROR: Requires PostgresManager")
        return
    
    # Determine Latest Data Date
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT MAX(start_date) FROM target_stats WHERE client_id = %s", (client_id,))
            row = cursor.fetchone()
            latest_data_date = row[0] if row and row[0] else date.today()
    
    print(f"\n{'='*60}")
    print(f"MARKET IMPACT DECOMPOSITION: {client_id}")
    print(f"Latest Data Date: {latest_data_date}")
    print(f"{'='*60}\n")
    
    # Define Periods
    analysis_end = datetime.combine(latest_data_date, datetime.min.time())
    analysis_start = analysis_end - timedelta(days=days)
    prior_end = analysis_start
    prior_start = prior_end - timedelta(days=days)
    
    print(f"Prior Period:   {prior_start.date()} to {prior_end.date()}")
    print(f"Current Period: {analysis_start.date()} to {analysis_end.date()}\n")
    
    # ==========================================
    # QUERY 1: SPEND & ROAS COMPARISON
    # ==========================================
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            # Prior Period
            cursor.execute("""
                SELECT 
                    SUM(spend) AS total_spend,
                    SUM(sales) AS total_sales,
                    SUM(clicks) AS total_clicks,
                    SUM(impressions) AS total_impressions,
                    SUM(sales) / NULLIF(SUM(spend), 0) AS roas,
                    SUM(spend) / NULLIF(SUM(clicks), 0) AS cpc,
                    SUM(orders) / NULLIF(SUM(clicks), 0) AS cvr
                FROM target_stats 
                WHERE client_id = %s 
                  AND start_date >= %s AND start_date < %s
            """, (client_id, prior_start.date(), prior_end.date()))
            prior_row = cursor.fetchone()
            
            # Current Period
            cursor.execute("""
                SELECT 
                    SUM(spend) AS total_spend,
                    SUM(sales) AS total_sales,
                    SUM(clicks) AS total_clicks,
                    SUM(impressions) AS total_impressions,
                    SUM(sales) / NULLIF(SUM(spend), 0) AS roas,
                    SUM(spend) / NULLIF(SUM(clicks), 0) AS cpc,
                    SUM(orders) / NULLIF(SUM(clicks), 0) AS cvr
                FROM target_stats 
                WHERE client_id = %s 
                  AND start_date >= %s AND start_date <= %s
            """, (client_id, analysis_start.date(), analysis_end.date()))
            current_row = cursor.fetchone()
    
    if not prior_row or not current_row or prior_row[0] is None or current_row[0] is None:
        print("ERROR: Insufficient data for analysis.")
        return
    
    prior_spend, prior_sales, prior_clicks, prior_imps, prior_roas, prior_cpc, prior_cvr = prior_row
    curr_spend, curr_sales, curr_clicks, curr_imps, curr_roas, curr_cpc, curr_cvr = current_row
    
    # Handle None values
    prior_spend = prior_spend or 0
    prior_sales = prior_sales or 0
    prior_roas = prior_roas or 0
    prior_cpc = prior_cpc or 0
    prior_cvr = prior_cvr or 0
    
    curr_spend = curr_spend or 0
    curr_sales = curr_sales or 0
    curr_roas = curr_roas or 0
    curr_cpc = curr_cpc or 0
    curr_cvr = curr_cvr or 0
    
    print("="*60)
    print("QUERY 1: PERIOD COMPARISON")
    print("="*60)
    print(f"{'Metric':<20} | {'Prior':<15} | {'Current':<15} | {'Change %':<10}")
    print("-"*60)
    
    spend_change = ((curr_spend - prior_spend) / prior_spend * 100) if prior_spend > 0 else 0
    roas_change = ((curr_roas - prior_roas) / prior_roas * 100) if prior_roas > 0 else 0
    cpc_change = ((curr_cpc - prior_cpc) / prior_cpc * 100) if prior_cpc > 0 else 0
    cvr_change = ((curr_cvr - prior_cvr) / prior_cvr * 100) if prior_cvr > 0 else 0
    
    print(f"{'Spend':<20} | ${prior_spend:,.2f}     | ${curr_spend:,.2f}     | {spend_change:+.1f}%")
    print(f"{'Sales':<20} | ${prior_sales:,.2f}     | ${curr_sales:,.2f}     | {((curr_sales - prior_sales) / prior_sales * 100) if prior_sales > 0 else 0:+.1f}%")
    print(f"{'ROAS':<20} | {prior_roas:.2f}          | {curr_roas:.2f}          | {roas_change:+.1f}%")
    print(f"{'CPC':<20} | ${prior_cpc:.2f}         | ${curr_cpc:.2f}         | {cpc_change:+.1f}%")
    print(f"{'CVR':<20} | {prior_cvr*100:.2f}%        | {curr_cvr*100:.2f}%        | {cvr_change:+.1f}%")
    
    # ==========================================
    # FLAG ANALYSIS
    # ==========================================
    print("\n" + "="*60)
    print("FLAG ANALYSIS")
    print("="*60)
    
    flags = []
    
    # Flag 1: Scale Effect
    if abs(spend_change) > 30:
        flags.append(f"🚩 SCALE EFFECT: Spend changed {spend_change:+.1f}% (>30% threshold)")
        flags.append(f"   → This is NOT market impact. It's scale dilution.")
    
    # Flag 2: CPC Shift (Genuine Market)
    if abs(cpc_change) > 20:
        flags.append(f"✓ GENUINE CPC SHIFT: CPC changed {cpc_change:+.1f}% (>20% = real market pressure)")
    else:
        flags.append(f"⚪ CPC Stable: CPC changed only {cpc_change:+.1f}% (<20% = not CPC driven)")
    
    # Flag 3: CVR Shift (Demand Change)
    if abs(cvr_change) > 15:
        flags.append(f"✓ CVR SHIFT: CVR changed {cvr_change:+.1f}% (>15% = demand/quality shift)")
    else:
        flags.append(f"⚪ CVR Stable: CVR changed only {cvr_change:+.1f}% (<15% = not demand driven)")
    
    for flag in flags:
        print(flag)
    
    # ==========================================
    # QUERY 2: CAMPAIGN STRUCTURE
    # ==========================================
    print("\n" + "="*60)
    print("QUERY 2: CAMPAIGN STRUCTURE CHANGES")
    print("="*60)
    
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            # Prior unique campaigns
            cursor.execute("""
                SELECT COUNT(DISTINCT campaign_name) 
                FROM target_stats 
                WHERE client_id = %s 
                  AND start_date >= %s AND start_date < %s
            """, (client_id, prior_start.date(), prior_end.date()))
            prior_campaigns = cursor.fetchone()[0] or 0
            
            # Current unique campaigns
            cursor.execute("""
                SELECT COUNT(DISTINCT campaign_name) 
                FROM target_stats 
                WHERE client_id = %s 
                  AND start_date >= %s AND start_date <= %s
            """, (client_id, analysis_start.date(), analysis_end.date()))
            curr_campaigns = cursor.fetchone()[0] or 0
    
    campaign_change = ((curr_campaigns - prior_campaigns) / prior_campaigns * 100) if prior_campaigns > 0 else 0
    
    print(f"Prior Active Campaigns:   {prior_campaigns}")
    print(f"Current Active Campaigns: {curr_campaigns}")
    print(f"Change: {campaign_change:+.1f}%")
    
    if abs(campaign_change) > 20:
        print(f"🚩 PORTFOLIO EFFECT: Campaign count changed {campaign_change:+.1f}% (>20%)")
        print(f"   → This is structural, not pure market impact.")
    else:
        print(f"⚪ Campaign structure stable.")
    
    # ==========================================
    # QUERY 3: DECOMPOSE ROAS CHANGE
    # ==========================================
    print("\n" + "="*60)
    print("QUERY 3: ROAS CHANGE DECOMPOSITION")
    print("="*60)
    
    # Total ROAS Change
    total_roas_change = curr_roas - prior_roas
    
    # Approximate decomposition:
    # ROAS = Sales / Spend = (Clicks * CVR * AOV) / (Clicks * CPC) = CVR * AOV / CPC
    # Simplified: ROAS ≈ f(CPC, CVR) holding AOV constant
    
    # CPC Impact on ROAS (inverse relationship)
    # If CPC goes up, ROAS goes down
    cpc_roas_impact = 0
    if prior_cpc > 0 and curr_cpc > 0:
        # ROAS ∝ 1/CPC, so delta ROAS from CPC = baseline_roas * (1 - curr_cpc/prior_cpc)
        cpc_roas_impact = prior_roas * (1 - curr_cpc / prior_cpc)
    
    # CVR Impact on ROAS (direct relationship)
    # If CVR goes up, ROAS goes up
    cvr_roas_impact = 0
    if prior_cvr > 0 and curr_cvr > 0:
        # ROAS ∝ CVR, so delta ROAS from CVR = baseline_roas * (curr_cvr/prior_cvr - 1)
        cvr_roas_impact = prior_roas * (curr_cvr / prior_cvr - 1)
    
    explained_roas_change = cpc_roas_impact + cvr_roas_impact
    unexplained = total_roas_change - explained_roas_change
    
    print(f"Total ROAS Change:      {total_roas_change:+.2f}")
    print(f"  └─ From CPC Change:   {cpc_roas_impact:+.2f}")
    print(f"  └─ From CVR Change:   {cvr_roas_impact:+.2f}")
    print(f"  └─ Unexplained (AOV/Scale/Other): {unexplained:+.2f}")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    explained_pct = (abs(explained_roas_change) / abs(total_roas_change) * 100) if total_roas_change != 0 else 0
    
    if explained_pct > 70:
        print(f"✅ Market forces (CPC/CVR) explain {explained_pct:.0f}% of ROAS change.")
        print(f"   → 'Market Impact' label is VALID.")
    elif explained_pct > 40:
        print(f"⚠️ Market forces explain only {explained_pct:.0f}% of ROAS change.")
        print(f"   → 'Market Impact' label is PARTIALLY VALID. Consider other factors.")
    else:
        print(f"🚩 Market forces explain only {explained_pct:.0f}% of ROAS change.")
        print(f"   → 'Market Impact' label is MISLEADING. Likely Scale/Portfolio effect.")
    
    return {
        'client_id': client_id,
        'spend_change_pct': spend_change,
        'roas_change': total_roas_change,
        'cpc_roas_impact': cpc_roas_impact,
        'cvr_roas_impact': cvr_roas_impact,
        'unexplained': unexplained,
        'explained_pct': explained_pct,
        'is_scale_effect': abs(spend_change) > 30,
        'is_genuine_market': explained_pct > 70
    }


if __name__ == "__main__":
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    clients_to_check = ["digiaansh_test", "s2c_test", "s2c_uae_test"]
    
    results = []
    for client in clients_to_check:
        try:
            res = decompose_market_impact(client, days=30)
            if res:
                results.append(res)
        except Exception as e:
            print(f"Error for {client}: {e}")
    
    # Summary Table
    print("\n\n" + "="*80)
    print("SUMMARY: MARKET IMPACT VALIDITY CHECK")
    print("="*80)
    print(f"{'Client':<20} | {'Spend Δ%':<10} | {'ROAS Δ':<10} | {'Explained%':<12} | {'Verdict':<20}")
    print("-"*80)
    
    for r in results:
        verdict = "✅ Genuine Market" if r['is_genuine_market'] else ("🚩 Scale Effect" if r['is_scale_effect'] else "⚠️ Mixed")
        print(f"{r['client_id']:<20} | {r['spend_change_pct']:+.1f}%     | {r['roas_change']:+.2f}     | {r['explained_pct']:.0f}%         | {verdict}")
