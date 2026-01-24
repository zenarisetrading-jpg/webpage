"""
Phase 2: Refined Attribution Framework
Decomposes ROAS change into 5 components:
1. Decision Impact (what we controlled)
2. Market Forces (CPC/CVR/AOV changes)
3. Scale Effects (spend dilution)
4. Portfolio Effects (campaign mix)
5. Unexplained Residual
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


def get_period_metrics(db, client_id: str, start_date: date, end_date: date) -> dict:
    """Get aggregated metrics for a period."""
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    SUM(spend) AS total_spend,
                    SUM(sales) AS total_sales,
                    SUM(clicks) AS total_clicks,
                    SUM(impressions) AS total_impressions,
                    SUM(COALESCE(orders, 0)) AS total_orders,
                    COUNT(DISTINCT campaign_name) AS active_campaigns,
                    -- Derived metrics
                    SUM(sales) / NULLIF(SUM(spend), 0) AS roas,
                    SUM(spend) / NULLIF(SUM(clicks), 0) AS cpc,
                    -- CVR: Use FLOAT division
                    CAST(SUM(COALESCE(orders, 0)) AS FLOAT) / NULLIF(SUM(clicks), 0) AS cvr,
                    SUM(sales) / NULLIF(SUM(COALESCE(orders, 0)), 0) AS aov
                FROM target_stats 
                WHERE client_id = %s 
                  AND start_date >= %s AND start_date <= %s
            """, (client_id, start_date, end_date))
            row = cursor.fetchone()
            
            if not row or row[0] is None:
                return None
            
            # Debug: Check actual values
            orders = row[4] or 0
            clicks = row[2] or 0
            cvr_calc = orders / clicks if clicks > 0 else 0
            
            return {
                'spend': row[0] or 0,
                'sales': row[1] or 0,
                'clicks': row[2] or 0,
                'impressions': row[3] or 0,
                'orders': row[4] or 0,
                'active_campaigns': row[5] or 0,
                'roas': row[6] or 0,
                'cpc': row[7] or 0,
                'cvr': cvr_calc,  # Use Python calculated CVR
                'aov': row[9] or 0
            }


def calculate_cpc_impact(prior: dict, current: dict) -> float:
    """CPC impact on ROAS (inverse relationship)."""
    if prior['cpc'] == 0 or current['cpc'] == 0:
        return 0
    cpc_change_pct = (current['cpc'] - prior['cpc']) / prior['cpc']
    # ROAS ∝ 1/CPC, so higher CPC = lower ROAS
    return -cpc_change_pct * prior['roas']


def calculate_cvr_impact(prior: dict, current: dict) -> float:
    """CVR impact on ROAS (direct relationship)."""
    if prior['cvr'] == 0 or current['cvr'] == 0:
        return None  # Can't calculate without CVR
    cvr_change_pct = (current['cvr'] - prior['cvr']) / prior['cvr']
    return cvr_change_pct * prior['roas']


def calculate_aov_impact(prior: dict, current: dict) -> float:
    """AOV impact on ROAS (direct relationship)."""
    if prior['aov'] == 0 or current['aov'] == 0:
        return None  # Can't calculate without AOV
    aov_change_pct = (current['aov'] - prior['aov']) / prior['aov']
    return aov_change_pct * prior['roas']


def estimate_scale_effect(spend_change_pct: float, baseline_roas: float) -> float:
    """
    Estimate ROAS decline from scaling spend.
    Based on diminishing returns curve.
    Conservative: ~0.5% ROAS decline per 10% spend increase.
    """
    if abs(spend_change_pct) < 0.20:  # Below threshold
        return 0
    
    # Diminishing returns factor
    dilution_factor = 0.05 * (spend_change_pct / 1.0)  # 5% ROAS decline per 100% spend increase
    return -dilution_factor * baseline_roas


def estimate_portfolio_effect(campaign_change_pct: float, baseline_roas: float) -> float:
    """
    Estimate ROAS impact from launching/pausing campaigns.
    New campaigns typically start at 60-70% of account average.
    """
    if abs(campaign_change_pct) < 0.20:  # Below threshold
        return 0
    
    if campaign_change_pct > 0:  # Net new campaigns
        new_campaign_roas_factor = 0.65  # New campaigns at 65% of baseline
        portfolio_dilution = campaign_change_pct * (1 - new_campaign_roas_factor)
        return -portfolio_dilution * baseline_roas
    else:
        return 0  # Pausing campaigns not typically dilutive


def generate_flags(spend_change: float, campaign_change: float, unexplained: float, 
                   cvr_available: bool, aov_available: bool) -> list:
    """Generate warning flags for confounded attribution."""
    flags = []
    
    if abs(spend_change) > 0.30:
        flags.append('⚠️ Scale Confounded')
    if abs(campaign_change) > 0.30:
        flags.append('⚠️ Portfolio Confounded')
    if abs(unexplained) > 0.15:
        flags.append('⚠️ Large Residual')
    if not cvr_available:
        flags.append('⚠️ CVR Estimated')
    if not aov_available:
        flags.append('⚠️ AOV Estimated')
    
    return flags if flags else ['✓ Clean Attribution']


def refined_roas_attribution(client_id: str, days: int = 30, decision_impact_value: float = 0):
    """
    Full ROAS attribution with 5-component decomposition.
    """
    db = get_db_manager(test_mode=False)
    if not isinstance(db, PostgresManager):
        print("ERROR: Requires PostgresManager")
        return None
    
    # Determine Latest Data Date
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT MAX(start_date) FROM target_stats WHERE client_id = %s", (client_id,))
            row = cursor.fetchone()
            latest_data_date = row[0] if row and row[0] else date.today()
    
    # Define Periods
    current_end = latest_data_date
    current_start = current_end - timedelta(days=days)
    prior_end = current_start
    prior_start = prior_end - timedelta(days=days)
    
    # Get Period Metrics
    prior = get_period_metrics(db, client_id, prior_start, prior_end)
    current = get_period_metrics(db, client_id, current_start, current_end)
    
    if not prior or not current:
        print(f"Insufficient data for {client_id}")
        return None
    
    # Calculate Change Percentages
    spend_change_pct = (current['spend'] - prior['spend']) / prior['spend'] if prior['spend'] > 0 else 0
    campaign_change_pct = (current['active_campaigns'] - prior['active_campaigns']) / prior['active_campaigns'] if prior['active_campaigns'] > 0 else 0
    
    # 1. Decision Impact (passed in from action-level calculation)
    decision_impact_roas = decision_impact_value / current['spend'] if current['spend'] > 0 else 0
    
    # 2. Market Forces
    cpc_impact = calculate_cpc_impact(prior, current)
    cvr_impact = calculate_cvr_impact(prior, current)
    aov_impact = calculate_aov_impact(prior, current)
    
    cvr_available = cvr_impact is not None
    aov_available = aov_impact is not None
    
    market_forces_total = (cpc_impact or 0) + (cvr_impact or 0) + (aov_impact or 0)
    
    # 3. Scale Effect
    scale_effect = estimate_scale_effect(spend_change_pct, prior['roas'])
    
    # 4. Portfolio Effect
    portfolio_effect = estimate_portfolio_effect(campaign_change_pct, prior['roas'])
    
    # 5. Unexplained Residual
    total_roas_change = current['roas'] - prior['roas']
    explained_change = decision_impact_roas + market_forces_total + scale_effect + portfolio_effect
    unexplained = total_roas_change - explained_change
    
    # Generate Flags
    flags = generate_flags(spend_change_pct, campaign_change_pct, unexplained, cvr_available, aov_available)
    
    return {
        'client_id': client_id,
        'periods': {
            'prior': {'start': prior_start, 'end': prior_end},
            'current': {'start': current_start, 'end': current_end}
        },
        'baseline_roas': prior['roas'],
        'actual_roas': current['roas'],
        'total_change': total_roas_change,
        'attribution': {
            'decision_impact': decision_impact_roas,
            'decision_impact_value': decision_impact_value,
            'market_forces': {
                'cpc': cpc_impact,
                'cvr': cvr_impact,
                'aov': aov_impact,
                'total': market_forces_total
            },
            'scale_effect': scale_effect,
            'portfolio_effect': portfolio_effect,
            'unexplained': unexplained
        },
        'metrics': {
            'prior': prior,
            'current': current
        },
        'changes': {
            'spend_pct': spend_change_pct * 100,
            'campaigns_pct': campaign_change_pct * 100,
            'cpc_pct': ((current['cpc'] - prior['cpc']) / prior['cpc'] * 100) if prior['cpc'] > 0 else 0,
            'cvr_pct': ((current['cvr'] - prior['cvr']) / prior['cvr'] * 100) if prior['cvr'] > 0 else None,
            'aov_pct': ((current['aov'] - prior['aov']) / prior['aov'] * 100) if prior['aov'] > 0 else None
        },
        'flags': flags,
        'reconciles': abs(unexplained) < 0.20  # Within tolerance
    }


def print_full_report(result: dict):
    """Print the full decomposition report (Mode 3: Power Users)."""
    if not result:
        return
    
    client = result['client_id']
    p = result['periods']
    attr = result['attribution']
    m = result['metrics']
    ch = result['changes']
    
    print("\n" + "="*75)
    print(f"ROAS ATTRIBUTION REPORT: {client}")
    print("="*75)
    print(f"Prior Period:   {p['prior']['start']} to {p['prior']['end']}")
    print(f"Current Period: {p['current']['start']} to {p['current']['end']}")
    print()
    
    # Header Metrics
    print(f"Baseline ROAS: {result['baseline_roas']:.2f}  →  Actual ROAS: {result['actual_roas']:.2f}  ({result['total_change']:+.2f})")
    print()
    
    # Changes Summary
    print("─── Period Changes ───────────────────────────────────────────────────────")
    print(f"  Spend:      ${m['prior']['spend']:,.0f} → ${m['current']['spend']:,.0f}  ({ch['spend_pct']:+.1f}%)")
    print(f"  Sales:      ${m['prior']['sales']:,.0f} → ${m['current']['sales']:,.0f}")
    print(f"  CPC:        ${m['prior']['cpc']:.2f} → ${m['current']['cpc']:.2f}  ({ch['cpc_pct']:+.1f}%)")
    print(f"  CVR:        {m['prior']['cvr']*100:.2f}% → {m['current']['cvr']*100:.2f}%  ({ch['cvr_pct']:+.1f}%)" if ch['cvr_pct'] else "  CVR:        N/A (missing orders)")
    print(f"  AOV:        ${m['prior']['aov']:.0f} → ${m['current']['aov']:.0f}  ({ch['aov_pct']:+.1f}%)" if ch['aov_pct'] else "  AOV:        N/A")
    print(f"  Campaigns:  {m['prior']['active_campaigns']} → {m['current']['active_campaigns']}  ({ch['campaigns_pct']:+.1f}%)")
    print()
    
    # Attribution Breakdown
    print("─── Attribution Breakdown ────────────────────────────────────────────────")
    print(f"  ├─ Decision Impact:    {attr['decision_impact']:+.2f}  (${attr['decision_impact_value']:+,.0f})")
    print(f"  │")
    print(f"  ├─ Market Forces:      {attr['market_forces']['total']:+.2f}")
    print(f"  │   ├─ CPC:            {attr['market_forces']['cpc']:+.2f}  (CPC {ch['cpc_pct']:+.1f}%)")
    print(f"  │   ├─ CVR:            {attr['market_forces']['cvr']:+.2f}" if attr['market_forces']['cvr'] else "  │   ├─ CVR:            N/A (estimated)")
    print(f"  │   └─ AOV:            {attr['market_forces']['aov']:+.2f}" if attr['market_forces']['aov'] else "  │   └─ AOV:            N/A (estimated)")
    print(f"  │")
    print(f"  ├─ Scale Effect:       {attr['scale_effect']:+.2f}  (Spend {ch['spend_pct']:+.1f}%)")
    print(f"  ├─ Portfolio Effect:   {attr['portfolio_effect']:+.2f}  (Campaigns {ch['campaigns_pct']:+.1f}%)")
    print(f"  │")
    print(f"  └─ Unexplained:        {attr['unexplained']:+.2f}")
    print()
    
    # Reconciliation
    recon = (result['baseline_roas'] + attr['decision_impact'] + 
             attr['market_forces']['total'] + attr['scale_effect'] + 
             attr['portfolio_effect'] + attr['unexplained'])
    print("─── Reconciliation ───────────────────────────────────────────────────────")
    print(f"  {result['baseline_roas']:.2f} + {attr['decision_impact']:+.2f} + {attr['market_forces']['total']:+.2f} + {attr['scale_effect']:+.2f} + {attr['portfolio_effect']:+.2f} + {attr['unexplained']:+.2f} = {recon:.2f}")
    print(f"  Matches Actual ({result['actual_roas']:.2f}): {'✓' if result['reconciles'] else '✗ INVESTIGATE'}")
    print()
    
    # Flags
    print("─── Flags ────────────────────────────────────────────────────────────────")
    for flag in result['flags']:
        print(f"  {flag}")
    print()


if __name__ == "__main__":
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    clients = ['digiaansh_test', 's2c_test', 's2c_uae_test']
    
    all_results = []
    
    for client in clients:
        # For now, pass 0 as decision impact (needs integration with action-level calc)
        # In production, this would come from validate_roas_attribution.py
        result = refined_roas_attribution(client, days=30, decision_impact_value=0)
        if result:
            all_results.append(result)
            print_full_report(result)
    
    # Summary Table
    print("\n" + "="*100)
    print("SUMMARY: REFINED ATTRIBUTION")
    print("="*100)
    print(f"{'Client':<18} | {'ROAS Δ':<8} | {'Decision':<10} | {'Market':<10} | {'Scale':<10} | {'Portfolio':<10} | {'Unexpl':<8} | {'Flags':<25}")
    print("-"*100)
    
    for r in all_results:
        a = r['attribution']
        flags_str = ', '.join(r['flags'][:2])  # First 2 flags
        print(f"{r['client_id']:<18} | {r['total_change']:+.2f}    | {a['decision_impact']:+.2f}      | {a['market_forces']['total']:+.2f}      | {a['scale_effect']:+.2f}      | {a['portfolio_effect']:+.2f}       | {a['unexplained']:+.2f}    | {flags_str:<25}")
