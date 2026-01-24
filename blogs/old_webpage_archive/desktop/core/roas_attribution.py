"""
ROAS Attribution Module
Provides clean, reusable functions for calculating ROAS decomposition.
Used by both validation scripts and the Impact Dashboard.
"""

from datetime import date, timedelta
from typing import Dict, Any, Optional
import pandas as pd


def get_period_metrics(db, client_id: str, start_date: date, end_date: date) -> Optional[Dict[str, Any]]:
    """
    Get aggregated metrics for a specific period.
    
    Returns dict with: spend, sales, clicks, orders, roas, cpc, cvr, aov, active_campaigns
    """
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    SUM(spend), SUM(sales), SUM(clicks), SUM(impressions),
                    SUM(COALESCE(orders, 0)), COUNT(DISTINCT campaign_name)
                FROM target_stats 
                WHERE client_id = %s AND start_date >= %s AND start_date <= %s
            """, (client_id, start_date, end_date))
            row = cursor.fetchone()
            
            if not row or row[0] is None:
                return None
            
            spend = row[0] or 0
            sales = row[1] or 0
            clicks = row[2] or 0
            impressions = row[3] or 0
            orders = row[4] or 0
            active_campaigns = row[5] or 0
            
            return {
                'spend': spend,
                'sales': sales,
                'clicks': clicks,
                'impressions': impressions,
                'orders': orders,
                'active_campaigns': active_campaigns,
                'roas': sales / spend if spend > 0 else 0,
                'cpc': spend / clicks if clicks > 0 else 0,
                'cvr': orders / clicks if clicks > 0 else 0,
                'aov': sales / orders if orders > 0 else 0
            }


def calculate_cpc_impact(prior: Dict, current: Dict) -> float:
    """CPC impact on ROAS (inverse relationship - higher CPC = lower ROAS)."""
    if prior['cpc'] == 0 or current['cpc'] == 0:
        return 0
    cpc_change_pct = (current['cpc'] - prior['cpc']) / prior['cpc']
    return -cpc_change_pct * prior['roas']


def calculate_cvr_impact(prior: Dict, current: Dict) -> Optional[float]:
    """CVR impact on ROAS (direct relationship - higher CVR = higher ROAS)."""
    if prior['cvr'] == 0 or current['cvr'] == 0:
        return None
    cvr_change_pct = (current['cvr'] - prior['cvr']) / prior['cvr']
    return cvr_change_pct * prior['roas']


def calculate_aov_impact(prior: Dict, current: Dict) -> Optional[float]:
    """AOV impact on ROAS (direct relationship)."""
    if prior['aov'] == 0 or current['aov'] == 0:
        return None
    aov_change_pct = (current['aov'] - prior['aov']) / prior['aov']
    return aov_change_pct * prior['roas']


def calculate_scale_effect(spend_change_pct: float, baseline_roas: float) -> float:
    """Estimate ROAS decline from scaling spend (diminishing returns)."""
    if abs(spend_change_pct) < 0.20:
        return 0
    dilution_factor = 0.05 * (spend_change_pct / 1.0)
    return -dilution_factor * baseline_roas


def calculate_portfolio_effect(campaign_change_pct: float, baseline_roas: float) -> float:
    """Estimate ROAS impact from new campaigns (typically start at 65% of baseline)."""
    if abs(campaign_change_pct) < 0.20:
        return 0
    if campaign_change_pct > 0:
        new_campaign_roas_factor = 0.65
        return -campaign_change_pct * (1 - new_campaign_roas_factor) * baseline_roas
    return 0


def get_roas_attribution(client_id: str, days: int = 30, decision_impact_value: float = 0) -> Optional[Dict[str, Any]]:
    """
    Get full ROAS attribution for a client.
    
    Returns dict with:
        - baseline_roas, actual_roas, total_change
        - cpc_impact, cvr_impact, aov_impact, market_impact_roas
        - scale_effect, portfolio_effect
        - decision_impact_roas
        - unexplained
        - periods (prior/current with dates)
        - flags, reconciles
    """
    from pathlib import Path
    from dotenv import load_dotenv
    import os
    
    # Ensure .env is loaded - it's at saddle/saddle/.env (two levels up from core/)
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    from core.db_manager import get_db_manager
    from core.postgres_manager import PostgresManager
    
    db = get_db_manager(test_mode=False)
    
    # Check if we got PostgresManager (either directly or need to access underlying)
    if not isinstance(db, PostgresManager):
        # Try to get postgres connection directly
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            try:
                db = PostgresManager(db_url)
            except Exception as e:
                print(f"Failed to create PostgresManager: {e}")
                return None
        else:
            print("DATABASE_URL not found, cannot calculate market decomposition")
            return None
    
    # Get latest data date
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT MAX(start_date) FROM target_stats WHERE client_id = %s", (client_id,))
            row = cursor.fetchone()
            latest_date = row[0] if row and row[0] else date.today()
    
    # Define periods
    current_end = latest_date
    current_start = current_end - timedelta(days=days)
    prior_end = current_start
    prior_start = prior_end - timedelta(days=days)
    
    # Get period metrics
    prior = get_period_metrics(db, client_id, prior_start, prior_end)
    current = get_period_metrics(db, client_id, current_start, current_end)
    
    if not prior or not current:
        return None
    
    # Calculate change percentages
    spend_change_pct = (current['spend'] - prior['spend']) / prior['spend'] if prior['spend'] > 0 else 0
    campaign_change_pct = (current['active_campaigns'] - prior['active_campaigns']) / prior['active_campaigns'] if prior['active_campaigns'] > 0 else 0
    
    # Market forces decomposition
    cpc_impact = calculate_cpc_impact(prior, current)
    cvr_impact = calculate_cvr_impact(prior, current)
    aov_impact = calculate_aov_impact(prior, current)
    
    market_impact_roas = (cpc_impact or 0) + (cvr_impact or 0) + (aov_impact or 0)
    
    # Scale and portfolio effects
    scale_effect = calculate_scale_effect(spend_change_pct, prior['roas'])
    portfolio_effect = calculate_portfolio_effect(campaign_change_pct, prior['roas'])
    
    # Decision impact in ROAS terms
    decision_impact_roas = decision_impact_value / current['spend'] if current['spend'] > 0 else 0
    
    # Unexplained residual
    total_change = current['roas'] - prior['roas']
    explained = decision_impact_roas + market_impact_roas + scale_effect + portfolio_effect
    unexplained = total_change - explained
    
    # Flags
    flags = []
    if abs(spend_change_pct) > 0.30:
        flags.append('⚠️ Scale Confounded')
    if abs(campaign_change_pct) > 0.30:
        flags.append('⚠️ Portfolio Confounded')
    if abs(unexplained) > 0.15:
        flags.append('⚠️ Large Residual')
    if cvr_impact is None:
        flags.append('⚠️ CVR Estimated')
    if not flags:
        flags.append('✓ Clean Attribution')
    
    return {
        # Core ROAS values
        'baseline_roas': round(prior['roas'], 2),
        'actual_roas': round(current['roas'], 2),
        'total_change': round(total_change, 2),
        
        # Market decomposition
        'cpc_impact': round(cpc_impact or 0, 2),
        'cvr_impact': round(cvr_impact or 0, 2),
        'aov_impact': round(aov_impact or 0, 2),
        'market_impact_roas': round(market_impact_roas, 2),
        
        # Other effects
        'scale_effect': round(scale_effect, 2),
        'portfolio_effect': round(portfolio_effect, 2),
        'decision_impact_roas': round(decision_impact_roas, 2),
        'unexplained': round(unexplained, 2),
        
        # Period info
        'periods': {
            'prior_start': prior_start,
            'prior_end': prior_end,
            'current_start': current_start,
            'current_end': current_end
        },
        
        # Metrics
        'prior_metrics': prior,
        'current_metrics': current,
        
        # Status
        'flags': flags,
        'reconciles': abs(unexplained) < 0.20
    }
