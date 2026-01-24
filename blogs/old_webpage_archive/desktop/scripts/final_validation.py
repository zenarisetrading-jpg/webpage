"""
Final Validation: CVR Time Series and Action Count Discrepancy
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from core.db_manager import get_db_manager
from core.postgres_manager import PostgresManager


def validate_cvr_surge(client_id: str = 's2c_test'):
    """Check if CVR surge is real or data artifact."""
    
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    db = get_db_manager(test_mode=False)
    
    print("="*70)
    print(f"CVR TIME SERIES VALIDATION: {client_id}")
    print("="*70)
    
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            # Get date range
            cursor.execute("SELECT MIN(start_date), MAX(start_date) FROM target_stats WHERE client_id = %s", (client_id,))
            min_date, max_date = cursor.fetchone()
            print(f"Data Range: {min_date} to {max_date}\n")
            
            # Weekly CVR breakdown
            cursor.execute("""
                SELECT
                    DATE_TRUNC('week', start_date) AS week,
                    SUM(COALESCE(orders, 0)) AS weekly_orders,
                    SUM(clicks) AS weekly_clicks,
                    CAST(SUM(COALESCE(orders, 0)) AS FLOAT) / NULLIF(SUM(clicks), 0) * 100 AS weekly_cvr,
                    COUNT(DISTINCT campaign_name) AS active_campaigns,
                    SUM(sales) AS weekly_sales,
                    SUM(spend) AS weekly_spend
                FROM target_stats
                WHERE client_id = %s
                GROUP BY DATE_TRUNC('week', start_date)
                ORDER BY week
            """, (client_id,))
            
            print(f"{'Week':<12} | {'Orders':<8} | {'Clicks':<8} | {'CVR%':<8} | {'Campaigns':<10} | {'Sales':<12}")
            print("-"*70)
            
            rows = cursor.fetchall()
            cvr_values = []
            for row in rows:
                week, orders, clicks, cvr, camps, sales, spend = row
                cvr = cvr or 0
                cvr_values.append(cvr)
                print(f"{str(week)[:10]:<12} | {orders or 0:<8} | {clicks or 0:<8} | {cvr:<8.2f} | {camps:<10} | ${sales or 0:<11,.0f}")
            
            # Trend Analysis
            print("\n" + "="*70)
            print("TREND ANALYSIS")
            print("="*70)
            
            if len(cvr_values) >= 2:
                first_half = sum(cvr_values[:len(cvr_values)//2]) / (len(cvr_values)//2)
                second_half = sum(cvr_values[len(cvr_values)//2:]) / (len(cvr_values) - len(cvr_values)//2)
                
                change = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
                
                print(f"First Half Avg CVR:  {first_half:.2f}%")
                print(f"Second Half Avg CVR: {second_half:.2f}%")
                print(f"Change: {change:+.1f}%")
                
                if change > 50:
                    print("\n🚨 LARGE CVR SURGE DETECTED - Investigate if this aligns with:")
                    print("   - New product launches")
                    print("   - Harvest campaign rollouts")
                    print("   - Seasonal spikes")
                else:
                    print("\n✓ CVR change appears gradual and normal")


def validate_action_counts():
    """Check why action counts differ between script and dashboard."""
    
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    db = get_db_manager(test_mode=False)
    
    print("\n\n" + "="*70)
    print("ACTION COUNT DISCREPANCY ANALYSIS")
    print("="*70)
    print("Script shows: s2c_uae=182, digi=165, s2c=51")
    print("Dashboard shows: 122, 830, 342")
    print("-"*70)
    
    clients = ['s2c_uae_test', 'digiaansh_test', 's2c_test']
    
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            for client in clients:
                print(f"\n{client}:")
                
                # Total actions in actions_log
                cursor.execute("""
                    SELECT COUNT(*) FROM actions_log WHERE client_id = %s
                """, (client,))
                total_actions = cursor.fetchone()[0]
                
                # Actions in last 30 days
                cursor.execute("""
                    SELECT COUNT(*) FROM actions_log 
                    WHERE client_id = %s 
                      AND action_date >= CURRENT_DATE - INTERVAL '30 days'
                """, (client,))
                recent_30d = cursor.fetchone()[0]
                
                # Actions in last 14 days
                cursor.execute("""
                    SELECT COUNT(*) FROM actions_log 
                    WHERE client_id = %s 
                      AND action_date >= CURRENT_DATE - INTERVAL '14 days'
                """, (client,))
                recent_14d = cursor.fetchone()[0]
                
                # Validated actions (based on validation_status pattern)
                cursor.execute("""
                    SELECT COUNT(*) FROM actions_log 
                    WHERE client_id = %s 
                      AND (COALESCE(validation_status, '') LIKE '%%✓%%'
                           OR COALESCE(validation_status, '') LIKE '%%Validated%%'
                           OR COALESCE(validation_status, '') LIKE '%%Confirmed%%'
                           OR COALESCE(validation_status, '') LIKE '%%Match%%'
                           OR COALESCE(validation_status, '') LIKE '%%Directional%%'
                           OR COALESCE(validation_status, '') LIKE '%%Normalized%%'
                           OR COALESCE(validation_status, '') LIKE '%%Volume%%')
                """, (client,))
                validated_total = cursor.fetchone()[0]
                
                # Check date range of actions
                cursor.execute("""
                    SELECT MIN(action_date), MAX(action_date) FROM actions_log WHERE client_id = %s
                """, (client,))
                min_date, max_date = cursor.fetchone()
                
                print(f"  Total Actions: {total_actions}")
                print(f"  Last 30 Days:  {recent_30d}")
                print(f"  Last 14 Days:  {recent_14d}")
                print(f"  Validated:     {validated_total}")
                print(f"  Date Range:    {min_date} to {max_date}")
    
    print("\n" + "="*70)
    print("HYPOTHESIS")
    print("="*70)
    print("""
The discrepancy likely comes from:
1. Dashboard uses CURRENT_DATE for filtering, script uses latest_data_date
2. Dashboard may count ALL actions (not just period-filtered)
3. Maturity threshold differs (script: action_date <= latest - 17 days)
4. Script filters to validated+mature only, dashboard may show all pending too
""")


if __name__ == "__main__":
    validate_cvr_surge('s2c_test')
    validate_action_counts()
