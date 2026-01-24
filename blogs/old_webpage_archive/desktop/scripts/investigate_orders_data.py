"""
Phase 1: CVR Data Investigation
Find where orders data lives and assess data quality.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta, date

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from core.db_manager import get_db_manager
from core.postgres_manager import PostgresManager


def investigate_orders_data():
    """Find where orders data lives in the database."""
    
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    db = get_db_manager(test_mode=False)
    if not isinstance(db, PostgresManager):
        print("ERROR: Requires PostgresManager")
        return
    
    print("="*70)
    print("PHASE 1: CVR DATA INVESTIGATION")
    print("="*70)
    
    clients = ['digiaansh_test', 's2c_test', 's2c_uae_test']
    
    # Step 1: Check all tables for orders column
    print("\n--- STEP 1: Checking Tables for Orders Column ---\n")
    
    tables_to_check = [
        'target_stats',
        'weekly_stats', 
        'actions_log',
        'account_health_metrics'
    ]
    
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            # Get all table names first
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            all_tables = [row[0] for row in cursor.fetchall()]
            print(f"All tables in database: {all_tables}\n")
            
            # Check each table for 'orders' column
            for table in all_tables:
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """, (table,))
                columns = [row[0] for row in cursor.fetchall()]
                
                has_orders = 'orders' in columns
                has_sales = 'sales' in columns
                has_clicks = 'clicks' in columns
                
                if has_orders or has_sales:
                    print(f"Table: {table}")
                    print(f"  Has 'orders': {'✓' if has_orders else '✗'}")
                    print(f"  Has 'sales': {'✓' if has_sales else '✗'}")
                    print(f"  Has 'clicks': {'✓' if has_clicks else '✗'}")
                    
                    # If orders column exists, check if it has data
                    if has_orders:
                        try:
                            cursor.execute(f"""
                                SELECT 
                                    COUNT(*) AS total_rows,
                                    SUM(CASE WHEN orders > 0 THEN 1 ELSE 0 END) AS rows_with_orders,
                                    SUM(orders) AS total_orders
                                FROM {table}
                            """)
                            row = cursor.fetchone()
                            print(f"  Total Rows: {row[0]}")
                            print(f"  Rows with Orders > 0: {row[1]}")
                            print(f"  Total Orders: {row[2]}")
                        except Exception as e:
                            print(f"  Error querying: {e}")
                    print()
    
    # Step 2: Check target_stats specifically (our main table)
    print("\n--- STEP 2: target_stats Orders Analysis ---\n")
    
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            # Check schema
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'target_stats'
                ORDER BY ordinal_position
            """)
            print("target_stats schema:")
            for col_name, dtype in cursor.fetchall():
                print(f"  {col_name}: {dtype}")
            
            print()
            
            # Check sample data for orders
            cursor.execute("""
                SELECT 
                    client_id,
                    COUNT(*) AS rows,
                    SUM(COALESCE(orders, 0)) AS total_orders,
                    SUM(sales) AS total_sales,
                    SUM(clicks) AS total_clicks,
                    -- Derived CVR if orders exist
                    SUM(COALESCE(orders, 0)) / NULLIF(SUM(clicks), 0) * 100 AS cvr_pct,
                    -- Derived AOV if orders exist
                    SUM(sales) / NULLIF(SUM(COALESCE(orders, 0)), 0) AS aov
                FROM target_stats
                WHERE client_id IN ('digiaansh_test', 's2c_test', 's2c_uae_test')
                GROUP BY client_id
            """)
            
            print(f"{'Client':<20} | {'Rows':<10} | {'Orders':<12} | {'Sales':<15} | {'CVR%':<8} | {'AOV':<10}")
            print("-"*80)
            
            for row in cursor.fetchall():
                client, rows, orders, sales, clicks, cvr, aov = row
                orders = orders or 0
                cvr = cvr or 0
                aov = aov or 0
                print(f"{client:<20} | {rows:<10} | {orders:<12.0f} | ${sales or 0:<14,.2f} | {cvr:<8.2f} | ${aov:<10.2f}")
    
    # Step 3: Check if orders might be stored differently (e.g., as 'conversions')
    print("\n--- STEP 3: Alternative Order Column Names ---\n")
    
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT column_name, table_name
                FROM information_schema.columns 
                WHERE table_schema = 'public'
                  AND (column_name LIKE '%order%' 
                       OR column_name LIKE '%conversion%'
                       OR column_name LIKE '%purchase%'
                       OR column_name LIKE '%unit%')
            """)
            results = cursor.fetchall()
            
            if results:
                print("Found potential order-related columns:")
                for col, table in results:
                    print(f"  {table}.{col}")
            else:
                print("No alternative order columns found.")
    
    # Step 4: Sample actual data to see what's there
    print("\n--- STEP 4: Sample Data from target_stats ---\n")
    
    with db._get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    client_id, 
                    start_date, 
                    campaign_name,
                    spend, 
                    sales, 
                    clicks,
                    orders
                FROM target_stats
                WHERE client_id = 's2c_test'
                  AND spend > 0
                ORDER BY start_date DESC
                LIMIT 5
            """)
            
            print("Sample rows from s2c_test:")
            for row in cursor.fetchall():
                print(f"  Date: {row[1]}, Campaign: {row[2][:30]}...")
                print(f"    Spend: ${row[3]:.2f}, Sales: ${row[4]:.2f}, Clicks: {row[5]}, Orders: {row[6]}")
                print()


if __name__ == "__main__":
    investigate_orders_data()
