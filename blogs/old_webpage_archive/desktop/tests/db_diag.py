import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
from core.postgres_manager import PostgresManager

db = PostgresManager()
client_id = "110 actions 30 day window demo_Account"

with db._get_connection() as conn:
    print("--- Date Check ---")
    latest_stats = pd.read_sql("SELECT MAX(start_date) as latest FROM target_stats WHERE client_id = %s", conn, params=(client_id,))
    print(f"Latest target_stats date: {latest_stats['latest'].iloc[0]}")
    
    latest_actions = pd.read_sql("SELECT MAX(action_date) as latest, COUNT(*) FROM actions_log WHERE client_id = %s", conn, params=(client_id,))
    print(f"Latest actions_log date: {latest_actions['latest'].iloc[0]}")
    print(f"Total actions in log: {latest_actions['count'].iloc[0]}")
    
    # Check sample of actions
    sample = pd.read_sql("SELECT action_date, action_type, target_text FROM actions_log WHERE client_id = %s LIMIT 5", conn, params=(client_id,))
    print("\nSample actions:")
    print(sample)
    
    # Check what window_days=30 would produce
    w = 30
    w_minus_1 = w - 1
    w2 = 2 * w - 1
    
    windows = pd.read_sql(f"""
        SELECT 
            MAX(start_date) as latest_date,
            MAX(start_date) - INTERVAL '{w_minus_1} days' as after_start,  
            MAX(start_date) - INTERVAL '{w} days' as before_end,   
            MAX(start_date) - INTERVAL '{w2} days' as before_start 
        FROM target_stats 
        WHERE client_id = %s
    """, conn, params=(client_id,))
    print("\nCalculated Windows for 30D:")
    print(windows)
    
    after_start = windows['after_start'].iloc[0]
    viable_actions = pd.read_sql("""
        SELECT COUNT(*) FROM actions_log 
        WHERE client_id = %s AND DATE(action_date) < %s
    """, conn, params=(client_id, after_start))
    print(f"Actions taken BEFORE after_start ({after_start}): {viable_actions['count'].iloc[0]}")
