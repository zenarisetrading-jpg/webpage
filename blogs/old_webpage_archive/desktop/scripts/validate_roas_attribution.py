import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.roas_attribution import get_roas_attribution
from core.db_manager import get_db_manager
from core.postgres_manager import PostgresManager

def validate_attribution_for_client(client_id: str):
    """
    Run validation using the SHARED core logic (Single Source of Truth).
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING CLIENT: {client_id}")
    print(f"{'='*60}")
    
    # Get Decision Impact Value (Dollar) first, as it's an input to the attribution model
    # We use the same logic as the dashboard: query get_action_impact
    db = get_db_manager()
    if not isinstance(db, PostgresManager):
        print("Error: DB is not PostgresManager")
        return

    # Fetch impact data (mimic dashboard default of 30 days)
    print("Fetching decision impact...")
    impact_df = db.get_action_impact(client_id, before_days=14, after_days=14)
    
    decision_value = 0
    if not impact_df.empty:
        # Replicate dashboard logic: Sum of decision_impact
        decision_value = impact_df['decision_impact'].sum()
    
    print(f"Input Decision Value: ${decision_value:,.2f}")
    
    # CALL THE CORE MODULE
    print("Running get_roas_attribution...")
    attribution = get_roas_attribution(client_id, days=30, decision_impact_value=decision_value)
    
    if not attribution:
        print("Failed to get attribution (likely no data or incorrect client ID).")
        return

    # PRINT METRICS
    print("\n--- CORE METRICS ---")
    print(f"Periods: {attribution['periods']['prior_start']} to {attribution['periods']['current_end']}")
    print(f"Prior Spend:   ${attribution['prior_metrics']['spend']:,.2f}")
    print(f"Current Spend: ${attribution['current_metrics']['spend']:,.2f}")
    
    spend_change = (attribution['current_metrics']['spend'] - attribution['prior_metrics']['spend']) / attribution['prior_metrics']['spend']
    print(f"Spend Change:     {spend_change:.1%}")
    
    print(f"Baseline ROAS:    {attribution['baseline_roas']:.2f}")
    print(f"Actual ROAS:      {attribution['actual_roas']:.2f}")
    print(f"Total Change:     {attribution['total_change']:+.2f}")
    
    print("\n--- MARKET DECOMPOSITION ---")
    print(f"Market Impact:    {attribution['market_impact_roas']:+.2f}")
    print(f"  • CPC Impact:   {attribution['cpc_impact']:+.2f}")
    print(f"  • CVR Impact:   {attribution['cvr_impact']:+.2f}")
    print(f"  • AOV Impact:   {attribution['aov_impact']:+.2f}")
    
    print("\n--- DECISION IMPACT ---")
    print(f"Decision Impact:  {attribution['decision_impact_roas']:+.2f}")
    
    print("\n--- OTHER FACTORS (The 'Gap') ---")
    print(f"Scale Effect:     {attribution['scale_effect']:+.2f}")
    print(f"Portfolio Effect: {attribution['portfolio_effect']:+.2f}")
    print(f"Unexplained:      {attribution['unexplained']:+.2f}")
    
    other_total = attribution['scale_effect'] + attribution['portfolio_effect'] + attribution['unexplained']
    print(f"TOTAL OTHER:      {other_total:+.2f}")
    
    # RECONCILIATION
    calc_actual = (attribution['baseline_roas'] + 
                   attribution['market_impact_roas'] + 
                   attribution['decision_impact_roas'] + 
                   other_total)
                   
    print("\n--- RECONCILIATION ---")
    print(f"Calc Actual:      {calc_actual:.2f}")
    print(f"Observed Actual:  {attribution['actual_roas']:.2f}")
    print(f"Match?            {'✓ YES' if abs(calc_actual - attribution['actual_roas']) < 0.01 else '❌ NO'}")

if __name__ == "__main__":
    # Load .env
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
if __name__ == "__main__":
    # Load .env
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    
    # 1. Run Standard Validation (Raw Data)
    print("\n>>> RAW DATA VALIDATION (All Actions) <<<")
    validate_attribution_for_client("s2c_test")
    
    # 2. Run Simulation to Match Dashboard (Validated Actions Only)
    # Dashboard shows Decision Impact +0.57 on Spend $1889.44 => Value approx +1080
    print("\n>>> DASHBOARD SIMULATION (Validated Only) <<<")
    # We cheat slightly by passing the expected value to see if the math holds
    
    client_id = "s2c_test"
    simulated_value = 1080.0
    
    print(f"Simulating Dashboard State for {client_id} with Decision Value: ${simulated_value:,.2f}")
    attribution = get_roas_attribution(client_id, days=30, decision_impact_value=simulated_value)
    
    if attribution:
        print("\n--- SIMULATED METRICS ---")
        print(f"Baseline ROAS:    {attribution['baseline_roas']:.2f}")
        print(f"Actual ROAS:      {attribution['actual_roas']:.2f}")
        
        print("\n--- DECOMPOSITION ---")
        print(f"Market Impact:    {attribution['market_impact_roas']:+.2f}")
        print(f"Decision Impact:  {attribution['decision_impact_roas']:+.2f} (Matches Dashboard +0.57?)")
        
        other_total = attribution['scale_effect'] + attribution['portfolio_effect'] + attribution['unexplained']
        print(f"Total Other:      {other_total:+.2f}")
        print(f"  • Scale:        {attribution['scale_effect']:+.2f}")
        print(f"  • Portfolio:    {attribution['portfolio_effect']:+.2f}")
        print(f"  • Unexplained:  {attribution['unexplained']:+.2f}")
        
        # Check Reconciliation
        calc = attribution['baseline_roas'] + attribution['market_impact_roas'] + attribution['decision_impact_roas'] + other_total
        print(f"\nReconciliation:   {calc:.2f} vs Actual {attribution['actual_roas']:.2f}")
        
        # User's Math Check
        # 2.62 + 1.01 + 0.57 + Other = 3.27
        # Other = 3.27 - 4.20 = -0.93
        print(f"\nTarget 'Other' for Dashboard Match: -0.93")
        print(f"Calculated 'Other': {other_total:+.2f}")
        print(f"Difference: {abs(-0.93 - other_total):.2f}")
