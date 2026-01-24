
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from core.data_loader import SmartMapper

def test_mapper():
    # Simulated headers from a standard Amazon Bulk File
    headers = [
        "Product", "Entity", "Operation", "Campaign Id", "Ad Group Id", "Portfolio Id", 
        "Campaign Name (Informational only)", "Ad Group Name (Informational only)", 
        "Start Date", "End Date", "Targeting Type", "State", "Daily Budget", "SKU", 
        "Ad Group Default Bid", "Bid", "Keyword Text", "Match Type", "Bidding Strategy", 
        "Placement", "Percentage", "Product Targeting Expression"
    ]
    
    df = pd.DataFrame(columns=headers)
    print("Original Columns:", df.columns.tolist())
    
    mapping = SmartMapper.map_columns(df)
    print("\nMapping Result:")
    for k, v in mapping.items():
        print(f"  {k} -> {v}")
        
    if "Campaign" not in mapping:
        print("\n❌ FAILED: 'Campaign' not mapped!")
    else:
        print(f"\n✅ 'Campaign' mapped to: {mapping['Campaign']}")

    if "CampaignId" not in mapping:
        print("❌ FAILED: 'CampaignId' not mapped!")
    else:
        print(f"✅ 'CampaignId' mapped to: {mapping['CampaignId']}")

if __name__ == "__main__":
    test_mapper()
