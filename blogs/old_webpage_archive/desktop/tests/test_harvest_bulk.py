import pandas as pd
from features.bulk_export import generate_harvest_bulk, EXPORT_COLUMNS

def test_harvest_bulk_logic():
    # Input Data: 2 Terms for same AG, different SKUs
    # Expectation: 
    # - 1 Campaign Row (Manual)
    # - 1 Ad Group Row (Avg Bid)
    # - 1 Product Ad Row (Winner SKU by Sales)
    # - 2 Keyword Rows
    
    data = [
        # Term 1: SKU A (Low Sales)
        {
            "Customer Search Term": "water bottle 1l",
            "Campaign Name": "Harvest Camp 1",
            "Ad Group Name": "Harvest AG 1",
            "SKU_advertised": "SKU-A",
            "Sales": 100.0,
            "Suggested Bid": 1.50
        },
        # Term 2: SKU B (High Sales - WINNER)
        {
            "Customer Search Term": "large water bottle",
            "Campaign Name": "Harvest Camp 1",
            "Ad Group Name": "Harvest AG 1",
            "SKU_advertised": "SKU-B",
            "Sales": 500.0,
            "Suggested Bid": 2.50
        }
    ]
    
    df = pd.DataFrame(data)
    
    print("--- Input Harvest Data ---")
    print(df)
    
    bulk_df = generate_harvest_bulk(df)
    
    print("\n--- Output Bulk File ---")
    print(bulk_df[["Entity", "Campaign Name", "Ad Group Name", "SKU", "Targeting Type", "Ad Group Default Bid", "Keyword Text"]])
    
    # Assertions
    
    # 1. Structure Count
    # Expected: 1 Camp + 1 AG + 1 PA + 2 KW = 5 rows
    assert len(bulk_df) == 5, f"Expected 5 rows, got {len(bulk_df)}"
    
    # 2. Campaign Row
    camp_row = bulk_df[bulk_df["Entity"] == "Campaign"].iloc[0]
    assert camp_row["Targeting Type"] == "Manual", "Campaign Targeting Type must be Manual"
    
    # 3. Ad Group Default Bid
    # Avg of 1.50 and 2.50 = 2.00
    ag_row = bulk_df[bulk_df["Entity"] == "Ad Group"].iloc[0]
    assert ag_row["Ad Group Default Bid"] == 2.00, f"Expected AG Default Bid 2.00, got {ag_row['Ad Group Default Bid']}"
    
    # 4. Winner SKU
    # SKU-B has 500 sales vs SKU-A 100 sales -> Winner is SKU-B
    pa_rows = bulk_df[bulk_df["Entity"] == "Product Ad"]
    assert len(pa_rows) == 1, "Should have exactly 1 Product Ad row"
    assert pa_rows.iloc[0]["SKU"] == "SKU-B", f"Expected Winner SKU-B, got {pa_rows.iloc[0]['SKU']}"
    
    # 5. Keyword Rows
    kw_rows = bulk_df[bulk_df["Entity"] == "Keyword"]
    assert len(kw_rows) == 2, "Should have 2 Keyword rows"
    # Check default bid presence on KW rows
    assert kw_rows.iloc[0]["Ad Group Default Bid"] == 2.00
    
    print("\n✅ Verification PASSED: Hierarchical Harvest Logic works.")

if __name__ == "__main__":
    test_harvest_bulk_logic()
