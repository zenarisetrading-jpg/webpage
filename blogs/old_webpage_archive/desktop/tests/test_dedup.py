import pandas as pd
from features.bulk_export import generate_bids_bulk

def test_deduplication():
    # Create test data with conflicting bids for same ID
    cols = ["Campaign Name", "Ad Group Name", "Targeting", "Bucket", "Match Type", 
            "Spend", "KeywordId", "New Bid", "Reason", "CampaignId", "AdGroupId"]
    
    data = [
        # Conflict Group 1: KW-123 (Higher spend first)
        ["C1", "A1", "water bottle", "High", "exact", 100.0, "123", 1.50, "Update", "1", "1"],
        ["C1", "A1", "1l water bottle", "High", "exact", 50.0, "123", 1.20, "Update", "1", "1"],
        
        # Conflict Group 2: KW-456 (Highest spend last)
        ["C2", "A2", "gym bottle", "Low", "broad", 20.0, "456", 0.80, "Update", "2", "2"],
        ["C2", "A2", "bottle gym", "Low", "broad", 80.0, "456", 0.90, "Update", "2", "2"],
        
        # No Conflict: KW-789
        ["C3", "A3", "shaker", "High", "exact", 200.0, "789", 2.00, "Update", "3", "3"],
    ]
    
    df = pd.DataFrame(data, columns=cols)
    
    print("--- Input Data (5 rows) ---")
    print(df[["Targeting", "KeywordId", "Spend", "New Bid"]])
    
    # Run Generation
    bulk_df, _ = generate_bids_bulk(df)
    
    print("\n--- Output Bulk File ---")
    print(bulk_df[["Keyword Text", "Keyword Id", "Bid"]])
    
    # Assertions
    assert len(bulk_df) == 3, f"Expected 3 unique rows, got {len(bulk_df)}"
    
    # Check KW-123: Should assume bid from 'water bottle' (Spend 100, Bid 1.50)
    row_123 = bulk_df[bulk_df["Keyword Id"] == "123"].iloc[0]
    assert row_123["Bid"] == 1.50, f"KW-123 Bid Mismatch. Expected 1.50, got {row_123['Bid']}"
    
    # Check KW-456: Should assume bid from 'bottle gym' (Spend 80, Bid 0.90)
    row_456 = bulk_df[bulk_df["Keyword Id"] == "456"].iloc[0]
    assert row_456["Bid"] == 0.90, f"KW-456 Bid Mismatch. Expected 0.90, got {row_456['Bid']}"
    
    print("\n✅ Verification PASSED: Deduplication worked correctly based on MAX SPEND.")

if __name__ == "__main__":
    test_deduplication()
