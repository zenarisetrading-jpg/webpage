import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

def normalize(text):
    if pd.isna(text): return ""
    import re
    return re.sub(r'[^a-z0-9]', '', str(text).lower().strip())

def run_simulation(db_path, client_id):
    conn = sqlite3.connect(db_path)
    
    # 1. Load Stats
    print(f"Loading target_stats for {client_id}...")
    stats_query = f"SELECT campaign_name, ad_group_name, spend, sales, orders FROM target_stats WHERE client_id = '{client_id}' AND start_date >= '2025-11-09'"
    df_stats = pd.read_sql(stats_query, conn)
    
    # 2. Load APR Cache
    print("Loading advertised_product_cache...")
    apr_query = f"SELECT campaign_name, ad_group_name, sku, asin FROM advertised_product_cache WHERE client_id = '{client_id}'"
    df_apr = pd.read_sql(apr_query, conn)
    
    # 3. Load Category Mappings
    print("Loading category_mappings...")
    cat_query = f"SELECT sku, category FROM category_mappings WHERE client_id = '{client_id}'"
    df_cat = pd.read_sql(cat_query, conn)
    
    # --- ENRICHMENT LOGIC ---
    
    # Normalize APR for lookup
    df_apr['sku_norm'] = df_apr['sku'].apply(normalize)
    df_apr['asin_norm'] = df_apr['asin'].apply(normalize)
    df_apr['_camp_norm'] = df_apr['campaign_name'].apply(normalize)
    df_apr['_ag_norm'] = df_apr['ad_group_name'].apply(normalize)
    
    # Normalize Categories
    df_cat['id_norm'] = df_cat['sku'].apply(normalize) # Mapping table uses 'sku' column for both SKU and ASIN
    cat_lookup = df_cat.groupby('id_norm')['category'].first()
    
    # Map APR to Categories (Try SKU then ASIN)
    df_apr['Category'] = df_apr['sku_norm'].map(cat_lookup)
    df_apr['Category'] = df_apr['Category'].fillna(df_apr['asin_norm'].map(cat_lookup))
    
    # Aggregate APR to Campaign/AdGroup level via normalized keys
    apr_lookup = df_apr.groupby(['_camp_norm', '_ag_norm'])['Category'].first().reset_index()
    
    # Merge Stats with Categories via normalized keys
    df_stats['_camp_norm'] = df_stats['campaign_name'].apply(normalize)
    df_stats['_ag_norm'] = df_stats['ad_group_name'].apply(normalize)
    
    df_stats = df_stats.merge(apr_lookup, on=['_camp_norm', '_ag_norm'], how='left')
    
    # -----------------------------------------------------------------
    # DIAGNOSIS: Why is Category 'Other'?
    # -----------------------------------------------------------------
    # Check if SKU was even found for these campaigns
    sku_lookup = df_apr.groupby(['_camp_norm', '_ag_norm'])['sku'].first().reset_index()
    df_stats = df_stats.merge(sku_lookup, on=['_camp_norm', '_ag_norm'], how='left', suffixes=('', '_check'))
    
    def diagnose(row):
        if not pd.isna(row['Category']) and row['Category'] != 'Other':
            return row['Category']
        if pd.isna(row['sku']):
            return "[Gap] Missing SKU Link"
        return "[Gap] SKU exists but no Category Mapping"

    df_stats['Diagnosis'] = df_stats.apply(diagnose, axis=1)
    df_stats['Category'] = df_stats['Category'].fillna('Other')
    
    # Group by Diagnosis for clarity
    diag_result = df_stats.groupby('Diagnosis').agg({
        'spend': 'sum',
        'sales': 'sum'
    }).reset_index().sort_values('spend', ascending=False)
    
    print("\n--- MAPPING DIAGNOSIS ---")
    print(diag_result.to_markdown(index=False))
    
    # DEBUG: Show top campaigns in 'Other'
    print("\n--- TOP CAMPAIGNS IN 'OTHER' (potential misses) ---")
    other_df = df_stats[df_stats['Category'] == 'Other'].groupby('campaign_name').agg({'spend':'sum'}).sort_values('spend', ascending=False).head(10)
    print(other_df)

    conn.close()

if __name__ == "__main__":
    db = "/Users/zayaanyousuf/Documents/Amazon PPC/saddle/saddle/data/ppc_test.db"
    run_simulation(db, "test1")
