import sqlite3
import pandas as pd
import numpy as np
import os
import sys
import re

# Mock streamlit session state and hub
class MockHub:
    def __init__(self, adv_df, cat_df):
        self.adv_df = adv_df
        self.cat_df = cat_df
    def is_loaded(self, key):
        if key == 'advertised_product_report': return self.adv_df is not None
        if key == 'category_mapping': return self.cat_df is not None
        return False
    def get_data(self, key):
        if key == 'advertised_product_report': return self.adv_df
        if key == 'category_mapping': return self.cat_df
        return None

def normalize(text):
    if pd.isna(text): return ""
    return re.sub(r'[^a-z0-9]', '', str(text).lower().strip())

def diagnose(db_path, client_id):
    conn = sqlite3.connect(db_path)
    
    # 1. Load Stats as the app does
    print(f"--- Diagnosing PerformanceSnapshot for {client_id} ---")
    db_data = pd.read_sql(f"""
        SELECT 
            start_date as Date,
            campaign_name as 'Campaign Name',
            ad_group_name as 'Ad Group Name',
            target_text as Targeting,
            match_type as 'Match Type',
            spend as Spend,
            sales as Sales,
            orders as Orders,
            clicks as Clicks,
            impressions as Impressions
        FROM target_stats 
        WHERE client_id = '{client_id}'
    """, conn)
    
    # 2. Load Reports from DB (as DataHub.load_from_database does)
    adv_report = pd.read_sql(f"SELECT campaign_name as 'Campaign Name', ad_group_name as 'Ad Group Name', sku as SKU, asin as ASIN FROM advertised_product_cache WHERE client_id = '{client_id}'", conn)
    cat_map = pd.read_sql(f"SELECT sku as SKU, category as Category, sub_category as 'Sub-Category' FROM category_mappings WHERE client_id = '{client_id}'", conn)
    
    hub = MockHub(adv_report, cat_map)
    data = db_data.copy()
    
    # --- REPLICATE RENDER_UI ENRICHMENT (lines 51-210) ---
    print(f"Step 1: SKU Merge (Report rows: {len(adv_report)})")
    if hub.is_loaded('advertised_product_report'):
        agg_dict = {}
        if 'SKU' in adv_report.columns:
            agg_dict['SKU'] = lambda x: ', '.join(x.dropna().astype(str).unique())
        
        if 'Campaign Name' in adv_report.columns and 'Ad Group Name' in adv_report.columns:
            data['Camp_Norm'] = data['Campaign Name'].astype(str).str.strip().str.lower()
            data['AG_Norm'] = data['Ad Group Name'].astype(str).str.strip().str.lower()
            
            adv_report_c = adv_report.copy()
            adv_report_c['Camp_Norm'] = adv_report_c['Campaign Name'].astype(str).str.strip().str.lower()
            adv_report_c['AG_Norm'] = adv_report_c['Ad Group Name'].astype(str).str.strip().str.lower()
            
            sku_lookup = adv_report_c.groupby(['Camp_Norm', 'AG_Norm']).agg(agg_dict).reset_index()
            sku_lookup.columns = ['Camp_Norm', 'AG_Norm', 'SKU_advertised']
            
            data = data.merge(sku_lookup, on=['Camp_Norm', 'AG_Norm'], how='left')
            data.drop(columns=['Camp_Norm', 'AG_Norm'], inplace=True, errors='ignore')

    sku_fill = data['SKU_advertised'].notna().mean() if 'SKU_advertised' in data.columns else 0
    print(f"SKU Fill Rate: {sku_fill:.1%}")
    
    print(f"Step 2: Category Merge (Mapping rows: {len(cat_map)})")
    if hub.is_loaded('category_mapping'):
        product_id_col = None
        if 'SKU_advertised' in data.columns and data['SKU_advertised'].notna().sum() > 0:
            product_id_col = 'SKU_advertised'
        elif 'ASIN_advertised' in data.columns and data['ASIN_advertised'].notna().sum() > 0:
            product_id_col = 'ASIN_advertised'
        
        print(f"Selected Product ID Col: {product_id_col}")
        
        if product_id_col:
            # Find product ID column in category map
            cat_id_candidates = [c for c in cat_map.columns if any(s in c.lower() for s in ['sku', 'asin', 'product'])]
            cat_id_col = cat_id_candidates[0] if cat_id_candidates else None
            
            if cat_id_col:
                print(f"Using Category Map Col: {cat_id_col}")
                
                data['ID_List'] = data[product_id_col].astype(str).str.split(',')
                exploded = data.explode('ID_List')
                exploded['ID_Clean'] = exploded['ID_List'].apply(normalize)
                
                cat_map_c = cat_map.copy()
                cat_map_c['ID_Clean'] = cat_map_c[cat_id_col].apply(normalize)
                
                print(f"Sample data['ID_Clean']: {exploded['ID_Clean'].dropna().head(5).tolist()}")
                print(f"Sample cat_map['ID_Clean']: {cat_map_c['ID_Clean'].dropna().head(5).tolist()}")
                
                merged = exploded.merge(cat_map_c[['ID_Clean', 'Category']], on='ID_Clean', how='left')
                match_count = merged['Category'].notna().sum()
                print(f"Internal Merge - Successful matches: {match_count}")
                
                first_match = merged.groupby(level=0).first()
                data['Category'] = first_match['Category']
    
    cat_fill = data['Category'].notna().mean() if 'Category' in data.columns else 0
    print(f"Category Fill Rate: {cat_fill:.1%}")
    
    # 3. Simulate performance metrics for the breakdown
    if 'Category' in data.columns:
        # Group by Category as the app does
        agg_cols = {'Spend': 'sum', 'Sales': 'sum', 'Orders': 'sum'}
        grouped = data.groupby('Category').agg(agg_cols).reset_index()
        print("\n--- GROUPED BREAKDOWN ---")
        print(grouped)
        if grouped.empty:
            print("WARNING: Grouped table is empty!")
    
    conn.close()

if __name__ == "__main__":
    db = "/Users/zayaanyousuf/Documents/Amazon PPC/saddle/saddle/data/ppc_test.db"
    diagnose(db, "test1")
