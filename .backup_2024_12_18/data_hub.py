"""
Unified Data Hub

Central data management for all modules.
Upload once, use everywhere.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Optional, Tuple
from core.data_loader import load_uploaded_file, SmartMapper, safe_numeric
from core.db_manager import get_db_manager
from core.mapping_engine import MappingEngine
from api.rainforest_client import ASINCache

class DataHub:
    """Central data management system."""
    
    def __init__(self):
        """Initialize data hub with session state."""
        if 'unified_data' not in st.session_state:
            st.session_state.unified_data = {
                'search_term_report': None,
                'advertised_product_report': None,
                'bulk_id_mapping': None,
                'category_mapping': None,
                'enriched_data': None,  # Merged/enriched dataset
                'upload_status': {
                    'search_term_report': False,
                    'advertised_product_report': False,
                    'bulk_id_mapping': False,
                    'category_mapping': False
                },
                'upload_timestamps': {
                    'search_term_report': None,
                    'advertised_product_report': None,
                    'bulk_id_mapping': None,
                    'category_mapping': None
                }
            }
    
    def get_data(self, data_type: str) -> Optional[pd.DataFrame]:
        """Get specific dataset."""
        return st.session_state.unified_data.get(data_type)
    
    def get_enriched_data(self) -> Optional[pd.DataFrame]:
        """Get the fully merged/enriched dataset."""
        return st.session_state.unified_data.get('enriched_data')
    
    def is_loaded(self, data_type: str) -> bool:
        """Check if a specific dataset is loaded."""
        return st.session_state.unified_data['upload_status'].get(data_type, False)
    
    def get_upload_status(self) -> Dict[str, bool]:
        """Get upload status for all datasets."""
        return st.session_state.unified_data['upload_status']
    
    def upload_search_term_report(self, uploaded_file) -> Tuple[bool, str]:
        """Upload and validate search term report."""
        df = load_uploaded_file(uploaded_file)
        if df is None:
            return False, "Failed to load file"
        
        # Map columns
        col_map = SmartMapper.map_columns(df)
        
        # Invert map for renaming (Found -> Standard)
        rename_map = {v: k for k, v in col_map.items()}
        df_renamed = df.rename(columns=rename_map)

        # -----------------------------------------------------------
        # REFINED MATCH TYPE LOGIC (Fix for "OTHER" buckets)
        # -----------------------------------------------------------
        # Add helper to infer match type from expression or targeting
        def infer_mt(row):
            curr = str(row.get('Match Type', '')).upper()
            
            # 1. Try TargetingExpression first (most accurate for PT)
            expr = str(row.get('TargetingExpression', '')).lower()
            if not expr or expr == 'nan':
                # 2. Fallback to Targeting column
                expr = str(row.get('Targeting', '')).lower()
            
            # Trust explicit strong types
            if curr in ['EXACT', 'BROAD', 'PHRASE']:
                return curr
            
            # Infer from Expression/Targeting
            if 'asin=' in expr or (len(expr) == 10 and expr.startswith('b0')):
                return 'PT'
            if 'category=' in expr:
                return 'CATEGORY'
            if any(x in expr for x in ['close-match', 'loose-match', 'substitutes', 'complements', '*']):
                return 'AUTO'
            
            return curr if curr and curr != 'NAN' else '-'

        df_renamed['Match Type'] = df_renamed.apply(infer_mt, axis=1)
        # -----------------------------------------------------------
        
        # Validate critical columns
        missing_critical = []
        for crit in ["Spend", "Clicks", "Sales", "Orders"]:
             if crit not in col_map:
                 missing_critical.append(crit)
        
        if missing_critical:
            # st.toast(f"⚠️ Warning: Missing data for {', '.join(missing_critical)}", icon="⚠️")
            st.info(f"Could not find columns for: {', '.join(missing_critical)}. Please checks report headers.")
        
        # CLEANUP: Enforce numeric types
        numeric_cols = ["Spend", "Sales", "Clicks", "Impressions", "Orders", "CPC", "RoAS", "ACOS"]
        for col in numeric_cols:
            if col in df_renamed.columns:
                df_renamed[col] = safe_numeric(df_renamed[col])
        
        # Store
        st.session_state.unified_data['search_term_report'] = df_renamed
        st.session_state.unified_data['upload_status']['search_term_report'] = True
        
        # Timestamp tracking (defensive check for existing sessions)
        if 'upload_timestamps' not in st.session_state.unified_data:
            st.session_state.unified_data['upload_timestamps'] = {}
        st.session_state.unified_data['upload_timestamps']['search_term_report'] = datetime.now()
        
        # Trigger enrichment
        self._enrich_data()
        
        # SAVE TO DB (Persistence)
        try:
            db = get_db_manager(st.session_state.get('test_mode', False))
            
            # Prepare records
            records = []
            
            # Determine date column
            date_col = None
            for col in ['Date', 'Start Date', 'date']:
                if col in df_renamed.columns:
                    date_col = col
                    break
            
            # Determine key columns
            camp_col = next((c for c in ['Campaign Name', 'campaign_name'] if c in df_renamed.columns), None)
            ag_col = next((c for c in ['Ad Group Name', 'ad_group_name'] if c in df_renamed.columns), None)
            target_col = next((c for c in ['Targeting', 'Customer Search Term'] if c in df_renamed.columns), None)
            mt_col = 'Match Type' if 'Match Type' in df_renamed.columns else None
            
            # Save to DB using the NEW signature (df, client_id, start_date)
            client_id = st.session_state.get('active_account_id')
            if not client_id:
                st.error("❌ No Active Account Selected! Please select an account in the sidebar.")
                return 0
            
            # DEBUG: Show which account we're saving to
            # st.toast(f"💾 Saving to account: {client_id}", icon="💾")
            
            # Get the date from the first row
            start_date = None
            if date_col and not df_renamed.empty:
                first_date = pd.to_datetime(df_renamed[date_col].iloc[0], errors='coerce')
                if pd.notna(first_date):
                    start_date = first_date.date()
            
            saved_count = db.save_target_stats_batch(df_renamed, client_id, start_date)
            
            # Store client_id for Account Overview
            if 'last_stats_save' not in st.session_state:
                st.session_state.last_stats_save = {}
            st.session_state.last_stats_save['client_id'] = client_id
            st.session_state.last_stats_save['start_date'] = start_date
            
            return True, f"Loaded & Saved {saved_count:,} rows to Account '{client_id}' ({len(col_map)} cols mapped)"
            
        except Exception as e:
            # Fallback to just memory if DB fails, but warn
            st.warning(f"Data saved to memory but DB save failed: {e}")
            return True, f"Loaded {len(df_renamed):,} rows (Mem only), {len(col_map)} cols mapped"
    
    def _populate_asin_cache(self, df: pd.DataFrame):
        """Populate ASIN cache with user's own products to prevent API lookups."""
        if df is None:
            return
            
        try:
            cache = ASINCache()
            
            # Identify columns
            asin_col = 'ASIN' if 'ASIN' in df.columns else None
            sku_col = 'SKU' if 'SKU' in df.columns else None
            title_col = 'Product Name' if 'Product Name' in df.columns else None
            
            # If no ASIN col, try to use SKU if it looks like ASIN
            target_col = asin_col
            if not target_col and sku_col:
                target_col = sku_col
                
            if not target_col:
                return

            count = 0
            for _, row in df.iterrows():
                val = str(row[target_col]).strip().upper()
                
                # Basic ASIN validation (starts with B, 10 chars)
                if len(val) == 10 and val.startswith('B'):
                    asin = val
                else:
                    continue
                    
                # Create a "dummy" but valid cache entry
                # This makes the ASIN Mapper think we already fetched it
                # We flag it as 'YOUR_PRODUCT' via brand/seller if needed, 
                # but simply having it in cache prevents API call.
                
                cache_data = {
                    'asin': asin,
                    'title': str(row[title_col]) if title_col and pd.notna(row[title_col]) else f"Your Product ({row.get(sku_col, 'Unknown SKU')})",
                    'brand': 'Your Brand', 
                    'seller': 'Your Seller ID',
                    'price': None,
                    'currency': 'AED',
                    'rating': None,
                    'reviews_count': None,
                    'category': 'Advertised Product',
                    'availability': 'In Stock',
                    'product_url': f"https://www.amazon.ae/dp/{asin}",
                    'status': 'success',
                    'is_own_product': True # Custom flag we can check
                }
                
                # Check if exists first to avoid overwriting rich API data with dummy data
                # Only set if NOT exists
                if not cache.get(asin):
                    cache.set(asin, 'AE', cache_data)
                    count += 1
            
            cache.close()
            if count > 0:
                pass # st.toast(f"Cached {count} brand ASINs to save API logic", icon="🛡️")
                
        except Exception as e:
            print(f"Failed to populate ASIN cache: {e}")

    def upload_advertised_product_report(self, uploaded_file) -> Tuple[bool, str]:
        """Upload and process advertised product report."""
        df = load_uploaded_file(uploaded_file)
        if df is None:
            return False, "Failed to load file"
        
        # Map columns
        col_map = SmartMapper.map_columns(df)
        
        # Look for ASIN and SKU columns
        if 'ASIN' not in col_map and 'SKU' not in col_map:
            return False, "No ASIN or SKU column found"
        
        # Rename columns
        df_renamed = df.rename(columns={v: k for k, v in col_map.items()})
        
        # Store
        st.session_state.unified_data['advertised_product_report'] = df_renamed
        st.session_state.unified_data['upload_status']['advertised_product_report'] = True
        
        if 'upload_timestamps' not in st.session_state.unified_data:
            st.session_state.unified_data['upload_timestamps'] = {}
        st.session_state.unified_data['upload_timestamps']['advertised_product_report'] = datetime.now()
        
        # 1. Populate ASIN Cache (Shielding)
        self._populate_asin_cache(df_renamed)
        
        # 2. Persist to DB
        try:
             client_id = st.session_state.get('active_account_id')
             if client_id:
                 db = get_db_manager(st.session_state.get('test_mode', False))
                 db.save_advertised_product_map(df_renamed, client_id)
        except Exception as e:
             st.warning(f"Could not persist to DB: {e}")

        # Trigger enrichment
        self._enrich_data()
        
        # Extract unique ASINs/SKUs
        asins = []
        if 'ASIN' in df_renamed.columns:
            asins.extend(df_renamed['ASIN'].dropna().unique().tolist())
        
        return True, f"Loaded {len(df_renamed):,} products, {len(asins)} unique ASINs"

    def upload_bulk_id_mapping(self, uploaded_file) -> Tuple[bool, str]:
        """Upload bulk upload ID mapping file."""
        df = load_uploaded_file(uploaded_file)
        if df is None:
            return False, "Failed to load file"
        
        # SmartMapper
        col_map = SmartMapper.map_columns(df)
        df_renamed = df.rename(columns={v: k for k, v in col_map.items()})
        
        # Validation: Need minimally Campaign Name and CampaignId
        required_cols = ['Campaign Name', 'CampaignId'] 
        found_cols = [c for c in required_cols if c in df_renamed.columns]
        
        if not all(col in df_renamed.columns for col in required_cols):
             # Fallback: Try to be lenient
             pass

        # Fallback Check: If 'Campaign Name' is still missing, try manual recovery
        if 'Campaign Name' not in df_renamed.columns:
            # Look for any column containing 'Campaign' and 'Name'
            candidates = [c for c in df.columns if 'Campaign' in c and 'Name' in c]
            if candidates:
                df_renamed['Campaign Name'] = df[candidates[0]]
            else:
                 # Last resort: Look for just 'Campaign'
                 candidates_simple = [c for c in df.columns if 'Campaign' in c and not 'ID' in c and not 'Id' in c]
                 if candidates_simple:
                    df_renamed['Campaign Name'] = df[candidates_simple[0]]

        # Check if we found decent content
        if 'CampaignId' not in df_renamed.columns:
            return False, f"Could not find 'Campaign ID' column. Found: {list(df_renamed.columns)}"

        # Store
        st.session_state.unified_data['bulk_id_mapping'] = df_renamed
        st.session_state.unified_data['upload_status']['bulk_id_mapping'] = True
        
        if 'upload_timestamps' not in st.session_state.unified_data:
            st.session_state.unified_data['upload_timestamps'] = {}
        st.session_state.unified_data['upload_timestamps']['bulk_id_mapping'] = datetime.now()
        
        # PERSIST TO DB
        try:
             client_id = st.session_state.get('active_account_id')
             if client_id:
                 db = get_db_manager(st.session_state.get('test_mode', False))
                 db.save_bulk_mapping(df_renamed, client_id)
        except Exception as e:
             st.warning(f"Could not persist to DB: {e}")
        
        # Trigger enrichment
        self._enrich_data()
        
        campaigns = df_renamed['Campaign Name'].nunique() if 'Campaign Name' in df_renamed.columns else 0
        adgroups = df_renamed['Ad Group Name'].nunique() if 'Ad Group Name' in df_renamed.columns else 0
        campaign_col_status = "Found" if 'Campaign Name' in df_renamed.columns else "MISSING"
        
        return True, f"Loaded {len(df_renamed):,} rows. Campaigns: {campaigns} ({campaign_col_status}), AdGroups: {adgroups}"
    
    def upload_category_mapping(self, uploaded_file) -> Tuple[bool, str]:
        """Upload internal category/SKU mapping."""
        df = load_uploaded_file(uploaded_file)
        if df is None:
            return False, "Failed to load file"
        
        # Flexible - just store whatever they upload
        # Expected: SKU, Category, Subcategory columns
        
        st.session_state.unified_data['category_mapping'] = df
        st.session_state.unified_data['upload_status']['category_mapping'] = True
        
        if 'upload_timestamps' not in st.session_state.unified_data:
            st.session_state.unified_data['upload_timestamps'] = {}
        st.session_state.unified_data['upload_timestamps']['category_mapping'] = datetime.now()
        
        # 1. Populate ASIN Cache (if SKUs are ASINs)
        self._populate_asin_cache(df)
        
        # PERSIST TO DB
        try:
             client_id = st.session_state.get('active_account_id')
             if client_id:
                 db = get_db_manager(st.session_state.get('test_mode', False))
                 db.save_category_mapping(df, client_id)
        except Exception as e:
             st.warning(f"Could not persist to DB: {e}")
        
        # Trigger enrichment
        self._enrich_data()
        
        skus = df.iloc[:, 0].nunique() if len(df.columns) > 0 else 0
        
        return True, f"Loaded {len(df):,} rows, {skus} unique SKUs"
    
    def _enrich_data(self):
        """Merge additional datasets into search term report using MappingEngine."""
        st_report = self.get_data('search_term_report')
        
        if st_report is None:
            return
            
        enriched = st_report.copy()
        
        # =============================================
        # 1. SKU from Advertised Product Report
        # =============================================
        apr = self.get_data('advertised_product_report')
        if apr is not None:
            enriched, sku_stats = MappingEngine.map_sku_from_apr(enriched, apr)
            # Show stats in UI
            if sku_stats['matched'] > 0:
                pass # st.toast(f"✅ SKU Mapping: {sku_stats['matched']}/{sku_stats['total']} matched", icon="📦")
        
        # =============================================
        # 2. IDs from Bulk File
        # =============================================
        bulk = self.get_data('bulk_id_mapping')
        if bulk is not None:
            enriched, id_stats = MappingEngine.map_ids_from_bulk(enriched, bulk)
            # Show stats in UI
            pass # st.toast(f"🔗 ID Mapping: Campaign={id_stats['campaign_id_matched']}, KW={id_stats['keyword_id_matched']}, PT={id_stats['targeting_id_matched']}", icon="🆔")
        
        # =============================================
        # 3. Category from Category Mapping
        # =============================================
        category_map = self.get_data('category_mapping')
        if category_map is not None:
            enriched, cat_stats = MappingEngine.map_category(enriched, category_map)
            # Show stats in UI
            if cat_stats['matched'] > 0:
                pass # st.toast(f"📊 Category Mapping: {cat_stats['matched']}/{cat_stats['total']} matched", icon="📁")
        
        # Store enriched data
        st.session_state.unified_data['enriched_data'] = enriched
    
    def clear_all(self):
        """Clear all uploaded data."""
        st.session_state.unified_data = {
            'search_term_report': None,
            'advertised_product_report': None,
            'bulk_id_mapping': None,
            'category_mapping': None,
            'enriched_data': None,
            'upload_status': {
                'search_term_report': False,
                'advertised_product_report': False,
                'bulk_id_mapping': False,
                'category_mapping': False
            }
        }
    
    def load_from_database(self, account_id: str) -> bool:
        """Load account's RECENT data (last 4 weeks) from database into session state."""
        try:
            db = get_db_manager(st.session_state.get('test_mode', False))
            
            # --- 1. Load TARGET STATS (Existing Logic) ---
            # Get last 4 weeks of data for accurate monthly baseline
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT start_date FROM target_stats 
                    WHERE client_id = ? 
                    ORDER BY start_date DESC 
                    LIMIT 4
                ''', (account_id,))
                recent_dates = [row[0] for row in cursor.fetchall()]
            
            if not recent_dates:
                self.clear_all()
                return False
            
            df = db.get_target_stats_by_account(account_id, limit=100000)
            df = df[df['start_date'].isin(recent_dates)].copy()
            
            if df.empty:
                self.clear_all()
                return False
            
            column_mapping = {
                'campaign_name': 'Campaign Name',
                'ad_group_name': 'Ad Group Name',
                'target_text': 'Customer Search Term',
                'match_type': 'Match Type',
                'spend': 'Spend',
                'sales': 'Sales',
                'orders': 'Orders',
                'clicks': 'Clicks',
                'impressions': 'Impressions',
                'start_date': 'Date'
            }
            
            df_renamed = df.rename(columns=column_mapping)
            
            # CRITICAL: Also add 'Targeting' column (needed by mapping engine)
            if 'Customer Search Term' in df_renamed.columns:
                df_renamed['Targeting'] = df_renamed['Customer Search Term']
            
            st.session_state.unified_data['search_term_report'] = df_renamed
            st.session_state.unified_data['upload_status']['search_term_report'] = True
            st.session_state.unified_data['upload_timestamps']['search_term_report'] = datetime.now()
            
            # --- 2. Load BULK ID MAPPING (FIRST - before other mappings) ---
            bulk_map = db.get_bulk_mapping(account_id)
            if not bulk_map.empty:
                 st.session_state.unified_data['bulk_id_mapping'] = bulk_map
                 st.session_state.unified_data['upload_status']['bulk_id_mapping'] = True
                 st.session_state.unified_data['upload_timestamps']['bulk_id_mapping'] = datetime.now()
                 pass # st.toast(f"🔗 Loaded {len(bulk_map)} bulk ID mappings from DB", icon="🆔")
            
            # --- 3. Load ADVERTISED PRODUCT MAP ---
            adv_map = db.get_advertised_product_map(account_id)
            if not adv_map.empty:
                 st.session_state.unified_data['advertised_product_report'] = adv_map
                 st.session_state.unified_data['upload_status']['advertised_product_report'] = True
                 st.session_state.unified_data['upload_timestamps']['advertised_product_report'] = datetime.now()
                 pass # st.toast(f"📦 Loaded {len(adv_map)} advertised products from DB", icon="📦")
            
            # --- 4. Load CATEGORY MAPPING ---
            cat_map = db.get_category_mappings(account_id)
            if not cat_map.empty:
                 st.session_state.unified_data['category_mapping'] = cat_map
                 st.session_state.unified_data['upload_status']['category_mapping'] = True
                 st.session_state.unified_data['upload_timestamps']['category_mapping'] = datetime.now()
                 pass # st.toast(f"📁 Loaded {len(cat_map)} category mappings from DB", icon="📁")

            # SET LAST_STATS_SAVE
            if 'last_stats_save' not in st.session_state:
                st.session_state.last_stats_save = {}
            st.session_state.last_stats_save['client_id'] = account_id
            st.session_state.last_stats_save['start_date'] = recent_dates[0] if recent_dates else None
            
            # CRITICAL: Trigger enrichment to merge all data
            self._enrich_data()
            
            # Verify enrichment worked
            enriched = self.get_enriched_data()
            if enriched is not None:
                id_cols = ['CampaignId', 'AdGroupId', 'KeywordId', 'TargetingId', 'SKU_advertised']
                found_cols = [c for c in id_cols if c in enriched.columns and enriched[c].notna().any()]
                if found_cols:
                    pass # st.toast(f"✅ Enrichment successful: {', '.join(found_cols)}", icon="✅")
                else:
                    st.warning("⚠️ Enrichment ran but no IDs/SKUs were mapped. Check your mapping files.")
            
            return True
            
        except Exception as e:
            st.error(f"Failed to load data from database: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary statistics of loaded data."""
        summary = {}
        
        # Search term report
        str_report = self.get_data('search_term_report')
        if str_report is not None:
            summary['search_terms'] = len(str_report)
            summary['total_clicks'] = str_report['Clicks'].sum() if 'Clicks' in str_report.columns else 0
            summary['total_spend'] = str_report['Spend'].sum() if 'Spend' in str_report.columns else 0
            summary['campaigns'] = str_report['Campaign Name'].nunique() if 'Campaign Name' in str_report.columns else 0
        
        # Advertised products
        adv_report = self.get_data('advertised_product_report')
        if adv_report is not None:
            summary['advertised_products'] = len(adv_report)
            summary['unique_asins'] = adv_report['ASIN'].nunique() if 'ASIN' in adv_report.columns else 0
        
        # Bulk IDs
        bulk_ids = self.get_data('bulk_id_mapping')
        if bulk_ids is not None:
            summary['mapped_campaigns'] = bulk_ids['Campaign Name'].nunique() if 'Campaign Name' in bulk_ids.columns else 0
            summary['mapped_adgroups'] = bulk_ids['Ad Group Name'].nunique() if 'Ad Group Name' in bulk_ids.columns else 0
        
        # Category mapping
        cat_map = self.get_data('category_mapping')
        if cat_map is not None:
            summary['categorized_skus'] = len(cat_map)
        
        return summary
