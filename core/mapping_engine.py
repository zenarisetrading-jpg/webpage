"""
Mapping Engine Module

Centralized data enrichment logic for:
1. SKU from Advertised Product Report
2. IDs from Bulk Upload File
3. Category from Category Mapping File

All matching uses normalized keys (case-insensitive, alphanumeric-only).
"""

import pandas as pd
import numpy as np
import re
from typing import Optional, Tuple


class MappingEngine:
    """Centralized mapping engine for data enrichment."""
    
    # =========================================================================
    # HELPER: Normalize Text for Matching
    # =========================================================================
    @staticmethod
    def normalize(series: pd.Series) -> pd.Series:
        """Normalize a Series for robust matching (lowercase, alphanumeric only)."""
        return series.astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True).str.strip()
    
    @staticmethod
    def normalize_targeting(series: pd.Series) -> pd.Series:
        """Normalize targeting expressions (remove 'asin=', 'category=', quotes)."""
        s = series.astype(str).str.lower().str.strip()
        s = s.str.replace(r'asin\s*=\s*', '', regex=True)
        s = s.str.replace(r'category\s*=\s*', '', regex=True)
        s = s.str.replace(r'["\']', '', regex=True)
        return s.str.strip()
    
    # =========================================================================
    # METHOD 1: Map SKU from Advertised Product Report
    # =========================================================================
    @staticmethod
    def map_sku_from_apr(df: pd.DataFrame, apr: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Maps SKU_advertised from the Advertised Product Report.
        
        Args:
            df: Search Term Report DataFrame
            apr: Advertised Product Report DataFrame
            
        Returns:
            Tuple of (enriched DataFrame, stats dict)
        """
        stats = {'method': 'apr', 'matched': 0, 'total': len(df), 'columns_added': []}
        
        if apr is None or df is None:
            return df, stats
            
        if 'Campaign Name' not in df.columns or 'Campaign Name' not in apr.columns:
            return df, stats
        if 'Ad Group Name' not in df.columns or 'Ad Group Name' not in apr.columns:
            return df, stats
            
        # Build aggregation dict
        agg_dict = {}
        if 'SKU' in apr.columns:
            agg_dict['SKU'] = lambda x: ', '.join(x.dropna().astype(str).unique())
        if 'ASIN' in apr.columns:
            agg_dict['ASIN'] = lambda x: ', '.join(x.dropna().astype(str).unique())
            
        if not agg_dict:
            return df, stats
        
        # Create lookup table (Campaign + AdGroup -> SKU/ASIN)
        lookup = apr.groupby(['Campaign Name', 'Ad Group Name']).agg(agg_dict).reset_index()
        lookup.columns = ['Campaign Name', 'Ad Group Name'] + [f'{col}_advertised' for col in agg_dict.keys()]
        
        # CRITICAL: Normalize campaign and ad group names for robust matching
        df = df.copy()
        df['_camp_norm'] = df['Campaign Name'].astype(str).str.strip().str.lower()
        df['_ag_norm'] = df['Ad Group Name'].astype(str).str.strip().str.lower()
        
        lookup['_camp_norm'] = lookup['Campaign Name'].astype(str).str.strip().str.lower()
        lookup['_ag_norm'] = lookup['Ad Group Name'].astype(str).str.strip().str.lower()
        
        # Merge on normalized keys
        before_cols = set(df.columns)
        enriched = df.merge(
            lookup.drop(columns=['Campaign Name', 'Ad Group Name']),
            on=['_camp_norm', '_ag_norm'],
            how='left'
        )
        
        # Cleanup normalization columns
        enriched.drop(columns=['_camp_norm', '_ag_norm'], inplace=True, errors='ignore')
        
        after_cols = set(enriched.columns)
        
        # Stats
        stats['columns_added'] = list(after_cols - before_cols)
        if 'SKU_advertised' in enriched.columns:
            stats['matched'] = enriched['SKU_advertised'].notna().sum()
        
        return enriched, stats
    
    # =========================================================================
    # METHOD 2: Map IDs from Bulk Upload File
    # =========================================================================
    @staticmethod
    def map_ids_from_bulk(df: pd.DataFrame, bulk: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Maps CampaignId, AdGroupId, KeywordId, TargetingId from Bulk file.
        
        Args:
            df: Search Term Report DataFrame
            bulk: Bulk Upload File DataFrame
            
        Returns:
            Tuple of (enriched DataFrame, stats dict)
        """
        stats = {'method': 'bulk', 'campaign_id_matched': 0, 'keyword_id_matched': 0, 
                 'targeting_id_matched': 0, 'total': len(df)}
        
        if bulk is None or df is None:
            return df, stats
            
        enriched = df.copy()
        
        # ========== PHASE 1: Campaign & AdGroup IDs ===========
        if 'Campaign Name' in enriched.columns and 'Campaign Name' in bulk.columns:
            if 'CampaignId' in bulk.columns:
                # Normalize keys
                enriched['_camp_norm'] = MappingEngine.normalize(enriched['Campaign Name'])
                
                bulk_norm = bulk.copy()
                bulk_norm['_camp_norm'] = MappingEngine.normalize(bulk_norm['Campaign Name'])
                
                # Prepare merge columns
                merge_cols = ['_camp_norm', 'CampaignId']
                on_keys = ['_camp_norm']
                
                if 'Ad Group Name' in enriched.columns and 'Ad Group Name' in bulk.columns and 'AdGroupId' in bulk.columns:
                    enriched['_ag_norm'] = MappingEngine.normalize(enriched['Ad Group Name'])
                    bulk_norm['_ag_norm'] = MappingEngine.normalize(bulk_norm['Ad Group Name'])
                    merge_cols.extend(['_ag_norm', 'AdGroupId'])
                    on_keys.append('_ag_norm')
                
                # Dedupe and merge - CRITICAL: Use groupby().first() to enforce 1-to-1 mapping
                # drop_duplicates() alone keeps duplicates if IDs differ for same name
                # Fix: Exclude keys from value selection to avoid 'already exists' error on reset_index
                val_cols = [c for c in merge_cols if c not in on_keys]
                id_lookup = bulk_norm.groupby(on_keys)[val_cols].first().reset_index()
                enriched = enriched.merge(id_lookup, on=on_keys, how='left')
                
                # Stats
                if 'CampaignId' in enriched.columns:
                    stats['campaign_id_matched'] = enriched['CampaignId'].notna().sum()
                
                # Cleanup
                enriched.drop(columns=['_camp_norm', '_ag_norm'], inplace=True, errors='ignore')
        
        # ========== PHASE 2: Keyword & Targeting IDs ===========
        if 'Targeting' in enriched.columns:
            # Detect keyword column in bulk (try multiple names)
            kw_col = next((c for c in ['Keyword Text', 'Customer Search Term', 'keyword_text', 'Keyword'] if c in bulk.columns), None)
            pt_col = next((c for c in ['Product Targeting Expression', 'TargetingExpression', 'targeting_expression'] if c in bulk.columns), None)
            
            # Detect ID columns - TRY BOTH FORMATS (with and without spaces)
            kwid_col = next((c for c in ['Keyword ID', 'KeywordId', 'keyword_id', 'Keyword Id'] if c in bulk.columns), None)
            ptid_col = next((c for c in ['Product Targeting ID', 'TargetingId', 'targeting_id', 'Product Targeting Id'] if c in bulk.columns), None)
            
            # Prepare enriched normalized keys
            enriched['_camp_norm'] = MappingEngine.normalize(enriched['Campaign Name'])
            enriched['_ag_norm'] = MappingEngine.normalize(enriched['Ad Group Name'])
            enriched['_target_norm'] = MappingEngine.normalize_targeting(enriched['Targeting'])
            
            # ----- Keyword ID -----
            if kw_col and kwid_col:
                # Filter bulk to only keyword rows (non-empty keyword text)
                kw_lookup = bulk[bulk[kw_col].notna() & (bulk[kw_col].astype(str).str.strip() != '')].copy()
                
                if not kw_lookup.empty:
                    kw_lookup = kw_lookup[['Campaign Name', 'Ad Group Name', kw_col, kwid_col, 'Match Type']].copy()
                    
                    kw_lookup['_camp_norm'] = MappingEngine.normalize(kw_lookup['Campaign Name'])
                    kw_lookup['_ag_norm'] = MappingEngine.normalize(kw_lookup['Ad Group Name'])
                    kw_lookup['_target_norm'] = MappingEngine.normalize_targeting(kw_lookup[kw_col])
                    kw_lookup['_match_norm'] = kw_lookup['Match Type'].astype(str).str.lower().str.strip()
                    
                    # ENRICHED (STR) Normalization
                    enriched['_match_norm'] = enriched['Match Type'].astype(str).str.lower().str.strip() if 'Match Type' in enriched.columns else ''
                    
                    # CRITICAL: Standardize to 'KeywordId' (no space) for internal use
                    kw_lookup = kw_lookup.rename(columns={kwid_col: 'KeywordId'})
                    
                    # STRATEGY 1: Strict Match (Campaign + AG + Keyword + Match Type)
                    # This handles the "phrase" vs "exact" collision correctly
                    strict_lookup = kw_lookup.groupby(['_camp_norm', '_ag_norm', '_target_norm', '_match_norm'])['KeywordId'].first().reset_index()
                    enriched = enriched.merge(strict_lookup, on=['_camp_norm', '_ag_norm', '_target_norm', '_match_norm'], how='left', suffixes=('', '_strict'))
                    
                    if 'KeywordId_strict' in enriched.columns:
                        if 'KeywordId' not in enriched.columns:
                            enriched['KeywordId'] = enriched['KeywordId_strict']
                        else:
                            enriched['KeywordId'] = enriched['KeywordId'].fillna(enriched['KeywordId_strict'])
                        enriched.drop(columns=['KeywordId_strict'], inplace=True, errors='ignore')
                        
                    # STRATEGY 2: Relaxed Match (Campaign + AG + Keyword) - Fallback
                    # Only for rows that didn't match strictly (e.g. if Match Type is missing or formatted differently)
                    relaxed_lookup = kw_lookup.groupby(['_camp_norm', '_ag_norm', '_target_norm'])['KeywordId'].first().reset_index()
                    enriched = enriched.merge(relaxed_lookup, on=['_camp_norm', '_ag_norm', '_target_norm'], how='left', suffixes=('', '_relaxed'))
                    
                    if 'KeywordId_relaxed' in enriched.columns:
                        if 'KeywordId' not in enriched.columns:
                            enriched['KeywordId'] = enriched['KeywordId_relaxed']
                        else:
                            enriched['KeywordId'] = enriched['KeywordId'].fillna(enriched['KeywordId_relaxed'])
                        enriched.drop(columns=['KeywordId_relaxed'], inplace=True, errors='ignore')
                    
                    # Cleanup specific normalization col
                    enriched.drop(columns=['_match_norm'], inplace=True, errors='ignore')
                    
                    stats['keyword_id_matched'] = enriched['KeywordId'].notna().sum()
            
            # ----- Targeting ID -----
            if pt_col and ptid_col:
                pt_lookup = bulk[bulk[pt_col].notna() & (bulk[pt_col].astype(str).str.strip() != '')].copy()
                
                if not pt_lookup.empty:
                    pt_lookup = pt_lookup[['Campaign Name', 'Ad Group Name', pt_col, ptid_col]].copy()
                    pt_lookup['_camp_norm'] = MappingEngine.normalize(pt_lookup['Campaign Name'])
                    pt_lookup['_ag_norm'] = MappingEngine.normalize(pt_lookup['Ad Group Name'])
                    pt_lookup['_target_norm'] = MappingEngine.normalize_targeting(pt_lookup[pt_col])
                    
                    # CRITICAL: Standardize to 'TargetingId' (no space) for internal use
                    pt_lookup = pt_lookup.rename(columns={ptid_col: 'TargetingId'})
                    
                    # Check if it's auto targeting (close-match, loose-match, etc.)
                    pt_lookup['_is_auto'] = pt_lookup['_target_norm'].str.contains('closematch|loosematch|substitutes|complements', regex=True)
                    
                    # Split: Auto PT vs ASIN/Category PT
                    auto_pt = pt_lookup[pt_lookup['_is_auto']].copy()
                    specific_pt = pt_lookup[~pt_lookup['_is_auto']].copy()
                    
                    # For ASIN/Category PT: Strict match on targeting expression
                    if not specific_pt.empty:
                        # CRITICAL: Enforce uniqueness on join keys
                        specific_pt = specific_pt.groupby(['_camp_norm', '_ag_norm', '_target_norm'])['TargetingId'].first().reset_index()
                        enriched = enriched.merge(specific_pt, on=['_camp_norm', '_ag_norm', '_target_norm'], how='left', suffixes=('', '_spec'))
                        
                        if 'TargetingId_spec' in enriched.columns:
                            if 'TargetingId' not in enriched.columns:
                                enriched['TargetingId'] = enriched['TargetingId_spec']
                            else:
                                enriched['TargetingId'] = enriched['TargetingId'].fillna(enriched['TargetingId_spec'])
                            enriched.drop(columns=['TargetingId_spec'], inplace=True, errors='ignore')
                    
                    # For Auto PT: Match on campaign + ad group only
                    if not auto_pt.empty:
                        auto_agg = auto_pt.groupby(['_camp_norm', '_ag_norm'])['TargetingId'].first().reset_index()
                        enriched = enriched.merge(auto_agg, on=['_camp_norm', '_ag_norm'], how='left', suffixes=('', '_auto'))
                        
                        if 'TargetingId_auto' in enriched.columns:
                            if 'TargetingId' not in enriched.columns:
                                enriched['TargetingId'] = enriched['TargetingId_auto']
                            else:
                                enriched['TargetingId'] = enriched['TargetingId'].fillna(enriched['TargetingId_auto'])
                            enriched.drop(columns=['TargetingId_auto'], inplace=True, errors='ignore')
                        
                    stats['targeting_id_matched'] = enriched['TargetingId'].notna().sum()
            
            # Cleanup
            enriched.drop(columns=['_camp_norm', '_ag_norm', '_target_norm'], inplace=True, errors='ignore')
        
        # ========== PHASE 3: Bid Columns ===========
        # Map Ad Group Default Bid and Keyword Bid from bulk file
        if 'Ad Group Default Bid' in bulk.columns or 'Bid' in bulk.columns:
            # Normalize keys for matching
            enriched['_camp_norm'] = MappingEngine.normalize(enriched['Campaign Name'])
            enriched['_ag_norm'] = MappingEngine.normalize(enriched['Ad Group Name'])
            
            bulk_norm = bulk.copy()
            bulk_norm['_camp_norm'] = MappingEngine.normalize(bulk_norm['Campaign Name'])
            bulk_norm['_ag_norm'] = MappingEngine.normalize(bulk_norm['Ad Group Name'])
            
            # Build bid lookup: Campaign + Ad Group -> Bid values
            bid_cols = ['_camp_norm', '_ag_norm']
            if 'Ad Group Default Bid' in bulk.columns:
                bid_cols.append('Ad Group Default Bid')
            if 'Bid' in bulk.columns:
                bid_cols.append('Bid')
            
            # Aggregate: take mean of available bids per Campaign + Ad Group
            agg_dict = {col: 'mean' for col in bid_cols if col not in ['_camp_norm', '_ag_norm']}
            bid_lookup = bulk_norm[bid_cols].groupby(['_camp_norm', '_ag_norm']).agg(agg_dict).reset_index()
            
            # Merge bid data
            enriched = enriched.merge(bid_lookup, on=['_camp_norm', '_ag_norm'], how='left', suffixes=('', '_bulk'))
            
            # Handle suffix conflicts
            for col in ['Ad Group Default Bid', 'Bid']:
                bulk_col = f'{col}_bulk'
                if bulk_col in enriched.columns:
                    if col not in enriched.columns:
                        enriched[col] = enriched[bulk_col]
                    else:
                        enriched[col] = enriched[col].fillna(enriched[bulk_col])
                    enriched.drop(columns=[bulk_col], inplace=True, errors='ignore')
            
            # Stats
            if 'Ad Group Default Bid' in enriched.columns:
                stats['default_bid_matched'] = enriched['Ad Group Default Bid'].notna().sum()
            if 'Bid' in enriched.columns:
                stats['bid_matched'] = enriched['Bid'].notna().sum()
            
            # Cleanup
            enriched.drop(columns=['_camp_norm', '_ag_norm'], inplace=True, errors='ignore')
        
        return enriched, stats
    
    # =========================================================================
    # METHOD 3: Map Category from Category Mapping File
    # =========================================================================
    @staticmethod
    def map_category(df: pd.DataFrame, category_map: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Maps Category and Sub-Category from the Category Mapping file.
        
        Args:
            df: Enriched DataFrame (should already have SKU_advertised)
            category_map: Category Mapping DataFrame (SKU -> Category, Sub-Category)
            
        Returns:
            Tuple of (enriched DataFrame, stats dict)
        """
        stats = {'method': 'category', 'matched': 0, 'total': len(df)}
        
        if category_map is None or df is None:
            return df, stats
            
        # Find join key in enriched data
        join_key = None
        for col in ['SKU_advertised', 'SKU', 'Advertised SKU', 'ASIN_advertised', 'ASIN']:
            if col in df.columns:
                join_key = col
                break
        
        if join_key is None:
            return df, stats
        
        # Find SKU column in category_map
        cat_sku_col = next((c for c in ['SKU', 'sku', 'ASIN', 'asin'] if c in category_map.columns), None)
        if cat_sku_col is None:
            return df, stats
            
        # Find category columns
        cat_col = next((c for c in ['Category', 'category'] if c in category_map.columns), None)
        subcat_col = next((c for c in ['Sub-Category', 'Sub Category', 'SubCategory', 'sub_category'] if c in category_map.columns), None)
        
        if cat_col is None and subcat_col is None:
            return df, stats
        
        # Build lookup
        lookup_cols = [cat_sku_col]
        if cat_col:
            lookup_cols.append(cat_col)
        if subcat_col:
            lookup_cols.append(subcat_col)
            
        lookup = category_map[lookup_cols].drop_duplicates()
        
        # Normalize SKUs for matching
        df['_sku_norm'] = MappingEngine.normalize(df[join_key])
        lookup['_sku_norm'] = MappingEngine.normalize(lookup[cat_sku_col])
        
        # Rename to standard names
        rename = {cat_sku_col: '_drop'}
        if cat_col:
            rename[cat_col] = 'Category'
        if subcat_col:
            rename[subcat_col] = 'Sub-Category'
        lookup = lookup.rename(columns=rename)
        
        # CRITICAL: Enforce uniqueness on SKU (Join Key)
        # If an SKU maps to multiple categories, take the first one to avoid row explosion
        lookup = lookup.groupby('_sku_norm').first().reset_index()
        
        # Merge
        enriched = df.merge(lookup[['_sku_norm', 'Category', 'Sub-Category'] if 'Category' in lookup.columns and 'Sub-Category' in lookup.columns 
                                   else ['_sku_norm'] + [c for c in ['Category', 'Sub-Category'] if c in lookup.columns]], 
                           on='_sku_norm', how='left', suffixes=('', '_cat'))
        
        # Stats
        if 'Category' in enriched.columns:
            stats['matched'] = enriched['Category'].notna().sum()
        
        # Cleanup
        enriched.drop(columns=['_sku_norm'], inplace=True, errors='ignore')
        
        return enriched, stats