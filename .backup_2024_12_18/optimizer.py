"""
Optimizer Module - Complete Implementation

Migrated from ppcsuite_v3.2.py with full feature parity:
- Harvest detection with winner campaign/SKU selection
- Isolation negatives (unique per campaign/ad group)
- Performance negatives (bleeders)
- Bid optimization (Exact/PT direct, Aggregated for broad/phrase/auto)
- Heatmap with action tracking
- Advanced simulation with scenarios, sensitivity, risk analysis

Architecture: features/_base.py template
Data Source: DataHub (enriched data with SKUs)
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, Any, Tuple, Optional, Set, List
from datetime import datetime, timedelta
from features._base import BaseFeature
from core.data_hub import DataHub
from core.data_loader import SmartMapper, safe_numeric, is_asin
from utils.formatters import format_currency, format_percentage, dict_to_excel
from utils.matchers import ExactMatcher
from ui.components import metric_card
import uuid

# ==========================================
# CONSTANTS
# ==========================================

BULK_COLUMNS = [
    "Product", "Entity", "Operation", "Campaign Id", "Ad Group Id", 
    "Campaign Name", "Ad Group Name", "Ad Group Default Bid", "Bid", 
    "Keyword Text", "Match Type", "Product Targeting Expression",
    "Keyword Id", "Product Targeting Id", "State"
]

DEFAULT_CONFIG = {
    # Harvest thresholds (Tier 2)
    "HARVEST_CLICKS": 10,
    "HARVEST_ORDERS": 3,
    "HARVEST_SALES": 150.0,
    "HARVEST_ROAS_MULT": 0.8,  # vs campaign median (80% = less strict)
    "DEDUPE_SIMILARITY": 0.90,
    
    # Negative thresholds
    "NEGATIVE_CLICKS_THRESHOLD": 10,
    "NEGATIVE_SPEND_THRESHOLD": 10.0,
    
    # Bid optimization
    "ALPHA_EXACT": 0.25,
    "ALPHA_BROAD": 0.20,
    "ALPHA": 0.20,
    "MAX_BID_CHANGE": 0.20,
    "MIN_CLICKS_BID": 3,
    "TARGET_ROAS": 2.50,
    
    # Harvest forecast
    "HARVEST_EFFICIENCY_MULTIPLIER": 1.30,  # 30% efficiency gain from exact match
}

# Elasticity scenarios for simulation
ELASTICITY_SCENARIOS = {
    'conservative': {
        'cpc': 0.3,
        'clicks': 0.5,
        'cvr': 0.0,
        'probability': 0.15
    },
    'expected': {
        'cpc': 0.5,
        'clicks': 0.85,
        'cvr': 0.1,
        'probability': 0.70
    },
    'aggressive': {
        'cpc': 0.6,
        'clicks': 0.95,
        'cvr': 0.15,
        'probability': 0.15
    }
}


# ==========================================
# DATA PREPARATION
# ==========================================

def prepare_data(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Validate and prepare data for optimization.
    Returns prepared DataFrame and date_info dict.
    """
    df = df.copy()
    
    # Ensure numeric columns
    for col in ["Impressions", "Clicks", "Spend", "Sales", "Orders"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = safe_numeric(df[col])
    
    # CPC calculation
    df["CPC"] = np.where(df["Clicks"] > 0, df["Spend"] / df["Clicks"], 0)
    
    # Standardize column names
    col_map = {
        "Campaign": "Campaign Name",
        "AdGroup": "Ad Group Name", 
        "Term": "Customer Search Term",
        "Match": "Match Type"
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    # Ensure Match Type exists
    if "Match Type" not in df.columns:
        df["Match Type"] = "broad"
    df["Match Type"] = df["Match Type"].fillna("broad").astype(str)
    
    # Targeting column normalization
    if "Targeting" not in df.columns:
        if "Keyword" in df.columns:
            df["Targeting"] = df["Keyword"].replace("", np.nan)
        else:
            df["Targeting"] = pd.Series([np.nan] * len(df))
    else:
        # If Targeting exists but has empty strings, ensure they are NaN for filling later
        df["Targeting"] = df["Targeting"].replace("", np.nan)
    
    if "TargetingExpression" in df.columns:
        # Prefer Expression over generic Targeting which might be "*"
        df["Targeting"] = df["TargetingExpression"].fillna(df["Targeting"])
    
    # CRITICAL FIX: Only fallback to Search Term for EXACT match types
    # For Auto/Broad/Phrase, we MUST NOT use Search Term as it breaks aggregation
    df["Targeting"] = df["Targeting"].fillna("")
    
    # 1. For Exact matches, missing Targeting can be filled with Search Term
    exact_mask = df["Match Type"].str.lower() == "exact"
    missing_targeting = (df["Targeting"] == "") | (df["Targeting"] == "*")
    df.loc[exact_mask & missing_targeting, "Targeting"] = df.loc[exact_mask & missing_targeting, "Customer Search Term"]
    
    # 2. For Auto/Broad/Phrase, if Targeting is missing, use Match Type as fallback grouping key
    # This prevents "fighter jet toy" appearing in Targeting for an auto campaign
    # But checking for '*' is important too as that is generic
    generic_targeting = (df["Targeting"] == "") | (df["Targeting"] == "*")
    df.loc[~exact_mask & generic_targeting, "Targeting"] = df.loc[~exact_mask & generic_targeting, "Match Type"]
    
    df["Targeting"] = df["Targeting"].astype(str)
    
    # 3. Normalize Auto targeting types for consistent grouping
    # e.g., "Close-Match" -> "close-match", "Close Match" -> "close-match"
    AUTO_TARGETING_TYPES = {'close-match', 'loose-match', 'substitutes', 'complements'}
    
    def normalize_auto_targeting(val):
        """Normalize auto targeting types to canonical lowercase-hyphen form."""
        val_norm = str(val).strip().lower().replace(" ", "-").replace("_", "-")
        if val_norm in AUTO_TARGETING_TYPES:
            return val_norm
        return val  # Keep original for non-auto types
    
    df["Targeting"] = df["Targeting"].apply(normalize_auto_targeting)
    
    # Sales/Orders attributed columns
    df["Sales_Attributed"] = df["Sales"]
    df["Orders_Attributed"] = df["Orders"]
    
    # Derived metrics
    df["CTR"] = np.where(df["Impressions"] > 0, df["Clicks"] / df["Impressions"], 0)
    df["ROAS"] = np.where(df["Spend"] > 0, df["Sales"] / df["Spend"], 0)
    df["CVR"] = np.where(df["Clicks"] > 0, df["Orders"] / df["Clicks"], 0)
    df["ACoS"] = np.where(df["Sales"] > 0, df["Spend"] / df["Sales"] * 100, 0)
    
    # Campaign-level metrics
    camp_stats = df.groupby("Campaign Name")[["Sales", "Spend"]].transform("sum")
    df["Campaign_ROAS"] = np.where(
        camp_stats["Spend"] > 0, 
        camp_stats["Sales"] / camp_stats["Spend"], 
        config["TARGET_ROAS"]
    )
    
    # Detect date range
    date_info = detect_date_range(df)
    
    # ==========================================
    # DATA SAVING: Handled by Data Hub. 
    # Logic removed to prevent implicit ID extraction from filenames.
    # ==========================================
    
    return df, date_info


def detect_date_range(df: pd.DataFrame) -> dict:
    """Detect date range from data for weekly normalization."""
    # Added 'start_date' for DB loaded frames
    date_cols = ["Date", "Start Date", "date", "Report Date", "start_date"]
    date_col = None
    
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        return {"weeks": 1.0, "label": "Period Unknown", "days": 7, "start_date": None, "end_date": None}
    
    try:
        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        if dates.empty:
            return {"weeks": 1.0, "label": "Period Unknown", "days": 7, "start_date": None, "end_date": None}
        
        min_date = dates.min()
        max_date = dates.max()
        days = (max_date - min_date).days + 1
        weeks = max(days / 7, 1.0)
        
        label = f"{days} days ({min_date.strftime('%b %d')} - {max_date.strftime('%b %d')})"
        
        return {
            "weeks": weeks, 
            "label": label, 
            "days": days,
            "start_date": min_date.date().isoformat(),  # ISO format string
            "end_date": max_date.date().isoformat()
        }
    except:
        return {"weeks": 1.0, "label": "Period Unknown", "days": 7, "start_date": None, "end_date": None}


# ==========================================
# HARVEST DETECTION
# ==========================================

def identify_harvest_candidates(
    df: pd.DataFrame, 
    config: dict, 
    matcher: ExactMatcher
) -> pd.DataFrame:
    """
    Identify high-performing search terms to harvest as exact match keywords.
    Winner campaign/SKU trumps others based on performance when KW appears in multiple campaigns.
    """
    
    # Filter for discovery campaigns (non-exact)
    auto_pattern = r'close-match|loose-match|substitutes|complements|category=|asin|b0'
    discovery_mask = (
        (~df["Match Type"].str.contains("exact", case=False, na=False)) |
        (df["Targeting"].str.contains(auto_pattern, case=False, na=False))
    )
    discovery_df = df[discovery_mask].copy()
    
    if discovery_df.empty:
        return pd.DataFrame()
    
    # FIXED: Aggregate by Targeting (not Customer Search Term) to match bid optimization
    # For Auto campaigns, Targeting contains the actual targeting expression
    # that bid optimization groups by
    agg_cols = {
        "Impressions": "sum", "Clicks": "sum", "Spend": "sum",
        "Sales": "sum", "Orders": "sum", "CPC": "mean"
    }
    
    # Also keep Customer Search Term for reference (use first value)
    if "Customer Search Term" in discovery_df.columns:
        agg_cols["Customer Search Term"] = "first"
    
    grouped = discovery_df.groupby("Targeting", as_index=False).agg(agg_cols)
    grouped["ROAS"] = np.where(grouped["Spend"] > 0, grouped["Sales"] / grouped["Spend"], 0)
    
    # Rename Targeting to Customer Search Term for compatibility with downstream code
    grouped = grouped.rename(columns={"Targeting": "Harvest_Term"})
    if "Customer Search Term" not in grouped.columns:
        grouped["Customer Search Term"] = grouped["Harvest_Term"]
    
    # Get metadata from BEST performing instance (winner selection)
    # Rank by Sales (primary), then ROAS (secondary)
    discovery_df["_perf_score"] = discovery_df["Sales"] + (discovery_df["ROAS"] * 10)
    discovery_df["_rank"] = discovery_df.groupby("Targeting")["_perf_score"].rank(
        method="first", ascending=False
    )
    
    # Build metadata columns list
    meta_cols = ["Targeting", "Campaign Name", "Ad Group Name", "Campaign_ROAS"]
    if "CampaignId" in discovery_df.columns:
        meta_cols.append("CampaignId")
    if "AdGroupId" in discovery_df.columns:
        meta_cols.append("AdGroupId")
    if "SKU_advertised" in discovery_df.columns:
        meta_cols.append("SKU_advertised")
    if "ASIN_advertised" in discovery_df.columns:
        meta_cols.append("ASIN_advertised")
    
    # Get winner row for each Targeting value
    meta_df = discovery_df[discovery_df["_rank"] == 1][meta_cols].drop_duplicates("Targeting")
    merged = pd.merge(grouped, meta_df, left_on="Harvest_Term", right_on="Targeting", how="left")
    
    # Ensure Customer Search Term column exists for downstream compatibility
    if "Customer Search Term" not in merged.columns:
        merged["Customer Search Term"] = merged["Harvest_Term"]
    
    # Apply harvest thresholds (Tier 2)
    baseline_roas = merged["Campaign_ROAS"].fillna(config["TARGET_ROAS"])
    
    # Individual threshold checks for debugging
    pass_clicks = merged["Clicks"] >= config["HARVEST_CLICKS"]
    pass_orders = merged["Orders"] >= config["HARVEST_ORDERS"]
    pass_sales = merged["Sales"] >= config["HARVEST_SALES"]
    pass_roas = merged["ROAS"] >= (baseline_roas * config["HARVEST_ROAS_MULT"])
    
    harvest_mask = pass_clicks & pass_orders & pass_sales & pass_roas
    
    candidates = merged[harvest_mask].copy()
    
    # DEBUG: Show why terms fail
    print(f"\n=== HARVEST DEBUG ===")
    print(f"Discovery rows: {len(discovery_df)}")
    print(f"Grouped search terms: {len(grouped)}")
    print(f"Threshold config: Clicks>={config['HARVEST_CLICKS']}, Orders>={config['HARVEST_ORDERS']}, Sales>=${config['HARVEST_SALES']}, ROAS>={config['HARVEST_ROAS_MULT']}x baseline")
    print(f"Pass clicks: {pass_clicks.sum()}, Pass orders: {pass_orders.sum()}, Pass sales: {pass_sales.sum()}, Pass ROAS: {pass_roas.sum()}")
    print(f"After ALL thresholds: {len(candidates)} candidates")
    
    # DEBUG: Check for specific terms
    test_terms = ['water cups for kids', 'water cups', 'steel water bottle', 'painting set for kids']
    print(f"\n--- Checking specific terms ---")
    for test_term in test_terms:
        in_grouped = merged[merged["Customer Search Term"].str.contains(test_term, case=False, na=False)]
        if len(in_grouped) > 0:
            for _, r in in_grouped.iterrows():
                req_roas = (r.get("Campaign_ROAS") or config["TARGET_ROAS"]) * config["HARVEST_ROAS_MULT"]
                pass_all = (r["Clicks"] >= config["HARVEST_CLICKS"] and 
                           r["Orders"] >= config["HARVEST_ORDERS"] and 
                           r["Sales"] >= config["HARVEST_SALES"] and 
                           r["ROAS"] >= req_roas)
                print(f"  '{r['Customer Search Term']}': Clicks={r['Clicks']}, Orders={r['Orders']}, Sales=${r['Sales']:.2f}, ROAS={r['ROAS']:.2f} vs {req_roas:.2f} | PASS={pass_all}")
        else:
            print(f"  '{test_term}' - NOT FOUND in Customer Search Term column")
    
    # Show sample of terms that pass all but ROAS
    almost_pass = pass_clicks & pass_orders & pass_sales & (~pass_roas)
    if almost_pass.sum() > 0:
        print(f"\nTerms failing ONLY on ROAS ({almost_pass.sum()} total):")
        for _, r in merged[almost_pass].head(5).iterrows():
            req_roas = r.get("Campaign_ROAS", config["TARGET_ROAS"]) * config["HARVEST_ROAS_MULT"]
            print(f"  - '{r['Customer Search Term']}': ROAS {r['ROAS']:.2f} < required {req_roas:.2f}")
    
    if len(candidates) > 0:
        print(f"\nTop 5 candidates BEFORE dedupe:")
        for _, r in candidates.head(5).iterrows():
            print(f"  - '{r['Customer Search Term']}': {r['Clicks']} clicks, {r['Orders']} orders, ${r['Sales']:.2f} sales")
    
    # Dedupe against existing exact keywords
    survivors = []
    deduped = []
    for _, row in candidates.iterrows():
        matched, match_info = matcher.find_match(row["Customer Search Term"], config["DEDUPE_SIMILARITY"])
        if not matched:
            survivors.append(row)
        else:
            deduped.append((row["Customer Search Term"], match_info))
    
    print(f"\nDedupe results:")
    print(f"  - Survivors (new harvest): {len(survivors)}")
    print(f"  - Deduped (already exist): {len(deduped)}")
    if deduped:
        print(f"  - Sample deduped terms:")
        for term, match in deduped[:5]:
            print(f"    '{term}' matched to: {match}")
    print(f"=== END HARVEST DEBUG ===\n")
    
    survivors_df = pd.DataFrame(survivors)
    
    if not survivors_df.empty:
        survivors_df["New Bid"] = survivors_df["CPC"] * 1.1
        survivors_df = survivors_df.sort_values("Sales", ascending=False)
    
    return survivors_df


# ==========================================
# NEGATIVE DETECTION
# ==========================================

def identify_negative_candidates(
    df: pd.DataFrame, 
    config: dict, 
    harvest_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd. DataFrame, pd.DataFrame]:
    """
    Identify negative keyword candidates:
    1. Isolation negatives - harvest terms to negate in source campaigns (unique per campaign/ad group)
    2. Performance negatives - bleeders with 0 sales, high spend
    3. ASIN Mapper integration - competitor ASINs flagged for negation
    
    Returns: (keyword_negatives_df, product_target_negatives_df, your_products_review_df)
    """
    negatives = []
    your_products_review = []
    seen_keys = set()  # Track (campaign, ad_group, term) for uniqueness
    
    # Stage 1: Isolation negatives
    if not harvest_df.empty:
        harvested_terms = set(
            harvest_df["Customer Search Term"].astype(str).str.strip().str.lower()
        )
        
        # Find all occurrences in non-exact campaigns
        isolation_mask = (
            df["Customer Search Term"].astype(str).str.strip().str.lower().isin(harvested_terms) &
            (~df["Match Type"].str.contains("exact", case=False, na=False))
        )
        
        isolation_df = df[isolation_mask].copy()
        
        # Aggregate logic for Isolation Negatives (Fix for "metrics broken down by date")
        if not isolation_df.empty:
            agg_cols = {"Clicks": "sum", "Spend": "sum"}
            meta_cols = {c: "first" for c in ["CampaignId", "AdGroupId"] if c in isolation_df.columns}
            
            isolation_agg = isolation_df.groupby(
                ["Campaign Name", "Ad Group Name", "Customer Search Term"], as_index=False
            ).agg({**agg_cols, **meta_cols})
            
            # Get winner campaign for each term (to exclude from negation)
            winner_camps = dict(zip(
                harvest_df["Customer Search Term"].str.lower(),
                harvest_df["Campaign Name"]
            ))
            
            for _, row in isolation_agg.iterrows():
                campaign = row["Campaign Name"]
                ad_group = row["Ad Group Name"]
                term = str(row["Customer Search Term"]).strip().lower()
                
                # Skip the winner campaign - don't negate where we're promoting
                if campaign == winner_camps.get(term):
                    continue
                
                # Unique key per campaign/ad group (redundant after groupby but good for safety)
                key = (campaign, ad_group, term)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                
                negatives.append({
                    "Type": "Isolation",
                    "Campaign Name": campaign,
                    "Ad Group Name": ad_group,
                    "Term": term,
                    "Is_ASIN": is_asin(term),
                    "Clicks": row["Clicks"],
                    "Spend": row["Spend"],
                    "CampaignId": row.get("CampaignId", ""),
                    "AdGroupId": row.get("AdGroupId", ""),
                })
    
    # Stage 2: Performance negatives (bleeders)
    non_exact_mask = ~df["Match Type"].str.contains("exact", case=False, na=False)
    # Don't filter Sales==0 yet - wait until aggregated
    bleeders = df[non_exact_mask].copy()
    
    if not bleeders.empty:
        # Aggregate by campaign + ad group + term
        agg_cols = {"Clicks": "sum", "Spend": "sum", "Impressions": "sum", "Sales": "sum"}
        meta_cols = {c: "first" for c in ["CampaignId", "AdGroupId"] if c in bleeders.columns}
        
        bleeder_agg = bleeders.groupby(
            ["Campaign Name", "Ad Group Name", "Customer Search Term"], as_index=False
        ).agg({**agg_cols, **meta_cols})
        
        # Apply thresholds (Sales == 0 AND Clicks/Spend > threshold)
        hard_stop_clicks = max(15, int(config["NEGATIVE_CLICKS_THRESHOLD"] * 1.5))
        
        bleeder_mask = (
            (bleeder_agg["Sales"] == 0) &
            (
                (bleeder_agg["Clicks"] >= config["NEGATIVE_CLICKS_THRESHOLD"]) |
                (bleeder_agg["Spend"] >= config["NEGATIVE_SPEND_THRESHOLD"])
            )
        )
        
        for _, row in bleeder_agg[bleeder_mask].iterrows():
            campaign = row["Campaign Name"]
            ad_group = row["Ad Group Name"]
            term = str(row["Customer Search Term"]).strip().lower()
            
            key = (campaign, ad_group, term)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            reason = "Hard Stop" if row["Clicks"] >= hard_stop_clicks else "Performance"
            negatives.append({
                "Type": f"Bleeder ({reason})",
                "Campaign Name": campaign,
                "Ad Group Name": ad_group,
                "Term": term,
                "Is_ASIN": is_asin(term),
                "Clicks": row["Clicks"],
                "Spend": row["Spend"],
                "CampaignId": row.get("CampaignId", ""),
                "AdGroupId": row.get("AdGroupId", ""),
            })
    
    # Stage 3: ASIN Mapper Integration
    asin_mapper_stats = {'total': 0, 'added': 0, 'duplicates': 0}
    
    if 'latest_asin_analysis' in st.session_state:
        asin_results = st.session_state['latest_asin_analysis']
        
        # DEBUG
        print(f"DEBUG - Optimizer Stage 3: Found ASIN analysis in session state, keys: {list(asin_results.keys())}")
        
        if 'optimizer_negatives' in asin_results:
            optimizer_data = asin_results['optimizer_negatives']
            
            # DEBUG  
            print(f"DEBUG - Optimizer Stage 3: Found optimizer_negatives with {len(optimizer_data.get('competitor_asins', []))} competitor ASINs")
            
            # Add competitor ASINs (auto-negate recommended)
            competitor_asins = optimizer_data.get('competitor_asins', [])
            asin_mapper_stats['total'] = len(competitor_asins)
            
            for asin_neg in competitor_asins:
                term = asin_neg['Term'].lower()
                campaign = asin_neg.get('Campaign Name', '')
                ad_group = asin_neg.get('Ad Group Name', '')
                
                key = (campaign, ad_group, term)
                if key in seen_keys:
                    asin_mapper_stats['duplicates'] += 1
                    continue
                seen_keys.add(key)
                
                negatives.append(asin_neg)
                asin_mapper_stats['added'] += 1
            
            # Collect your products for separate review section
            your_products_review = optimizer_data.get('your_products_review', [])
        else:
            print("DEBUG - Optimizer Stage 3: 'optimizer_negatives' key NOT found in asin_results")
    else:
        print("DEBUG - Optimizer Stage 3: 'latest_asin_analysis' NOT in session state")
    
    neg_df = pd.DataFrame(negatives)
    your_products_df = pd.DataFrame(your_products_review)
    
    if neg_df.empty:
        return pd.DataFrame(), pd.DataFrame(), your_products_df
    
    # CRITICAL: Map KeywordId and TargetingId for negatives
    # Negatives are at campaign+adgroup+term level, so we need to look up IDs
    from core.data_hub import DataHub
    hub = DataHub()
    bulk = hub.get_data('bulk_id_mapping')
    
    if bulk is not None and not bulk.empty:
        # Normalize for matching
        neg_df['_camp_norm'] = neg_df['Campaign Name'].astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
        neg_df['_ag_norm'] = neg_df['Ad Group Name'].astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
        neg_df['_term_norm'] = neg_df['Term'].astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
        
        bulk = bulk.copy()
        bulk['_camp_norm'] = bulk['Campaign Name'].astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
        bulk['_ag_norm'] = bulk['Ad Group Name'].astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
        
        # For keywords: try to match on campaign + ad group + keyword text
        if 'Customer Search Term' in bulk.columns:
            bulk['_kw_norm'] = bulk['Customer Search Term'].astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
            kw_lookup = bulk[bulk['KeywordId'].notna()][['_camp_norm', '_ag_norm', '_kw_norm', 'KeywordId']].drop_duplicates()
            
            # Try exact match on term first
            neg_df = neg_df.merge(
                kw_lookup.rename(columns={'_kw_norm': '_term_norm'}),
                on=['_camp_norm', '_ag_norm', '_term_norm'],
                how='left',
                suffixes=('', '_exact')
            )
        
        # For PT: match on campaign + ad group + PT expression
        if 'Product Targeting Expression' in bulk.columns:
            bulk['_pt_norm'] = bulk['Product Targeting Expression'].astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
            pt_lookup = bulk[bulk['TargetingId'].notna()][['_camp_norm', '_ag_norm', '_pt_norm', 'TargetingId']].drop_duplicates()
            
            neg_df = neg_df.merge(
                pt_lookup.rename(columns={'_pt_norm': '_term_norm'}),
                on=['_camp_norm', '_ag_norm', '_term_norm'],
                how='left',
                suffixes=('', '_exact')
            )
        
        # Fallback: If no exact match, get any ID from same campaign+adgroup
        if 'KeywordId' not in neg_df.columns or neg_df['KeywordId'].isna().any():
            id_fallback = bulk.groupby(['_camp_norm', '_ag_norm']).agg({
                'KeywordId': 'first',
                'TargetingId': 'first'
            }).reset_index()
            
            neg_df = neg_df.merge(
                id_fallback,
                on=['_camp_norm', '_ag_norm'],
                how='left',
                suffixes=('', '_fallback')
            )
            
            # Coalesce: use exact match if available, otherwise fallback
            if 'KeywordId_fallback' in neg_df.columns:
                if 'KeywordId' not in neg_df.columns:
                    neg_df['KeywordId'] = neg_df['KeywordId_fallback']
                else:
                    neg_df['KeywordId'] = neg_df['KeywordId'].fillna(neg_df['KeywordId_fallback'])
            
            if 'TargetingId_fallback' in neg_df.columns:
                if 'TargetingId' not in neg_df.columns:
                    neg_df['TargetingId'] = neg_df['TargetingId_fallback']
                else:
                    neg_df['TargetingId'] = neg_df['TargetingId'].fillna(neg_df['TargetingId_fallback'])
        
        # Cleanup
        neg_df.drop(columns=['_camp_norm', '_ag_norm', '_term_norm', 'KeywordId_fallback', 'TargetingId_fallback', 
                             'KeywordId_exact', 'TargetingId_exact'], inplace=True, errors='ignore')
    
    # Split into keywords vs product targets
    neg_kw = neg_df[~neg_df["Is_ASIN"]].copy()
    neg_pt = neg_df[neg_df["Is_ASIN"]].copy()
    
    # Format for output
    if not neg_kw.empty:
        neg_kw["Match Type"] = "negativeExact"
    if not neg_pt.empty:
        neg_pt["Match Type"] = "Negative Product Targeting"
        neg_pt["Term"] = neg_pt["Term"].apply(lambda x: f'asin="{x.upper()}"')
    
    # Store ASIN Mapper stats for UI display
    st.session_state['asin_mapper_integration_stats'] = asin_mapper_stats
    
    return neg_kw, neg_pt, your_products_df
# ==========================================
# BID OPTIMIZATION
# ==========================================

# ==========================================
# BID OPTIMIZATION (vNext)
# ==========================================

def calculate_bid_optimizations(
    df: pd.DataFrame, 
    config: dict, 
    harvested_terms: Set[str] = None,
    negative_terms: Set[Tuple[str, str, str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate optimal bid adjustments using vNext Bucketed Logic.
    
    Returns 4 DataFrames for 4 tabs:
    1. Exact Keywords (Match Type = exact, manual keywords only)
    2. Product Targeting (PT) - asin= or asin-expanded= syntax
    3. Aggregated Keywords (Broad/Phrase)
    4. Auto/Category (close-match, loose-match, substitutes, complements, category=)
    
    MANDATORY: Bleeders (Sales=0 with Clicks >= threshold) are EXCLUDED.
    """
    harvested_terms = harvested_terms or set()
    negative_terms = negative_terms or set()
    
    # Get bleeder thresholds from config
    # NOTE: Bleeders are identified separately in identify_negative_candidates
    # and passed via negative_terms. We don't re-exclude inline here because
    # rows with Sales=0 but low clicks still need optimization (Hold/Lower bid).

    # 1. Global Exclusions
    def is_excluded(row):
        # Get both Customer Search Term AND Targeting values
        cst = str(row.get("Customer Search Term", "")).strip().lower()
        targeting = str(row.get("Targeting", "")).strip().lower()
        
        # Check Harvest - if EITHER column matches harvested terms, exclude
        # This handles the mismatch where bid optimization groups by Targeting
        # but harvest identifies by Customer Search Term
        if cst in harvested_terms or targeting in harvested_terms:
            return True
            
        # Check Negatives (Campaign, AdGroup, Term)
        camp = str(row.get("Campaign Name", "")).strip()
        ag = str(row.get("Ad Group Name", "")).strip()
        
        # Check against both CST and Targeting
        neg_key_cst = (camp, ag, cst)
        neg_key_targeting = (camp, ag, targeting)
        if neg_key_cst in negative_terms or neg_key_targeting in negative_terms:
            return True
            
        return False
        
    # Apply Exclusion Filter
    mask_excluded = df.apply(is_excluded, axis=1)
    df_clean = df[~mask_excluded].copy()
    
    if df_clean.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # 2. Define bucket detection helpers
    
    # Auto/Category targeting types (normalized)
    AUTO_TYPES = {'close-match', 'loose-match', 'substitutes', 'complements', 'auto'}
    
    def is_auto_or_category(targeting_val):
        """Check if targeting indicates Auto campaign or Category targeting."""
        t = str(targeting_val).lower().strip()
        # Category targeting
        if t.startswith("category=") or "category" in t:
            return True
        # Auto targeting types
        if t in AUTO_TYPES:
            return True
        return False
    
    def is_pt_targeting(targeting_val):
        """Check if targeting indicates Product Targeting (ASIN-based)."""
        t = str(targeting_val).lower().strip()
        # PT syntax: asin="B0XXX", asin-expanded="B0XXX"
        if "asin=" in t or "asin-expanded=" in t:
            return True
        # Raw ASIN check (but not category)
        if is_asin(t) and not t.startswith("category"):
            return True
        return False
    
    # 3. Build mutually exclusive bucket masks
    # Order matters: Check Auto/Category FIRST, then PT, then Keywords
    
    # Bucket 4: Auto/Category - identified by Targeting value OR Match Type = 'auto'
    # For Auto campaigns, Match Type is 'auto' but Targeting may contain search terms
    mask_auto_by_targeting = df_clean["Targeting"].apply(is_auto_or_category)
    mask_auto_by_matchtype = df_clean["Match Type"].str.lower().isin(["auto", "-"])  # "-" sometimes used for auto
    mask_auto = mask_auto_by_targeting | mask_auto_by_matchtype
    
    # Bucket 2: Product Targeting (PT) - identified by ASIN syntax, NOT auto
    mask_pt = df_clean["Targeting"].apply(is_pt_targeting) & (~mask_auto)
    
    # Bucket 1: Exact Keywords - Match Type = exact, NOT PT, NOT Auto
    mask_exact = (
        (df_clean["Match Type"].str.lower() == "exact") & 
        (~mask_pt) & 
        (~mask_auto)
    )
    
    # Bucket 3: Aggregated Keywords (Broad/Phrase) - NOT PT, NOT Auto, NOT Exact
    mask_broad_phrase = (
        df_clean["Match Type"].str.lower().isin(["broad", "phrase"]) & 
        (~mask_pt) & 
        (~mask_auto)
    )
    
    # 4. Process each bucket with appropriate thresholds
    
    # Exact: Fast reaction (Min 5 clicks)
    bids_exact = _process_bucket(df_clean[mask_exact], config, min_clicks=5, bucket_name="Exact")
    
    # PT: Fast reaction (Min 5 clicks)
    bids_pt = _process_bucket(df_clean[mask_pt], config, min_clicks=5, bucket_name="Product Targeting")
    
    # Broad/Phrase (Aggregated): Conservative (Min 10 clicks)
    bids_agg = _process_bucket(df_clean[mask_broad_phrase], config, min_clicks=10, bucket_name="Broad/Phrase")
    
    # Auto/Category: Slow reaction (Min 15 clicks)
    bids_auto = _process_bucket(df_clean[mask_auto], config, min_clicks=15, bucket_name="Auto/Category")
    
    return bids_exact, bids_pt, bids_agg, bids_auto

def _process_bucket(segment_df: pd.DataFrame, config: dict, min_clicks: int, bucket_name: str) -> pd.DataFrame:
    """
    Unified bucket processor with Campaign Median ROAS classification.
    
    Classification vs campaign median ROAS:
    - Promote: ROAS >= 1.2× median → Bid UP
    - Stable: ROAS 0.8× – 1.2× median → HOLD
    - Bid Down: ROAS < 0.8× median → Bid DOWN
    """
    if segment_df.empty:
        return pd.DataFrame()
    
    segment_df = segment_df.copy()
        
    # Normalize targeting for grouping
    segment_df["_targeting_norm"] = (
        segment_df["Targeting"].astype(str).str.strip().str.lower()
    )
    
    has_keyword_id = "KeywordId" in segment_df.columns and segment_df["KeywordId"].notna().any()
    has_targeting_id = "TargetingId" in segment_df.columns and segment_df["TargetingId"].notna().any()
    
    if has_keyword_id or has_targeting_id:
        segment_df["_group_key"] = segment_df.apply(
            lambda r: str(r.get("KeywordId") or r.get("TargetingId") or r["_targeting_norm"]).strip(),
            axis=1
        )
    else:
        segment_df["_group_key"] = segment_df["_targeting_norm"]
    
    # Aggregation
    agg_cols = {"Clicks": "sum", "Spend": "sum", "Sales": "sum", "Impressions": "sum", "Orders": "sum"}
    meta_cols = {c: "first" for c in [
        "Campaign Name", "Ad Group Name", "CampaignId", "AdGroupId", 
        "KeywordId", "TargetingId", "Match Type", "Targeting"
    ] if c in segment_df.columns}
    
    if "Current Bid" in segment_df.columns:
        agg_cols["Current Bid"] = "max"
    if "CPC" in segment_df.columns:
        agg_cols["CPC"] = "mean"
        
    grouped = segment_df.groupby(["Campaign Name", "Ad Group Name", "_group_key"], as_index=False).agg({**agg_cols, **meta_cols})
    grouped = grouped.drop(columns=["_group_key"], errors="ignore")
    
    # Calculate ROAS
    grouped["ROAS"] = np.where(grouped["Spend"] > 0, grouped["Sales"] / grouped["Spend"], 0)
    
    # =====================================================
    # NEW: Use BUCKET-LEVEL Median ROAS as baseline
    # This is more stable than campaign-level and represents
    # typical performance for this match type (Auto, Exact, PT, etc.)
    # Apply 0.8x seasonal multiplier since benchmark is lifetime data
    # =====================================================
    SEASONAL_MULTIPLIER = 0.8  # Account for seasonal variance
    
    all_valid_roas = grouped[(grouped["Spend"] > 0) & (grouped["Sales"] > 0)]["ROAS"]
    bucket_median = all_valid_roas.median() if len(all_valid_roas) >= 1 else config.get("TARGET_ROAS", 2.5)
    
    # Apply seasonal adjustment
    baseline_roas = bucket_median * SEASONAL_MULTIPLIER
    
    print(f"[{bucket_name}] Bucket Median ROAS: {bucket_median:.2f}x → Baseline (×0.8): {baseline_roas:.2f}x")
    
    # Calculate Ad Group level aggregates for fallback
    adgroup_stats = grouped.groupby(["Campaign Name", "Ad Group Name"]).agg({
        "Clicks": "sum", "Spend": "sum", "Sales": "sum", "Orders": "sum"
    }).reset_index()
    adgroup_stats["AG_ROAS"] = np.where(
        adgroup_stats["Spend"] > 0, 
        adgroup_stats["Sales"] / adgroup_stats["Spend"], 0
    )
    adgroup_stats["AG_Clicks"] = adgroup_stats["Clicks"]
    adgroup_lookup = adgroup_stats.set_index(["Campaign Name", "Ad Group Name"])[["AG_ROAS", "AG_Clicks"]].to_dict('index')
    
    # Get alpha values
    alpha = config.get("ALPHA", config.get("ALPHA_EXACT", 0.20))
    if "Broad" in bucket_name or "Auto" in bucket_name:
        alpha = config.get("ALPHA_BROAD", alpha * 0.8)
    
    def apply_optimization(r):
        clicks = r["Clicks"]
        roas = r["ROAS"]
        adgroup = r.get("Ad Group Name", "")
        campaign = r["Campaign Name"]
        
        # Base bid determination
        current_bid = float(r.get("Current Bid", 0) or 0)
        avg_cpc = float(r.get("CPC", 0) or 0)
        base_bid = current_bid if current_bid > 0 else avg_cpc
        
        if base_bid <= 0:
            return 0.0, "Hold: No Bid/CPC Data", "none"
        
        # Use bucket-level baseline (already adjusted for seasonality)
        # Level 1: Try targeting-level data
        if clicks >= min_clicks and roas > 0:
            return _classify_and_bid(roas, baseline_roas, base_bid, alpha, f"targeting|{bucket_name}", config)
        
        # Level 2: Fallback to Ad Group aggregate (conservative)
        ag_key = (campaign, adgroup)
        ag_stats = adgroup_lookup.get(ag_key, {})
        ag_roas = ag_stats.get("AG_ROAS", 0)
        ag_clicks = ag_stats.get("AG_Clicks", 0)
        
        if ag_clicks >= min_clicks and ag_roas > 0:
            # Use conservative alpha for fallback
            return _classify_and_bid(ag_roas, baseline_roas, base_bid, alpha * 0.5, f"adgroup|{bucket_name}", config)
        
        # Level 3: Insufficient data - HOLD
        return base_bid, f"Hold: Insufficient data ({clicks} clicks)", "none"
    
    results = grouped.apply(apply_optimization, axis=1)
    
    grouped["New Bid"] = results.apply(lambda x: x[0])
    grouped["Reason"] = results.apply(lambda x: x[1])
    grouped["Data Source"] = results.apply(lambda x: x[2] if len(x) > 2 else "targeting")
    grouped["Bucket"] = bucket_name
    
    return grouped


def _classify_and_bid(roas: float, median_roas: float, base_bid: float, alpha: float, 
                      data_source: str, config: dict) -> Tuple[float, str, str]:
    """
    Classify ROAS vs bucket baseline and determine bid action.
    
    Thresholds are alpha-based:
    - Promote: ROAS >= (1 + alpha)× baseline → Bid UP
    - Stable: ROAS between (1-alpha) – (1+alpha) × baseline → HOLD
    - Bid Down: ROAS < (1 - alpha)× baseline → Bid DOWN
    """
    max_change = config.get("MAX_BID_CHANGE", 0.25)
    
    # If no valid baseline, use default target ROAS
    if median_roas is None or median_roas <= 0:
        median_roas = config.get("TARGET_ROAS", 2.5)
        data_source = f"{data_source}|default"
    
    # Alpha-based thresholds (e.g., alpha=0.25 → 1.25x and 0.75x)
    promote_threshold = median_roas * (1 + alpha)
    stable_threshold = median_roas * (1 - alpha)
    
    if roas >= promote_threshold:
        # PROMOTE: Bid UP by alpha
        adjustment = min(alpha, max_change)
        new_bid = base_bid * (1 + adjustment)
        reason = f"Promote: ROAS {roas:.2f} ≥ {promote_threshold:.2f} ({data_source})"
        
    elif roas >= stable_threshold:
        # STABLE: Hold bid
        new_bid = base_bid
        reason = f"Stable: ROAS {roas:.2f} ~ {median_roas:.2f} ({data_source})"
        
    else:
        # BID DOWN by alpha
        adjustment = min(alpha, max_change)
        new_bid = base_bid * (1 - adjustment)
        reason = f"Bid Down: ROAS {roas:.2f} < {stable_threshold:.2f} ({data_source})"
    
    # Safety floor
    new_bid = max(0.10, new_bid)
    
    return new_bid, reason, data_source


def _optimize_bid(row: dict, config: dict, alpha: float) -> Tuple[float, str]:
    """Legacy optimize bid function - kept for backward compatibility."""
    current_bid = float(row.get("Current Bid", 0) or 0)
    avg_cpc = float(row.get("CPC", 0) or 0)
    actual_roas = float(row.get("ROAS", 0) or 0)
    target_roas = config.get("TARGET_ROAS", 2.5)
    
    base_bid = current_bid if current_bid > 0 else avg_cpc
    if base_bid <= 0:
        return 0.5, "Hold: No Bid/CPC Data"
    
    if actual_roas > 0:
        delta = (actual_roas - target_roas) / target_roas
        delta = max(-0.5, min(1.0, delta))
        adjustment = alpha * delta
        
        if delta > 0:
            reason = f"↑ ROAS {actual_roas:.2f} > {target_roas:.2f}"
        else:
            reason = f"↓ ROAS {actual_roas:.2f} < {target_roas:.2f}"
    else:
        return base_bid, "Hold: Low Data"
    
    adjustment = max(-config.get("MAX_BID_CHANGE", 0.25), min(config.get("MAX_BID_CHANGE", 0.25), adjustment))
    new_bid = base_bid * (1 + adjustment)
    
    if avg_cpc > 0:
        safety_cap = avg_cpc * 2.0
        if new_bid > safety_cap:
            new_bid = safety_cap
            reason += " (Capped)"
    
    return new_bid, reason


# ==========================================
# HEATMAP WITH ACTION TRACKING
# ==========================================

def create_heatmap(
    df: pd.DataFrame,
    config: dict,
    harvest_df: pd.DataFrame,
    neg_kw: pd.DataFrame,
    neg_pt: pd.DataFrame,
    direct_bids: pd.DataFrame,
    agg_bids: pd.DataFrame
) -> pd.DataFrame:
    """
    Create performance heatmap with action tracking.
    Shows which campaigns/ad groups need attention and what optimizer is doing.
    """
    # Group by Campaign and Ad Group
    grouped = df.groupby(["Campaign Name", "Ad Group Name"]).agg({
        "Clicks": "sum",
        "Spend": "sum",
        "Sales_Attributed": "sum",
        "Orders_Attributed": "sum",
        "Impressions": "sum"
    }).reset_index()
    
    # Calculate metrics
    grouped["CTR"] = np.where(grouped["Impressions"] > 0, 
                               grouped["Clicks"] / grouped["Impressions"] * 100, 0)
    grouped["CVR"] = np.where(grouped["Clicks"] > 0, 
                               grouped["Orders_Attributed"] / grouped["Clicks"] * 100, 0)
    grouped["ROAS"] = np.where(grouped["Spend"] > 0, 
                                grouped["Sales_Attributed"] / grouped["Spend"], 0)
    grouped["ACoS"] = np.where(grouped["Sales_Attributed"] > 0, 
                                grouped["Spend"] / grouped["Sales_Attributed"] * 100, 999)
    
    # Initialize action tracking columns
    grouped["Harvest_Count"] = 0
    grouped["Negative_Count"] = 0
    grouped["Bid_Increase_Count"] = 0
    grouped["Bid_Decrease_Count"] = 0
    grouped["Actions_Taken"] = ""
    
    # Count harvest actions per campaign/ad group
    if not harvest_df.empty:
        for idx, row in grouped.iterrows():
            camp, ag = row["Campaign Name"], row["Ad Group Name"]
            match = harvest_df[
                (harvest_df["Campaign Name"] == camp) &
                (harvest_df.get("Ad Group Name", pd.Series([""])).fillna("") == ag)
            ]
            if not match.empty:
                grouped.at[idx, "Harvest_Count"] = len(match)
    
    # Count negative actions
    negatives_df = pd.concat([neg_kw, neg_pt]) if not neg_kw.empty or not neg_pt.empty else pd.DataFrame()
    if not negatives_df.empty:
        for idx, row in grouped.iterrows():
            camp, ag = row["Campaign Name"], row["Ad Group Name"]
            match = negatives_df[
                (negatives_df["Campaign Name"] == camp) &
                (negatives_df.get("Ad Group Name", pd.Series([""])).fillna("") == ag)
            ]
            if not match.empty:
                grouped.at[idx, "Negative_Count"] = len(match)
    
    # Count bid actions
    all_bids = pd.concat([direct_bids, agg_bids]) if not direct_bids.empty or not agg_bids.empty else pd.DataFrame()
    if not all_bids.empty and "New Bid" in all_bids.columns:
        for idx, row in grouped.iterrows():
            camp, ag = row["Campaign Name"], row["Ad Group Name"]
            match = all_bids[
                (all_bids["Campaign Name"] == camp) &
                (all_bids.get("Ad Group Name", pd.Series([""])).fillna("") == ag)
            ]
            if not match.empty and "Cost Per Click (CPC)" in match.columns:
                increases = match[match["New Bid"] > match["Cost Per Click (CPC)"]]
                decreases = match[match["New Bid"] < match["Cost Per Click (CPC)"]]
                grouped.at[idx, "Bid_Increase_Count"] = len(increases)
                grouped.at[idx, "Bid_Decrease_Count"] = len(decreases)
    
    # Build action summary text
    for idx, row in grouped.iterrows():
        actions = []
        
        if row["Harvest_Count"] > 0:
            actions.append(f"💎 {int(row['Harvest_Count'])} harvests")
        if row["Negative_Count"] > 0:
            actions.append(f"🛑 {int(row['Negative_Count'])} negatives")
        if row["Bid_Increase_Count"] > 0:
            actions.append(f"⬆️ {int(row['Bid_Increase_Count'])} bid increases")
        if row["Bid_Decrease_Count"] > 0:
            actions.append(f"⬇️ {int(row['Bid_Decrease_Count'])} bid decreases")
        
        is_low_volume = (
            row["Clicks"] < config.get("MIN_CLICKS_BID", 3) or 
            row["Orders_Attributed"] < 2
        )
        
        if actions:
            grouped.at[idx, "Actions_Taken"] = " | ".join(actions)
        elif is_low_volume:
            grouped.at[idx, "Actions_Taken"] = "⏸️ Hold (Low volume)"
        else:
            grouped.at[idx, "Actions_Taken"] = "✅ No action needed"
    
    # Calculate color scores using percentiles
    def get_color_score(value, series, higher_is_better=True):
        if pd.isna(value) or value == 0:
            return -1
        
        valid = series[series > 0]
        if len(valid) < 2:
            return 1
        
        p33, p67 = valid.quantile(0.33), valid.quantile(0.67)
        
        if higher_is_better:
            if value >= p67: return 2  # Green
            elif value >= p33: return 1  # Yellow
            else: return 0  # Red
        else:
            if value <= p33: return 2  # Green
            elif value <= p67: return 1  # Yellow
            else: return 0  # Red
    
    grouped["CTR_Score"] = grouped["CTR"].apply(lambda x: get_color_score(x, grouped["CTR"], True))
    grouped["CVR_Score"] = grouped["CVR"].apply(lambda x: get_color_score(x, grouped["CVR"], True))
    grouped["ROAS_Score"] = grouped["ROAS"].apply(lambda x: get_color_score(x, grouped["ROAS"], True))
    grouped["ACoS_Score"] = grouped["ACoS"].apply(lambda x: get_color_score(x, grouped["ACoS"], False))
    
    grouped["Overall_Score"] = (
        grouped["CTR_Score"] + grouped["CVR_Score"] + 
        grouped["ROAS_Score"] + grouped["ACoS_Score"]
    ) / 4
    
    # Priority classification
    grouped["Priority"] = grouped["Overall_Score"].apply(
        lambda x: "🔴 High" if x < 0.7 else ("🟡 Medium" if x < 1.3 else "🟢 Good")
    )
    
    # Sort worst first
    grouped = grouped.sort_values("Overall_Score", ascending=True)
    
    return grouped
# ==========================================
# SIMULATION & FORECASTING
# ==========================================

def run_simulation(
    df: pd.DataFrame,
    direct_bids: pd.DataFrame,
    agg_bids: pd.DataFrame,
    harvest_df: pd.DataFrame,
    config: dict,
    date_info: dict
) -> dict:
    """
    Simulate the impact of proposed bid changes on future performance.
    Uses elasticity model with scenario analysis.
    """
    num_weeks = date_info.get("weeks", 1.0)
    
    # Calculate current baseline (raw)
    current_raw = _calculate_baseline(df)
    current = _normalize_to_weekly(current_raw, num_weeks)
    
    # Combine bid changes
    all_bids = pd.concat([direct_bids, agg_bids]) if not direct_bids.empty or not agg_bids.empty else pd.DataFrame()
    
    if not all_bids.empty:
        all_bids = all_bids.copy()
        
        # Safely extract CPC and New Bid
        if "Cost Per Click (CPC)" in all_bids.columns:
            all_bids["CPC"] = pd.to_numeric(all_bids["Cost Per Click (CPC)"], errors="coerce").fillna(0)
        else:
            all_bids["CPC"] = pd.to_numeric(all_bids.get("CPC", 0), errors="coerce").fillna(0) if "CPC" in all_bids.columns else 0.0
            
        if "New Bid" in all_bids.columns:
             all_bids["New Bid"] = pd.to_numeric(all_bids["New Bid"], errors="coerce").fillna(0)
        else:
             all_bids["New Bid"] = 0.0
        all_bids["Bid_Change_Pct"] = np.where(
            all_bids["CPC"] > 0,
            (all_bids["New Bid"] - all_bids["CPC"]) / all_bids["CPC"],
            0
        )
    
    # Count recommendations
    total_recs = len(all_bids)
    hold_count = 0
    actual_changes = 0
    
    if not all_bids.empty and "Reason" in all_bids.columns:
        hold_mask = all_bids["Reason"].astype(str).str.contains("Hold", case=False, na=False)
        hold_count = hold_mask.sum()
        actual_changes = (~hold_mask).sum()
    
    # Run scenarios
    scenarios = {}
    for name, elasticity in ELASTICITY_SCENARIOS.items():
        forecast_raw = _forecast_scenario(all_bids, harvest_df, elasticity, current_raw, config)
        forecast = _normalize_to_weekly(forecast_raw, num_weeks)
        scenarios[name] = forecast
    
    scenarios["current"] = current
    
    # Calculate sensitivity
    sensitivity_df = _calculate_sensitivity(all_bids, harvest_df, ELASTICITY_SCENARIOS["expected"], current_raw, config, num_weeks)
    
    # Analyze risks
    risk_analysis = _analyze_risks(all_bids)
    
    return {
        "scenarios": scenarios,
        "sensitivity": sensitivity_df,
        "risk_analysis": risk_analysis,
        "date_info": date_info,
        "diagnostics": {
            "total_recommendations": total_recs,
            "actual_changes": actual_changes,
            "hold_count": hold_count,
            "harvest_count": len(harvest_df)
        }
    }


def _calculate_baseline(df: pd.DataFrame) -> dict:
    """Calculate current performance baseline."""
    total_clicks = df["Clicks"].sum()
    total_spend = df["Spend"].sum()
    total_sales = df["Sales_Attributed"].sum() if "Sales_Attributed" in df.columns else df["Sales"].sum()
    total_orders = df["Orders_Attributed"].sum() if "Orders_Attributed" in df.columns else df["Orders"].sum()
    total_impressions = df["Impressions"].sum() if "Impressions" in df.columns else 0
    
    return {
        "clicks": total_clicks,
        "spend": total_spend,
        "sales": total_sales,
        "orders": total_orders,
        "impressions": total_impressions,
        "cpc": total_spend / total_clicks if total_clicks > 0 else 0,
        "cvr": total_orders / total_clicks if total_clicks > 0 else 0,
        "roas": total_sales / total_spend if total_spend > 0 else 0,
        "acos": (total_spend / total_sales * 100) if total_sales > 0 else 0,
        "ctr": (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    }


def _normalize_to_weekly(metrics: dict, num_weeks: float) -> dict:
    """Normalize metrics to weekly averages."""
    if num_weeks <= 0:
        num_weeks = 1.0
    
    return {
        "clicks": metrics["clicks"] / num_weeks,
        "spend": metrics["spend"] / num_weeks,
        "sales": metrics["sales"] / num_weeks,
        "orders": metrics["orders"] / num_weeks,
        "impressions": metrics.get("impressions", 0) / num_weeks,
        "cpc": metrics.get("cpc", 0),
        "cvr": metrics.get("cvr", 0),
        "roas": metrics.get("roas", 0),
        "acos": metrics.get("acos", 0),
        "ctr": metrics.get("ctr", 0)
    }


def _forecast_scenario(
    bid_changes: pd.DataFrame,
    harvest_df: pd.DataFrame,
    elasticity: dict,
    baseline: dict,
    config: dict
) -> dict:
    """Forecast performance for a single scenario."""
    forecasted_changes = []
    
    # Part 1: Process bid changes
    if not bid_changes.empty:
        for _, row in bid_changes.iterrows():
            bid_change_pct = row.get("Bid_Change_Pct", 0)
            reason = str(row.get("Reason", "")).lower()
            
            # Skip holds
            if "hold" in reason or abs(bid_change_pct) < 0.005:
                continue
            
            current_clicks = float(row.get("Clicks", 0) or 0)
            current_spend = float(row.get("Spend", 0) or 0)
            current_orders = float(row.get("Orders_Attributed", 0) or 0)
            current_sales = float(row.get("Sales_Attributed", 0) or 0)
            current_cpc = float(row.get("CPC", 0) or row.get("Cost Per Click (CPC)", 0) or 0)
            
            if current_clicks == 0 and current_cpc == 0:
                continue
            
            current_cvr = current_orders / current_clicks if current_clicks > 0 else 0
            current_aov = current_sales / current_orders if current_orders > 0 else 0
            
            if current_aov == 0 and baseline["orders"] > 0:
                current_aov = baseline["sales"] / baseline["orders"]
            
            # Apply elasticity
            new_cpc = current_cpc * (1 + elasticity["cpc"] * bid_change_pct)
            new_clicks = current_clicks * (1 + elasticity["clicks"] * bid_change_pct)
            new_cvr = current_cvr * (1 + elasticity["cvr"] * bid_change_pct)
            
            new_orders = new_clicks * new_cvr
            new_sales = new_orders * current_aov
            new_spend = new_clicks * new_cpc
            
            forecasted_changes.append({
                "delta_clicks": new_clicks - current_clicks,
                "delta_spend": new_spend - current_spend,
                "delta_sales": new_sales - current_sales,
                "delta_orders": new_orders - current_orders
            })
    
    # Part 2: Process harvest campaigns
    if not harvest_df.empty:
        efficiency = config.get("HARVEST_EFFICIENCY_MULTIPLIER", 1.15)
        
        for _, row in harvest_df.iterrows():
            base_clicks = float(row.get("Clicks", 0) or 0)
            base_spend = float(row.get("Spend", 0) or 0)
            base_orders = float(row.get("Orders", 0) or 0)
            base_sales = float(row.get("Sales", 0) or 0)
            base_cpc = float(row.get("CPC", 0) or 0)
            
            if base_clicks < 5:
                continue
            
            new_bid = float(row.get("New Bid", base_cpc * 1.1) or base_cpc * 1.1)
            base_cvr = base_orders / base_clicks if base_clicks > 0 else 0
            base_aov = base_sales / base_orders if base_orders > 0 else 0
            
            # Harvest: same traffic, better efficiency
            fore_clicks = base_clicks
            fore_cpc = new_bid * 0.95
            fore_cvr = base_cvr * efficiency
            
            fore_orders = fore_clicks * fore_cvr
            fore_sales = fore_orders * base_aov
            fore_spend = fore_clicks * fore_cpc
            
            forecasted_changes.append({
                "delta_clicks": fore_clicks - base_clicks,
                "delta_spend": fore_spend - base_spend,
                "delta_sales": fore_sales - base_sales,
                "delta_orders": fore_orders - base_orders
            })
    
    # Aggregate changes
    if not forecasted_changes:
        return baseline.copy()
    
    total_delta = {
        "clicks": sum(fc["delta_clicks"] for fc in forecasted_changes),
        "spend": sum(fc["delta_spend"] for fc in forecasted_changes),
        "sales": sum(fc["delta_sales"] for fc in forecasted_changes),
        "orders": sum(fc["delta_orders"] for fc in forecasted_changes)
    }
    
    new_clicks = max(0, baseline["clicks"] + total_delta["clicks"])
    new_spend = max(0, baseline["spend"] + total_delta["spend"])
    new_sales = max(0, baseline["sales"] + total_delta["sales"])
    new_orders = max(0, baseline["orders"] + total_delta["orders"])
    
    return {
        "clicks": new_clicks,
        "spend": new_spend,
        "sales": new_sales,
        "orders": new_orders,
        "impressions": baseline.get("impressions", 0),
        "cpc": new_spend / new_clicks if new_clicks > 0 else 0,
        "cvr": new_orders / new_clicks if new_clicks > 0 else 0,
        "roas": new_sales / new_spend if new_spend > 0 else 0,
        "acos": (new_spend / new_sales * 100) if new_sales > 0 else 0,
        "ctr": baseline.get("ctr", 0)
    }


def _calculate_sensitivity(
    bid_changes: pd.DataFrame,
    harvest_df: pd.DataFrame,
    elasticity: dict,
    baseline: dict,
    config: dict,
    num_weeks: float
) -> pd.DataFrame:
    """Calculate sensitivity analysis at different bid adjustment levels."""
    adjustments = ["-30%", "-20%", "-10%", "+0%", "+10%", "+20%", "+30%"]
    multipliers = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    results = []
    for adj, mult in zip(adjustments, multipliers):
        # Scale bid changes
        if not bid_changes.empty:
            scaled_bids = bid_changes.copy()
            scaled_bids["Bid_Change_Pct"] = scaled_bids["Bid_Change_Pct"] * mult
        else:
            scaled_bids = pd.DataFrame()
        
        forecast = _forecast_scenario(scaled_bids, harvest_df, elasticity, baseline, config)
        normalized = _normalize_to_weekly(forecast, num_weeks)
        
        results.append({
            "Bid_Adjustment": adj,
            "Spend": normalized["spend"],
            "Sales": normalized["sales"],
            "ROAS": normalized["roas"],
            "Orders": normalized["orders"],
            "ACoS": normalized["acos"]
        })
    
    return pd.DataFrame(results)


def _analyze_risks(bid_changes: pd.DataFrame) -> dict:
    """Analyze risks in proposed bid changes."""
    if bid_changes.empty:
        return {"summary": {"high_risk_count": 0, "medium_risk_count": 0, "low_risk_count": 0}, "high_risk": []}
    
    high_risk = []
    medium_risk = 0
    low_risk = 0
    
    for _, row in bid_changes.iterrows():
        reason = str(row.get("Reason", "")).lower()
        if "hold" in reason:
            continue
        
        bid_change = row.get("Bid_Change_Pct", 0)
        clicks = row.get("Clicks", 0)
        
        risk_factors = []
        
        # Large bid change
        if abs(bid_change) > 0.25:
            risk_factors.append(f"Large change ({bid_change*100:+.0f}%)")
        
        # Low data
        if clicks < 10:
            risk_factors.append(f"Low data ({clicks} clicks)")
        
        # Classify
        if len(risk_factors) >= 2 or abs(bid_change) > 0.40:
            high_risk.append({
                "keyword": row.get("Targeting", row.get("Customer Search Term", "")),
                "campaign": row.get("Campaign Name", ""),
                "bid_change": f"{bid_change*100:+.0f}%",
                "current_bid": row.get("CPC", row.get("Cost Per Click (CPC)", 0)),
                "new_bid": row.get("New Bid", 0),
                "reasons": ", ".join(risk_factors) if risk_factors else "Large adjustment"
            })
        elif risk_factors:
            medium_risk += 1
        else:
            low_risk += 1
    
    return {
        "summary": {
            "high_risk_count": len(high_risk),
            "medium_risk_count": medium_risk,
            "low_risk_count": low_risk
        },
        "high_risk": high_risk
    }


# ==========================================
# BULK FILE GENERATION
# ==========================================

def generate_negatives_bulk(neg_kw: pd.DataFrame, neg_pt: pd.DataFrame) -> pd.DataFrame:
    """Generate Amazon bulk upload file for negatives."""
    rows = []
    
    for _, row in pd.concat([neg_kw, neg_pt]).iterrows():
        is_asin_neg = row.get("Is_ASIN", False) or "asin=" in str(row.get("Term", "")).lower()
        
        row_data = {
            "Product": "Sponsored Products",
            "Operation": "create",
            "Campaign Name": row.get("Campaign Name", ""),
            "Campaign Id": row.get("CampaignId", ""),
            "State": "enabled"
        }
        
        ad_group = row.get("Ad Group Name", "")
        ad_group_id = row.get("AdGroupId", "")
        
        if ad_group and str(ad_group_id):
            row_data["Ad Group Name"] = ad_group
            row_data["Ad Group Id"] = ad_group_id
            row_data["Entity"] = "Negative Product Targeting" if is_asin_neg else "Negative Keyword"
        else:
            row_data["Entity"] = "Campaign Negative Product Targeting" if is_asin_neg else "Campaign Negative Keyword"
        
        if is_asin_neg:
            term = row.get("Term", "")
            if is_asin(term) and not term.lower().startswith("asin="):
                row_data["Product Targeting Expression"] = f'asin="{term}"'
            else:
                row_data["Product Targeting Expression"] = term
                
            row_data["Product Targeting Id"] = str(row.get("TargetingId", "")).strip()
            row_data["Match Type"] = ""
        else:
            row_data["Keyword Text"] = row.get("Term", "")
            row_data["Keyword Id"] = str(row.get("KeywordId", "")).strip()
            row_data["Match Type"] = "negativeExact"
        
        rows.append(row_data)
    
    df_out = pd.DataFrame(rows)
    for col in BULK_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = ""
    
    return df_out[BULK_COLUMNS] if not df_out.empty else pd.DataFrame(columns=BULK_COLUMNS)


def generate_bids_bulk(bids_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Generate Amazon bulk upload file for bid updates."""
    rows = []
    skipped = 0
    
    # Auto targeting types
    AUTO_TYPES = {'close-match', 'loose-match', 'substitutes', 'complements'}
    
    for _, row in bids_df.iterrows():
        if "Hold" in str(row.get("Reason", "")):
            skipped += 1
            continue
        
        target_text = str(row.get("Targeting", "")).strip()
        match_type = str(row.get("Match Type", "")).strip()
        
        # Determine Entity Type and Fields
        # 1. Manual PT (High priority)
        is_pt_manual = (
            target_text.lower().startswith("asin=") or 
            target_text.lower().startswith("category=") or
            is_asin(target_text)
        )
        
        # 2. Auto Targeting
        # Check standard types, generic 'auto', or if target text itself is an auto type
        is_auto_specific = match_type in AUTO_TYPES
        is_auto_generic = (match_type.lower() == 'auto')
        is_target_auto = target_text in AUTO_TYPES
        
        is_auto = (is_auto_specific or is_target_auto or is_auto_generic) and not is_pt_manual
        
        row_data = {
            "Product": "Sponsored Products",
            "Entity": "",
            "Operation": "update",
            "Campaign Id": row.get("CampaignId", ""),
            "Ad Group Id": row.get("AdGroupId", ""),
            "Campaign Name": row.get("Campaign Name", ""),
            "Ad Group Name": row.get("Ad Group Name", ""),
            "Bid": f"{row.get('New Bid', 0):.2f}",
            "State": "enabled",
            # Initialize empty fields
            "Keyword Id": "",
            "Product Targeting Id": "",
            "Keyword Text": "",
            "Match Type": "",
            "Product Targeting Expression": ""
        }
        
        if is_auto:
            # Auto Targeting
            row_data["Entity"] = "Product Targeting"
            row_data["Product Targeting Id"] = str(row.get("TargetingId", "")).strip()
            
            # Determine expression
            expression = ""
            if match_type in AUTO_TYPES:
                expression = match_type
            elif target_text in AUTO_TYPES:
                expression = target_text
            else:
                # Fallback: Try to get from preserved expression column
                fallback = str(row.get("Product Targeting Expression", "") or row.get("TargetingExpression", "")).strip()
                if fallback:
                    expression = fallback
                else:
                    # Last resort: if we have valid ID, maybe safe to leave blank?
                    if target_text.lower() == 'auto':
                         pass 
                    else:
                         expression = target_text # Use whatever text we have

            row_data["Product Targeting Expression"] = expression
            # Match Type and Keyword Text remain empty
            
        elif is_pt_manual:
            # Manual Product Targeting
            row_data["Entity"] = "Product Targeting"
            row_data["Product Targeting Id"] = str(row.get("TargetingId", "")).strip()
            
            # Format Expression: ensure asin="B0..." if it's a raw ASIN
            if is_asin(target_text) and not target_text.lower().startswith("asin="):
                row_data["Product Targeting Expression"] = f'asin="{target_text}"'
            else:
                row_data["Product Targeting Expression"] = target_text
            # Match Type and Keyword Text remain empty
            
        else:
            # Keyword Targeting (Broad/Phrase/Exact)
            row_data["Entity"] = "Keyword"
            row_data["Keyword Id"] = str(row.get("KeywordId", "")).strip()
            row_data["Keyword Text"] = target_text
            row_data["Match Type"] = match_type
            # Product Targeting Expression remains empty

        rows.append(row_data)
    
    df_out = pd.DataFrame(rows)
    
    # Ensure all standard columns exist
    for col in BULK_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = ""
            
    # Return with correct column order
    return (df_out[BULK_COLUMNS] if not df_out.empty else pd.DataFrame(columns=BULK_COLUMNS)), skipped


# ==========================================
# ACTION LOGGING
# ==========================================

def _log_optimization_events(results: dict, client_id: str = "default_client", report_date: str = None) -> int:
    """
    Parse optimizer results and log actions to database.
    
    Parses:
    - direct_bids, agg_bids -> BID_UPDATE actions
    - neg_kw -> NEGATIVE (keyword) actions
    - neg_pt -> NEGATIVE (ASIN) actions
    - harvest -> HARVEST actions
    
    Args:
        results: Optimizer results dictionary
        client_id: Client identifier
        report_date: The report's start date (for time-lag matching). If None, uses current time.
        
    Returns:
        Number of actions logged
    """
    db_manager = st.session_state.get('db_manager')
    if db_manager is None:
        st.error("❌ DB Manager missing in logging!")
        return 0
    
    actions = []
    
    # 1. BID_UPDATE actions from direct_bids
    direct_bids = results.get('direct_bids', pd.DataFrame())
    
    # DEBUG LOG
    # st.warning(f"DEBUG: Processing {len(direct_bids)} bid rows for logging...")
    
    if not direct_bids.empty:
        count_skipped = 0
        for _, row in direct_bids.iterrows():
            reason = row.get('Reason', '')
            if isinstance(reason, str) and reason.startswith('Hold'):
                count_skipped += 1
                continue  # Skip holds
            
            actions.append({
                'entity_name': 'Keyword' if row.get('Match Type', '') in ['exact', 'broad', 'phrase', 'EXACT', 'BROAD', 'PHRASE'] else 'Target',
                'action_type': 'BID_UPDATE',
                'old_value': row.get('Current Bid', row.get('Cost Per Click (CPC)', '')),
                'new_value': row.get('New Bid', ''),
                'reason': reason,
                'campaign_name': row.get('Campaign Name', ''),
                'ad_group_name': row.get('Ad Group Name', ''),
                'target_text': row.get('Targeting', ''),
                'match_type': row.get('Match Type', '')
            })
    
    # 2. BID_UPDATE actions from agg_bids
    agg_bids = results.get('agg_bids', pd.DataFrame())
    if not agg_bids.empty:
        for _, row in agg_bids.iterrows():
            if row.get('Reason', '').startswith('Hold'):
                continue
            
            actions.append({
                'entity_name': 'Target',
                'action_type': 'BID_UPDATE',
                'old_value': row.get('Current Bid', row.get('Cost Per Click (CPC)', '')),
                'new_value': row.get('New Bid', ''),
                'reason': row.get('Reason', ''),
                'campaign_name': row.get('Campaign Name', ''),
                'ad_group_name': row.get('Ad Group Name', ''),
                'target_text': row.get('Targeting', ''),
                'match_type': row.get('Match Type', '')
            })
    
    # 3. NEGATIVE actions from neg_kw
    neg_kw = results.get('neg_kw', pd.DataFrame())
    if not neg_kw.empty:
        for _, row in neg_kw.iterrows():
            actions.append({
                'entity_name': 'Keyword',
                'action_type': 'NEGATIVE',
                'old_value': 'active',
                'new_value': 'negative',
                'reason': row.get('Type', 'bleeder'),
                'campaign_name': row.get('Campaign Name', ''),
                'ad_group_name': row.get('Ad Group Name', ''),
                'target_text': row.get('Customer Search Term', row.get('Term', '')),
                'match_type': row.get('Match Type', '')
            })
    
    # 4. NEGATIVE actions from neg_pt (ASINs)
    neg_pt = results.get('neg_pt', pd.DataFrame())
    if not neg_pt.empty:
        for _, row in neg_pt.iterrows():
            actions.append({
                'entity_name': 'ASIN',
                'action_type': 'NEGATIVE',
                'old_value': 'active',
                'new_value': 'negative',
                'reason': row.get('Type', 'competitor'),
                'campaign_name': row.get('Campaign Name', ''),
                'ad_group_name': row.get('Ad Group Name', ''),
                'target_text': row.get('Term', row.get('Customer Search Term', '')),
                'match_type': 'PT'
            })
    
    # 5. HARVEST actions
    harvest = results.get('harvest', pd.DataFrame())
    if not harvest.empty:
        for _, row in harvest.iterrows():
            actions.append({
                'entity_name': 'Keyword',
                'action_type': 'HARVEST',
                'old_value': row.get('Match Type', 'broad'),
                'new_value': 'exact',
                'reason': f"ROAS: {row.get('ROAS', 0):.2f}",
                'campaign_name': row.get('Winner Campaign', row.get('Campaign Name', '')),
                'ad_group_name': row.get('Ad Group Name', ''),
                'target_text': row.get('Customer Search Term', ''),
            'match_type': 'exact'
            })
    
    # Log all actions
    if actions:
        # Pass None for batch_id to let DB manager generate it
        logged = db_manager.log_action_batch(actions, client_id, batch_id=None, action_date=report_date)
        return logged
    else:
        return 0


# ==========================================
# STREAMLIT UI MODULE
# ==========================================

class OptimizerModule(BaseFeature):
    """
    Complete Bid Optimization Engine with:
    - Harvest detection
    - Isolation & performance negatives
    - Bid optimization (direct + aggregated)
    - Heatmap with action tracking
    - Advanced simulation & forecasting
    """
    
    def __init__(self):
        super().__init__()
        
        # Check if optimizer_config exists in session state (set by OH panel)
        if 'optimizer_config' in st.session_state:
            self.config = st.session_state['optimizer_config'].copy()  # Use OH values!
        else:
            self.config = DEFAULT_CONFIG.copy()  # Fallback to defaults
        
        self.results = {}
        
        # Initialize session state with defaults if not set (once per session)
        # Check if optimizer_config exists in session state to use those values
        if 'optimizer_config' in st.session_state:
            config_source = st.session_state['optimizer_config']  # Use OH values!
        else:
            config_source = DEFAULT_CONFIG
        
        widget_defaults = {
            "opt_harvest_clicks": config_source.get("HARVEST_CLICKS", DEFAULT_CONFIG["HARVEST_CLICKS"]),
            "opt_harvest_orders": config_source.get("HARVEST_ORDERS", DEFAULT_CONFIG["HARVEST_ORDERS"]),
            "opt_harvest_sales": config_source.get("HARVEST_SALES", DEFAULT_CONFIG["HARVEST_SALES"]),
            "opt_harvest_roas_mult": config_source.get("HARVEST_ROAS_MULT", DEFAULT_CONFIG["HARVEST_ROAS_MULT"]),
            "opt_alpha_exact": config_source.get("ALPHA_EXACT", DEFAULT_CONFIG["ALPHA_EXACT"]),
            "opt_alpha_broad": config_source.get("ALPHA_BROAD", DEFAULT_CONFIG["ALPHA_BROAD"]),
            "opt_max_bid_change": config_source.get("MAX_BID_CHANGE", DEFAULT_CONFIG["MAX_BID_CHANGE"]),
            "opt_min_clicks_bid": config_source.get("MIN_CLICKS_BID", DEFAULT_CONFIG["MIN_CLICKS_BID"]),
            "opt_target_roas": config_source.get("TARGET_ROAS", DEFAULT_CONFIG["TARGET_ROAS"]),
            "opt_neg_clicks_threshold": config_source.get("NEGATIVE_CLICKS_THRESHOLD", DEFAULT_CONFIG["NEGATIVE_CLICKS_THRESHOLD"]),
            "opt_neg_spend_threshold": config_source.get("NEGATIVE_SPEND_THRESHOLD", DEFAULT_CONFIG["NEGATIVE_SPEND_THRESHOLD"]),
            "opt_run_simulation": True,
        }
        for key, default in widget_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default
    
    def render_ui(self):
        """Render the feature UI."""
        self.render_header("PPC Optimizer", "optimizer")
        
        # CHECKPOINT: Ensure results belong to active account
        if 'optimizer_results' in st.session_state:
            results = st.session_state['optimizer_results']
            results_account = results.get('account_id')
            active_account = st.session_state.get('active_account_id')
            
            # If account changed or results have no account ID (legacy), CLEAR IT
            if active_account and results_account != active_account:
                st.toast(f"🔄 Clearing data for new account: {active_account}", icon="🧹")
                if 'optimizer_results' in st.session_state: del st.session_state['optimizer_results']
                if 'latest_optimizer_run' in st.session_state: del st.session_state['latest_optimizer_run']
                if 'run_optimizer' in st.session_state: del st.session_state['run_optimizer']
                st.rerun()

        # Get enriched data
        hub = DataHub()
        if not hub.is_loaded("search_term_report"):
            st.warning("⚠️ Please upload a Search Term Report in the Data Hub first.")
            st.info("Go to **Data Hub** → Upload files → Return here")
            return
        
        # Get enriched data
        df = hub.get_enriched_data()
        if df is None:
            df = hub.get_data("search_term_report")
        
        # Sidebar configuration (Always render to allow changing settings)
        self._render_sidebar()
        
        # Share config globally
        st.session_state['optimizer_config'] = self.config
        
        # Logic Flow:
        # 1. Triggered? -> Run Analysis -> Reset Trigger -> Rerun
        # 2. Results Cached? -> Display Cached Results
        # 3. Default -> Show Summary
        
        if st.session_state.get("run_optimizer"):
            self._run_analysis(df)
            # CRITICAL FIX: Reset explicit trigger so it doesn't auto-run on module swap
            st.session_state["run_optimizer"] = False
            # st.rerun()  <-- REMOVED: Do not rerun, just display results inline to preserve widget state
            if self.results:
                 self._display_results()
            
        elif 'optimizer_results' in st.session_state:
            self.results = st.session_state['optimizer_results']
            self._display_results()
            
        else:
            self._display_summary(df)
            st.info("👈 Configure settings and click **Run Optimization** to start")
    
    def _render_sidebar(self):
        """Render sidebar configuration panels."""
        with st.sidebar:
            st.divider()
            st.markdown("##### OPTIMIZER SETTINGS")
            
            # Harvest Thresholds
            with st.expander("Harvest Thresholds", expanded=False):
                st.number_input(
                    "Min Clicks", min_value=1,
                    help="Minimum total clicks a search term needs to be considered for harvesting.",
                    key="opt_harvest_clicks"
                )
                st.number_input(
                    "Min Orders", min_value=1,
                    help="Minimum total orders required to validate a term.",
                    key="opt_harvest_orders"
                )
                st.number_input(
                    "Min Sales ($)", min_value=0.0, step=10.0,
                    help="Minimum sales revenue required.",
                    key="opt_harvest_sales"
                )
                st.number_input(
                    "ROAS vs Median (Multiplier)", 
                    min_value=0.5, max_value=2.0, step=0.1,
                    help="Term ROAS must be this many times better than the median to be harvested (e.g. 1.2x).",
                    key="opt_harvest_roas_mult"
                )
            
            # Bid Optimization
            with st.expander("Bid Optimization", expanded=False):
                st.slider(
                    "Alpha (Exact/PT)", min_value=0.05, max_value=0.50, step=0.05,
                    help="Aggressiveness of bid changes for Exact Match. Higher = larger bid jumps toward target.",
                    key="opt_alpha_exact"
                )
                st.slider(
                    "Alpha (Broad/Agg)", min_value=0.05, max_value=0.50, step=0.05,
                    help="Aggressiveness for Broad/Phrase match types. Usually lower to be conservative.",
                    key="opt_alpha_broad"
                )
                st.slider(
                    "Max Change %", min_value=0.05, max_value=0.50, step=0.05,
                    help="Hard cap on percentage change per optimization run (e.g. 0.20 = max 20% change).",
                    key="opt_max_bid_change"
                )
                st.number_input(
                    "Min Clicks for Bid", min_value=1,
                    key="opt_min_clicks_bid"
                )
                st.number_input(
                    "Default Target ROAS (if 0)", 
                    min_value=0.5, max_value=10.0, step=0.1,
                    help="Fallback Target ROAS for campaigns with no calculated target history.",
                    key="opt_target_roas"
                )
            
            # Negative Thresholds
            with st.expander("Negative Thresholds", expanded=False):
                st.number_input(
                    "Min Clicks (0 Sales)", min_value=5,
                    key="opt_neg_clicks_threshold"
                )
                st.number_input(
                    "Min Spend (0 Sales)", min_value=1.0,
                    key="opt_neg_spend_threshold"
                )
            
            st.checkbox("Include Simulation", key="opt_run_simulation")
            
            # Sync session state values to self.config for use in analysis
            self.config["HARVEST_CLICKS"] = st.session_state["opt_harvest_clicks"]
            self.config["HARVEST_ORDERS"] = st.session_state["opt_harvest_orders"]
            self.config["HARVEST_SALES"] = st.session_state["opt_harvest_sales"]
            self.config["HARVEST_ROAS_MULT"] = st.session_state["opt_harvest_roas_mult"]
            self.config["ALPHA_EXACT"] = st.session_state["opt_alpha_exact"]
            self.config["ALPHA_BROAD"] = st.session_state["opt_alpha_broad"]
            self.config["MAX_BID_CHANGE"] = st.session_state["opt_max_bid_change"]
            self.config["MIN_CLICKS_BID"] = st.session_state["opt_min_clicks_bid"]
            self.config["TARGET_ROAS"] = st.session_state["opt_target_roas"]
            self.config["NEGATIVE_CLICKS_THRESHOLD"] = st.session_state["opt_neg_clicks_threshold"]
            self.config["NEGATIVE_SPEND_THRESHOLD"] = st.session_state["opt_neg_spend_threshold"]
            
            # Teal button styling
            st.markdown("""
            <style>
            [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%) !important;
                border: none !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            if st.button("Run Optimization", type="primary", use_container_width=True, key="opt_run_btn"):
                st.session_state["run_optimizer"] = True
                st.rerun()  # Rerun to trigger analysis with current session state values
    
    def _display_summary(self, df: pd.DataFrame):
        """Display data summary metrics."""
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Search Terms", f"{len(df):,}")
        c2.metric("Total Spend", format_currency(df["Spend"].sum() if "Spend" in df.columns else 0))
        c3.metric("Total Sales", format_currency(df["Sales"].sum() if "Sales" in df.columns else 0))
        
        spend = df["Spend"].sum() if "Spend" in df.columns else 0
        sales = df["Sales"].sum() if "Sales" in df.columns else 0
        roas = sales / spend if spend > 0 else 0
        c4.metric("ROAS", f"{roas:.2f}x")
    
    def _run_analysis(self, df: pd.DataFrame):
        """Execute the full optimization pipeline."""
        with st.spinner("Running Optimization Engine..."):
            # Prepare data
            df, date_info = prepare_data(df, self.config)
            
            # Create matcher
            matcher = ExactMatcher(df)
            
            # 1. Harvest candidates (with winner selection)
            harvest_df = identify_harvest_candidates(df, self.config, matcher)
            harvested_terms = set(harvest_df["Customer Search Term"].str.lower()) if not harvest_df.empty else set()
            
            # 2. Negative candidates (unique per campaign/ad group + ASIN Mapper integration)
            neg_kw, neg_pt, your_products_review = identify_negative_candidates(df, self.config, harvest_df)
            
            # Construct negative exclusion set (Single-Ownership Rule)
            negative_set = set()
            for _, row in pd.concat([neg_kw, neg_pt]).iterrows():
                camp = str(row.get("Campaign Name", "")).strip()
                ag = str(row.get("Ad Group Name", "")).strip()
                term = str(row.get("Term", "")).strip().lower()
                if camp and ag and term:
                    negative_set.add((camp, ag, term))
            
            # 3. Bid optimizations (direct + aggregated, NOT at search term level for non-exact)
            direct_bids, agg_bids = calculate_bid_optimizations(
                df,
                self.config,
                harvested_terms=harvested_terms,
                negative_terms=negative_set
            )
            
            # 4. Heatmap with action tracking
            heatmap_df = create_heatmap(df, self.config, harvest_df, neg_kw, neg_pt, direct_bids, agg_bids)
            
            # 5. Simulation (optional)
            simulation = None
            if st.session_state.get("run_simulation", True):
                simulation = run_simulation(df, direct_bids, agg_bids, harvest_df, self.config, date_info)
            
            # Store results
            self.results = {
                "account_id": st.session_state.get('active_account_id'), # CRITICAL FOR CHECKPOINT
                "df": df,
                "date_info": date_info,
                "harvest": harvest_df,
                "neg_kw": neg_kw,
                "neg_pt": neg_pt,
                "your_products_review": your_products_review,
                "direct_bids": direct_bids,
                "agg_bids": agg_bids,
                "heatmap": heatmap_df,
                "simulation": simulation
            }
            
            # Persist for other modules (e.g. AI Analyst)
            st.session_state['optimizer_results'] = self.results
        
        # ==========================================
        # LOG OPTIMIZATION ACTIONS FOR IMPACT TRACKING
        # ==========================================
        # NOTE: This is OUTSIDE the spinner so it runs after spinner completes
        try:
            # USE ACTIVE ACCOUNT ID - STRICT MODE
            client_id = st.session_state.get('active_account_id')
            
            if not client_id:
                st.error("❌ No active account detected. Please select an account in the sidebar.")
                return

            # STRICT DB CHECK: Ensure account actually exists
            db = st.session_state.get('db_manager')
            if db:
                valid_accts = [a[0] for a in db.get_all_accounts()]
                if client_id not in valid_accts:
                    st.error(f"❌ Invalid Account ID ('{client_id}'). This account does not exist in the database. Please refresh and re-select.")
                    return

            # Use report start date from DETECTED DATE RANGE
            date_info = self.results.get('date_info', {})
            report_date = date_info.get('start_date')
            
            # If Report Date is missing (e.g. unknown format), try fallback but warn
            if not report_date:
                fallback_date = st.session_state.get('last_stats_save', {}).get('start_date')
                if fallback_date:
                    report_date = fallback_date
                    st.warning(f"⚠️ Report date detection failed. Using upload date: {report_date}")
            
            if not report_date:
                st.warning("⚠️ Could not determine Report Date. Impact Analysis may not match correctly with future data.")
            
            # Log with correct date
            logged = _log_optimization_events(self.results, client_id, report_date)
            
            account_name = st.session_state.get('active_account_name', client_id)
            if logged > 0:
                st.success(f"📊 Logged {logged} actions for {account_name} (Date: {report_date})")
            else:
                st.info(f"ℹ️ No actions to log (all Hold or no recommendations)")
                    
        except Exception as e:
            st.error(f"❌ Action logging error: {e}")
        
        # Display results
        self._display_results()
    
    def _display_results(self):
        """Display all optimization results."""
        r = self.results
        df = r["df"]
        
        # Key metrics
        self._display_key_metrics(r)
        
        # Add sticky tabs CSS
        st.markdown("""
        <style>
        /* Sticky tabs container */
        div[data-baseweb="tab-list"] {
            position: sticky !important;
            top: 0 !important;
            z-index: 999 !important;
            background-color: #0e1117 !important;
            padding-top: 1rem !important;
            padding-bottom: 0.5rem !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Tabs
        tabs = st.tabs([
            "📊 Dashboard",
            "🌾 Harvest",
            "🛑 Negatives",
            "💰 Bids",
            "🔥 Heatmap",
            "🎯 Simulation",
            "📥 Downloads"
        ])
        
        with tabs[0]:
            st.markdown("**What this does:** An executive summary of your campaign performance, highlighting key metrics like Spend, Sales, ROAS, and ACoS over time.")
            self._display_dashboard(r)
        with tabs[1]:
            st.markdown("**What this does:** Identifies high-performing search terms that should be promoted to keywords for better control and scalability.")
            self._display_harvest(r["harvest"])
        with tabs[2]:
            st.markdown("**What this does:** Flags search terms and ASINs that are wasting spend with little to no return, suggesting them as negative targets.")
            self._display_negatives(r["neg_kw"], r["neg_pt"])
        with tabs[3]:
            st.markdown("**What this does:** Recommends bid adjustments for your targets based on performance data to optimize for your target ROAS.")
            self._display_bids(r["direct_bids"], r["agg_bids"])
        with tabs[4]:
            st.markdown("**What this does:** Visualizes performance patterns by day of week and hour of day to identify the most effective times for your ads.")
            self._display_heatmap(r["heatmap"])
        with tabs[5]:
            st.markdown("**What this does:** Forecasts the potential impact of applying the recommended bid changes and harvest actions on your future performance.")
            self._display_simulation(r["simulation"], r["date_info"])
        with tabs[6]:
            st.markdown("**What this does:** Download the comprehensive optimization reports, including bulk upload files for immediate implementation.")
            self._display_downloads(r)
    
    def _display_key_metrics(self, r: dict):
        """Display summary metrics bar."""
        df = r["df"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Spend", format_currency(df["Spend"].sum()))
        c2.metric("Total Sales", format_currency(df["Sales_Attributed"].sum()))
        roas = df["Sales_Attributed"].sum() / df["Spend"].sum() if df["Spend"].sum() > 0 else 0
        c3.metric("ROAS", f"{roas:.2f}x")
        acos = df["Spend"].sum() / df["Sales_Attributed"].sum() * 100 if df["Sales_Attributed"].sum() > 0 else 0
        c4.metric("ACoS", f"{acos:.1f}%")
        
        st.divider()
    
    def _calculate_account_health(self, df: pd.DataFrame, r: dict) -> dict:
        """
        Calculate account health diagnostics for dashboard display.
        Returns factual metrics about the current state (not projections).
        """
        total_spend = df['Spend'].sum() if 'Spend' in df.columns else 0
        total_sales = df['Sales'].sum() if 'Sales' in df.columns else 0
        total_orders = df['Orders'].sum() if 'Orders' in df.columns else 0
        
        # Current ROAS and ACoS
        current_roas = total_sales / total_spend if total_spend > 0 else 0
        current_acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
        
        # Waste Ratio: Spend on terms with 0 orders / Total Spend
        zero_order_mask = df['Orders'] == 0
        wasted_spend = df.loc[zero_order_mask, 'Spend'].sum()
        waste_ratio = (wasted_spend / total_spend * 100) if total_spend > 0 else 0
        
        # CVR
        total_clicks = df['Clicks'].sum() if 'Clicks' in df.columns else 0
        cvr = (total_orders / total_clicks * 100) if total_clicks > 0 else 0
        
        # Health Score (0-100)
        # Factors: ROAS, Waste Ratio, CVR
        roas_score = min(100, current_roas / 4.0 * 100)  # 4.0 ROAS = 100 points
        waste_score = max(0, 100 - waste_ratio * 3)  # 0% waste = 100, 33% waste = 0
        cvr_score = min(100, cvr / 5.0 * 100)  # 5% CVR = 100 points
        
        health_score = (roas_score * 0.4 + waste_score * 0.4 + cvr_score * 0.2)
        
        return {
            "health_score": health_score,
            "waste_ratio": waste_ratio,
            "current_roas": current_roas,
            "current_acos": current_acos,
            "cvr": cvr
        }
    
    def _display_dashboard_v2(self, r: dict):
        """Display overview dashboard."""
        
        # Unpack Data
        df = r.get("df", pd.DataFrame())
        harvest = r.get("harvest", pd.DataFrame())
        neg_kw = r.get("neg_kw", pd.DataFrame())
        direct_bids = r.get("direct_bids", pd.DataFrame())
        agg_bids = r.get("agg_bids", pd.DataFrame())
        
        if df.empty:
            st.warning("No data available for dashboard.")
            return

        # ==========================================
        # 2. SUMMARY TILES
        # ==========================================
        st.markdown("##### Summary")
        
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        
        # CHANGED: Show Ad Groups instead of Keywords (since Keywords often duplicates Search Terms in DB load)
        ad_groups_count = df["Ad Group Name"].nunique()
        terms_count = df["Customer Search Term"].nunique()
        negatives_count = len(neg_kw)
        bids_count = len(direct_bids) + len(agg_bids)
        harvest_count = len(harvest)
        
        with col_s1: metric_card("Ad Groups", f"{ad_groups_count:,}", border_color="#444")
        with col_s2: metric_card("Search Terms", f"{terms_count:,}", border_color="#444")
        with col_s3: metric_card("Negatives", f"{negatives_count}", border_color="#444")
        with col_s4: metric_card("Bid Changes", f"{bids_count}", border_color="#444")
        with col_s5: metric_card("Harvest Ops", f"{harvest_count}", border_color="#444")

        # ==========================================
        # 3. ACCOUNT HEALTH DIAGNOSTICS
        # ==========================================
        st.markdown("##### 📊 Account Health")
        
        # Calculate health diagnostics
        health = self._calculate_account_health(df, r)
        
        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        
        with col_h1:
            # Health Score (0-100)
            score = health['health_score']
            if score >= 80:
                status_emoji, status_label, score_color = "🟢", "Excellent", "#22c55e"
            elif score >= 60:
                status_emoji, status_label, score_color = "🟡", "Good", "#eab308"
            elif score >= 40:
                status_emoji, status_label, score_color = "🟠", "Fair", "#f59e0b"
            else:
                status_emoji, status_label, score_color = "🔴", "Needs Work", "#ef4444"
            
            metric_card(
                "Health Score",
                f"{score:.0f}/100",
                subtitle=f"{status_emoji} {status_label}",
                border_color=score_color
            )
        
        with col_h2:
            # Waste Ratio
            waste_pct = health['waste_ratio']
            waste_color = "#22c55e" if waste_pct < 10 else "#f59e0b" if waste_pct < 20 else "#ef4444"
            metric_card(
                "Waste Ratio",
                f"{waste_pct:.1f}%",
                subtitle=f"of spend → 0 orders",
                border_color=waste_color
            )
        
        with col_h3:
            # Optimization Coverage
            flagged = negatives_count + harvest_count
            coverage_pct = (flagged / terms_count * 100) if terms_count > 0 else 0
            metric_card(
                "Flagged Terms",
                f"{flagged}",
                subtitle=f"{coverage_pct:.1f}% of terms",
                border_color="#3b82f6"
            )
        
        with col_h4:
            # Current ROAS
            current_roas = health['current_roas']
            roas_color = "#22c55e" if current_roas >= 3.0 else "#f59e0b" if current_roas >= 2.0 else "#ef4444"
            metric_card(
                "Current ROAS",
                f"{current_roas:.2f}x",
                subtitle=f"ACoS: {health['current_acos']:.1f}%",
                border_color=roas_color
            )
        
        # Summary insight
        action_summary = []
        if waste_pct >= 15:
            action_summary.append(f"🔴 High waste ({waste_pct:.0f}%) - prioritize negatives")
        if harvest_count > 0:
            action_summary.append(f"🟢 {harvest_count} terms ready to scale via Exact match")
        if current_roas < 2.0:
            action_summary.append(f"🟠 ROAS below target - review bids")
        
        if action_summary:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                        border: 1px solid #334155; border-radius: 8px; padding: 16px; margin: 16px 0;">
                <p style="margin: 0; color: #94a3b8; font-size: 14px;">
                    <strong style="color: #f1f5f9;">🎯 Quick Insights:</strong> 
                    {' • '.join(action_summary)}
                </p>
            </div>
            """, unsafe_allow_html=True)

    
    def _display_harvest(self, df: pd.DataFrame):
        """Display harvest candidates."""
        st.subheader("🌾 Harvest Candidates (Winners)")
        
        if not df.empty:
            st.success(f"Found **{len(df)}** terms to migrate to Exact Match")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("📦 Send to Campaign Creator", type="primary"):
                    st.session_state["harvest_payload"] = df
                    st.session_state["current_module"] = "creator"
                    st.rerun()
            
            cols = ["Customer Search Term", "Campaign Name", "Orders", "Sales", "ROAS", "CPC", "New Bid"]
            if "SKU_advertised" in df.columns:
                cols.insert(2, "SKU_advertised")
            elif "ASIN_advertised" in df.columns:
                cols.insert(2, "ASIN_advertised")
            else:
                st.warning("⚠️ No SKU/ASIN mapping found. Upload 'Advertised Product Report' in Data Hub for SKU mapping.")
            
            display_cols = [c for c in cols if c in df.columns]
            st.dataframe(df[display_cols], use_container_width=True)
            
            st.caption(f"Estimated Revenue Upside: {format_currency(df['Sales'].sum() * 0.15)} (15% efficiency gain)")
        else:
            st.info("No harvest candidates found matching thresholds")
    
    def _display_negatives(self, kw_df: pd.DataFrame, pt_df: pd.DataFrame):
        """Display negative candidates."""
        st.subheader("🛑 Negative Candidates")
        
        tab1, tab2 = st.tabs([f"Keywords ({len(kw_df)})", f"Product Targets ({len(pt_df)})"])
        
        with tab1:
            if not kw_df.empty:
                st.dataframe(
                    kw_df[["Type", "Campaign Name", "Ad Group Name", "Term", "Clicks", "Spend"]],
                    use_container_width=True
                )
                st.caption(f"Potential Savings: {format_currency(kw_df['Spend'].sum())}")
            else:
                st.info("No negative keywords found")
        
        with tab2:
            if not pt_df.empty:
                # DEBUG: Check what columns we have
                print(f"DEBUG - _display_negatives PT: DataFrame has {len(pt_df)} rows, columns: {list(pt_df.columns)}")
                print(f"DEBUG - _display_negatives PT: Types present: {pt_df['Type'].unique().tolist() if 'Type' in pt_df.columns else 'NO TYPE COLUMN'}")
                
                # Select only columns that exist
                desired_cols = ["Type", "Campaign Name", "Ad Group Name", "Term", "Clicks", "Spend"]
                available_cols = [c for c in desired_cols if c in pt_df.columns]
                
                if available_cols:
                    st.dataframe(
                        pt_df[available_cols],
                        use_container_width=True
                    )
                    st.caption(f"Potential Savings: {format_currency(pt_df['Spend'].sum() if 'Spend' in pt_df.columns else 0)}")
                else:
                    st.error(f"Column mismatch! Expected {desired_cols}, got {list(pt_df.columns)}")
            else:
                st.info("No negative ASINs found")
        
        # Show ASIN Mapper integration stats
        st.divider()
        if 'asin_mapper_integration_stats' in st.session_state:
            stats = st.session_state['asin_mapper_integration_stats']
            if stats['total'] > 0:
                if stats['added'] == 0:
                    st.info(f"ℹ️ **ASIN Mapper Integration**: All {stats['total']} ASIN(s) from ASIN Mapper were already included in the bleeder/hard-stop list above. No duplicates added.")
                else:
                    msg_parts = [f"✅ **ASIN Mapper Integration**: Added {stats['added']} ASIN(s) from ASIN Mapper"]
                    if stats['duplicates'] > 0:
                        msg_parts.append(f" ({stats['duplicates']} already in list, skipped)")
                    st.success("".join(msg_parts))
    
    def _display_bids(self, bids_exact: pd.DataFrame = None, bids_pt: pd.DataFrame = None, 
                       bids_agg: pd.DataFrame = None, bids_auto: pd.DataFrame = None,
                       direct_df: pd.DataFrame = None, agg_df: pd.DataFrame = None):
        """Display bid optimizations in 4 tabs: Exact KW, PT, Aggregated KW, Auto/Category."""
        st.subheader("💰 Bid Optimizations")
        
        # Handle backward compatibility - if old 2-df signature is used
        if bids_exact is None and direct_df is not None:
            bids_exact = direct_df
            bids_agg = agg_df
            bids_pt = pd.DataFrame()
            bids_auto = pd.DataFrame()
        
        # Ensure all DFs exist
        bids_exact = bids_exact if bids_exact is not None else pd.DataFrame()
        bids_pt = bids_pt if bids_pt is not None else pd.DataFrame()
        bids_agg = bids_agg if bids_agg is not None else pd.DataFrame()
        bids_auto = bids_auto if bids_auto is not None else pd.DataFrame()
        
        tab1, tab2, tab3, tab4 = st.tabs([
            f"Exact Keywords ({len(bids_exact)})", 
            f"Product Targeting ({len(bids_pt)})",
            f"Aggregated KW ({len(bids_agg)})", 
            f"Auto/Category ({len(bids_auto)})"
        ])
        
        cols = ["Targeting", "Campaign Name", "Match Type", "Clicks", "Orders", "Sales", "ROAS", "Current Bid", "CPC", "New Bid", "Reason", "Bucket"]
        
        with tab1:
            st.markdown("*Exact match keywords - fast optimization*")
            if not bids_exact.empty:
                display_cols = [c for c in cols if c in bids_exact.columns]
                st.dataframe(bids_exact[display_cols], use_container_width=True)
            else:
                st.info("No exact keyword bid optimizations")
        
        with tab2:
            st.markdown("*Product Targeting (ASIN-based) - fast optimization*")
            if not bids_pt.empty:
                display_cols = [c for c in cols if c in bids_pt.columns]
                st.dataframe(bids_pt[display_cols], use_container_width=True)
            else:
                st.info("No product targeting bid optimizations")
        
        with tab3:
            st.markdown("*Broad/Phrase keywords - conservative optimization*")
            if not bids_agg.empty:
                display_cols = [c for c in cols if c in bids_agg.columns]
                st.dataframe(bids_agg[display_cols], use_container_width=True)
            else:
                st.info("No aggregated keyword bid optimizations")
        
        with tab4:
            st.markdown("*Auto campaigns and Category targeting - slow optimization*")
            if not bids_auto.empty:
                display_cols = [c for c in cols if c in bids_auto.columns]
                st.dataframe(bids_auto[display_cols], use_container_width=True)
            else:
                st.info("No auto/category bid optimizations")
    
    def _display_heatmap(self, df: pd.DataFrame):
        """Display heatmap with action tracking."""
        st.subheader("🔥 Wasted Spend Heatmap with Action Tracking")
        
        if df.empty:
            st.info("No heatmap data available")
            return
        
        # Description
        st.markdown("""
        <div style="background-color: #1e293b; border-left: 4px solid #3b82f6; padding: 15px; margin-bottom: 20px; border-radius: 4px; color: #e2e8f0;">
        Visual performance heatmap showing which campaigns/ad groups need attention.<br>
        🔴 Red = Fix immediately | 🟡 Yellow = Monitor | 🟢 Green = Good performance<br>
        <b>NEW:</b> See which issues the optimizer is already addressing (harvests, negatives, bids)
        </div>
        """, unsafe_allow_html=True)
        
        # Priority counts
        high_priority = len(df[df["Priority"] == "🔴 High"])
        medium_priority = len(df[df["Priority"] == "🟡 Medium"])
        good_performance = len(df[df["Priority"] == "🟢 Good"])
        
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("🔴 High Priority", high_priority)
        with c2: metric_card("🟡 Medium Priority", medium_priority)
        with c3: metric_card("🟢 Good Performance", good_performance)
        
        st.divider()
        
        # Action status
        st.markdown("### 🎯 Optimizer Actions Status")
        actions_addressed = len(df[df["Actions_Taken"] != "✅ No action needed"])
        needs_attention = len(df[
            (df["Priority"] == "🔴 High") & 
            (df["Actions_Taken"].isin(["✅ No action needed", "⏸️ Hold (Low volume)"]))
        ])
        high_fixed = len(df[
            (df["Priority"] == "🔴 High") & 
            (~df["Actions_Taken"].isin(["✅ No action needed", "⏸️ Hold (Low volume)"]))
        ])
        coverage = (actions_addressed / len(df) * 100) if len(df) > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("✅ Being Addressed", actions_addressed)
        with c2: metric_card("⚠️ Needs Attention", needs_attention)
        with c3: metric_card("🔴 High Priority Fixed", f"{high_fixed}/{high_priority}")
        with c4: metric_card("Coverage", f"{coverage:.0f}%")
        
        st.divider()
        
        # Filters
        c1, c2 = st.columns(2)
        with c1:
            filter_priority = st.multiselect(
                "Filter by Priority",
                ["🔴 High", "🟡 Medium", "🟢 Good"],
                default=["🔴 High", "🟡 Medium"]
            )
        with c2:
            filter_actions = st.selectbox(
                "Filter by Actions",
                ["All", "Has Actions", "Needs Actions"]
            )
        
        # Apply filters
        filtered = df[df["Priority"].isin(filter_priority)] if filter_priority else df
        
        if filter_actions == "Has Actions":
            filtered = filtered[~filtered["Actions_Taken"].isin(["✅ No action needed", "⏸️ Hold (Low volume)"])]
        elif filter_actions == "Needs Actions":
            filtered = filtered[filtered["Actions_Taken"].isin(["✅ No action needed", "⏸️ Hold (Low volume)"])]
        
        # Display table
        st.markdown("### 📊 Performance Heatmap with Actions")
        
        display_cols = [
            "Priority", "Campaign Name", "Ad Group Name", "Actions_Taken",
            "Spend", "Sales_Attributed", "ROAS", "ACoS", "CTR", "CVR"
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]
        
        # Format columns
        styled = filtered[display_cols].style.format({
            "Spend": "${:,.2f}",
            "Sales_Attributed": "${:,.2f}",
            "ROAS": "{:.2f}x",
            "ACoS": "{:.1f}%",
            "CTR": "{:.2f}%",
            "CVR": "{:.2f}%"
        })
        
        st.dataframe(styled, use_container_width=True)
    
    
    # _display_simulation moved to features/simulator.py
            

    
    def _display_downloads(self, r: dict):
        """Display download buttons with preview."""
        st.subheader("📥 Download Bulk Files")
        
        # 1. Negative Keywords
        neg_kw = r["neg_kw"]
        if not neg_kw.empty:
            st.markdown("### 🛑 Negative Keywords Bulk")
            kw_bulk = generate_negatives_bulk(neg_kw, pd.DataFrame())
            
            with st.expander("👁️ Preview Negative Keywords", expanded=False):
                st.dataframe(kw_bulk.head(5), use_container_width=True)
            
            buf = BytesIO()
            kw_bulk.to_excel(buf, index=False)
            st.download_button(
                "📥 Download Negative Keywords (.xlsx)",
                buf.getvalue(),
                "negative_keywords_bulk.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.caption(f"Count: {len(neg_kw)}")
            
        st.divider()
        
        # 2. Negative Product Targets
        neg_pt = r["neg_pt"]
        if not neg_pt.empty:
            st.markdown("### 🛑 Negative Product Targets Bulk")
            pt_bulk = generate_negatives_bulk(pd.DataFrame(), neg_pt)
            
            with st.expander("👁️ Preview Negative PT", expanded=False):
                st.dataframe(pt_bulk.head(5), use_container_width=True)
            
            buf = BytesIO()
            pt_bulk.to_excel(buf, index=False)
            st.download_button(
                "📥 Download Negative PT (.xlsx)",
                buf.getvalue(),
                "negative_pt_bulk.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.caption(f"Count: {len(neg_pt)}")
            
            # Show ASIN Mapper integration stats
            if 'asin_mapper_integration_stats' in st.session_state:
                stats = st.session_state['asin_mapper_integration_stats']
                if stats['total'] > 0:
                    if stats['added'] == 0:
                        st.info(f"ℹ️ **ASIN Mapper Integration**: All {stats['total']} ASIN(s) from ASIN Mapper were already included in the bleeder/hard-stop list above. No duplicates added.")
                    else:
                        msg_parts = [f"✅ **ASIN Mapper Integration**: Added {stats['added']} ASIN(s) from ASIN Mapper"]
                        if stats['duplicates'] > 0:
                            msg_parts.append(f" ({stats['duplicates']} already in list, skipped)")
                        st.success("".join(msg_parts))
            
        st.divider()
        
        # 3. Your Products - Review Required
        your_products = r.get("your_products_review", pd.DataFrame())
        if not your_products.empty:
            st.markdown("### 🟡 Your Products - Review Required")
            st.warning(f"Found **{len(your_products)} of YOUR ASINs** that are non-converting. Manual review needed to decide:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("✅ **Keep if:**\n- It's a cross-sell/complement\n- Strategic intent")
            with col2:
                st.error("❌ **Negate if:**\n- Wrong category entirely\n- Direct substitute with another SKU")
            
            # Display table with recommendations
            display_cols = ['Term', 'Brand', 'Product', 'Clicks', 'Spend', 'Recommendation']
            available_cols = [c for c in display_cols if c in your_products.columns]
            
            if available_cols:
                review_df = your_products[available_cols].copy()
                st.dataframe(review_df, use_container_width=True)
            
            st.caption("💡 These are detected as YOUR products based on brand/ASIN matching. Review before negating.")
            
        st.divider()
        
        # Bids
        all_bids = pd.concat([r["direct_bids"], r["agg_bids"]]) if not r["direct_bids"].empty or not r["agg_bids"].empty else pd.DataFrame()
        if not all_bids.empty:
            st.markdown("### 💰 Bids Bulk File")
            bids_bulk, skipped = generate_bids_bulk(all_bids)
            
            with st.expander("👁️ Preview Bids Bulk", expanded=False):
                st.dataframe(bids_bulk.head(5), use_container_width=True)
            
            buf = BytesIO()
            bids_bulk.to_excel(buf, index=False)
            st.download_button(
                "📥 Download Bids Bulk (.xlsx)",
                buf.getvalue(),
                "bids_bulk.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            if skipped > 0:
                st.warning(f"Note: {skipped} recommendations were skipped (Hold/Low Data).")
        
        st.divider()
        
        # Harvest
        if not r["harvest"].empty:
            st.markdown("### 🌾 Harvest Candidates")
            with st.expander("👁️ Preview Harvest List", expanded=False):
                st.dataframe(r["harvest"].head(5), use_container_width=True)
            
            buf = BytesIO()
            r["harvest"].to_excel(buf, index=False)
            st.download_button(
                "📥 Download Harvest List",
                buf.getvalue(),
                "harvest_candidates.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.divider()
        
        # Full report
        if st.button("📊 Generate Full Excel Report"):
            report_data = {
                "Harvest": r["harvest"],
                "Negative_Keywords": r["neg_kw"],
                "Negative_PT": r["neg_pt"],
                "Direct_Bids": r["direct_bids"],
                "Agg_Bids": r["agg_bids"],
                "Heatmap": r["heatmap"]
            }
            
            if r.get("simulation"):
                sens = r["simulation"].get("sensitivity", pd.DataFrame())
                if not sens.empty:
                    report_data["Sensitivity"] = sens
            
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                for name, data in report_data.items():
                    if not data.empty:
                        data.to_excel(writer, sheet_name=name[:31], index=False)
            
            st.download_button(
                "📥 Download Combined Report",
                buf.getvalue(),
                "optimization_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # BaseFeature interface
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, str]:
        required = ["Campaign Name", "Customer Search Term", "Clicks", "Spend", "Sales"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            return False, f"Missing columns: {', '.join(missing)}"
        return True, ""
    
    def analyze(self, data: pd.DataFrame) -> dict:
        return self.results
    
    def display_results(self, results: dict):
        self.results = results
        self._display_results()
