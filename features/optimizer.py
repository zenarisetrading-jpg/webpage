"""
Optimizer Module - Complete Implementation

Migrated from ppcsuite_v3.2.py with full feature parity:
- Harvest detection with winner campaign/SKU selection
- Isolation negatives (unique per campaign/ad group)
- Performance negatives (bleeders)
- Bid optimization (Exact/PT direct, Aggregated for broad/phrase/auto)
- Heatmap with action tracking
- Advanced simulation with scenarios, sensitivity, risk analysis

Note: Bulk file generation moved to features/bulk_export.py

Architecture: features/_base.py template
Data Source: DataHub (enriched data with SKUs)
"""

# Import bulk export functions from separate module
from features.bulk_export import (
    EXPORT_COLUMNS,
    generate_negatives_bulk,
    generate_bids_bulk,
    generate_harvest_bulk,
    strip_targeting_prefix
)

# Import shared constants
from features.constants import AUTO_TARGETING_TYPES, normalize_auto_targeting

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, Any, Tuple, Optional, Set, List
from features._base import BaseFeature
from core.data_hub import DataHub
from core.data_loader import safe_numeric, is_asin
from utils.formatters import format_currency, dataframe_to_excel
from utils.matchers import ExactMatcher
from utils.metrics import calculate_ppc_metrics, ensure_numeric_columns
from ui.components import metric_card
import plotly.graph_objects as go

# Validation Architecture
from bulk_validation_spec import (
    OptimizationRecommendation,
    RecommendationType,
    validate_recommendation,
    ValidationResult
)


# ==========================================
# CONSTANTS
# ==========================================

BULK_COLUMNS = [
    "Product", "Entity", "Operation", "Campaign Id", "Ad Group Id", 
    "Campaign Name", "Ad Group Name", "Ad Group Default Bid", "Bid", 
    "Keyword Text", "Match Type", "Product Targeting Expression",
    "Keyword Id", "Product Targeting Id", "State"
]

# Bid Safety Limits (Option C: Hybrid - relative + absolute floor)
BID_LIMITS = {
    "MIN_BID_FLOOR": 0.30,        # Never bid below $0.30 (Amazon minimum)
    "MIN_BID_MULTIPLIER": 0.50,   # Never below 50% of current bid
    "MAX_BID_MULTIPLIER": 3.00,   # Never above 300% of current bid
}

# CVR-Based Threshold Configuration
CVR_CONFIG = {
    "CVR_FLOOR": 0.01,             # Minimum CVR for calculations (1%)
    "CVR_CEILING": 0.20,           # Maximum CVR for calculations (20%)
    "HARD_STOP_MULTIPLIER": 3.0,   # Hard stop = 3× expected clicks to convert
    "SOFT_NEGATIVE_FLOOR": 10,     # Minimum clicks for soft negative
    "HARD_STOP_FLOOR": 15,         # Minimum clicks for hard stop
}

DEFAULT_CONFIG = {
    # Harvest thresholds (Tier 2)
    "HARVEST_CLICKS": 10,
    "HARVEST_ORDERS": 3,           # Will be dynamic based on CVR
    # HARVEST_SALES removed - currency threshold doesn't work across geos
    "HARVEST_ROAS_MULT": 0.90,     # vs BUCKET median (90% = moderately strict)
    "MAX_BID_CHANGE": 0.25,        # Max 25% change per run
    "DEDUPE_SIMILARITY": 0.85,     # ExactMatcher threshold
    "TARGET_ROAS": 2.5,
    
    # Negative thresholds (now CVR-based, no currency dependency)
    "NEGATIVE_CLICKS_THRESHOLD": 10,  # Baseline for legacy compatibility
    # NEGATIVE_SPEND_THRESHOLD removed - currency threshold doesn't work across geos
    
    # Bid optimization
    "ALPHA_EXACT": 0.25,           # 25% step size for exact
    "ALPHA_BROAD": 0.20,           # 20% step size for broad
    "ALPHA": 0.20,
    "MAX_BID_CHANGE": 0.25,        # 25% safety cap
    "TARGET_ROAS": 2.50,
    
    # Min clicks thresholds per bucket (user-configurable)
    "MIN_CLICKS_EXACT": 5,
    "MIN_CLICKS_PT": 5,
    "MIN_CLICKS_BROAD": 8,
    "MIN_CLICKS_AUTO": 8,
    
    # Harvest forecast
    "HARVEST_EFFICIENCY_MULTIPLIER": 1.30,  # 30% efficiency gain from exact match
    "HARVEST_LAUNCH_MULTIPLIER": 2.0,  # Bid multiplier for new harvest keywords to compete for impressions
    
    # Bucket median sanity check
    "BUCKET_MEDIAN_FLOOR_MULTIPLIER": 0.5,  # Bucket median must be >= 50% of target ROAS
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

@st.cache_data(show_spinner=False)
def prepare_data(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and prepare data for optimization.
    Returns prepared DataFrame and date_info dict.
    """
    df = df.copy()
    # Ensure numeric columns (using shared utility)
    df = ensure_numeric_columns(df, inplace=True)

    # CPC calculation (will be replaced by calculate_ppc_metrics later)
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
    # Using shared normalize_auto_targeting from features.constants
    df["Targeting"] = df["Targeting"].apply(normalize_auto_targeting)
    
    # Sales/Orders attributed columns
    df["Sales_Attributed"] = df["Sales"]
    df["Orders_Attributed"] = df["Orders"]

    # Derived metrics (using shared utility)
    # optimizer.py uses decimal format: 0.05 = 5%
    df = calculate_ppc_metrics(df, percentage_format='decimal', inplace=True)
    
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


@st.cache_data(show_spinner=False)
def calculate_account_benchmarks(df: pd.DataFrame, config: dict) -> dict:
    """
    Calculate account-level CVR benchmarks for dynamic thresholds.
    
    Returns dict with:
        - account_cvr: Clamped account-wide conversion rate
        - expected_clicks: Expected clicks needed for first conversion
        - soft_threshold: Clicks threshold for soft negative
        - hard_stop_threshold: Clicks threshold for hard stop
        - harvest_min_orders: Dynamic min orders for harvest based on CVR
    """
    # Calculate account-level CVR
    total_clicks = df['Clicks'].sum()
    total_orders = df['Orders'].sum()
    
    raw_cvr = total_orders / total_clicks if total_clicks > 0 else 0.03
    
    # Apply safety clamps (1% - 20%)
    account_cvr = np.clip(raw_cvr, CVR_CONFIG["CVR_FLOOR"], CVR_CONFIG["CVR_CEILING"])
    
    # Calculate thresholds
    expected_clicks = 1 / account_cvr
    soft_threshold = max(CVR_CONFIG["SOFT_NEGATIVE_FLOOR"], expected_clicks)
    hard_stop_threshold = max(CVR_CONFIG["HARD_STOP_FLOOR"], expected_clicks * CVR_CONFIG["HARD_STOP_MULTIPLIER"])
    
    # Dynamic harvest min orders: Based on harvest_clicks × account_cvr
    # Floor at 3 orders minimum
    harvest_clicks = config.get("HARVEST_CLICKS", 10)
    harvest_min_orders = max(3, int(harvest_clicks * account_cvr))
    
    # Calculate universal (account-wide) ROAS using spend-weighted average (Total Sales / Total Spend)
    # This gives realistic baseline that matches actual account performance
    valid_rows = df[df["Spend"] > 0].copy()
    total_spend = valid_rows["Spend"].sum()
    total_sales = valid_rows["Sales"].sum()
    
    if total_spend >= 100:  # Need meaningful spend for reliable ROAS
        universal_median_roas = total_sales / total_spend
    else:
        universal_median_roas = config.get("TARGET_ROAS", 2.5)

    print(f"\n=== ACCOUNT BENCHMARKS (CVR-Based) ===")
    print(f"Account CVR: {account_cvr:.1%} (raw: {raw_cvr:.1%})")
    print(f"Expected clicks to convert: {expected_clicks:.1f}")
    print(f"Soft negative threshold: {soft_threshold:.0f} clicks")
    print(f"Hard stop threshold: {hard_stop_threshold:.0f} clicks")
    print(f"Harvest min orders (dynamic): {harvest_min_orders}")
    print(f"Universal Median ROAS: {universal_median_roas:.2f}x (n={len(valid_rows)})")
    print(f"=== END BENCHMARKS ===\n")
    
    return {
        'account_cvr': account_cvr,
        'raw_cvr': raw_cvr,
        'expected_clicks': expected_clicks,
        'soft_threshold': soft_threshold,
        'hard_stop_threshold': hard_stop_threshold,
        'harvest_min_orders': harvest_min_orders,
        'universal_median_roas': universal_median_roas,
        'was_clamped': raw_cvr != account_cvr
    }


# ==========================================
# HARVEST DETECTION
# ==========================================

def identify_harvest_candidates(
    df: pd.DataFrame, 
    config: dict, 
    matcher: ExactMatcher,
    account_benchmarks: dict = None
) -> pd.DataFrame:
    """
    Identify high-performing search terms to harvest as exact match keywords.
    Winner campaign/SKU trumps others based on performance when KW appears in multiple campaigns.
    
    CHANGES:
    - Uses BUCKET median ROAS (not campaign ROAS) for consistent baseline
    - Uses CVR-based dynamic min orders
    - Winner score: Sales + ROAS×5 (reduced from ×10)
    """
    
    # Use benchmarks if provided
    if account_benchmarks is None:
        account_benchmarks = calculate_account_benchmarks(df, config)
    
    universal_median_roas = account_benchmarks.get('universal_median_roas', config.get("TARGET_ROAS", 2.5))
    
    # Use dynamic min orders from CVR analysis
    min_orders_threshold = account_benchmarks.get('harvest_min_orders', config["HARVEST_ORDERS"])
    
    # Filter for discovery campaigns (non-exact)
    auto_pattern = r'close-match|loose-match|substitutes|complements|category=|asin|b0'
    discovery_mask = (
        (~df["Match Type"].str.contains("exact", case=False, na=False)) |
        (df["Targeting"].str.contains(auto_pattern, case=False, na=False))
    )
    discovery_df = df[discovery_mask].copy()
    
    if discovery_df.empty:
        return pd.DataFrame(columns=["Harvest_Term", "Campaign Name", "Ad Group Name", "ROAS", "Spend", "Sales", "Orders"])
    
    # CRITICAL: Use Customer Search Term for harvest (actual user queries)
    # NOT Targeting (which contains targeting expressions like close-match, category=, etc.)
    harvest_column = "Customer Search Term" if "Customer Search Term" in discovery_df.columns else "Targeting"
    
    # PT PREFIX STRIPPING: Strip asin= and asin-expanded= prefixes so clean ASINs can be harvested
    from features.bulk_export import strip_targeting_prefix
    discovery_df[harvest_column] = discovery_df[harvest_column].apply(strip_targeting_prefix)
    
    # CRITICAL: Filter OUT targeting expressions that are NOT actual search queries
    # NOTE: asin= and asin-expanded= are now ALLOWED after prefix stripping
    targeting_expression_patterns = [
        r'^close-match$', r'^loose-match$', r'^substitutes$', r'^complements$', r'^auto$',
        r'^category=', r'^keyword-group=',  # PT (asin=) now allowed
    ]
    
    # Create mask for rows that are actual search queries (not targeting expressions)
    is_actual_search_query = ~discovery_df[harvest_column].str.lower().str.strip().str.match(
        '|'.join(targeting_expression_patterns), na=False
    )
    
    # Filter to only actual search queries
    discovery_df = discovery_df[is_actual_search_query].copy()
    
    if discovery_df.empty:
        return pd.DataFrame(columns=["Harvest_Term", "Campaign Name", "Ad Group Name", "ROAS", "Spend", "Sales", "Orders"])
    
    # Aggregate by Customer Search Term for harvest
    agg_cols = {
        "Impressions": "sum", "Clicks": "sum", "Spend": "sum",
        "Sales": "sum", "Orders": "sum", "CPC": "mean"
    }
    
    # Also keep Targeting for reference
    if "Targeting" in discovery_df.columns and harvest_column != "Targeting":
        agg_cols["Targeting"] = "first"
    
    grouped = discovery_df.groupby(harvest_column, as_index=False).agg(agg_cols)
    grouped["ROAS"] = np.where(grouped["Spend"] > 0, grouped["Sales"] / grouped["Spend"], 0)
    
    # Rename to Harvest_Term for consistency
    grouped = grouped.rename(columns={harvest_column: "Harvest_Term"})
    grouped["Customer Search Term"] = grouped["Harvest_Term"]
    
    # CHANGE #3: Winner selection score rebalanced (ROAS×5 instead of ×10)
    # Get metadata from BEST performing instance (winner selection)
    # Rank by Sales (primary), then ROAS (secondary)
    discovery_df["_perf_score"] = discovery_df["Sales"] + (discovery_df["ROAS"] * 5)
    discovery_df["_rank"] = discovery_df.groupby("Customer Search Term")["_perf_score"].rank(
        method="first", ascending=False
    )
    
    # Build metadata columns list
    meta_cols = ["Customer Search Term", "Campaign Name", "Ad Group Name", "Campaign_ROAS"]
    if "CampaignId" in discovery_df.columns:
        meta_cols.append("CampaignId")
    if "AdGroupId" in discovery_df.columns:
        meta_cols.append("AdGroupId")
    if "SKU_advertised" in discovery_df.columns:
        meta_cols.append("SKU_advertised")
    if "ASIN_advertised" in discovery_df.columns:
        meta_cols.append("ASIN_advertised")
    
    # Get winner row for each Customer Search Term value
    meta_df = discovery_df[discovery_df["_rank"] == 1][meta_cols].drop_duplicates("Customer Search Term")
    merged = pd.merge(grouped, meta_df, on="Customer Search Term", how="left")
    
    # Ensure Customer Search Term column exists for downstream compatibility
    if "Customer Search Term" not in merged.columns:
        merged["Customer Search Term"] = merged["Harvest_Term"]
    
    # Step 2: Calculate bucket ROAS using spend-weighted average (Total Sales / Total Spend)
    # This matches the actual bucket performance shown in UI, not skewed by many 0-sale rows
    bucket_with_spend = merged[merged["Spend"] > 0]
    bucket_sample_size = len(bucket_with_spend)
    total_spend = bucket_with_spend["Spend"].sum()
    total_sales = bucket_with_spend["Sales"].sum()
    bucket_weighted_roas = total_sales / total_spend if total_spend > 0 else 0

    # Step 3: Stat sig check - need minimum data for reliable bucket ROAS
    MIN_SAMPLE_SIZE_FOR_STAT_SIG = 20
    MIN_SPEND_FOR_STAT_SIG = 100  # Need at least AED 100 spend for reliable bucket ROAS
    OUTLIER_THRESHOLD_MULTIPLIER = 1.5

    if bucket_sample_size < MIN_SAMPLE_SIZE_FOR_STAT_SIG or total_spend < MIN_SPEND_FOR_STAT_SIG:
        baseline_roas = universal_median_roas  # Use universal
        baseline_source = "Universal Median (insufficient bucket data)"
    else:
        # Step 4: Outlier detection
        if bucket_weighted_roas > universal_median_roas * OUTLIER_THRESHOLD_MULTIPLIER:
            baseline_roas = universal_median_roas  # Outlier, use universal
            baseline_source = "Universal Median (bucket is outlier)"
        else:
            baseline_roas = bucket_weighted_roas  # Valid, use bucket weighted ROAS
            baseline_source = f"Bucket Weighted ROAS (spend={total_spend:.0f})"

    print(f"\n=== HARVEST BASELINE ===")
    print(f"Baseline ROAS: {baseline_roas:.2f}x ({baseline_source})")
    print(f"Required ROAS: {baseline_roas * config['HARVEST_ROAS_MULT']:.2f}x")
    print(f"=== END HARVEST BASELINE ===\n")
    
    # Apply harvest thresholds (Tier 2)
    # High-ROAS term exception
    def calculate_roas_threshold(row):
        term_roas = row["ROAS"]
        if term_roas >= universal_median_roas:
            return term_roas >= (universal_median_roas * config["HARVEST_ROAS_MULT"])
        else:
            return term_roas >= (baseline_roas * config["HARVEST_ROAS_MULT"])

    # Individual threshold checks for debugging
    pass_clicks = merged["Clicks"] >= config["HARVEST_CLICKS"]
    pass_orders = merged["Orders"] >= min_orders_threshold  # CHANGE #5: CVR-based dynamic threshold
    # pass_sales = merged["Sales"] >= config["HARVEST_SALES"]  # REMOVED: Currency threshold doesn't work across geos
    pass_roas = merged.apply(calculate_roas_threshold, axis=1)
    
    # Currency-based threshold (HARVEST_SALES) removed - only clicks, orders, ROAS matter
    harvest_mask = pass_clicks & pass_orders & pass_roas
    
    candidates = merged[harvest_mask].copy()
    
    # DEBUG: Show why terms fail
    print(f"\n=== HARVEST DEBUG ===")
    print(f"Discovery rows: {len(discovery_df)}")
    print(f"Grouped search terms: {len(grouped)}")
    print(f"Threshold config: Clicks>={config['HARVEST_CLICKS']}, Orders>={min_orders_threshold} (CVR-based), ROAS>{config['HARVEST_ROAS_MULT']}x bucket median")
    print(f"Pass clicks: {pass_clicks.sum()}, Pass orders: {pass_orders.sum()}, Pass ROAS: {pass_roas.sum()}")
    print(f"After ALL thresholds: {len(candidates)} candidates")
    
    # DEBUG: Check for specific terms
    test_terms = ['water cups for kids', 'water cups', 'steel water bottle', 'painting set for kids']
    print(f"\n--- Checking specific terms ---")
    for test_term in test_terms:
        in_grouped = merged[merged["Customer Search Term"].str.contains(test_term, case=False, na=False)]
        if len(in_grouped) > 0:
            for _, r in in_grouped.iterrows():
                req_roas = baseline_roas * config["HARVEST_ROAS_MULT"]
                pass_all = (r["Clicks"] >= config["HARVEST_CLICKS"] and 
                           r["Orders"] >= min_orders_threshold and 
                           r["ROAS"] >= req_roas)
                print(f"  '{r['Customer Search Term']}': Clicks={r['Clicks']}, Orders={r['Orders']}, Sales={r['Sales']:.2f}, ROAS={r['ROAS']:.2f} vs {req_roas:.2f} | PASS={pass_all}")
        else:
            print(f"  '{test_term}' - NOT FOUND in Customer Search Term column")
    
    # Show sample of terms that pass all but ROAS
    almost_pass = pass_clicks & pass_orders & (~pass_roas)
    if almost_pass.sum() > 0:
        print(f"\\nTerms failing ONLY on ROAS ({almost_pass.sum()} total):")
        for _, r in merged[almost_pass].head(5).iterrows():
            req_roas = baseline_roas * config["HARVEST_ROAS_MULT"]
            print(f"  - '{r['Customer Search Term']}': ROAS {r['ROAS']:.2f} < required {req_roas:.2f}")
    
    if len(candidates) > 0:
        print(f"\\nTop 5 candidates BEFORE dedupe:")
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
    
    print(f"\\nDedupe results:")
    print(f"  - Survivors (new harvest): {len(survivors)}")
    print(f"  - Deduped (already exist): {len(deduped)}")
    if deduped:
        print(f"  - Sample deduped terms:")
        for term, match in deduped[:5]:
            print(f"    '{term}' matched to: {match}")
    print(f"=== END HARVEST DEBUG ===\\n")
    
    survivors_df = pd.DataFrame(survivors)
    
    if not survivors_df.empty:
        # Apply launch multiplier (2x by default) to ensure new harvest keywords can compete
        launch_mult = DEFAULT_CONFIG.get("HARVEST_LAUNCH_MULTIPLIER", 2.0)
        survivors_df["New Bid"] = survivors_df["CPC"] * launch_mult
        survivors_df = survivors_df.sort_values("Sales", ascending=False)
    
    return survivors_df


# ==========================================
# NEGATIVE DETECTION
# ==========================================

def enrich_with_ids(df: pd.DataFrame, bulk: pd.DataFrame) -> pd.DataFrame:
    """
    Unified high-precision ID mapping helper.
    Matches by Campaign, Ad Group, and Targeting Text (Keyword/PT).
    Synchronizes OptimizationRecommendation objects.
    """
    if df.empty or bulk is None or bulk.empty:
        return df
        
    def normalize_for_mapping(series):
        """
        Normalize and strip prefixes for robust matching.
        e.g., 'asin="B0123"' -> 'b0123', 'category="123"' -> '123'
        """
        s = series.astype(str).str.strip().str.lower()
        # Remove common prefixes and quotes
        s = s.str.replace(r'^(asin|category|asin-expanded|keyword-group)=', '', regex=True)
        s = s.str.replace(r'^"', '', regex=True).str.replace(r'"$', '', regex=True)
        # Final alphanumeric cleanup
        return s.str.replace(r'[^a-z0-9]', '', regex=True)
    
    df = df.copy()
    bulk = bulk.copy()
    
    # Normalize for mapping
    df['_camp_norm'] = normalize_for_mapping(df['Campaign Name'])
    df['_ag_norm'] = normalize_for_mapping(df.get('Ad Group Name', pd.Series([''] * len(df))))
    
    # Initialize ID columns if missing to avoid KeyError during resolution
    for col in ['KeywordId', 'TargetingId']:
        if col not in df.columns:
            df[col] = np.nan
    
    # For general targeting (Search Term or Targeting column)
    target_col = 'Term' if 'Term' in df.columns else 'Targeting'
    df['_target_norm'] = normalize_for_mapping(df[target_col])
    
    bulk['_camp_norm'] = normalize_for_mapping(bulk['Campaign Name'])
    bulk['_ag_norm'] = normalize_for_mapping(bulk.get('Ad Group Name', pd.Series([''] * len(bulk))))
    
    # Precise Match 1: Keywords
    # CRITICAL: Include Match Type to distinguish phrase/exact versions of the same keyword
    kw_col = next((c for c in ['Customer Search Term', 'Keyword Text', 'keyword_text'] if c in bulk.columns), None)
    if kw_col:
        bulk['_kw_norm'] = normalize_for_mapping(bulk[kw_col])
        # Normalize Match Type for both df and bulk
        df['_match_norm'] = df['Match Type'].astype(str).str.lower().str.strip() if 'Match Type' in df.columns else ''
        bulk['_match_norm'] = bulk['Match Type'].astype(str).str.lower().str.strip() if 'Match Type' in bulk.columns else ''
        
        # STRICT LOOKUP: Include Match Type to prevent collision between phrase/exact/broad
        # Use groupby().first() to ensure 1-to-1 mapping and prevent row explosion
        kw_base = bulk[bulk['KeywordId'].notna() & (bulk['KeywordId'] != "") & (bulk['KeywordId'] != "nan")][
            ['_camp_norm', '_ag_norm', '_kw_norm', '_match_norm', 'KeywordId', 'CampaignId', 'AdGroupId']
        ]
        kw_lookup = kw_base.groupby(['_camp_norm', '_ag_norm', '_kw_norm', '_match_norm']).first().reset_index()
        
        # STRATEGY 1: Strict match (Campaign + AG + Keyword + Match Type)
        df = df.merge(
            kw_lookup.rename(columns={'_kw_norm': '_target_norm'}),
            on=['_camp_norm', '_ag_norm', '_target_norm', '_match_norm'],
            how='left',
            suffixes=('', '_bulk_kw')
        )
        
        # STRATEGY 2: Fallback for unmatched rows (ignore Match Type)
        # Only fill if KeywordId is still missing after strict match
        strict_matched = df.get('KeywordId_bulk_kw', pd.Series([np.nan]*len(df))).notna()
        if (~strict_matched).any():
            # Relaxed lookup: Campaign + AG + Keyword only (take first ID for each)
            kw_lookup_relaxed = kw_base.groupby(['_camp_norm', '_ag_norm', '_kw_norm'])[
                ['KeywordId', 'CampaignId', 'AdGroupId']
            ].first().reset_index()
            kw_lookup_relaxed = kw_lookup_relaxed.rename(columns={
                '_kw_norm': '_target_norm', 
                'KeywordId': 'KeywordId_relaxed', 
                'CampaignId': 'CampaignId_relaxed', 
                'AdGroupId': 'AdGroupId_relaxed'
            })
            df = df.merge(
                kw_lookup_relaxed,
                on=['_camp_norm', '_ag_norm', '_target_norm'],
                how='left',
                suffixes=('', '_relax')
            )
            # Only use relaxed match if strict match failed
            for col in ['KeywordId', 'CampaignId', 'AdGroupId']:
                relaxed_col = f'{col}_relaxed'
                bulk_kw_col = f'{col}_bulk_kw'
                if relaxed_col in df.columns:
                    if bulk_kw_col in df.columns:
                        df[bulk_kw_col] = df[bulk_kw_col].fillna(df[relaxed_col])
                    else:
                        df[bulk_kw_col] = df[relaxed_col]
                    df.drop(columns=[relaxed_col], inplace=True, errors='ignore')
        
    # Precise Match 2: Product Targeting
    pt_col = next((c for c in ['TargetingExpression', 'Product Targeting Expression', 'targeting_expression'] if c in bulk.columns), None)
    if pt_col:
        bulk['_pt_norm'] = normalize_for_mapping(bulk[pt_col])
        pt_lookup = bulk[bulk['TargetingId'].notna() & (bulk['TargetingId'] != "") & (bulk['TargetingId'] != "nan")][['_camp_norm', '_ag_norm', '_pt_norm', 'TargetingId', 'CampaignId', 'AdGroupId']].drop_duplicates()
        
        df = df.merge(
            pt_lookup.rename(columns={'_pt_norm': '_target_norm'}),
            on=['_camp_norm', '_ag_norm', '_target_norm'],
            how='left',
            suffixes=('', '_bulk_pt')
        )

    # Resolve IDs
    id_cols = ['CampaignId', 'AdGroupId', 'KeywordId', 'TargetingId']
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].replace('', np.nan).replace('nan', np.nan)
            
        col_kw = f'{col}_bulk_kw'
        col_pt = f'{col}_bulk_pt'
        
        # Priority: Exact Keyword Match > Exact PT Match > Existing
        if col_kw in df.columns:
            df[col] = df[col].fillna(df[col_kw])
        if col_pt in df.columns:
            df[col] = df[col].fillna(df[col_pt])
            
    # Fallback: Campaign/Ad Group IDs if still missing
    missing_basics = df.get('CampaignId', pd.Series([np.nan]*len(df))).isna() | df.get('AdGroupId', pd.Series([np.nan]*len(df))).isna()
    if missing_basics.any():
        fallback_lookup = bulk.groupby(['_camp_norm', '_ag_norm'])[['CampaignId', 'AdGroupId']].first().reset_index()
        df = df.merge(fallback_lookup, on=['_camp_norm', '_ag_norm'], how='left', suffixes=('', '_fallback'))
        
        df['CampaignId'] = df.get('CampaignId', pd.Series([np.nan]*len(df))).fillna(df.get('CampaignId_fallback', pd.Series([np.nan]*len(df))))
        df['AdGroupId'] = df.get('AdGroupId', pd.Series([np.nan]*len(df))).fillna(df.get('AdGroupId_fallback', pd.Series([np.nan]*len(df))))

    # Sync Recommendation Objects
    if 'recommendation' in df.columns:
        def sync_rec(row):
            rec = row['recommendation']
            if not isinstance(rec, OptimizationRecommendation):
                return rec
            
            # Use mapped IDs - handle empty strings as None
            def get_id(val):
                if pd.isna(val) or str(val).strip() == "":
                    return None
                return str(val)

            rec.campaign_id = get_id(row.get('CampaignId')) or rec.campaign_id
            rec.ad_group_id = get_id(row.get('AdGroupId')) or rec.ad_group_id
            rec.keyword_id = get_id(row.get('KeywordId')) or rec.keyword_id
            rec.product_targeting_id = get_id(row.get('TargetingId')) or rec.product_targeting_id
            
            # Re-validate
            rec.validation_result = validate_recommendation(rec)
            return rec
            
        df['recommendation'] = df.apply(sync_rec, axis=1)

    # Cleanup internal columns
    drop_cols = [c for c in df.columns if c.startswith('_') or c.endswith('_bulk_kw') or c.endswith('_bulk_pt') or c.endswith('_fallback')]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    return df


def identify_negative_candidates(
    df: pd.DataFrame, 
    config: dict, 
    harvest_df: pd.DataFrame,
    account_benchmarks: dict = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Identify negative keyword candidates:
    1. Isolation negatives - harvest terms to negate in source campaigns (unique per campaign/ad group)
    2. Performance negatives - bleeders with 0 sales, high spend (CVR-based thresholds)
    3. ASIN Mapper integration - competitor ASINs flagged for negation
    
    CHANGES:
    - Uses CVR-based dynamic thresholds for hard stop
    
    Returns: (keyword_negatives_df, product_target_negatives_df, your_products_review_df)
    """
    # Get account benchmarks for CVR-based thresholds
    if account_benchmarks is None:
        account_benchmarks = calculate_account_benchmarks(df, config)
    
    soft_threshold = account_benchmarks['soft_threshold']
    hard_stop_threshold = account_benchmarks['hard_stop_threshold']
    
    negatives = []
    your_products_review = []
    seen_keys = set()  # Track (campaign, ad_group, term) for uniqueness
    
    # Stage 1: Isolation negatives
    if not harvest_df.empty:
        harvested_terms = set(
            harvest_df["Customer Search Term"].astype(str).str.strip().str.lower()
        )
        
        # Strip PT prefixes from main df for accurate matching
        from features.bulk_export import strip_targeting_prefix
        df_cst_clean = df["Customer Search Term"].apply(strip_targeting_prefix).astype(str).str.strip().str.lower()
        
        # Find all occurrences in non-exact campaigns
        isolation_mask = (
            df_cst_clean.isin(harvested_terms) &
            (~df["Match Type"].str.contains("exact", case=False, na=False))
        )
        
        isolation_df = df[isolation_mask].copy()
        # Store the cleaned term for grouping
        isolation_df["_cst_clean"] = df_cst_clean[isolation_mask]
        
        # Aggregate logic for Isolation Negatives (Fix for "metrics broken down by date")
        if not isolation_df.empty:
            # Group by CLEANED term to match harvest (prefixes already stripped)
            agg_cols = {"Clicks": "sum", "Spend": "sum"}
            meta_cols = {c: "first" for c in ["CampaignId", "AdGroupId", "KeywordId", "TargetingId", "Campaign Targeting Type"] if c in isolation_df.columns}
            
            isolation_agg = isolation_df.groupby(
                ["Campaign Name", "Ad Group Name", "_cst_clean"], as_index=False
            ).agg({**agg_cols, **meta_cols})
            isolation_agg = isolation_agg.rename(columns={"_cst_clean": "Customer Search Term"})
            
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
                
                
                # Create recommendation object for validation
                rec = OptimizationRecommendation(
                    recommendation_id=f"iso_{campaign}_{term}",
                    recommendation_type=RecommendationType.NEGATIVE_ISOLATION,
                    campaign_name=campaign,
                    campaign_id=row.get("CampaignId", ""),
                    campaign_targeting_type=row.get("Campaign Targeting Type", "Manual"),
                    ad_group_name=None, # Isolation is campaign-level
                    keyword_text=term,
                    match_type="campaign negative exact",
                    currency=config.get("currency", "AED")
                )
                rec.validation_result = validate_recommendation(rec)
                
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
                    "KeywordId": row.get("KeywordId", ""),
                    "TargetingId": row.get("TargetingId", ""),
                    "recommendation": rec # Store for UI and Export
                })
    
    # Stage 2: Performance negatives (bleeders) - CVR-BASED THRESHOLDS
    non_exact_mask = ~df["Match Type"].str.contains("exact", case=False, na=False)
    # Don't filter Sales==0 yet - wait until aggregated
    bleeders = df[non_exact_mask].copy()
    
    if not bleeders.empty:
        # Group by lowercased term to avoid case-based duplicates
        bleeders['_term_norm_group'] = bleeders['Customer Search Term'].astype(str).str.strip().str.lower()
        
        # Aggregate by campaign + ad group + term
        agg_cols = {"Clicks": "sum", "Spend": "sum", "Impressions": "sum", "Sales": "sum"}
        meta_cols = {c: "first" for c in ["CampaignId", "AdGroupId", "KeywordId", "TargetingId", "Campaign Targeting Type"] if c in bleeders.columns}
        
        bleeder_agg = bleeders.groupby(
            ["Campaign Name", "Ad Group Name", "_term_norm_group"], as_index=False
        ).agg({**agg_cols, **meta_cols})
        bleeder_agg = bleeder_agg.rename(columns={"_term_norm_group": "Customer Search Term"})
        
        # Apply CVR-based thresholds (Sales == 0 AND Clicks > threshold)
        # Currency threshold removed - only using clicks-based logic
        bleeder_mask = (
            (bleeder_agg["Sales"] == 0) &
            (bleeder_agg["Clicks"] >= soft_threshold)
        )
        
        for _, row in bleeder_agg[bleeder_mask].iterrows():
            campaign = row["Campaign Name"]
            ad_group = row["Ad Group Name"]
            term = str(row["Customer Search Term"]).strip().lower()
            
            key = (campaign, ad_group, term)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            
            # Use CVR-based hard stop threshold
            reason = "Hard Stop" if row["Clicks"] >= hard_stop_threshold else "Performance"
            
            # Create recommendation object for validation
            rec = OptimizationRecommendation(
                recommendation_id=f"bld_{campaign}_{ad_group}_{term}",
                recommendation_type=RecommendationType.NEGATIVE_BLEEDER,
                campaign_name=campaign,
                campaign_id=row.get("CampaignId", ""),
                campaign_targeting_type=row.get("Campaign Targeting Type", "Manual"),
                ad_group_name=ad_group,
                ad_group_id=row.get("AdGroupId", ""),
                keyword_text=term,
                match_type="negative exact",
                currency=config.get("currency", "AED")
            )
            rec.validation_result = validate_recommendation(rec)
            
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
                "KeywordId": row.get("KeywordId", ""),
                "TargetingId": row.get("TargetingId", ""),
                "recommendation": rec # Store for UI and Export
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
        empty = pd.DataFrame(columns=["Campaign Name", "Ad Group Name", "Term", "Match Type"])
        return empty.copy(), empty.copy(), your_products_df
    
    # CRITICAL: Map KeywordId and TargetingId for negatives
    # Negatives are at campaign+adgroup+term level, so we need to look up IDs
    # FINAL ENRICHMENT: Map IDs from Bulk for export
    neg_df = enrich_with_ids(neg_df, DataHub().get_data('bulk_id_mapping'))
    
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
# BID OPTIMIZATION (vNext)
# ==========================================

def calculate_bid_optimizations(
    df: pd.DataFrame, 
    config: dict, 
    harvested_terms: Set[str] = None,
    negative_terms: Set[Tuple[str, str, str]] = None,
    universal_median_roas: float = None,
    data_days: int = 7  # Number of days in dataset for visibility boost detection
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
    
    # 1. Global Exclusions
    def is_excluded(row):
        # Get both Customer Search Term AND Targeting values
        cst = str(row.get("Customer Search Term", "")).strip().lower()
        targeting = str(row.get("Targeting", "")).strip().lower()
        
        # Check Harvest - if EITHER column matches harvested terms, exclude
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
        empty = pd.DataFrame(columns=["Campaign Name", "Ad Group Name", "Targeting", "Match Type", "Current Bid", "New Bid"])
        return empty.copy(), empty.copy(), empty.copy(), empty.copy()
    
    
    # Calculate universal median if not provided (outlier-resistant)
    if universal_median_roas is None:
        valid_rows = df_clean[(df_clean["Spend"] > 0) & (df_clean["Sales"] > 0)].copy()
        
        if len(valid_rows) >= 10:
            # Filter to rows with meaningful spend (>= $5) to avoid low-spend outliers
            substantial_rows = valid_rows[valid_rows["Spend"] >= 5.0]
            
            if len(substantial_rows) >= 10:
                # Use winsorized median (cap at 99th percentile to remove extreme outliers)
                roas_values = substantial_rows["ROAS"].values
                cap_value = np.percentile(roas_values, 99)
                winsorized_roas = np.clip(roas_values, 0, cap_value)
                universal_median_roas = np.median(winsorized_roas)
                
                print(f"\n=== UNIVERSAL MEDIAN CALCULATION ===")
                print(f"Total valid rows: {len(valid_rows)}")
                print(f"Substantial spend rows (>=$5): {len(substantial_rows)}")
                print(f"Raw median: {valid_rows['ROAS'].median():.2f}x")
                print(f"99th percentile cap: {cap_value:.2f}x")
                print(f"Winsorized median: {universal_median_roas:.2f}x")
                print(f"=== END UNIVERSAL MEDIAN ===\n")
            else:
                # Not enough substantial data, fall back to all rows
                universal_median_roas = valid_rows["ROAS"].median()
                print(f"⚠️ Using all rows median: {universal_median_roas:.2f}x (only {len(substantial_rows)} rows with spend >=$5)")
        else:
            universal_median_roas = config.get("TARGET_ROAS", 2.5)
            print(f"⚠️ Insufficient data, using TARGET_ROAS: {universal_median_roas:.2f}x")
    
    # 2. Define bucket detection helpers
    AUTO_TYPES = {'close-match', 'loose-match', 'substitutes', 'complements', 'auto'}
    
    def is_auto_or_category(targeting_val):
        t = str(targeting_val).lower().strip()
        if t.startswith("category=") or "category" in t:
            return True
        if t in AUTO_TYPES:
            return True
        return False
    
    def is_pt_targeting(targeting_val):
        t = str(targeting_val).lower().strip()
        if "asin=" in t or "asin-expanded=" in t:
            return True
        if is_asin(t) and not t.startswith("category"):
            return True
        return False
    
    def is_category_targeting(targeting_val):
        t = str(targeting_val).lower().strip()
        return t.startswith("category=") or (t.startswith("category") and "=" in t)
    
    # 3. Build mutually exclusive bucket masks
    # CRITICAL: Auto bucket should ONLY include genuine auto targeting types (close-match, loose-match, etc.)
    # NOT asin-expanded or category targets, even if match_type is "auto" or "-"
    
    # First identify PT and Category targets (takes precedence)
    mask_pt_targeting = df_clean["Targeting"].apply(is_pt_targeting)
    mask_category_targeting = df_clean["Targeting"].apply(is_category_targeting)
    
    # Auto bucket: targeting type is in AUTO_TYPES AND NOT a PT/Category target
    mask_auto_by_targeting = df_clean["Targeting"].apply(lambda x: str(x).lower().strip() in AUTO_TYPES)
    mask_auto_by_matchtype = df_clean["Match Type"].str.lower().isin(["auto", "-"])
    mask_auto = (mask_auto_by_targeting | mask_auto_by_matchtype) & (~mask_pt_targeting) & (~mask_category_targeting)
    
    # PT bucket: PT targeting AND not auto
    mask_pt = mask_pt_targeting & (~mask_auto)
    
    # Category bucket: Category targeting AND not auto/PT
    mask_category = mask_category_targeting & (~mask_auto) & (~mask_pt)
    
    # Exact bucket: Match Type is exact AND not PT/Category/Auto
    mask_exact = (
        (df_clean["Match Type"].str.lower() == "exact") & 
        (~mask_pt) & 
        (~mask_category) &
        (~mask_auto)
    )
    
    # Broad/Phrase bucket: Match Type is broad/phrase AND not PT/Category/Auto
    mask_broad_phrase = (
        df_clean["Match Type"].str.lower().isin(["broad", "phrase"]) & 
        (~mask_pt) & 
        (~mask_category) &
        (~mask_auto)
    )
    
    # 4. Process each bucket
    bids_exact = _process_bucket(df_clean[mask_exact], config, 
                                  min_clicks=config.get("MIN_CLICKS_EXACT", 5), 
                                  bucket_name="Exact",
                                  universal_median_roas=universal_median_roas,
                                  data_days=data_days)
    
    bids_pt = _process_bucket(df_clean[mask_pt], config, 
                               min_clicks=config.get("MIN_CLICKS_PT", 5), 
                               bucket_name="Product Targeting",
                               universal_median_roas=universal_median_roas,
                               data_days=data_days)
    
    bids_agg = _process_bucket(df_clean[mask_broad_phrase], config, 
                                min_clicks=config.get("MIN_CLICKS_BROAD", 10), 
                                bucket_name="Broad/Phrase",
                                universal_median_roas=universal_median_roas,
                                data_days=data_days)
    
    bids_auto = _process_bucket(df_clean[mask_auto], config, 
                                 min_clicks=config.get("MIN_CLICKS_AUTO", 10), 
                                 bucket_name="Auto",
                                 universal_median_roas=universal_median_roas,
                                 data_days=data_days)
    
    bids_category = _process_bucket(df_clean[mask_category], config, 
                                     min_clicks=config.get("MIN_CLICKS_CATEGORY", 10), 
                                     bucket_name="Category",
                                     universal_median_roas=universal_median_roas,
                                     data_days=data_days)
    
    # Combine auto and category for backwards compatibility (displayed as "Auto/Category")
    bids_auto_combined = pd.concat([bids_auto, bids_category], ignore_index=True) if not bids_category.empty else bids_auto
    
    # FINAL ENRICHMENT: Ensure IDs are present for Bulk Export
    bulk = DataHub().get_data('bulk_id_mapping')
    
    bids_exact = enrich_with_ids(bids_exact, bulk)
    bids_pt = enrich_with_ids(bids_pt, bulk)
    bids_agg = enrich_with_ids(bids_agg, bulk)
    bids_auto_combined = enrich_with_ids(bids_auto_combined, bulk)

    return bids_exact, bids_pt, bids_agg, bids_auto_combined

def _process_bucket(segment_df: pd.DataFrame, config: dict, min_clicks: int, bucket_name: str, universal_median_roas: float, data_days: int = 7) -> pd.DataFrame:
    """Unified bucket processor with Bucket Median ROAS classification."""
    if segment_df.empty:
        return pd.DataFrame()
    
    segment_df = segment_df.copy()
    segment_df["_targeting_norm"] = segment_df["Targeting"].astype(str).str.strip().str.lower()
    
    has_keyword_id = "KeywordId" in segment_df.columns and segment_df["KeywordId"].notna().any()
    has_targeting_id = "TargetingId" in segment_df.columns and segment_df["TargetingId"].notna().any()
    
    # CRITICAL FIX: For Auto/Category campaigns, group by Targeting TYPE (from Targeting column)
    # NOT by TargetingId, which contains individual ASIN IDs that can't be bid-adjusted
    is_auto_bucket = bucket_name in ["Auto/Category", "Auto", "Category"]
    
    if is_auto_bucket:
        # For auto campaigns: Use the Targeting column value (close-match, loose-match, substitutes, complements)
        # This preserves targeting type while avoiding individual ASIN grouping
        segment_df["_group_key"] = segment_df["_targeting_norm"]
    elif has_keyword_id or has_targeting_id:
        # For keywords/PT: use IDs for grouping
        segment_df["_group_key"] = segment_df.apply(
            lambda r: str(r.get("KeywordId") or r.get("TargetingId") or r["_targeting_norm"]).strip(),
            axis=1
        )
    else:
        # Fallback: use normalized targeting text
        segment_df["_group_key"] = segment_df["_targeting_norm"]
    
    agg_cols = {"Clicks": "sum", "Spend": "sum", "Sales": "sum", "Impressions": "sum", "Orders": "sum"}
    meta_cols = {c: "first" for c in [
        "Campaign Name", "Ad Group Name", "CampaignId", "AdGroupId", 
        "KeywordId", "TargetingId", "Match Type", "Targeting", "Campaign Targeting Type"
    ] if c in segment_df.columns}
    
    if "Current Bid" in segment_df.columns:
        agg_cols["Current Bid"] = "max"
    if "CPC" in segment_df.columns:
        agg_cols["CPC"] = "mean"
    # NEW: Include bid columns from bulk file if available
    if "Ad Group Default Bid" in segment_df.columns:
        agg_cols["Ad Group Default Bid"] = "first"
    if "Bid" in segment_df.columns:
        agg_cols["Bid"] = "first"
        
    grouped = segment_df.groupby(["Campaign Name", "Ad Group Name", "_group_key"], as_index=False).agg({**agg_cols, **meta_cols})
    grouped = grouped.drop(columns=["_group_key"], errors="ignore")
    grouped["ROAS"] = np.where(grouped["Spend"] > 0, grouped["Sales"] / grouped["Spend"], 0)
    
    # Post-aggregation cleanup for auto campaigns
    if is_auto_bucket:
        # Keep TargetingId if it was successfully mapped earlier
        # (Previously we cleared this which caused bulk export failures)
        pass 
    
    # Calculate bucket ROAS using spend-weighted average (Total Sales / Total Spend)
    # This matches actual bucket performance, not skewed by many 0-sale rows
    bucket_with_spend = grouped[grouped["Spend"] > 0]
    bucket_sample_size = len(bucket_with_spend)
    total_spend = bucket_with_spend["Spend"].sum()
    total_sales = bucket_with_spend["Sales"].sum()
    bucket_weighted_roas = total_sales / total_spend if total_spend > 0 else 0
    
    # Stat sig check
    MIN_SAMPLE_SIZE_FOR_STAT_SIG = 20
    MIN_SPEND_FOR_STAT_SIG = 100  # Need at least AED 100 spend for reliable bucket ROAS
    OUTLIER_THRESHOLD_MULTIPLIER = 1.5
    
    if bucket_sample_size < MIN_SAMPLE_SIZE_FOR_STAT_SIG or total_spend < MIN_SPEND_FOR_STAT_SIG:
        baseline_roas = universal_median_roas
        baseline_source = f"Universal Weighted ROAS (insufficient bucket data: {bucket_sample_size} rows, {total_spend:.0f} spend)"
    else:
        if bucket_weighted_roas > universal_median_roas * OUTLIER_THRESHOLD_MULTIPLIER:
            baseline_roas = universal_median_roas
            baseline_source = f"Universal Weighted ROAS (bucket {bucket_weighted_roas:.2f}x is outlier)"
        else:
            baseline_roas = bucket_weighted_roas
            baseline_source = f"Bucket Weighted ROAS (n={bucket_sample_size}, spend={total_spend:.0f})"
    
    # Sanity check floor
    target_roas = config.get("TARGET_ROAS", 2.5)
    min_acceptable_roas = target_roas * config.get("BUCKET_MEDIAN_FLOOR_MULTIPLIER", 0.5)
    
    if baseline_roas < min_acceptable_roas:
        baseline_roas = min_acceptable_roas
        baseline_source += " [FLOORED]"
    
    print(f"[{bucket_name}] Baseline: {baseline_roas:.2f}x ({baseline_source})")
    
    adgroup_stats = grouped.groupby(["Campaign Name", "Ad Group Name"]).agg({
        "Clicks": "sum", "Spend": "sum", "Sales": "sum", "Orders": "sum"
    }).reset_index()
    adgroup_stats["AG_ROAS"] = np.where(adgroup_stats["Spend"] > 0, adgroup_stats["Sales"] / adgroup_stats["Spend"], 0)
    adgroup_stats["AG_Clicks"] = adgroup_stats["Clicks"]
    adgroup_lookup = adgroup_stats.set_index(["Campaign Name", "Ad Group Name"])[["AG_ROAS", "AG_Clicks"]].to_dict('index')
    
    alpha = config.get("ALPHA", config.get("ALPHA_EXACT", 0.20))
    if "Broad" in bucket_name or "Auto" in bucket_name:
        alpha = config.get("ALPHA_BROAD", alpha * 0.8)
    
    # VISIBILITY BOOST CONFIG
    # Targets with LOW/NO impressions over 2+ weeks = bid not competitive
    # (High impressions + low clicks = CTR problem, not a bid problem)
    # ONLY for explicit keyword campaigns + close-match (advertiser chose these)
    # NOT for loose-match, substitutes, complements, ASIN, category (Amazon decides relevance)
    VISIBILITY_BOOST_MIN_DAYS = 14  # Need at least 2 weeks of data
    VISIBILITY_BOOST_MAX_IMPRESSIONS = 100  # Below this = not winning auctions
    VISIBILITY_BOOST_PCT = 0.30  # 30% boost
    VISIBILITY_BOOST_ELIGIBLE_TYPES = {"exact", "phrase", "broad", "close-match"}  # Only these get boosted
    
    def apply_optimization(r):
        clicks = r["Clicks"]
        impressions = r.get("Impressions", 0)
        roas = r["ROAS"]
        targeting = str(r.get("Targeting", "")).strip().lower()
        match_type = str(r.get("Match Type", "")).strip().lower()
        
        # Priority: Bid (from bulk) → Ad Group Default Bid (from bulk) → CPC (from STR)
        base_bid = float(
            r.get("Bid") if pd.notna(r.get("Bid")) and r.get("Bid") > 0 else
            r.get("Ad Group Default Bid") if pd.notna(r.get("Ad Group Default Bid")) and r.get("Ad Group Default Bid") > 0 else
            r.get("Current Bid", 0) or r.get("CPC", 0) or 0
        )
        
        if base_bid <= 0:
            return 0.0, "Hold: No Bid/CPC Data", "Hold (No Data)"
        
        # VISIBILITY BOOST: 2+ weeks data, <100 impressions = bid not competitive
        # Only for keywords (exact/phrase/broad) and close-match auto
        # Exclude: loose-match, substitutes, complements, ASIN targeting, category targeting
        # Note: 0 impressions = bid SO low it can't even enter auctions (needs boost even more)
        # Paused targets can be identified by state='paused', not by impressions=0
        is_eligible_for_boost = (
            match_type in VISIBILITY_BOOST_ELIGIBLE_TYPES or
            targeting in VISIBILITY_BOOST_ELIGIBLE_TYPES
        )
        
        if (is_eligible_for_boost and 
            data_days >= VISIBILITY_BOOST_MIN_DAYS and 
            impressions < VISIBILITY_BOOST_MAX_IMPRESSIONS):
            new_bid = round(base_bid * (1 + VISIBILITY_BOOST_PCT), 2)
            return new_bid, f"Visibility Boost: Only {impressions} impressions in {data_days} days", "Visibility Boost (+30%)"
        
        if clicks >= min_clicks and roas > 0:
            return _classify_and_bid(roas, baseline_roas, base_bid, alpha, f"targeting|{bucket_name}", config)
        
        ag_key = (r["Campaign Name"], r.get("Ad Group Name", ""))
        ag_stats = adgroup_lookup.get(ag_key, {})
        if ag_stats.get("AG_Clicks", 0) >= min_clicks and ag_stats.get("AG_ROAS", 0) > 0:
            return _classify_and_bid(ag_stats["AG_ROAS"], baseline_roas, base_bid, alpha * 0.5, f"adgroup|{bucket_name}", config)
        
        return base_bid, f"Hold: Insufficient data ({clicks} clicks)", "Hold (Insufficient Data)"
    
    opt_results = grouped.apply(apply_optimization, axis=1)
    grouped["New Bid"] = opt_results.apply(lambda x: x[0])
    grouped["Reason"] = opt_results.apply(lambda x: x[1])
    grouped["Decision_Basis"] = opt_results.apply(lambda x: x[2])
    grouped["Bucket"] = bucket_name
    
    # SYNC: Create recommendation objects for validation
    if not grouped.empty:
        def create_bid_rec(row):
            rec_id = f"bid_{row['Campaign Name']}_{row['Ad Group Name']}_{row['Targeting']}"
            
            # Determine recommendation type
            new_bid = row['New Bid']
            current_bid = row.get('Current Bid', 0)
            
            rec_type = RecommendationType.BID_INCREASE if new_bid > current_bid else RecommendationType.BID_DECREASE
            if new_bid == current_bid:
                # No change, but still create for validation check if desired
                # Actually, only validate if it's an actionable change
                if row['Decision_Basis'] == "Hold (Insufficient Data)" or row['Decision_Basis'] == "Hold (No Data)":
                    return None
            
            # Determine if this is PT or Keyword based on bucket
            is_pt = row['Bucket'] in ["Product Targeting", "Auto"]
            
            rec = OptimizationRecommendation(
                recommendation_id=rec_id,
                recommendation_type=rec_type,
                campaign_name=row['Campaign Name'],
                campaign_id=row.get('CampaignId', ""),
                campaign_targeting_type=row.get('Campaign Targeting Type', "Manual"),
                ad_group_name=row['Ad Group Name'],
                ad_group_id=row.get('AdGroupId', ""),
                keyword_text=row['Targeting'] if not is_pt else None,
                product_targeting_expression=row['Targeting'] if is_pt else None,
                match_type=row['Match Type'],
                current_bid=float(current_bid) if pd.notna(current_bid) else 0.0,
                new_bid=float(new_bid),
                currency=config.get("currency", "AED")
            )
            rec.validation_result = validate_recommendation(rec)
            return rec
            
        grouped['recommendation'] = grouped.apply(create_bid_rec, axis=1)
    
    return grouped


def _classify_and_bid(roas: float, median_roas: float, base_bid: float, alpha: float, 
                      data_source: str, config: dict) -> Tuple[float, str, str]:
    """Classify ROAS vs bucket baseline and determine bid action."""
    max_change = config.get("MAX_BID_CHANGE", 0.25)
    THRESHOLD_BAND = 0.10
    promote_threshold = median_roas * (1 + THRESHOLD_BAND)
    stable_threshold = median_roas * (1 - THRESHOLD_BAND)
    
    if roas >= promote_threshold:
        adjustment = min(alpha, max_change)
        new_bid = base_bid * (1 + adjustment)
        reason = f"Promote: ROAS {roas:.2f} ≥ {promote_threshold:.2f} ({data_source})"
        action = "promote"
    elif roas >= stable_threshold:
        new_bid = base_bid
        reason = f"Stable: ROAS {roas:.2f} ~ {median_roas:.2f} ({data_source})"
        action = "stable"
    else:
        adjustment = min(alpha, max_change)
        new_bid = base_bid * (1 - adjustment)
        reason = f"Bid Down: ROAS {roas:.2f} < {stable_threshold:.2f} ({data_source})"
        action = "bid_down"
    
    min_allowed = max(BID_LIMITS["MIN_BID_FLOOR"], base_bid * BID_LIMITS["MIN_BID_MULTIPLIER"])
    max_allowed = base_bid * BID_LIMITS["MAX_BID_MULTIPLIER"]
    new_bid = np.clip(new_bid, min_allowed, max_allowed)
    
    return new_bid, reason, action


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
    """Create performance heatmap with action tracking."""
    grouped = df.groupby(["Campaign Name", "Ad Group Name"]).agg({
        "Clicks": "sum", "Spend": "sum", "Sales_Attributed": "sum",
        "Orders_Attributed": "sum", "Impressions": "sum"
    }).reset_index()
    
    # Rename to standard names for consistency
    grouped = grouped.rename(columns={"Sales_Attributed": "Sales", "Orders_Attributed": "Orders"})
    
    grouped["CTR"] = np.where(grouped["Impressions"] > 0, grouped["Clicks"] / grouped["Impressions"] * 100, 0)
    grouped["CVR"] = np.where(grouped["Clicks"] > 0, grouped["Orders"] / grouped["Clicks"] * 100, 0)
    grouped["ROAS"] = np.where(grouped["Spend"] > 0, grouped["Sales"] / grouped["Spend"], 0)
    grouped["ACoS"] = np.where(grouped["Sales"] > 0, grouped["Spend"] / grouped["Sales"] * 100, 999)
    
    grouped["Harvest_Count"] = 0
    grouped["Negative_Count"] = 0
    grouped["Bid_Increase_Count"] = 0
    grouped["Bid_Decrease_Count"] = 0
    grouped["Actions_Taken"] = ""
    
    all_bids = pd.concat([direct_bids, agg_bids]) if not direct_bids.empty or not agg_bids.empty else pd.DataFrame()
    negatives_df = pd.concat([neg_kw, neg_pt]) if not neg_kw.empty or not neg_pt.empty else pd.DataFrame()

    for idx, row in grouped.iterrows():
        camp, ag = row["Campaign Name"], row["Ad Group Name"]
        
        # Safely filter dataframes even if empty or missing columns
        h_match = pd.DataFrame()
        if not harvest_df.empty and "Campaign Name" in harvest_df.columns:
            h_match = harvest_df[(harvest_df["Campaign Name"] == camp) & (harvest_df.get("Ad Group Name", "") == ag)]
            
        n_match = pd.DataFrame()
        if not negatives_df.empty and "Campaign Name" in negatives_df.columns:
            n_match = negatives_df[(negatives_df["Campaign Name"] == camp) & (negatives_df.get("Ad Group Name", "") == ag)]
            
        b_match = pd.DataFrame()
        if not all_bids.empty and "Campaign Name" in all_bids.columns:
            b_match = all_bids[(all_bids["Campaign Name"] == camp) & (all_bids.get("Ad Group Name", "") == ag)]
        
        grouped.at[idx, "Harvest_Count"] = len(h_match)
        grouped.at[idx, "Negative_Count"] = len(n_match)
        
        # Collect reasons for Actions
        reasons = []
        if not h_match.empty and "Reason" in h_match.columns:
            reasons.extend(h_match["Reason"].dropna().astype(str).unique().tolist())
            
        if not n_match.empty and "Reason" in n_match.columns:
            reasons.extend(n_match["Reason"].dropna().astype(str).unique().tolist())
            
        if not b_match.empty and "New Bid" in b_match.columns:
            cur_bids = b_match.get("Current Bid", b_match.get("CPC", 0))
            grouped.at[idx, "Bid_Increase_Count"] = (b_match["New Bid"] > cur_bids).sum()
            grouped.at[idx, "Bid_Decrease_Count"] = (b_match["New Bid"] < cur_bids).sum()
            
            if "Reason" in b_match.columns:
                reasons.extend(b_match["Reason"].dropna().astype(str).unique().tolist())
            
        actions = []
        if grouped.at[idx, "Harvest_Count"] > 0: actions.append(f"💎 {int(grouped.at[idx, 'Harvest_Count'])} harvests")
        if grouped.at[idx, "Negative_Count"] > 0: actions.append(f"🛑 {int(grouped.at[idx, 'Negative_Count'])} negatives")
        if grouped.at[idx, "Bid_Increase_Count"] > 0: actions.append(f"⬆️ {int(grouped.at[idx, 'Bid_Increase_Count'])} increases")
        if grouped.at[idx, "Bid_Decrease_Count"] > 0: actions.append(f"⬇️ {int(grouped.at[idx, 'Bid_Decrease_Count'])} decreases")
        
        if actions:
            grouped.at[idx, "Actions_Taken"] = " | ".join(actions)
            # Summarize reasons (top 3 unique)
            unique_reasons = sorted(list(set([r for r in reasons if r])))
            if unique_reasons:
                grouped.at[idx, "Reason_Summary"] = "; ".join(unique_reasons[:3]) + ("..." if len(unique_reasons) > 3 else "")
            else:
                grouped.at[idx, "Reason_Summary"] = "Multiple actions"
        elif row["Clicks"] < config.get("MIN_CLICKS_EXACT", 5):
            grouped.at[idx, "Actions_Taken"] = "⏸️ Hold (Low volume)"
            grouped.at[idx, "Reason_Summary"] = "Low data volume"
        else:
            grouped.at[idx, "Actions_Taken"] = "✅ No action needed"
            
            # Provide more specific status based on performance
            if row["Sales"] == 0 and row["Spend"] > 10:
                grouped.at[idx, "Reason_Summary"] = "Zero Sales (Monitoring)"
            elif row["ROAS"] < config.get("TARGET_ROAS", 2.5) * 0.8:
                grouped.at[idx, "Reason_Summary"] = "Low Efficiency (Monitoring)"
            else:
                grouped.at[idx, "Reason_Summary"] = "Stable Performance"

    # Priority Scoring
    def score(val, series, high_is_better=True):
        valid = series[series > 0]
        if len(valid) < 2: return 1
        p33, p67 = valid.quantile(0.33), valid.quantile(0.67)
        return (2 if val >= p67 else 1 if val >= p33 else 0) if high_is_better else (2 if val <= p33 else 1 if val <= p67 else 0)

    grouped["Overall_Score"] = (grouped.apply(lambda r: score(r["CTR"], grouped["CTR"]), axis=1) + 
                                grouped.apply(lambda r: score(r["CVR"], grouped["CVR"]), axis=1) + 
                                grouped.apply(lambda r: score(r["ROAS"], grouped["ROAS"]), axis=1) + 
                                grouped.apply(lambda r: score(r["ACoS"], grouped["ACoS"], False), axis=1)) / 4
    
    grouped["Priority"] = grouped["Overall_Score"].apply(lambda x: "🔴 High" if x < 0.7 else ("🟡 Medium" if x < 1.3 else "🟢 Good"))
    return grouped.sort_values("Overall_Score")

# ==========================================
# BULK GENERATION & LOGGING
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
    total_sales = df["Sales"].sum()
    total_orders = df["Orders"].sum()
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
            current_orders = float(row.get("Orders", 0) or 0)
            current_sales = float(row.get("Sales", 0) or 0)
            current_cpc = float(row.get("CPC", 0) or row.get("Cost Per Click (CPC)", 0) or 0)
            
            if current_clicks == 0 and current_cpc == 0:
                continue
            
            current_cvr = current_orders / current_clicks if current_clicks > 0 else 0
            current_aov = current_sales / current_orders if current_orders > 0 else 0
            current_roas = current_sales / current_spend if current_spend > 0 else 0
            
            if current_aov == 0 and baseline["orders"] > 0:
                current_aov = baseline["sales"] / baseline["orders"]
            
            # Calculate baseline ROAS for comparison
            baseline_roas = baseline["sales"] / baseline["spend"] if baseline["spend"] > 0 else 1.0
            
            # Apply elasticity with ROAS-aware adjustment
            new_cpc = current_cpc * (1 + elasticity["cpc"] * bid_change_pct)
            new_clicks = current_clicks * (1 + elasticity["clicks"] * bid_change_pct)
            
            # KEY FIX: When DECREASING bids on LOW-ROAS targets, CVR doesn't decrease
            # because we're cutting wasteful traffic, not good traffic
            if bid_change_pct < 0 and current_roas < baseline_roas:
                # Below-average ROAS target: cutting this traffic is GOOD
                # CVR stays same or improves slightly (removing untargeted clicks)
                new_cvr = current_cvr * (1 + abs(elasticity["cvr"] * bid_change_pct * 0.2))
            else:
                # Normal case: CVR follows elasticity
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
            
            # Use launch multiplier (2x) for harvest bids
            launch_mult = config.get("HARVEST_LAUNCH_MULTIPLIER", 2.0)
            new_bid = float(row.get("New Bid", base_cpc * launch_mult) or base_cpc * launch_mult)
            base_cvr = base_orders / base_clicks if base_clicks > 0 else 0
            base_aov = base_sales / base_orders if base_orders > 0 else 0
            base_roas = base_sales / base_spend if base_spend > 0 else 0
            
            # FIXED HARVEST LOGIC:
            # Harvest moves proven performers to exact match
            # They already convert well - we're just isolating them for better control
            # 
            # Key insight: Harvest terms are ABOVE-AVERAGE performers
            # Moving them to exact match gives:
            # 1. Same/similar traffic (they're proven keywords)
            # 2. Slightly LOWER CPC (exact match is more efficient than broad)
            # 3. HIGHER CVR (focused traffic, no irrelevant queries)
            
            # Model: Same clicks, 10% lower CPC, 15%+ better CVR
            fore_clicks = base_clicks
            fore_cpc = base_cpc * 0.90  # Exact match is typically more efficient
            fore_cvr = base_cvr * efficiency  # 1.15x CVR improvement (default)
            
            fore_orders = fore_clicks * fore_cvr
            fore_sales = fore_orders * base_aov
            fore_spend = fore_clicks * fore_cpc
            
            forecasted_changes.append({
                "delta_clicks": fore_clicks - base_clicks,
                "delta_spend": fore_spend - base_spend,  # Should be NEGATIVE (savings!)
                "delta_sales": fore_sales - base_sales,  # Should be POSITIVE (better CVR!)
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
                "factors": ", ".join(risk_factors)
            })
        elif len(risk_factors) == 1:
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

# NOTE: EXPORT_COLUMNS, generate_negatives_bulk, generate_bids_bulk, generate_harvest_bulk
# are imported from features/bulk_export.py at the top of this file

def _log_optimization_events(results: dict, client_id: str, report_date: str):
    """
    Standardizes and logs optimization actions (bids, negatives, harvests).
    
    If user has already accepted actions this session (optimizer_actions_accepted=True),
    writes directly to DB and shows undo toast.
    Otherwise, stores in session state for confirmation when leaving optimizer tab.
    """
    from core.db_manager import get_db_manager
    import uuid
    import streamlit as st
    import time
    
    batch_id = str(uuid.uuid4())[:8]
    actions_to_log = []

    # 1. Process Negative Keywords
    for _, row in results.get('neg_kw', pd.DataFrame()).iterrows():
        actions_to_log.append({
            'entity_name': 'Keyword',
            'action_type': 'NEGATIVE',
            'old_value': 'ENABLED',
            'new_value': 'PAUSED',
            'reason': row.get('Reason', 'Low efficiency / Waste'),
            'campaign_name': row.get('Campaign Name', ''),
            'ad_group_name': row.get('Ad Group Name', ''),
            'target_text': row.get('Term', ''),
            'match_type': row.get('Match Type', 'NEGATIVE')
        })

    # 2. Process Negative Product Targets (ASINs)
    for _, row in results.get('neg_pt', pd.DataFrame()).iterrows():
        actions_to_log.append({
            'entity_name': 'ASIN',
            'action_type': 'NEGATIVE',
            'old_value': 'ENABLED',
            'new_value': 'PAUSED',
            'reason': row.get('Reason', 'Low efficiency / Waste'),
            'campaign_name': row.get('Campaign Name', ''),
            'ad_group_name': row.get('Ad Group Name', ''),
            'target_text': row.get('Term', ''),
            'match_type': 'TARGETING_EXPRESSION'
        })

    # 3. Process Bid Optimizations (Combined)
    bid_dfs = [
        results.get('bids_exact', pd.DataFrame()),
        results.get('bids_pt', pd.DataFrame()),
        results.get('bids_agg', pd.DataFrame()),
        results.get('bids_auto', pd.DataFrame())
    ]
    for b_df in bid_dfs:
        if b_df.empty: continue
        for _, row in b_df.iterrows():
            actions_to_log.append({
                'entity_name': 'Target',
                'action_type': 'BID_CHANGE',
                'old_value': str(row.get('Current Bid', '')),
                'new_value': str(row.get('New Bid', '')),
                'reason': row.get('Reason', 'Portfolio Optimization'),
                'campaign_name': row.get('Campaign Name', ''),
                'ad_group_name': row.get('Ad Group Name', ''),
                'target_text': row.get('Targeting', ''),
                'match_type': row.get('Match Type', '')
            })

    # 4. Process Harvests - WITH WINNER SOURCE TRACKING
    for _, row in results.get('harvest', pd.DataFrame()).iterrows():
        # Determine winner source campaign and new campaign name
        winner_campaign = row.get('Campaign Name', '')
        search_term = row.get('Customer Search Term', '')
        
        # Generate new campaign name (you can customize this logic)
        new_campaign = f"Harvest_Exact_{winner_campaign}" if winner_campaign else "Harvest_Exact_Campaign"
        
        actions_to_log.append({
            'entity_name': 'Keyword',
            'action_type': 'HARVEST',
            'old_value': 'DISCOVERY',
            'new_value': 'PROMOTED',
            'reason': f"Conv: {row.get('Orders', 0)} orders",
            'campaign_name': winner_campaign,  # Source campaign
            'ad_group_name': row.get('Ad Group Name', ''),
            'target_text': search_term,
            'match_type': 'EXACT',
            # NEW FIELDS FOR IMPACT ANALYSIS:
            'winner_source_campaign': winner_campaign,  # Which campaign won
            'new_campaign_name': new_campaign,  # Where it's being moved
            'before_match_type': row.get('Match Type', 'broad'),  # Original match type
            'after_match_type': 'exact'  # Harvested to exact
        })


    # === DEDUPLICATE ACTIONS ===
    # Remove duplicates that would violate the unique constraint:
    # (client_id, action_date, target_text, action_type, campaign_name)
    # Keep the last occurrence (most recent values for the same target)
    seen_keys = {}
    for i, action in enumerate(actions_to_log):
        key = (
            action.get('target_text', '').lower().strip(),
            action.get('action_type', ''),
            action.get('campaign_name', '').strip()
        )
        seen_keys[key] = i  # Overwrite with latest index
    
    # Build deduplicated list (keeping only the last occurrence of each key)
    unique_indices = set(seen_keys.values())
    actions_to_log = [a for i, a in enumerate(actions_to_log) if i in unique_indices]

    if not actions_to_log:
        return 0
    
    # PENDING ACTIONS WORKFLOW: Store actions in session state for confirmation on tab exit
    # This allows user to run multiple scenarios before committing to actions_log table
    st.session_state['pending_actions'] = {
        'actions': actions_to_log,
        'client_id': client_id,
        'batch_id': batch_id,
        'report_date': report_date
    }
    
    # Show user feedback that actions are ready (not saved yet)
    st.info(f"📋 {len(actions_to_log)} actions ready. Will prompt to save when you leave this tab.", icon="📋")
    
    return len(actions_to_log)


# ==========================================
# STREAMLIT UI MODULE
# ==========================================

class OptimizerModule(BaseFeature):
    """Complete Bid Optimization Engine."""
    
    def __init__(self):
        super().__init__()
        
        # Check if optimizer_config exists in session state
        if 'optimizer_config' in st.session_state:
            self.config = st.session_state['optimizer_config'].copy()
        else:
            self.config = DEFAULT_CONFIG.copy()
            
        self.results = {}
        
        # Initialize session state with defaults for widgets
        config_source = self.config
        widget_defaults = {
            "opt_harvest_clicks": config_source.get("HARVEST_CLICKS", 10),
            "opt_harvest_orders": config_source.get("HARVEST_ORDERS", 3),
            # opt_harvest_sales removed - no currency thresholds
            "opt_harvest_roas_mult": int(config_source.get("HARVEST_ROAS_MULT", 0.8) * 100),
            "opt_alpha_exact": int(config_source.get("ALPHA_EXACT", 0.15) * 100),
            "opt_alpha_broad": int(config_source.get("ALPHA_BROAD", 0.10) * 100),
            "opt_max_bid_change": int(config_source.get("MAX_BID_CHANGE", 0.20) * 100),
            "opt_target_roas": config_source.get("TARGET_ROAS", 2.5),
            "opt_neg_clicks_threshold": config_source.get("NEGATIVE_CLICKS_THRESHOLD", 10),
            # opt_neg_spend_threshold removed - no currency thresholds
            "opt_min_clicks_exact": config_source.get("MIN_CLICKS_EXACT", 5),
            "opt_min_clicks_pt": config_source.get("MIN_CLICKS_PT", 5),
            "opt_min_clicks_broad": config_source.get("MIN_CLICKS_BROAD", 10),
            "opt_min_clicks_auto": config_source.get("MIN_CLICKS_AUTO", 10),
        }
        for key, default in widget_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default
    
    def render_ui(self):
        self.render_header("PPC Optimizer", "optimizer")
        hub = DataHub()
        if not hub.is_loaded("search_term_report"):
            st.warning("⚠️ Please upload a Search Term Report first.")
            return
        
        df = hub.get_enriched_data() or hub.get_data("search_term_report")
        self._render_sidebar()
        
        # Share config globally
        st.session_state['optimizer_config'] = self.config
        
        if st.session_state.get("run_optimizer"):
            self._run_analysis(df)
            st.session_state["run_optimizer"] = False
            self._display_results()
        elif 'optimizer_results' in st.session_state:
            self.results = st.session_state['optimizer_results']
            self._display_results()
        else:
            self._display_summary(df)
            st.info("👈 Click **Run Optimizer** to start")
    
    def _render_sidebar(self):
        """Render sidebar configuration panels."""
        # SVG Icons for Sidebar
        icon_color = "#8F8CA3"
        settings_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>'
        bolt_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>'

        with st.sidebar:
            st.divider()
            st.markdown(f'<div style="color: #8F8CA3; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; margin-bottom: 12px;">{settings_icon}Optimizer Settings</div>', unsafe_allow_html=True)
            
            # === PRESETS ===
            st.markdown(f'<div style="color: #F5F5F7; font-size: 0.85rem; font-weight: 600; margin-bottom: 8px;">{bolt_icon}Quick Presets</div>', unsafe_allow_html=True)
            
            preset_options = ["Conservative", "Balanced", "Aggressive"]
            active_preset = st.session_state.get("last_preset", "Conservative")
            preset_idx = preset_options.index(active_preset) if active_preset in preset_options else 0
            
            preset = st.radio(
                "preset_selector",
                preset_options,
                index=preset_idx,
                horizontal=True,
                label_visibility="collapsed",
                key="opt_preset"
            )
            
            # Define preset values (no currency thresholds - only clicks-based)
            preset_configs = {
                "Conservative": {
                    "harvest_clicks": 15, "harvest_orders": 4, "harvest_roas": 90,
                    "alpha_exact": 15, "alpha_broad": 12, "max_change": 15, "target_roas": 2.5,
                    "neg_clicks": 15,
                    "min_clicks_exact": 8, "min_clicks_pt": 8, "min_clicks_broad": 12, "min_clicks_auto": 12
                },
                "Balanced": {
                    "harvest_clicks": 10, "harvest_orders": 3, "harvest_roas": 80,
                    "alpha_exact": 20, "alpha_broad": 16, "max_change": 20, "target_roas": 2.5,
                    "neg_clicks": 10,
                    "min_clicks_exact": 5, "min_clicks_pt": 5, "min_clicks_broad": 10, "min_clicks_auto": 10
                },
                "Aggressive": {
                    "harvest_clicks": 8, "harvest_orders": 2, "harvest_roas": 70,
                    "alpha_exact": 25, "alpha_broad": 20, "max_change": 25, "target_roas": 2.5,
                    "neg_clicks": 8,
                    "min_clicks_exact": 3, "min_clicks_pt": 3, "min_clicks_broad": 8, "min_clicks_auto": 8
                }
            }
            
            # Apply preset to session state if changed
            if "last_preset" not in st.session_state or st.session_state["last_preset"] != preset:
                st.session_state["last_preset"] = preset
                config = preset_configs[preset]
                st.session_state["opt_harvest_clicks"] = config["harvest_clicks"]
                st.session_state["opt_harvest_orders"] = config["harvest_orders"]
                st.session_state["opt_harvest_roas_mult"] = config["harvest_roas"]
                st.session_state["opt_alpha_exact"] = config["alpha_exact"]
                st.session_state["opt_alpha_broad"] = config["alpha_broad"]
                st.session_state["opt_max_bid_change"] = config["max_change"]
                st.session_state["opt_target_roas"] = config["target_roas"]
                st.session_state["opt_neg_clicks_threshold"] = config["neg_clicks"]
                st.session_state["opt_min_clicks_exact"] = config["min_clicks_exact"]
                st.session_state["opt_min_clicks_pt"] = config["min_clicks_pt"]
                st.session_state["opt_min_clicks_broad"] = config["min_clicks_broad"]
                st.session_state["opt_min_clicks_auto"] = config["min_clicks_auto"]
            
            st.caption("*Select preset or customize below*")
            st.divider()
            
            # === PRIMARY ACTION PANEL ===
            st.subheader("Ready to optimize")
            
            st.markdown(
                "The system will adjust bids, add negatives, and harvest high-performing terms "
                "based on current account performance."
            )
            
            # Brand purple/wine palette: 
            # Primary: #5B556F (Wine/Slate Purple)
            # Secondary: rgba(91, 85, 111, 0.8)
            
            st.markdown("""
            <style>
            /* Primary CTA Button - Brand Wine Gradient */
            [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
                background: linear-gradient(135deg, #5B556F 0%, #464156 100%) !important;
                border: 1px solid rgba(255, 255, 255, 0.05) !important;
                font-weight: 600 !important;
                letter-spacing: 0.3px !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
            }
            [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:hover {
                background: linear-gradient(135deg, #6A6382 0%, #5B556F 100%) !important;
                transform: translateY(-1px);
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15) !important;
            }

            /* Slider Styling - Brand Wine Accents */
            /* Track / Base */
            div[data-testid="stSlider"] div[aria-label="slider-track"] {
                background: rgba(91, 85, 111, 0.15) !important;
            }
            /* Progress Bar */
            div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div:nth-child(2) {
                background: #5B556F !important;
            }
            /* Handle / Thumb */
            div[data-testid="stSlider"] div[role="slider"] {
                background-color: #5B556F !important;
                border: 2px solid #F5F5F7 !important;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
            }
            /* Value Label */
            div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
                color: #B6B4C2 !important;
            }
            div[data-testid="stSlider"] span[data-baseweb="typography"] {
                color: #5B556F !important;
                font-weight: 700 !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Primary CTA
            if st.button(
                "Run Optimizer",
                type="primary",
                use_container_width=True,
                key="opt_run_primary",
                help="Run optimization with current settings"
            ):
                st.session_state["run_optimizer"] = True
                st.rerun()
            
            # Preview metrics (if results available from previous run)
            if "optimizer_results" in st.session_state:
                results = st.session_state["optimizer_results"]
                harvest = results.get("harvest", pd.DataFrame())
                neg_kw = results.get("neg_kw", pd.DataFrame())
                neg_pt = results.get("neg_pt", pd.DataFrame())
                direct_bids = results.get("direct_bids", pd.DataFrame())
                agg_bids = results.get("agg_bids", pd.DataFrame())
                
                bid_count = len(direct_bids) + len(agg_bids) if direct_bids is not None and agg_bids is not None else 0
                neg_count = len(neg_kw) + len(neg_pt) if neg_kw is not None and neg_pt is not None else 0
                harvest_count = len(harvest) if harvest is not None else 0
                
                # Paused count (targets with new bid = 0 or state = paused)
                pause_count = 0
                if direct_bids is not None and not direct_bids.empty and "New Bid" in direct_bids.columns:
                    pause_count += (direct_bids["New Bid"] == 0).sum()
                if agg_bids is not None and not agg_bids.empty and "New Bid" in agg_bids.columns:
                    pause_count += (agg_bids["New Bid"] == 0).sum()
                
                st.caption("**Last run preview:**")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Bid updates", f"{bid_count:,}")
                with c2:
                    st.metric("Negatives", f"{neg_count:,}")
                with c3:
                    st.metric("Harvests", f"{harvest_count:,}")
                with c4:
                    st.metric("Paused", f"{pause_count:,}")
            

            # Sync all session state values to self.config (convert integers to decimals)
            self.config["HARVEST_CLICKS"] = st.session_state["opt_harvest_clicks"]
            self.config["HARVEST_ORDERS"] = st.session_state["opt_harvest_orders"]
            # self.config["HARVEST_SALES"] = st.session_state["opt_harvest_sales"]  # Removed: currency threshold
            self.config["HARVEST_ROAS_MULT"] = st.session_state["opt_harvest_roas_mult"] / 100.0
            self.config["ALPHA_EXACT"] = st.session_state["opt_alpha_exact"] / 100.0
            self.config["ALPHA_BROAD"] = st.session_state["opt_alpha_broad"] / 100.0
            self.config["MAX_BID_CHANGE"] = st.session_state["opt_max_bid_change"] / 100.0
            self.config["TARGET_ROAS"] = st.session_state["opt_target_roas"]
            self.config["NEGATIVE_CLICKS_THRESHOLD"] = st.session_state["opt_neg_clicks_threshold"]
            # self.config["NEGATIVE_SPEND_THRESHOLD"] = st.session_state["opt_neg_spend_threshold"]  # Removed: currency threshold
            self.config["MIN_CLICKS_EXACT"] = st.session_state["opt_min_clicks_exact"]
            self.config["MIN_CLICKS_PT"] = st.session_state["opt_min_clicks_pt"]
            self.config["MIN_CLICKS_BROAD"] = st.session_state["opt_min_clicks_broad"]
            self.config["MIN_CLICKS_AUTO"] = st.session_state["opt_min_clicks_auto"]


    def _calculate_account_health(self, df: pd.DataFrame, r: dict) -> dict:
        """Calculate account health diagnostics for dashboard display (Last 30 Days from DB)."""
        from datetime import timedelta
        from core.db_manager import get_db_manager
        
        # Get data from database for accurate last 30 days (not just uploaded CSV)
        try:
            db = get_db_manager(st.session_state.get('test_mode', False))
            client_id = st.session_state.get('active_account_id')
            
            if not db or not client_id:
                # Fallback to uploaded data if DB not available
                df_filtered = df.copy()
            else:
                # Pull from database to get full historical context
                df_db = db.get_target_stats_by_account(client_id, limit=50000)
                
                if df_db is None or df_db.empty:
                    # No DB data, use uploaded CSV
                    df_filtered = df.copy()
                else:
                    # Use database data for last 30 days
                    df_db['start_date'] = pd.to_datetime(df_db['start_date'], errors='coerce')
                    valid_dates = df_db['start_date'].dropna()
                    
                    if not valid_dates.empty:
                        max_date = valid_dates.max()
                        cutoff_date = max_date - timedelta(days=30)
                        df_filtered = df_db[df_db['start_date'] >= cutoff_date].copy()
                        
                        # Map DB columns to expected optimizer columns
                        df_filtered = df_filtered.rename(columns={
                            'spend': 'Spend',
                            'sales': 'Sales',
                            'orders': 'Orders',
                            'clicks': 'Clicks'
                        })
                    else:
                        df_filtered = df.copy()
        except Exception as e:
            # On any error, fall back to uploaded data
            print(f"Health calc DB error: {e}")
            df_filtered = df.copy()
        
        # Ensure we have required columns
        if 'Spend' not in df_filtered.columns or 'Sales' not in df_filtered.columns:
            # Return empty health if data is invalid
            return {
                "health_score": 0,
                "roas_score": 0,
                "efficiency_score": 0,
                "cvr_score": 0,
                "efficiency_rate": 0,
                "waste_ratio": 100,
                "wasted_spend": 0,
                "current_roas": 0,
                "current_acos": 0,
                "cvr": 0,
                "total_spend": 0,
                "total_sales": 0
            }
        
        # Calculate metrics
        total_spend = df_filtered['Spend'].sum()
        total_sales = df_filtered['Sales'].sum()
        total_orders = df_filtered.get('Orders', pd.Series([0])).sum()
        total_clicks = df_filtered.get('Clicks', pd.Series([0])).sum()
        
        current_roas = total_sales / total_spend if total_spend > 0 else 0
        current_acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
        
        # Efficiency calculation - ROW LEVEL, not aggregated
        # Each row is Campaign->AdGroup->Target->Date, check conversion at that granularity
        converting_spend = df_filtered.loc[df_filtered.get('Orders', 0) > 0, 'Spend'].sum()
        
        efficiency_rate = (converting_spend / total_spend * 100) if total_spend > 0 else 0
        wasted_spend = total_spend - converting_spend
        waste_ratio = 100 - efficiency_rate
        
        cvr = (total_orders / total_clicks * 100) if total_clicks > 0 else 0
        
        roas_score = min(100, current_roas / 4.0 * 100)
        efficiency_score = efficiency_rate
        cvr_score = min(100, cvr / 10.0 * 100)
        health_score = (roas_score * 0.4 + efficiency_score * 0.4 + cvr_score * 0.2)
        
        health_metrics = {
            "health_score": health_score,
            "roas_score": roas_score,
            "efficiency_score": efficiency_score,
            "cvr_score": cvr_score,
            "efficiency_rate": efficiency_rate,
            "waste_ratio": waste_ratio,
            "wasted_spend": wasted_spend,
            "current_roas": current_roas,
            "current_acos": current_acos,
            "cvr": cvr,
            "total_spend": total_spend,
            "total_sales": total_sales
        }
        
        # Persist to database for Home tab cockpit
        try:
            if db and client_id:
                db.save_account_health(client_id, health_metrics)
        except Exception:
            pass  # Don't break optimizer if DB save fails
        
        return health_metrics

    def _run_analysis(self, df):
        df, date_info = prepare_data(df, self.config)
        benchmarks = calculate_account_benchmarks(df, self.config)
        universal_median = benchmarks.get('universal_median_roas', self.config.get("TARGET_ROAS", 2.5))
        
        matcher = ExactMatcher(df)
        
        harvest = identify_harvest_candidates(df, self.config, matcher, benchmarks)
        neg_kw, neg_pt, your_products = identify_negative_candidates(df, self.config, harvest, benchmarks)
        
        neg_set = set(zip(neg_kw["Campaign Name"], neg_kw["Ad Group Name"], neg_kw["Term"].str.lower()))
        data_days = date_info.get("days", 7) if date_info else 7
        bids_ex, bids_pt, bids_agg, bids_auto = calculate_bid_optimizations(df, self.config, set(harvest["Customer Search Term"].str.lower()), neg_set, universal_median, data_days=data_days)
        
        heatmap = create_heatmap(df, self.config, harvest, neg_kw, neg_pt, pd.concat([bids_ex, bids_pt]), pd.concat([bids_agg, bids_auto]))
        
        self.results = {
            "df": df, "date_info": date_info, "harvest": harvest, "neg_kw": neg_kw, "neg_pt": neg_pt,
            "your_products_review": your_products, 
            "bids_exact": bids_ex, "bids_pt": bids_pt, "bids_agg": bids_agg, "bids_auto": bids_auto,
            "direct_bids": pd.concat([bids_ex, bids_pt]),
            "agg_bids": pd.concat([bids_agg, bids_auto]), "heatmap": heatmap,
            "simulation": run_simulation(df, pd.concat([bids_ex, bids_pt]), pd.concat([bids_agg, bids_auto]), harvest, self.config, date_info)
        }
        st.session_state['optimizer_results'] = self.results

    def _display_dashboard_v2(self, results):
        """Delegate to standalone overview module for lazy loading."""
        from features.overview_tab import render_overview_tab
        render_overview_tab(results)

    def _extract_validation_info(self, df):
        """Extract status icon and issues from recommendation objects in DataFrame."""
        if df.empty or 'recommendation' not in df.columns:
            return df
            
        df = df.copy()
        def get_status(rec):
            if not isinstance(rec, OptimizationRecommendation):
                return "✅"
            return rec.get_status_icon()
            
        def get_issues(rec):
            if not isinstance(rec, OptimizationRecommendation):
                return ""
            all_msgs = []
            if rec.errors:
                all_msgs.extend([e['message'] for e in rec.errors])
            if rec.warnings:
                all_msgs.extend([w['message'] for w in rec.warnings])
            return "; ".join(all_msgs)
            
        df['Status'] = df['recommendation'].apply(get_status)
        df['Validation Issues'] = df['recommendation'].apply(get_issues)
        
        # Ensure Status is at the front
        cols = ['Status'] + [c for c in df.columns if c not in ['Status', 'recommendation']]
        return df[cols]

    def _display_negatives(self, neg_kw, neg_pt):
        """Delegate to standalone negatives module for lazy loading."""
        from features.negatives_tab import render_negatives_tab
        render_negatives_tab(neg_kw, neg_pt, extract_validation_fn=self._extract_validation_info)

    def _display_bids(self, bids_exact=None, bids_pt=None, bids_agg=None, bids_auto=None):
        """Delegate to standalone bids module for lazy loading."""
        from features.bids_tab import render_bids_tab
        render_bids_tab(bids_exact, bids_pt, bids_agg, bids_auto, extract_validation_fn=self._extract_validation_info)

    def _display_harvest(self, harvest_df):
        """Delegate to standalone harvest module for lazy loading."""
        from features.harvest_tab import render_harvest_tab
        render_harvest_tab(harvest_df)

    def _display_heatmap(self, heatmap_df):
        """Delegate to standalone audit module for lazy loading."""
        from features.audit_tab import render_audit_tab
        render_audit_tab(heatmap_df)

    def _group_issues(self, issues):
        """Aggregate identical issues to prevent UI clutter."""
        if not issues:
            return []
        
        # Filter out "row: -1" which are already summary issues from the validator
        summaries = [i for i in issues if i.get('row') == -1]
        row_issues = [i for i in issues if i.get('row') != -1]
        
        grouped = {}
        for issue in row_issues:
            rule = issue.get('rule') or issue.get('code', 'UNKNOWN')
            msg = issue.get('msg') or issue.get('message', '')
            severity = issue.get('severity', 'warning')
            
            key = (rule, msg, severity)
            if key not in grouped:
                grouped[key] = 0
            grouped[key] += 1
            
        result = summaries.copy()
        for (rule, msg, severity), count in grouped.items():
            if count > 1:
                display_msg = f"{count} rows affected: {msg}"
            else:
                # If only one row, try to include the row number if available
                row_num = row_issues[0].get('row', '') # This is a bit lazy but fine for 1 item
                display_msg = msg
                
            result.append({'rule': rule, 'msg': display_msg, 'severity': severity})
        return result

    def _display_downloads(self, results):
        """Delegate to standalone downloads module for lazy loading."""
        from features.downloads_tab import render_downloads_tab
        render_downloads_tab(results, group_issues_fn=self._group_issues)

    def validate_data(self, data): return True, ""
    def analyze(self, data): return self.results
    def display_results(self, results):
        self.results = results
        # When called via run(), use the standard display
        self._display_results()
    
    def _display_results(self):
        """Internal router for multi-tab display."""
        tabs = st.tabs(["Overview", "Negatives", "Bids", "Harvest", "Audit", "Downloads"])
        with tabs[0]: self._display_dashboard_v2(self.results)
        with tabs[1]: self._display_negatives(self.results["neg_kw"], self.results["neg_pt"])
        with tabs[2]: self._display_bids(self.results["bids_exact"], self.results["bids_pt"], self.results["bids_agg"], self.results["bids_auto"])
        with tabs[3]: self._display_harvest(self.results["harvest"])
        with tabs[4]: self._display_heatmap(self.results["heatmap"])
        with tabs[5]: self._display_downloads(self.results)
