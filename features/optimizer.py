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
from ui.components import metric_card
import plotly.graph_objects as go

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
    "HARVEST_SALES": 150.0,
    "HARVEST_ROAS_MULT": 0.8,      # vs BUCKET median (80% = less strict)
    "MAX_BID_CHANGE": 0.20,      # Max % change per run
    "DEDUPE_SIMILARITY": 0.85,    # ExactMatcher threshold
    "TARGET_ROAS": 2.5,
    
    # Negative thresholds (now CVR-based)
    "NEGATIVE_CLICKS_THRESHOLD": 10,  # Baseline for legacy compatibility
    "NEGATIVE_SPEND_THRESHOLD": 10.0,
    
    # Bid optimization
    "ALPHA_EXACT": 0.25,
    "ALPHA_BROAD": 0.20,
    "ALPHA": 0.20,
    "MAX_BID_CHANGE": 0.20,
    "TARGET_ROAS": 2.50,
    
    # Min clicks thresholds per bucket (user-configurable)
    "MIN_CLICKS_EXACT": 5,
    "MIN_CLICKS_PT": 5,
    "MIN_CLICKS_BROAD": 10,
    "MIN_CLICKS_AUTO": 10,
    
    # Harvest forecast
    "HARVEST_EFFICIENCY_MULTIPLIER": 1.30,  # 30% efficiency gain from exact match
    
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
    
    # CRITICAL: Filter OUT targeting expressions that are NOT actual search queries
    # These should NOT be harvested as keywords
    targeting_expression_patterns = [
        r'^close-match$', r'^loose-match$', r'^substitutes$', r'^complements$', r'^auto$',
        r'^asin=', r'^asin-expanded=', r'^category=', r'^keyword-group=',
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
    pass_sales = merged["Sales"] >= config["HARVEST_SALES"]
    pass_roas = merged.apply(calculate_roas_threshold, axis=1)
    
    harvest_mask = pass_clicks & pass_orders & pass_sales & pass_roas
    
    candidates = merged[harvest_mask].copy()
    
    # DEBUG: Show why terms fail
    print(f"\\n=== HARVEST DEBUG ===")
    print(f"Discovery rows: {len(discovery_df)}")
    print(f"Grouped search terms: {len(grouped)}")
    print(f"Threshold config: Clicks>={config['HARVEST_CLICKS']}, Orders>={min_orders_threshold} (CVR-based), Sales>=${config['HARVEST_SALES']}, ROAS>={config['HARVEST_ROAS_MULT']}x bucket median")
    print(f"Pass clicks: {pass_clicks.sum()}, Pass orders: {pass_orders.sum()}, Pass sales: {pass_sales.sum()}, Pass ROAS: {pass_roas.sum()}")
    print(f"After ALL thresholds: {len(candidates)} candidates")
    
    # DEBUG: Check for specific terms
    test_terms = ['water cups for kids', 'water cups', 'steel water bottle', 'painting set for kids']
    print(f"\\n--- Checking specific terms ---")
    for test_term in test_terms:
        in_grouped = merged[merged["Customer Search Term"].str.contains(test_term, case=False, na=False)]
        if len(in_grouped) > 0:
            for _, r in in_grouped.iterrows():
                req_roas = baseline_roas * config["HARVEST_ROAS_MULT"]
                pass_all = (r["Clicks"] >= config["HARVEST_CLICKS"] and 
                           r["Orders"] >= min_orders_threshold and 
                           r["Sales"] >= config["HARVEST_SALES"] and 
                           r["ROAS"] >= req_roas)
                print(f"  '{r['Customer Search Term']}': Clicks={r['Clicks']}, Orders={r['Orders']}, Sales=${r['Sales']:.2f}, ROAS={r['ROAS']:.2f} vs {req_roas:.2f} | PASS={pass_all}")
        else:
            print(f"  '{test_term}' - NOT FOUND in Customer Search Term column")
    
    # Show sample of terms that pass all but ROAS
    almost_pass = pass_clicks & pass_orders & pass_sales & (~pass_roas)
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
        survivors_df["New Bid"] = survivors_df["CPC"] * 1.1
        survivors_df = survivors_df.sort_values("Sales", ascending=False)
    
    return survivors_df


# ==========================================
# NEGATIVE DETECTION
# ==========================================

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
    
    # Stage 2: Performance negatives (bleeders) - CVR-BASED THRESHOLDS
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
        
        # Apply CVR-based thresholds (Sales == 0 AND Clicks/Spend > threshold)
        bleeder_mask = (
            (bleeder_agg["Sales"] == 0) &
            (
                (bleeder_agg["Clicks"] >= soft_threshold) |
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
            
            # Use CVR-based hard stop threshold
            reason = "Hard Stop" if row["Clicks"] >= hard_stop_threshold else "Performance"
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
        empty = pd.DataFrame(columns=["Campaign Name", "Ad Group Name", "Term", "Match Type"])
        return empty.copy(), empty.copy(), your_products_df
    
    # CRITICAL: Map KeywordId and TargetingId for negatives
    # Negatives are at campaign+adgroup+term level, so we need to look up IDs
    from core.data_hub import DataHub
    hub = DataHub()
    bulk = hub.get_data('bulk_id_mapping')
    
    if bulk is not None and not bulk.empty:
        # Helper function to normalize strings for matching
        def normalize_for_matching(series):
            """Normalize text series for fuzzy matching (lowercase, alphanumeric only)."""
            return series.astype(str).str.strip().str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
        
        # Normalize for matching
        neg_df['_camp_norm'] = normalize_for_matching(neg_df['Campaign Name'])
        neg_df['_ag_norm'] = normalize_for_matching(neg_df['Ad Group Name'])
        neg_df['_term_norm'] = normalize_for_matching(neg_df['Term'])
        
        bulk = bulk.copy()
        bulk['_camp_norm'] = normalize_for_matching(bulk['Campaign Name'])
        bulk['_ag_norm'] = normalize_for_matching(bulk['Ad Group Name'])
        
        # For keywords: try to match on campaign + ad group + keyword text
        if 'Customer Search Term' in bulk.columns:
            bulk['_kw_norm'] = normalize_for_matching(bulk['Customer Search Term'])
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
            bulk['_pt_norm'] = normalize_for_matching(bulk['Product Targeting Expression'])
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
            for col in ['KeywordId', 'TargetingId']:
                fallback_col = f'{col}_fallback'
                if fallback_col in neg_df.columns:
                    if col not in neg_df.columns:
                        neg_df[col] = neg_df[fallback_col]
                    else:
                        neg_df[col] = neg_df[col].fillna(neg_df[fallback_col])
        
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
# BID OPTIMIZATION (vNext)
# ==========================================

def calculate_bid_optimizations(
    df: pd.DataFrame, 
    config: dict, 
    harvested_terms: Set[str] = None,
    negative_terms: Set[Tuple[str, str, str]] = None,
    universal_median_roas: float = None
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
                                  universal_median_roas=universal_median_roas)
    
    bids_pt = _process_bucket(df_clean[mask_pt], config, 
                               min_clicks=config.get("MIN_CLICKS_PT", 5), 
                               bucket_name="Product Targeting",
                               universal_median_roas=universal_median_roas)
    
    bids_agg = _process_bucket(df_clean[mask_broad_phrase], config, 
                                min_clicks=config.get("MIN_CLICKS_BROAD", 10), 
                                bucket_name="Broad/Phrase",
                                universal_median_roas=universal_median_roas)
    
    bids_auto = _process_bucket(df_clean[mask_auto], config, 
                                 min_clicks=config.get("MIN_CLICKS_AUTO", 10), 
                                 bucket_name="Auto",
                                 universal_median_roas=universal_median_roas)
    
    bids_category = _process_bucket(df_clean[mask_category], config, 
                                     min_clicks=config.get("MIN_CLICKS_CATEGORY", 10), 
                                     bucket_name="Category",
                                     universal_median_roas=universal_median_roas)
    
    # Combine auto and category for backwards compatibility (displayed as "Auto/Category")
    bids_auto_combined = pd.concat([bids_auto, bids_category], ignore_index=True) if not bids_category.empty else bids_auto
    
    return bids_exact, bids_pt, bids_agg, bids_auto_combined

def _process_bucket(segment_df: pd.DataFrame, config: dict, min_clicks: int, bucket_name: str, universal_median_roas: float) -> pd.DataFrame:
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
        "KeywordId", "TargetingId", "Match Type", "Targeting"
    ] if c in segment_df.columns}
    
    if "Current Bid" in segment_df.columns:
        agg_cols["Current Bid"] = "max"
    if "CPC" in segment_df.columns:
        agg_cols["CPC"] = "mean"
        
    grouped = segment_df.groupby(["Campaign Name", "Ad Group Name", "_group_key"], as_index=False).agg({**agg_cols, **meta_cols})
    grouped = grouped.drop(columns=["_group_key"], errors="ignore")
    grouped["ROAS"] = np.where(grouped["Spend"] > 0, grouped["Sales"] / grouped["Spend"], 0)
    
    # Post-aggregation cleanup for auto campaigns
    if is_auto_bucket:
        # Clear TargetingId to prevent individual ASIN IDs from appearing in bulk file
        # The Targeting column already has the correct targeting type (close-match, loose-match, etc.)
        if "TargetingId" in grouped.columns:
            grouped["TargetingId"] = ""
    
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
    
    def apply_optimization(r):
        clicks = r["Clicks"]
        roas = r["ROAS"]
        base_bid = float(r.get("Current Bid", 0) or r.get("CPC", 0) or 0)
        
        if base_bid <= 0:
            return 0.0, "Hold: No Bid/CPC Data", "Hold (No Data)"
        
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
    Standardizes and logs optimization actions (bids, negatives, harvests) to the database.
    This enables the Impact Analyzer to track performance 'before' and 'after' these actions.
    """
    from core.db_manager import get_db_manager
    import uuid
    import streamlit as st
    
    db = get_db_manager(st.session_state.get('test_mode', False))
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

    # 4. Process Harvests
    for _, row in results.get('harvest', pd.DataFrame()).iterrows():
        actions_to_log.append({
            'entity_name': 'Keyword',
            'action_type': 'HARVEST',
            'old_value': 'DISCOVERY',
            'new_value': 'PROMOTED',
            'reason': f"Conv: {row.get('Orders', 0)} orders",
            'campaign_name': row.get('Campaign Name', ''),
            'ad_group_name': row.get('Ad Group Name', ''),
            'target_text': row.get('Customer Search Term', ''),
            'match_type': 'EXACT'
        })

    if actions_to_log:
        try:
            db.log_action_batch(actions_to_log, client_id, batch_id, report_date)
            return len(actions_to_log)
        except Exception as e:
            st.error(f"Failed to log actions: {str(e)}")
            return 0
    return 0


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
            "opt_harvest_sales": config_source.get("HARVEST_SALES", 150.0),
            "opt_harvest_roas_mult": int(config_source.get("HARVEST_ROAS_MULT", 0.8) * 100),
            "opt_alpha_exact": int(config_source.get("ALPHA_EXACT", 0.15) * 100),
            "opt_alpha_broad": int(config_source.get("ALPHA_BROAD", 0.10) * 100),
            "opt_max_bid_change": int(config_source.get("MAX_BID_CHANGE", 0.20) * 100),
            "opt_target_roas": config_source.get("TARGET_ROAS", 2.5),
            "opt_neg_clicks_threshold": config_source.get("NEGATIVE_CLICKS_THRESHOLD", 10),
            "opt_neg_spend_threshold": config_source.get("NEGATIVE_SPEND_THRESHOLD", 25.0),
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
            st.info("👈 Click **Run Optimization** to start")
    
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
            preset = st.radio(
                "preset_selector",
                ["Conservative", "Balanced", "Aggressive"],
                horizontal=True,
                label_visibility="collapsed",
                key="opt_preset"
            )
            
            # Define preset values
            preset_configs = {
                "Conservative": {
                    "harvest_clicks": 15, "harvest_orders": 4, "harvest_sales": 200.0, "harvest_roas": 90,
                    "alpha_exact": 15, "alpha_broad": 12, "max_change": 15, "target_roas": 2.5,
                    "neg_clicks": 15, "neg_spend": 15.0,
                    "min_clicks_exact": 8, "min_clicks_pt": 8, "min_clicks_broad": 12, "min_clicks_auto": 12
                },
                "Balanced": {
                    "harvest_clicks": 10, "harvest_orders": 3, "harvest_sales": 150.0, "harvest_roas": 80,
                    "alpha_exact": 20, "alpha_broad": 16, "max_change": 20, "target_roas": 2.5,
                    "neg_clicks": 10, "neg_spend": 10.0,
                    "min_clicks_exact": 5, "min_clicks_pt": 5, "min_clicks_broad": 10, "min_clicks_auto": 10
                },
                "Aggressive": {
                    "harvest_clicks": 8, "harvest_orders": 2, "harvest_sales": 100.0, "harvest_roas": 70,
                    "alpha_exact": 25, "alpha_broad": 20, "max_change": 25, "target_roas": 2.5,
                    "neg_clicks": 8, "neg_spend": 8.0,
                    "min_clicks_exact": 3, "min_clicks_pt": 3, "min_clicks_broad": 8, "min_clicks_auto": 8
                }
            }
            
            # Apply preset to session state if changed
            if "last_preset" not in st.session_state or st.session_state["last_preset"] != preset:
                st.session_state["last_preset"] = preset
                config = preset_configs[preset]
                st.session_state["opt_harvest_clicks"] = config["harvest_clicks"]
                st.session_state["opt_harvest_orders"] = config["harvest_orders"]
                st.session_state["opt_harvest_sales"] = config["harvest_sales"]
                st.session_state["opt_harvest_roas_mult"] = config["harvest_roas"]
                st.session_state["opt_alpha_exact"] = config["alpha_exact"]
                st.session_state["opt_alpha_broad"] = config["alpha_broad"]
                st.session_state["opt_max_bid_change"] = config["max_change"]
                st.session_state["opt_target_roas"] = config["target_roas"]
                st.session_state["opt_neg_clicks_threshold"] = config["neg_clicks"]
                st.session_state["opt_neg_spend_threshold"] = config["neg_spend"]
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
                "Run optimization with recommended settings",
                type="primary",
                use_container_width=True,
                key="opt_run_primary"
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
            
            # === ADVANCED SETTINGS (ALL CONTROLS COLLAPSED) ===
            # Advanced Settings Icon
            sliders_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>'
            
            # Helper for Sidebar Chiclet Header
            def sidebar_chiclet_header(label, icon_html):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.15) 0%, rgba(91, 85, 111, 0.08) 100%); 
                            border: 1px solid rgba(91, 85, 111, 0.3); 
                            border-radius: 8px; 
                            padding: 8px 12px; 
                            margin-top: 20px; 
                            margin-bottom: -10px;
                            display: flex; 
                            align-items: center; 
                            gap: 8px;">
                    {icon_html}
                    <span style="color: #F5F5F7; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;">{label}</span>
                </div>
                """, unsafe_allow_html=True)

            sidebar_chiclet_header("Configuration", sliders_icon)
            with st.expander("Expand settings", expanded=False):
                st.caption("Fine-tune optimization behavior — defaults work well for most accounts")
                
                # === HARVEST SETTINGS ===
                leaf_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8a8 8 0 0 1-8 8Z"></path><path d="M11 20c0-2.5 2-5.5 2-5.5"></path></svg>'
                st.markdown(f'<div style="color: #F5F5F7; font-size: 0.8rem; font-weight: 600; margin-top: 10px; margin-bottom: 5px;">{leaf_icon}Harvest Graduation</div>', unsafe_allow_html=True)
                st.caption("Graduate keywords from Auto/Broad → Exact")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input(
                        "Min Clicks", 
                        min_value=5, 
                        max_value=30,
                        help="Need this many clicks before graduation",
                        key="opt_harvest_clicks"
                    )
                    st.number_input(
                        "Min Sales ($)", 
                        min_value=0.0, 
                        step=25.0,
                        help="Minimum revenue required",
                        key="opt_harvest_sales"
                    )
                with col2:
                    st.number_input(
                        "Min Orders", 
                        min_value=1, 
                        max_value=10,
                        help="Must show real conversions",
                        key="opt_harvest_orders"
                    )
                    st.slider(
                        "Efficiency Target", 
                        min_value=50, 
                        max_value=120, 
                        step=5,
                        help="Relative to account median (e.g. 80% means anything above 0.8x account median is a winner)",
                        key="opt_harvest_roas_mult",
                        format="%d%%"
                    )
            
            # === BID SETTINGS ===
            trending_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>'
            sidebar_chiclet_header("Bid Adjustments", trending_icon)
            with st.expander("Fine-tune bids", expanded=False):
                st.caption("How aggressively to change bids")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.slider(
                        "Step Size (Exact)", 
                        min_value=5, 
                        max_value=40, 
                        step=5,
                        help="Bid change % for Exact keywords",
                        key="opt_alpha_exact",
                        format="%d%%"
                    )
                    st.slider(
                        "Safety Cap", 
                        min_value=10, 
                        max_value=50, 
                        step=5,
                        help="Maximum single bid change allowed (%)",
                        key="opt_max_bid_change",
                        format="%d%%"
                    )
                with col2:
                    st.slider(
                        "Step Size (Broad)", 
                        min_value=5, 
                        max_value=40, 
                        step=5,
                        help="Bid change % for Broad/Auto",
                        key="opt_alpha_broad",
                        format="%d%%"
                    )
                    st.number_input(
                        "Target ROAS", 
                        min_value=1.0, 
                        max_value=10.0, 
                        step=0.5,
                        help="Your profitability target",
                        key="opt_target_roas"
                    )
            
            # === NEGATIVE SETTINGS ===
            shield_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
            sidebar_chiclet_header("Negative Blocking", shield_icon)
            with st.expander("Auto-block rules", expanded=False):
                st.caption("Auto-block keywords with zero sales")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input(
                        "Min Clicks (0 sales)", 
                        min_value=5, 
                        max_value=30,
                        help="Block after this many wasted clicks",
                        key="opt_neg_clicks_threshold"
                    )
                with col2:
                    st.number_input(
                        "Min Spend (0 sales)", 
                        min_value=5.0, 
                        max_value=50.0, 
                        step=5.0,
                        help="Or block after wasting this much $",
                        key="opt_neg_spend_threshold"
                    )
                
                st.info("💡 Thresholds auto-adjust based on your conversion rate")
            
            # === REACTION SPEED ===
            sidebar_chiclet_header("Reaction Speed", sliders_icon)
            with st.expander("Fine-tune reaction", expanded=False):
                st.caption("Fine-tune reaction speed")
                
                st.markdown("**Bid Data Requirements**")
                st.caption("*Clicks needed before adjusting bids*")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input(
                        "Exact Keywords", 
                        min_value=3, 
                        max_value=15,
                        help="Fast reaction for proven keywords",
                        key="opt_min_clicks_exact"
                    )
                    st.number_input(
                        "Broad/Phrase", 
                        min_value=5, 
                        max_value=20,
                        help="Slower reaction for discovery",
                        key="opt_min_clicks_broad"
                    )
                with col2:
                    st.number_input(
                        "Product Targeting", 
                        min_value=3, 
                        max_value=15,
                        help="Fast reaction for ASINs",
                        key="opt_min_clicks_pt"
                    )
                    st.number_input(
                        "Auto Campaigns", 
                        min_value=5, 
                        max_value=20,
                        help="Slower reaction for testing",
                        key="opt_min_clicks_auto"
                    )
                
                st.divider()
                st.markdown("**Safety Limits** *(Auto-applied)*")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_input("Min Bid", value="$0.30", disabled=True, help="Minimum bid floor")
                with col2:
                    st.text_input("Max Bid", value="3× current", disabled=True, help="Maximum bid ceiling")
                
                st.divider()
                st.checkbox(
                    "Include Simulation & Forecasting",
                    key="opt_run_simulation",
                    help="Generate impact projections and scenario analysis"
                )
            
            # Sync all session state values to self.config (convert integers to decimals)
            self.config["HARVEST_CLICKS"] = st.session_state["opt_harvest_clicks"]
            self.config["HARVEST_ORDERS"] = st.session_state["opt_harvest_orders"]
            self.config["HARVEST_SALES"] = st.session_state["opt_harvest_sales"]
            self.config["HARVEST_ROAS_MULT"] = st.session_state["opt_harvest_roas_mult"] / 100.0
            self.config["ALPHA_EXACT"] = st.session_state["opt_alpha_exact"] / 100.0
            self.config["ALPHA_BROAD"] = st.session_state["opt_alpha_broad"] / 100.0
            self.config["MAX_BID_CHANGE"] = st.session_state["opt_max_bid_change"] / 100.0
            self.config["TARGET_ROAS"] = st.session_state["opt_target_roas"]
            self.config["NEGATIVE_CLICKS_THRESHOLD"] = st.session_state["opt_neg_clicks_threshold"]
            self.config["NEGATIVE_SPEND_THRESHOLD"] = st.session_state["opt_neg_spend_threshold"]
            self.config["MIN_CLICKS_EXACT"] = st.session_state["opt_min_clicks_exact"]
            self.config["MIN_CLICKS_PT"] = st.session_state["opt_min_clicks_pt"]
            self.config["MIN_CLICKS_BROAD"] = st.session_state["opt_min_clicks_broad"]
            self.config["MIN_CLICKS_AUTO"] = st.session_state["opt_min_clicks_auto"]


    def _calculate_account_health(self, df: pd.DataFrame, r: dict) -> dict:
        """Calculate account health diagnostics for dashboard display (Last 30 Days)."""
        # Filter to last 30 days for consistency with Recent Impact and Key Insights
        from datetime import timedelta
        
        df_filtered = df.copy()
        
        # Find date column
        date_col = None
        for col in ['Date', 'Start Date', 'date', 'Report Date', 'start_date']:
            if col in df_filtered.columns:
                date_col = col
                break
        
        if date_col:
            try:
                df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors='coerce')
                valid_dates = df_filtered[date_col].dropna()
                if not valid_dates.empty:
                    max_date = valid_dates.max()
                    cutoff_date = max_date - timedelta(days=30)
                    df_filtered = df_filtered[df_filtered[date_col] >= cutoff_date]
            except:
                pass  # If date filtering fails, use full dataset as fallback
        
        # Use filtered data for all calculations
        total_spend = df_filtered['Spend'].sum()
        total_sales = df_filtered['Sales'].sum()
        total_orders = df_filtered['Orders'].sum()
        total_clicks = df_filtered['Clicks'].sum()
        
        current_roas = total_sales / total_spend if total_spend > 0 else 0
        current_acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
        
        # Efficiency calculation at TARGETING level (grouped)
        # Measures % of spend that goes to converting targets (orders > 0)
        if 'Targeting' in df_filtered.columns:
            targeting_agg = df_filtered.groupby('Targeting').agg({'Spend': 'sum', 'Orders': 'sum'}).reset_index()
            converting_spend = targeting_agg[targeting_agg['Orders'] > 0]['Spend'].sum()
        else:
            converting_spend = df_filtered.loc[df_filtered['Orders'] > 0, 'Spend'].sum()
        
        efficiency_rate = (converting_spend / total_spend * 100) if total_spend > 0 else 0
        wasted_spend = total_spend - converting_spend
        waste_ratio = 100 - efficiency_rate
        
        cvr = (total_orders / total_clicks * 100) if total_clicks > 0 else 0
        
        roas_score = min(100, current_roas / 4.0 * 100)
        efficiency_score = efficiency_rate  # Direct: 46% converting = score of 46
        cvr_score = min(100, cvr / 5.0 * 100)
        health_score = (roas_score * 0.4 + efficiency_score * 0.4 + cvr_score * 0.2)
        
        health_metrics = {
            "health_score": health_score,
            "roas_score": roas_score,
            "efficiency_score": efficiency_score,  # Renamed from waste_score
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
            from core.db_manager import get_db_manager
            db = get_db_manager(st.session_state.get('test_mode', False))
            client_id = st.session_state.get('active_account_id')
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
        bids_ex, bids_pt, bids_agg, bids_auto = calculate_bid_optimizations(df, self.config, set(harvest["Customer Search Term"].str.lower()), neg_set, universal_median)
        
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
        """Display overview dashboard with simulation and key metrics."""
        import streamlit as st
        
        icon_color = "#8F8CA3"
        overview_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 12v5"/><path d="M12 9v8"/><path d="M17 11v6"/></svg>'
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); 
                    border: 1px solid rgba(124, 58, 237, 0.2); 
                    border-radius: 8px; 
                    padding: 12px 16px; 
                    margin-bottom: 20px;
                    display: flex; 
                    align-items: center; 
                    gap: 10px;">
            {overview_icon}
            <span style="color: #F5F5F7; font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Optimization Overview</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Extract data
        direct_bids = results.get("direct_bids", pd.DataFrame())
        agg_bids = results.get("agg_bids", pd.DataFrame())
        harvest = results.get("harvest", pd.DataFrame())
        simulation = results.get("simulation")
        df = results.get("df", pd.DataFrame())
        neg_kw = results.get("neg_kw", pd.DataFrame())
        neg_pt = results.get("neg_pt", pd.DataFrame())
        
        # Calculate summary metrics
        total_bids = len(direct_bids) + len(agg_bids) if direct_bids is not None and agg_bids is not None else 0
        total_negatives = (len(neg_kw) if neg_kw is not None else 0) + (len(neg_pt) if neg_pt is not None else 0)
        total_harvests = len(harvest) if harvest is not None else 0
        
        # Current performance metrics
        if not df.empty:
            total_spend = df['Spend'].sum()
            total_sales = df['Sales'].sum()
            current_roas = total_sales / total_spend if total_spend > 0 else 0
            current_acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
        else:
            total_spend = total_sales = current_roas = current_acos = 0
        
        # Display simulation if available and has data
        if simulation and simulation.get('summary'):
            st.markdown("### 📊 Impact Forecast")
            
            # Extract simulation metrics
            summary = simulation.get('summary', {})
            
            st.markdown("""
            <div style="background: rgba(91, 85, 111, 0.05); border-left: 4px solid #5B556F; padding: 12px 20px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
                <p style="color: #B6B4C2; font-size: 0.9rem; margin: 0;">
                    <strong>Projected Impact</strong>: Based on historical performance and bid elasticity modeling
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display forecast metrics
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                spend_change = summary.get('spend_change_pct', 0)
                st.metric("Spend Change", f"{spend_change:+.1f}%", 
                         delta=f"${summary.get('spend_change_abs', 0):,.2f}")
            
            with c2:
                sales_change = summary.get('sales_change_pct', 0)
                st.metric("Sales Change", f"{sales_change:+.1f}%",
                         delta=f"${summary.get('sales_change_abs', 0):,.2f}")
            
            with c3:
                roas_new = summary.get('roas_new', 0)
                roas_current = summary.get('roas_current', 0)
                roas_delta = roas_new - roas_current
                st.metric("Projected ROAS", f"{roas_new:.2f}x",
                         delta=f"{roas_delta:+.2f}x")
            
            with c4:
                profit_impact = summary.get('profit_impact', 0)
                st.metric("Profit Impact", f"${profit_impact:,.2f}",
                         delta="Estimated" if profit_impact > 0 else None)
            
            st.divider()
        
        # Only show if simulation is completely disabled/not available
        # (Don't show if simulation ran but just has no data)
        
        # Quick actions
        st.markdown("### Quick Actions")
        st.markdown("""
        <p style="color: #B6B4C2; font-size: 0.9rem; margin-bottom: 16px;">
            Navigate to specific tabs above to review detailed recommendations
        </p>
        """, unsafe_allow_html=True)
        
        qa1, qa2, qa3 = st.columns(3)
        
        with qa1:
            if total_negatives > 0:
                st.info(f"🛡️ **{total_negatives}** negatives identified - Review in Defence tab")
        
        with qa2:
            if total_bids > 0:
                st.info(f"📊 **{total_bids}** bid adjustments ready - Review in Bids tab")
        
        with qa3:
            if total_harvests > 0:
                st.info(f"🌱 **{total_harvests}** harvest candidates - Review in Harvest tab")

    def _display_negatives(self, neg_kw, neg_pt):
        # Icons
        icon_color = "#8F8CA3"
        shield_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
        
        def tab_header(label, icon_html):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); 
                        border: 1px solid rgba(124, 58, 237, 0.2); 
                        border-radius: 8px; 
                        padding: 12px 16px; 
                        margin-bottom: 20px;
                        display: flex; 
                        align-items: center; 
                        gap: 10px;">
                {icon_html}
                <span style="color: #F5F5F7; font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">{label}</span>
            </div>
            """, unsafe_allow_html=True)

        # sub-navigation for negatives
        neg_tabs = [
            {"name": "Keyword Negatives", "icon": "🛑 "},
            {"name": "Product Targeting Negatives", "icon": "🎯 "}
        ]
        
        if 'active_neg_tab' not in st.session_state:
            st.session_state['active_neg_tab'] = "Keyword Negatives"
        
        # Use horizontal radio for clean tertiary navigation
        active_neg = st.radio(
            "Select Negative Type",
            options=["🛑 Keyword Negatives", "🎯 Product Targeting Negatives"],
            label_visibility="collapsed",
            horizontal=True,
            key="neg_radio_nav"
        )
        st.session_state['active_neg_tab'] = active_neg.split(" ", 1)[1]  # Strip emoji


        st.markdown("<br>", unsafe_allow_html=True)
        
        active_tab = st.session_state['active_neg_tab']
        if active_tab == "Keyword Negatives":
            tab_header("Negative Keywords Identified", shield_icon)
            if not neg_kw.empty:
                st.data_editor(neg_kw, use_container_width=True, height=400, disabled=True, hide_index=True)
            else:
                st.info("No negative keywords found.")
        else:
            tab_header("Product Targeting Candidates", shield_icon)
            if not neg_pt.empty:
                st.data_editor(neg_pt, use_container_width=True, height=400, disabled=True, hide_index=True)
            else:
                st.info("No product targeting negatives found.")

    def _display_bids(self, bids_exact=None, bids_pt=None, bids_agg=None, bids_auto=None):
        icon_color = "#8F8CA3"
        sliders_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>'
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); 
                    border: 1px solid rgba(124, 58, 237, 0.2); 
                    border-radius: 8px; 
                    padding: 12px 16px; 
                    margin-bottom: 20px;
                    display: flex; 
                    align-items: center; 
                    gap: 10px;">
            {sliders_icon}
            <span style="color: #F5F5F7; font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Bid Optimizations</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Define preferred column order
        preferred_cols = ["Targeting", "Campaign Name", "Match Type", "Clicks", "Orders", "Sales", "ROAS", "Current Bid", "CPC", "New Bid", "Reason", "Decision_Basis", "Bucket"]
        
        # sub-navigation for bids
        bid_tabs = [
            {"name": "Exact Keywords", "icon": "🎯 "},
            {"name": "Product Targeting", "icon": "📦 "},
            {"name": "Broad / Phrase", "icon": "📈 "},
            {"name": "Auto / Category", "icon": "⚡ "}
        ]
        
        if 'active_bid_tab' not in st.session_state:
            st.session_state['active_bid_tab'] = "Exact Keywords"
        
        # Use horizontal radio for clean tertiary navigation
        active_bid = st.radio(
            "Select Bid Category",
            options=["🎯 Exact Keywords", "📦 Product Targeting", "📈 Broad / Phrase", "⚡ Auto / Category"],
            label_visibility="collapsed",
            horizontal=True,
            key="bid_radio_nav"
        )
        st.session_state['active_bid_tab'] = active_bid.split(" ", 1)[1]  # Strip emoji


        st.markdown("<br>", unsafe_allow_html=True)
        
        def safe_display(df):
            if df is not None and not df.empty:
                available_cols = [c for c in preferred_cols if c in df.columns]
                st.data_editor(df[available_cols], use_container_width=True, height=400, disabled=True, hide_index=True)
            else:
                st.info("No bid adjustments needed for this bucket.")

        active_tab = st.session_state['active_bid_tab']
        if active_tab == "Exact Keywords": safe_display(bids_exact)
        elif active_tab == "Product Targeting": safe_display(bids_pt)
        elif active_tab == "Broad / Phrase": safe_display(bids_agg)
        elif active_tab == "Auto / Category": safe_display(bids_auto)

    def _display_harvest(self, harvest_df):
        icon_color = "#8F8CA3"
        leaf_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8a8 8 0 0 1-8 8Z"></path><path d="M11 20c0-2.5 2-5.5 2-5.5"></path></svg>'
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); 
                    border: 1px solid rgba(124, 58, 237, 0.2); 
                    border-radius: 8px; 
                    padding: 12px 16px; 
                    margin-bottom: 20px;
                    display: flex; 
                    align-items: center; 
                    gap: 10px;">
            {leaf_icon}
            <span style="color: #F5F5F7; font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Harvest Candidates</span>
        </div>
        """, unsafe_allow_html=True)

        if harvest_df is not None and not harvest_df.empty:
            st.markdown("""
            <div style="background: rgba(91, 85, 111, 0.05); border-left: 4px solid #5B556F; padding: 12px 20px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
                <p style="color: #B6B4C2; font-size: 0.9rem; margin: 0;">
                    <strong>Success Strategy</strong>: These high-performing search terms have been identified for promotion to Exact Match campaigns to secure placement and improve ROI.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Bridge to Campaign Creator with Streamlit button (no page reload)
            cta_left, cta_right = st.columns([3, 1])
            with cta_left:
                st.markdown("""
                <div style="background: rgba(124, 58, 237, 0.08); border: 1px solid rgba(124, 58, 237, 0.2); padding: 15px; border-radius: 12px; display: flex; align-items: center;">
                    <div style="color: #F5F5F7; font-size: 0.95rem;">
                        💡 <strong>Ready to Scale?</strong> Export these terms directly to the Campaign Creator.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with cta_right:
                # Store harvest data for Creator and navigate
                if st.button("OPEN CAMPAIGN CREATOR", type="primary", use_container_width=True):
                    st.session_state['harvest_payload'] = harvest_df
                    st.session_state['active_creator_tab'] = "Harvest Winners"
                    st.session_state['current_module'] = 'creator'
                    st.rerun()
            
            st.data_editor(harvest_df, use_container_width=True, height=400, disabled=True, hide_index=True)
        else:
            st.info("No harvest candidates met the performance criteria for this period.")

    def _display_heatmap(self, heatmap_df):
        icon_color = "#8F8CA3"
        search_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>'
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); 
                    border: 1px solid rgba(124, 58, 237, 0.2); 
                    border-radius: 8px; 
                    padding: 12px 16px; 
                    margin-bottom: 20px;
                    display: flex; 
                    align-items: center; 
                    gap: 10px;">
            {search_icon}
            <span style="color: #F5F5F7; font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Wasted Spend Heatmap</span>
        </div>
        """, unsafe_allow_html=True)

        if heatmap_df is not None and not heatmap_df.empty:
            st.markdown("""
            <div style="background: rgba(91, 85, 111, 0.05); border-left: 4px solid #5B556F; padding: 12px 20px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
                <p style="color: #B6B4C2; font-size: 0.9rem; margin: 0;">
                    <strong>Visual Intelligence</strong>: Red indicates immediate fix required, Yellow requires monitoring, and Green shows efficient performance.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # --- 1. Top Cards ---
            p1, p2, p3 = st.columns(3)
            high_count = len(heatmap_df[heatmap_df["Priority"].str.contains("High")])
            med_count = len(heatmap_df[heatmap_df["Priority"].str.contains("Medium")])
            good_count = len(heatmap_df[heatmap_df["Priority"].str.contains("Good")])
            
            with p1: metric_card("High Priority", str(high_count), "shield", color="#f87171")
            with p2: metric_card("Medium Priority", str(med_count), "shield", color="#fbbf24")
            with p3: metric_card("Good Performance", str(good_count), "check", color="#4ade80")
            
            st.divider()
            
            # --- 2. Action Status Cards ---
            addressed_mask = ~heatmap_df["Actions_Taken"].str.contains("Hold|No action", na=False)
            addressed_count = addressed_mask.sum()
            needs_attn_count = len(heatmap_df) - addressed_count
            high_addressed = (heatmap_df[heatmap_df["Priority"].str.contains("High") & addressed_mask]).shape[0]
            coverage = (addressed_count / len(heatmap_df) * 100) if len(heatmap_df) > 0 else 0

            # --- PREMIUM HERO TILES (Impact Dashboard Style) - Neutralized ---
            theme_mode = st.session_state.get('theme_mode', 'dark')
            # Saddle brand colors extracted from logo (Neutralized version)
            brand_purple = "#5B556F"
            brand_muted = "#8F8CA3"
            brand_slate = "#444357"
            brand_text = "#F5F5F7"
            brand_muted_text = "#B6B4C2"
            
            # Surface and Glow
            surface_glow = "rgba(91, 85, 111, 0.08)"
            
            icon_color = brand_muted
            
            # Action Icons
            check_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><polyline points="20 6 9 17 4 12"></polyline></svg>'
            warning_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
            bolt_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>'
            target_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 6px;"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>'

            st.markdown("""
            <style>
            .hero-tile {
                background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 4px 24px rgba(0,0,0,0.08);
                transition: all 0.3s ease;
                margin-bottom: 10px;
            }
            .hero-tile:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.12);
            }
            .hero-value {
                font-size: 1.25rem;
                font-weight: 700;
                margin-bottom: 4px;
                margin-top: 8px;
            }
            .hero-label {
                font-size: 0.75rem;
                opacity: 0.7;
                text-transform: uppercase;
                letter-spacing: 0.8px;
                font-weight: 600;
            }
            </style>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.markdown(f"""
                <div class="hero-tile" style="border-left: 4px solid {brand_purple}; background: linear-gradient(135deg, {surface_glow} 0%, rgba(255,255,255,0.02) 100%);">
                    <div class="hero-label">{check_icon}Being Addressed</div>
                    <div class="hero-value" style="color: {brand_text};">{addressed_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown(f"""
                <div class="hero-tile" style="border-left: 4px solid {brand_slate}; background: linear-gradient(135deg, {surface_glow} 0%, rgba(255,255,255,0.02) 100%);">
                    <div class="hero-label">{warning_icon}Needs Attention</div>
                    <div class="hero-value" style="color: {brand_muted_text};">{needs_attn_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with c3:
                st.markdown(f"""
                <div class="hero-tile" style="border-left: 4px solid {brand_purple}; background: linear-gradient(135deg, {surface_glow} 0%, rgba(255,255,255,0.02) 100%);">
                    <div class="hero-label">{bolt_icon}High Priority Fixed</div>
                    <div class="hero-value" style="color: {brand_text};">{high_addressed}/{high_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with c4:
                st.markdown(f"""
                <div class="hero-tile" style="border-left: 4px solid {brand_slate}; background: linear-gradient(135deg, {surface_glow} 0%, rgba(255,255,255,0.02) 100%);">
                    <div class="hero-label">{target_icon}Coverage</div>
                    <div class="hero-value" style="color: {brand_muted_text};">{coverage:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # --- 3. Interactive Filters ---
            f1, f2 = st.columns(2)
            with f1:
                p_filter = st.multiselect("Filter by Priority", ["🔴 High", "🟡 Medium", "🟢 Good"], default=["🔴 High", "🟡 Medium"])
            with f2:
                has_action = st.selectbox("Filter by Actions", ["All", "Only with Actions", "Only Holds/No Action"])
            
            filtered_df = heatmap_df.copy()
            if p_filter:
                filtered_df = filtered_df[filtered_df["Priority"].isin(p_filter)]
            
            if has_action == "Only with Actions":
                filtered_df = filtered_df[~filtered_df["Actions_Taken"].str.contains("Hold|No action", na=False)]
            elif has_action == "Only Holds/No Action":
                filtered_df = filtered_df[filtered_df["Actions_Taken"].str.contains("Hold|No action", na=False)]
                
            # Heatmap icon
            icon_color = "#8F8CA3"
            heat_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>'
            st.markdown(f"#### {heat_icon}Performance Heatmap with Actions ({len(filtered_df)} items)", unsafe_allow_html=True)
            
            cols = ["Priority", "Campaign Name", "Ad Group Name", "Actions_Taken", "Reason_Summary", "Spend", "Sales", "ROAS", "CVR"]
            display_df = filtered_df[[c for c in cols if c in filtered_df.columns]].copy()
            
            # Rename for display
            display_df = display_df.rename(columns={"Reason_Summary": "Reason"})
            
            def style_priority(val):
                if "High" in val: return "color: #ef4444; font-weight: bold"
                if "Medium" in val: return "color: #eab308; font-weight: bold"
                if "Good" in val: return "color: #22c55e; font-weight: bold"
                return ""

            styled = display_df.style.map(style_priority, subset=["Priority"]) \
                                     .background_gradient(subset=["ROAS"], cmap="RdYlGn") \
                                     .background_gradient(subset=["CVR"], cmap="YlGn") \
                                     .format({"Spend": "${:,.2f}", "Sales": "${:,.2f}", "ROAS": "{:.2f}x", "CVR": "{:.2f}%"})
            
            st.dataframe(styled, use_container_width=True, height=500)
        else:
            st.info("No heatmap data available.")

    def _display_downloads(self, results):
        icon_color = "#8F8CA3"
        dl_icon = f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>'
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(91, 85, 111, 0.1) 0%, rgba(91, 85, 111, 0.05) 100%); 
                    border: 1px solid rgba(124, 58, 237, 0.2); 
                    border-radius: 8px; 
                    padding: 12px 16px; 
                    margin-bottom: 20px;
                    display: flex; 
                    align-items: center; 
                    gap: 10px;">
            {dl_icon}
            <span style="color: #F5F5F7; font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Export optimizations</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <p style="color: #B6B4C2; font-size: 0.95rem; margin-bottom: 24px;">
            Download formatted Amazon Bulk Files to apply these optimizations instantly.
        </p>
        """, unsafe_allow_html=True)
        
        # 1. Negative Keywords
        neg_kw = results.get("neg_kw", pd.DataFrame())
        if not neg_kw.empty:
            shield_icon_sub = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
            st.markdown(f"<div style='color: #F5F5F7; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center;'>{shield_icon_sub}Negative Keywords Bulk</div>", unsafe_allow_html=True)
            kw_bulk = generate_negatives_bulk(neg_kw, pd.DataFrame())
            with st.expander("👁️ Preview File Content", expanded=False):
                st.dataframe(kw_bulk.head(5), use_container_width=True)
            
            buf = dataframe_to_excel(kw_bulk)
            st.download_button(
                label="📥 Download Negative Keywords (.xlsx)", 
                data=buf, 
                file_name="negative_keywords.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_neg_btn",
                type="primary",
                use_container_width=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

        # 2. Bids
        all_bids = pd.concat([results.get("direct_bids", pd.DataFrame()), results.get("agg_bids", pd.DataFrame())], ignore_index=True)
        if not all_bids.empty:
            sliders_icon_sub = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>'
            st.markdown(f"<div style='color: #F5F5F7; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center;'>{sliders_icon_sub}Bid Optimizations Bulk</div>", unsafe_allow_html=True)
            bid_bulk, _ = generate_bids_bulk(all_bids)
            with st.expander("👁️ Preview File Content", expanded=False):
                st.dataframe(bid_bulk.head(5), use_container_width=True)
            
            buf = dataframe_to_excel(bid_bulk)
            st.download_button(
                label="📥 Download Bid Adjustments (.xlsx)", 
                data=buf, 
                file_name="bid_optimizations.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_bid_btn",
                type="primary",
                use_container_width=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

        # 3. Harvest
        harvest = results.get("harvest", pd.DataFrame())
        if not harvest.empty:
            leaf_icon_sub = f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8a8 8 0 0 1-8 8Z"></path><path d="M11 20c0-2.5 2-5.5 2-5.5"></path></svg>'
            st.markdown(f"<div style='color: #F5F5F7; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center;'>{leaf_icon_sub}Harvest Candidates</div>", unsafe_allow_html=True)
            with st.expander("👁️ Preview Candidate List", expanded=False):
                st.dataframe(harvest.head(5), use_container_width=True)
            
            buf = dataframe_to_excel(harvest)
            st.download_button(
                label="📥 Download Harvest List (.xlsx)", 
                data=buf, 
                file_name="harvest_candidates.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_harvest_btn",
                type="primary",
                use_container_width=True
            )

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
