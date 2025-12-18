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
from features._base import BaseFeature
from core.data_hub import DataHub
from core.data_loader import safe_numeric, is_asin
from utils.formatters import format_currency, dataframe_to_excel
from utils.matchers import ExactMatcher
from ui.components import metric_card

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
    
    # Calculate universal (account-wide) median for outlier detection (outlier-resistant)
    valid_rows = df[(df["Spend"] > 0) & (df["Sales"] > 0)].copy()

    if len(valid_rows) >= 10:
        substantial_rows = valid_rows[valid_rows["Spend"] >= 5.0]
        
        if len(substantial_rows) >= 10:
            # Use winsorized median (cap at 99th percentile to remove extreme outliers)
            roas_values = substantial_rows["ROAS"].values
            cap_value = np.percentile(roas_values, 99)
            winsorized_roas = np.clip(roas_values, 0, cap_value)
            universal_median_roas = np.median(winsorized_roas)
        else:
            # Not enough substantial data, fall back to all rows
            universal_median_roas = valid_rows["ROAS"].median()
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
    
    # CHANGE #3: Winner selection score rebalanced (ROAS×5 instead of ×10)
    # Get metadata from BEST performing instance (winner selection)
    # Rank by Sales (primary), then ROAS (secondary)
    discovery_df["_perf_score"] = discovery_df["Sales"] + (discovery_df["ROAS"] * 5)
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
    
    # Step 2: Calculate bucket median
    bucket_valid_roas = merged[(merged["Spend"] > 0) & (merged["Sales"] > 0)]["ROAS"]
    bucket_sample_size = len(bucket_valid_roas)

    # Step 3: Stat sig check
    MIN_SAMPLE_SIZE_FOR_STAT_SIG = 20
    OUTLIER_THRESHOLD_MULTIPLIER = 1.5

    if bucket_sample_size < MIN_SAMPLE_SIZE_FOR_STAT_SIG:
        baseline_roas = universal_median_roas  # Use universal
        baseline_source = "Universal Median (insufficient bucket data)"
    else:
        bucket_median_roas = bucket_valid_roas.median()
        
        # Step 4: Outlier detection
        if bucket_median_roas > universal_median_roas * OUTLIER_THRESHOLD_MULTIPLIER:
            baseline_roas = universal_median_roas  # Outlier, use universal
            baseline_source = "Universal Median (bucket is outlier)"
        else:
            baseline_roas = bucket_median_roas  # Valid, use bucket
            baseline_source = "Bucket Median"

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
    
    # 3. Build mutually exclusive bucket masks
    mask_auto_by_targeting = df_clean["Targeting"].apply(is_auto_or_category)
    mask_auto_by_matchtype = df_clean["Match Type"].str.lower().isin(["auto", "-"])
    mask_auto = mask_auto_by_targeting | mask_auto_by_matchtype
    
    mask_pt = df_clean["Targeting"].apply(is_pt_targeting) & (~mask_auto)
    
    mask_exact = (
        (df_clean["Match Type"].str.lower() == "exact") & 
        (~mask_pt) & 
        (~mask_auto)
    )
    
    mask_broad_phrase = (
        df_clean["Match Type"].str.lower().isin(["broad", "phrase"]) & 
        (~mask_pt) & 
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
                                 bucket_name="Auto/Category",
                                 universal_median_roas=universal_median_roas)
    
    return bids_exact, bids_pt, bids_agg, bids_auto

def _process_bucket(segment_df: pd.DataFrame, config: dict, min_clicks: int, bucket_name: str, universal_median_roas: float) -> pd.DataFrame:
    """Unified bucket processor with Bucket Median ROAS classification."""
    if segment_df.empty:
        return pd.DataFrame()
    
    segment_df = segment_df.copy()
    segment_df["_targeting_norm"] = segment_df["Targeting"].astype(str).str.strip().str.lower()
    
    has_keyword_id = "KeywordId" in segment_df.columns and segment_df["KeywordId"].notna().any()
    has_targeting_id = "TargetingId" in segment_df.columns and segment_df["TargetingId"].notna().any()
    
    if has_keyword_id or has_targeting_id:
        segment_df["_group_key"] = segment_df.apply(
            lambda r: str(r.get("KeywordId") or r.get("TargetingId") or r["_targeting_norm"]).strip(),
            axis=1
        )
    else:
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
    
    all_valid_roas = grouped[(grouped["Spend"] > 0) & (grouped["Sales"] > 0)]["ROAS"]
    bucket_sample_size = len(all_valid_roas)
    
    # Stat sig check
    MIN_SAMPLE_SIZE_FOR_STAT_SIG = 20
    OUTLIER_THRESHOLD_MULTIPLIER = 1.5
    
    if bucket_sample_size < MIN_SAMPLE_SIZE_FOR_STAT_SIG:
        baseline_roas = universal_median_roas
        baseline_source = f"Universal Median (insufficient bucket data: {bucket_sample_size} < 20)"
    else:
        bucket_median_roas = all_valid_roas.median()
        
        if bucket_median_roas > universal_median_roas * OUTLIER_THRESHOLD_MULTIPLIER:
            baseline_roas = universal_median_roas
            baseline_source = f"Universal Median (bucket median {bucket_median_roas:.2f}x is outlier)"
        else:
            baseline_roas = bucket_median_roas
            baseline_source = f"Bucket Median (n={bucket_sample_size})"
    
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
    
    grouped["CTR"] = np.where(grouped["Impressions"] > 0, grouped["Clicks"] / grouped["Impressions"] * 100, 0)
    grouped["CVR"] = np.where(grouped["Clicks"] > 0, grouped["Orders_Attributed"] / grouped["Clicks"] * 100, 0)
    grouped["ROAS"] = np.where(grouped["Spend"] > 0, grouped["Sales_Attributed"] / grouped["Spend"], 0)
    grouped["ACoS"] = np.where(grouped["Sales_Attributed"] > 0, grouped["Spend"] / grouped["Sales_Attributed"] * 100, 999)
    
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
        
        if not b_match.empty and "New Bid" in b_match.columns:
            cur_bids = b_match.get("Current Bid", b_match.get("CPC", 0))
            grouped.at[idx, "Bid_Increase_Count"] = (b_match["New Bid"] > cur_bids).sum()
            grouped.at[idx, "Bid_Decrease_Count"] = (b_match["New Bid"] < cur_bids).sum()
            
        actions = []
        if grouped.at[idx, "Harvest_Count"] > 0: actions.append(f"💎 {int(grouped.at[idx, 'Harvest_Count'])} harvests")
        if grouped.at[idx, "Negative_Count"] > 0: actions.append(f"🛑 {int(grouped.at[idx, 'Negative_Count'])} negatives")
        if grouped.at[idx, "Bid_Increase_Count"] > 0: actions.append(f"⬆️ {int(grouped.at[idx, 'Bid_Increase_Count'])} increases")
        if grouped.at[idx, "Bid_Decrease_Count"] > 0: actions.append(f"⬇️ {int(grouped.at[idx, 'Bid_Decrease_Count'])} decreases")
        
        if actions:
            grouped.at[idx, "Actions_Taken"] = " | ".join(actions)
        elif row["Clicks"] < config.get("MIN_CLICKS_EXACT", 5):
            grouped.at[idx, "Actions_Taken"] = "⏸️ Hold (Low volume)"
        else:
            grouped.at[idx, "Actions_Taken"] = "✅ No action needed"

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

def generate_negatives_bulk(neg_kw, neg_pt):
    """Generate Amazon bulk upload file for negatives."""
    frames = []
    
    if neg_kw is not None and not neg_kw.empty:
        neg_kw = neg_kw.reset_index(drop=True)
        df = pd.DataFrame(columns=BULK_COLUMNS)
        df["Product"] = "Sponsored Products"
        df["Entity"] = "Negative Keyword"
        df["Operation"] = "Create"
        df["Campaign Id"] = neg_kw.get("CampaignId", "")
        df["Ad Group Id"] = neg_kw.get("AdGroupId", "")
        df["Campaign Name"] = neg_kw["Campaign Name"]
        df["Ad Group Name"] = neg_kw["Ad Group Name"]
        df["Keyword Text"] = neg_kw["Term"]
        df["Match Type"] = "negativeExact"
        df["State"] = "enabled"
        frames.append(df)
        
    if neg_pt is not None and not neg_pt.empty:
        neg_pt = neg_pt.reset_index(drop=True)
        df = pd.DataFrame(columns=BULK_COLUMNS)
        df["Product"] = "Sponsored Products"
        df["Entity"] = "Product Targeting"
        df["Operation"] = "Create"
        df["Campaign Id"] = neg_pt.get("CampaignId", "")
        df["Ad Group Id"] = neg_pt.get("AdGroupId", "")
        df["Campaign Name"] = neg_pt["Campaign Name"]
        df["Ad Group Name"] = neg_pt["Ad Group Name"]
        df["Product Targeting Expression"] = neg_pt["Term"]
        df["Match Type"] = "negativeExact" # For negative PT
        df["State"] = "enabled"
        frames.append(df)
        
    return pd.concat(frames) if frames else pd.DataFrame(columns=BULK_COLUMNS)

def generate_bids_bulk(bids_df):
    """Generate Amazon bulk upload file for bid updates."""
    if bids_df is None or bids_df.empty:
        return pd.DataFrame(columns=BULK_COLUMNS), 0
        
    # Filter for actual changes (non-hold)
    changes = bids_df[~bids_df["Reason"].str.contains("Hold", case=False, na=False)].copy()
    if changes.empty:
        return pd.DataFrame(columns=BULK_COLUMNS), 0
    
    changes = changes.reset_index(drop=True)
    df = pd.DataFrame(columns=BULK_COLUMNS)
    df["Product"] = "Sponsored Products"
    df["Entity"] = np.where(changes["Match Type"].str.contains("auto", case=False, na=False), "Ad Group", "Keyword")
    # Actually, for keywords/PT, we need to distinguish
    df["Entity"] = np.where(changes["Bucket"] == "Product Targeting", "Product Targeting", df["Entity"])
    
    df["Operation"] = "Update"
    df["Campaign Id"] = changes.get("CampaignId", "")
    df["Ad Group Id"] = changes.get("AdGroupId", "")
    df["Campaign Name"] = changes["Campaign Name"]
    df["Ad Group Name"] = changes["Ad Group Name"]
    df["Bid"] = changes["New Bid"]
    
    # ID Mapping
    df["Keyword Id"] = changes.get("KeywordId", "")
    df["Product Targeting Id"] = changes.get("TargetingId", "")
    
    return df, len(changes)

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
        with st.sidebar:
            st.divider()
            st.markdown("##### OPTIMIZER SETTINGS")
            
            # === PRESETS ===
            st.markdown("**⚙️ Quick Presets**")
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
            
            # === HARVEST SETTINGS ===
            with st.expander("🌾 Harvest Graduation", expanded=False):
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
            with st.expander("💰 Bid Adjustments", expanded=False):
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
            with st.expander("🛑 Negative Blocking", expanded=False):
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
            
            # === ADVANCED SETTINGS (COLLAPSED) ===
            with st.expander("⚙️ Advanced Settings", expanded=False):
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
            
            # === SIMULATION TOGGLE ===
            st.divider()
            st.checkbox(
                "📊 Include Simulation & Forecasting",
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
            
            # Teal button styling
            st.markdown("""
            <style>
            [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%) !important;
                border: none !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # RUN BUTTON
            st.divider()
            if st.button("🚀 Run Optimization", type="primary", use_container_width=True, key="opt_run_btn"):
                st.session_state["run_optimizer"] = True
                st.rerun()

    def _calculate_account_health(self, df: pd.DataFrame, r: dict) -> dict:
        """Calculate account health diagnostics for dashboard display."""
        total_spend = df['Spend'].sum()
        total_sales = df['Sales'].sum()
        total_orders = df['Orders'].sum()
        total_clicks = df['Clicks'].sum()
        
        current_roas = total_sales / total_spend if total_spend > 0 else 0
        current_acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
        
        zero_order_mask = df['Orders'] == 0
        wasted_spend = df.loc[zero_order_mask, 'Spend'].sum()
        waste_ratio = (wasted_spend / total_spend * 100) if total_spend > 0 else 0
        
        cvr = (total_orders / total_clicks * 100) if total_clicks > 0 else 0
        
        roas_score = min(100, current_roas / 4.0 * 100)
        waste_score = max(0, 100 - waste_ratio * 3)
        cvr_score = min(100, cvr / 5.0 * 100)
        health_score = (roas_score * 0.4 + waste_score * 0.4 + cvr_score * 0.2)
        
        return {
            "health_score": health_score,
            "waste_ratio": waste_ratio,
            "wasted_spend": wasted_spend,
            "current_roas": current_roas,
            "current_acos": current_acos,
            "cvr": cvr
        }

    def _display_summary(self, df):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Search Terms", f"{len(df):,}")
        c2.metric("Spend", format_currency(df["Spend"].sum()))
        c3.metric("Sales", format_currency(df["Sales"].sum()))
        roas = df["Sales"].sum() / df["Spend"].sum() if df["Spend"].sum() > 0 else 0
        c4.metric("ROAS", f"{roas:.2f}x")

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
        """Display rich overview dashboard with account health."""
        df = results.get("df", pd.DataFrame())
        harvest = results.get("harvest", pd.DataFrame())
        neg_kw = results.get("neg_kw", pd.DataFrame())
        direct_bids = results.get("direct_bids", pd.DataFrame())
        agg_bids = results.get("agg_bids", pd.DataFrame())
        sim = results.get("simulation", {})
        
        
        # 2. Key Action Stats (5 columns)
        st.markdown("##### 🎯 Optimization Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        
        with c1: metric_card("Ad Groups", f"{df['Ad Group Name'].nunique():,}")
        with c2: metric_card("Search Terms", f"{len(df):,}")
        with c3: metric_card("Negatives", f"{len(neg_kw)}")
        with c4: metric_card("Bid Changes", f"{len(direct_bids) + len(agg_bids)}")
        with c5: metric_card("Harvest Ops", f"{len(harvest)}")
        
        st.divider()
        
        # 3. Account Health (5 columns)
        st.markdown("##### 🩺 Account Health Diagnostics")
        health = self._calculate_account_health(df, results)
        hc1, hc2, hc3, hc4, hc5 = st.columns(5)
        
        with hc1:
            score = health['health_score']
            color = "#22c55e" if score >= 80 else "#eab308" if score >= 60 else "#ef4444"
            metric_card("Health Score", f"{score:.0f}/100", border_color=color)
        with hc2:
            waste = health['waste_ratio']
            w_color = "#22c55e" if waste < 15 else "#ef4444"
            metric_card("Waste Ratio", f"{waste:.1f}%", subtitle="Spend on 0 orders", border_color=w_color)
        with hc3:
             metric_card("Wasted Spend", format_currency(health.get('wasted_spend', 0)), subtitle="Drain on profit", border_color="#f43f5e")
        with hc4:
             metric_card("Current ROAS", f"{health['current_roas']:.2f}x", subtitle="Account Baseline")
        with hc5:
            metric_card("Target ROAS", f"{self.config.get('TARGET_ROAS', 2.5):.1f}x", subtitle="Config Goal", border_color="#a855f7")
        
        # 4. Forecasted Impact
        if sim and "forecast" in sim:
            st.markdown("---")
            st.markdown("#### 🚀 Forecasted Impact (Expected Scenario)")
            fc1, fc2, fc3 = st.columns(3)
            
            baseline = sim["baseline"]
            forecast = sim["forecast"]
            
            sales_delta = (forecast['sales'] / baseline['sales' ] - 1) * 100 if baseline['sales'] > 0 else 0
            roas_delta = (forecast['roas'] - baseline['roas'])
            
            fc1.metric("Projected Weekly Sales", format_currency(forecast['sales']), f"{sales_delta:+.1f}%")
            fc2.metric("Projected ROAS", f"{forecast['roas']:.2f}x", f"{roas_delta:+.2f}x")
            fc3.metric("Weekly Spend Adjust", format_currency(forecast['spend']), f"{(forecast['spend']/baseline['spend']-1)*100:+.1f}%")

    def _display_negatives(self, neg_kw, neg_pt):
        st.subheader("🛑 Negative Keywords")
        if not neg_kw.empty:
            st.dataframe(neg_kw, use_container_width=True)
        else:
            st.info("No negative keywords found.")
            
        st.subheader("🛑 Product Targeting Negatives")
        if not neg_pt.empty:
            st.dataframe(neg_pt, use_container_width=True)
        else:
            st.info("No product targeting negatives found.")

    def _display_bids(self, bids_exact=None, bids_pt=None, bids_agg=None, bids_auto=None):
        st.subheader("💰 Bid Optimizations")
        # Define preferred column order
        preferred_cols = ["Targeting", "Campaign Name", "Match Type", "Clicks", "Orders", "Sales", "ROAS", "Current Bid", "CPC", "New Bid", "Reason", "Decision_Basis", "Bucket"]
        
        t1, t2, t3, t4 = st.tabs(["Exact", "Product Targeting", "Broad/Phrase", "Auto/Category"])
        
        def safe_display(df):
            if df is not None and not df.empty:
                # Only show columns that exist in the dataframe
                available_cols = [c for c in preferred_cols if c in df.columns]
                st.dataframe(df[available_cols], use_container_width=True)
            else:
                st.info("No data available.")

        with t1: safe_display(bids_exact)
        with t2: safe_display(bids_pt)
        with t3: safe_display(bids_agg)
        with t4: safe_display(bids_auto)

    def _display_harvest(self, harvest_df):
        st.subheader("🌾 Harvest Candidates")
        if harvest_df is not None and not harvest_df.empty:
            st.markdown("**What this does:** Identifies high-performing search terms that should be promoted to Exact Match campaigns.")
            
            # Bridge to Campaign Creator
            st.info("💡 **Action Required**: You can download the list below or send these terms to the [**Campaign Creator**](/?feature=creator) to build new campaigns automatically.")
            
            st.dataframe(harvest_df, use_container_width=True)
        else:
            st.info("No harvest candidates found.")

    def _display_heatmap(self, heatmap_df):
        st.markdown("### 🔥 Wasted Spend Heatmap with Action Tracking")
        if heatmap_df is not None and not heatmap_df.empty:
            st.info("💡 **Visual Performance Heatmap**: Red = Fix immediately | Yellow = Monitor | Green = Good Performance\n\n**NEW:** See which issues the optimizer is already addressing (harvests, negatives, bids)")
            
            # --- 1. Top Cards ---
            p1, p2, p3 = st.columns(3)
            high_count = len(heatmap_df[heatmap_df["Priority"].str.contains("High")])
            med_count = len(heatmap_df[heatmap_df["Priority"].str.contains("Medium")])
            good_count = len(heatmap_df[heatmap_df["Priority"].str.contains("Good")])
            
            with p1: metric_card("High Priority", str(high_count), border_color="#ef4444")
            with p2: metric_card("Medium Priority", str(med_count), border_color="#eab308")
            with p3: metric_card("Good Performance", str(good_count), border_color="#22c55e")
            
            st.divider()
            
            # --- 2. Action Status Cards ---
            st.markdown("#### 🚀 Optimizer Actions Status")
            a1, a2, a3, a4 = st.columns(4)
            addressed_mask = ~heatmap_df["Actions_Taken"].str.contains("Hold|No action", na=False)
            addressed_count = addressed_mask.sum()
            needs_attn_count = len(heatmap_df) - addressed_count
            high_addressed = (heatmap_df[heatmap_df["Priority"].str.contains("High") & addressed_mask]).shape[0]
            coverage = (addressed_count / len(heatmap_df) * 100) if len(heatmap_df) > 0 else 0
            
            with a1: metric_card("✅ Being Addressed", str(addressed_count))
            with a2: metric_card("⚠️ Needs Attention", str(needs_attn_count), border_color="#eab308")
            with a3: metric_card("🔴 High Priority Fixed", f"{high_addressed}/{high_count}", border_color="#ef4444")
            with a4: metric_card("📊 Coverage", f"{coverage:.0f}%")
            
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
                
            st.markdown(f"#### 📊 Performance Heatmap with Actions ({len(filtered_df)} items)")
            
            cols = ["Priority", "Campaign Name", "Ad Group Name", "Actions_Taken", "Spend", "Sales", "ROAS", "CVR"]
            display_df = filtered_df[[c for c in cols if c in filtered_df.columns]].copy()
            
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

    def _display_simulation(self, results):
        """Display simulation results with weekly baseline vs forecast."""
        if not results:
            st.info("Run optimization to see simulation results.")
            return
            
        sim = results.get("simulation", {})
        if not sim:
            st.warning("Simulation data missing.")
            return
            
        st.subheader("📊 Optimization Forecast")
        st.markdown("**What this does:** Predicted weekly impact of applying all recommended bid changes and harvest actions.")
        
        baseline = sim["scenarios"]["current"]
        forecast = sim["scenarios"]["expected"]
        
        # 1. Weekly Comparison Cards
        c1, c2, c3, c4 = st.columns(4)
        
        def delta_metric(col, label, current, forecast, suffix="", prefix="", inverse=False):
            delta = (forecast / current - 1) * 100 if current > 0 else 0
            delta_color = "normal" if not inverse else "inverse"
            col.metric(label, f"{prefix}{forecast:,.1f}{suffix}", f"{delta:+.1f}%", delta_color=delta_color)

        with c1: delta_metric(c1, "Weekly Sales", baseline['sales'], forecast['sales'], prefix="$")
        with c2: delta_metric(c2, "Weekly Orders", baseline['orders'], forecast['orders'])
        with c3:
            roas_delta = forecast['roas'] - baseline['roas']
            c3.metric("Weekly ROAS", f"{forecast['roas']:.2f}x", f"{roas_delta:+.2f}x")
        with c4: delta_metric(c4, "Weekly Spend", baseline['spend'], forecast['spend'], prefix="$", inverse=True)
        
        st.divider()
        
        # 2. Risk & Diagnostics
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("##### ⚠️ Risk Analysis")
            risk = sim.get("risk_analysis", {})
            sumry = risk.get("summary", {})
            
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("High Risk", sumry.get("high_risk_count", 0))
            rc2.metric("Med Risk", sumry.get("medium_risk_count", 0))
            rc3.metric("Low Risk", sumry.get("low_risk_count", 0))
            
            if risk.get("high_risk"):
                with st.expander("View High Risk Items"):
                    st.table(risk["high_risk"])
                    
        with col_r2:
            st.markdown("##### 🔍 Forecast Diagnostics")
            diag = sim.get("diagnostics", {})
            dc1, dc2 = st.columns(2)
            dc1.write(f"✅ **Active Changes**: {diag.get('actual_changes', 0)}")
            dc1.write(f"⏸️ **Holds**: {diag.get('hold_count', 0)}")
            dc2.write(f"🌾 **Harvest Ops**: {diag.get('harvest_count', 0)}")
            dc2.write(f"📅 **Data Period**: {results.get('date_info', {}).get('days', 0)} days")

        # 3. Sensitivity Chart
        st.markdown("---")
        st.markdown("##### 📈 Bid Sensitivity Analysis (Total Account Impact)")
        sens_df = sim.get("sensitivity", pd.DataFrame())
        if not sens_df.empty:
            st.line_chart(sens_df.set_index("Bid_Adjustment")[["Spend", "Sales"]])
            st.caption("Simulated weekly impact if you scaled ALL bid overrides by the percentage on the X-axis.")
    def _display_downloads(self, results):
        st.subheader("📥 Download Bulk Files")
        st.markdown("Download the optimized bulk files for immediate upload to Amazon Advertising Console.")
        
        # 1. Negative Keywords
        neg_kw = results.get("neg_kw", pd.DataFrame())
        if not neg_kw.empty:
            st.markdown("### 🛑 Negative Keywords Bulk")
            kw_bulk = generate_negatives_bulk(neg_kw, pd.DataFrame())
            with st.expander("👁️ Preview Negative Keywords", expanded=False):
                st.dataframe(kw_bulk.head(5), use_container_width=True)
            
            buf = dataframe_to_excel(kw_bulk)
            st.download_button("📥 Download Negative Keywords (.xlsx)", buf, "negative_keywords.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # 2. Bids
        all_bids = pd.concat([results.get("direct_bids", pd.DataFrame()), results.get("agg_bids", pd.DataFrame())], ignore_index=True)
        if not all_bids.empty:
            st.markdown("### 💰 Bid Optimizations Bulk")
            bid_bulk, _ = generate_bids_bulk(all_bids)
            with st.expander("👁️ Preview Bid Bulk", expanded=False):
                st.dataframe(bid_bulk.head(5), use_container_width=True)
            
            buf = dataframe_to_excel(bid_bulk)
            st.download_button("📥 Download Bid Bulk (.xlsx)", buf, "bid_optimizations.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # 3. Harvest
        harvest = results.get("harvest", pd.DataFrame())
        if not harvest.empty:
            st.markdown("### 🌾 Harvest Candidates Bulk")
            # For harvest, we normally just export the list for Creator, 
            # but we can provide a basic excel export here too.
            with st.expander("👁️ Preview Harvest List", expanded=False):
                st.dataframe(harvest.head(5), use_container_width=True)
            
            buf = dataframe_to_excel(harvest)
            st.download_button("📥 Download Harvest List (.xlsx)", buf, "harvest_candidates.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    def validate_data(self, data): return True, ""
    def analyze(self, data): return self.results
    def display_results(self, results):
        self.results = results
        # When called via run(), use the standard display
        self._display_results()
    
    def _display_results(self):
        """Internal router for multi-tab display."""
        tabs = st.tabs(["Overview", "Negatives", "Bids", "Harvest", "Audit", "Simulation", "Downloads"])
        with tabs[0]: self._display_dashboard_v2(self.results)
        with tabs[1]: self._display_negatives(self.results["neg_kw"], self.results["neg_pt"])
        with tabs[2]: self._display_bids(self.results["bids_exact"], self.results["bids_pt"], self.results["bids_agg"], self.results["bids_auto"])
        with tabs[3]: self._display_harvest(self.results["harvest"])
        with tabs[4]: self._display_heatmap(self.results["heatmap"])
        with tabs[5]: self._display_simulation(self.results)
        with tabs[6]: self._display_downloads(self.results)
