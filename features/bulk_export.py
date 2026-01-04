"""
Bulk Export Module - Amazon Bulk Sheet Generation

Generates Amazon Advertising bulk upload files for:
- Negative keywords/product targeting
- Bid updates
- Harvest campaign creation

Separated from optimizer.py for cleaner maintenance.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List

# Import comprehensive validation engine
from core.bulk_validation import (
    validate_bulk_export,
    validate_isolation_negative,
    validate_bleeder_negative,
    validate_bid_update,
    detect_negative_type,
    NegativeType,
    Severity,
    ValidationResult,
    is_blank,
    ERROR_MESSAGES,
)

from bulk_validation_spec import OptimizationRecommendation


# ==========================================
# AMAZON BULK FILE SCHEMA
# ==========================================

EXPORT_COLUMNS = [
    "Product", "Entity", "Operation", "Campaign Id", "Ad Group Id", 
    "Campaign Name", "Ad Group Name", "Bid", "Ad Group Default Bid", 
    "Keyword Text", "Match Type", "Product Targeting Expression", 
    "Keyword Id", "Product Targeting Id", "State"
]


# ==========================================
# LEGACY VALIDATION (DEPRECATED - use validate_bulk_export)
# ==========================================

def validate_negatives_bulk(df: pd.DataFrame, currency: str = "USD") -> Tuple[pd.DataFrame, List[dict]]:
    """
    Validate negatives bulk export for Amazon compliance.
    
    Uses new comprehensive validation engine with Isolation/Bleeder detection.
    
    Returns:
        Tuple of (sorted DataFrame, list of validation issues)
    """
    if df is None or df.empty:
        return df, []
    
    df = df.copy()
    
    # Run comprehensive validation
    validated_df, result = validate_bulk_export(df, export_type="negatives", currency=currency)
    
    # Convert to unified format
    issues = result.to_dict_list()
    
    # Additional cleaning logic

    # --- NEG001: Mutual Exclusivity ---
    for idx, row in df.iterrows():
        entity = str(row.get("Entity", "")).lower()
        
        if "keyword" in entity:
            # KW entity: PT fields must be blank
            if not is_blank(row.get("Product Targeting Expression")):
                issues.append({"row": idx, "code": "NEG001", "msg": "Keyword entity has PT Expression filled (auto-cleared)", "severity": "warning"})
                df.at[idx, "Product Targeting Expression"] = ""
            if not is_blank(row.get("Product Targeting Id")):
                issues.append({"row": idx, "code": "NEG001", "msg": "Keyword entity has PT Id filled (auto-cleared)", "severity": "warning"})
                df.at[idx, "Product Targeting Id"] = ""
        
        elif "product" in entity or "targeting" in entity:
            # PT entity: KW fields must be blank
            if not is_blank(row.get("Keyword Text")):
                issues.append({"row": idx, "code": "NEG001", "msg": "PT entity has Keyword Text filled (auto-cleared)", "severity": "warning"})
                df.at[idx, "Keyword Text"] = ""
            if not is_blank(row.get("Keyword Id")):
                issues.append({"row": idx, "code": "NEG001", "msg": "PT entity has Keyword Id filled (auto-cleared)", "severity": "warning"})
                df.at[idx, "Keyword Id"] = ""
    
    # --- NEG002: Match Type ---
    for idx, row in df.iterrows():
        entity = str(row.get("Entity", "")).lower()
        
        if "keyword" in entity:
            if row.get("Match Type") != "negativeExact":
                issues.append({"row": idx, "code": "NEG002", "msg": f"Match Type corrected to negativeExact", "severity": "warning"})
                df.at[idx, "Match Type"] = "negativeExact"
        
        elif "product" in entity or "targeting" in entity:
            if not is_blank(row.get("Match Type")):
                issues.append({"row": idx, "code": "NEG002", "msg": "PT Match Type cleared (should be blank)", "severity": "warning"})
                df.at[idx, "Match Type"] = ""
    
    # --- NEG003: Bid Column Blank ---
    if "Bid" in df.columns:
        for idx, row in df.iterrows():
            if not is_blank(row.get("Bid")):
                issues.append({"row": idx, "code": "NEG003", "msg": "Bid cleared (must be blank for negatives)", "severity": "warning"})
                df.at[idx, "Bid"] = ""
    
    # --- GEN001: Flag Missing Campaign/AdGroup IDs ---
    df["_has_campaign_id"] = df["Campaign Id"].apply(lambda x: not is_blank(x))
    df["_has_adgroup_id"] = df["Ad Group Id"].apply(lambda x: not is_blank(x))
    df["_missing_ids"] = ~(df["_has_campaign_id"] & df["_has_adgroup_id"])
    
    missing_id_count = df["_missing_ids"].sum()
    if missing_id_count > 0:
        issues.append({"row": -1, "code": "GEN001", "msg": f"{missing_id_count} rows missing Campaign/Ad Group IDs (moved to bottom)", "severity": "warning"})
    
    # Sort: rows with IDs first, missing IDs last
    df = df.sort_values("_missing_ids", ascending=True).reset_index(drop=True)
    df.drop(columns=["_has_campaign_id", "_has_adgroup_id", "_missing_ids"], inplace=True)
    
    # --- GEN002: No Dual IDs ---
    for idx, row in df.iterrows():
        has_kwid = not is_blank(row.get("Keyword Id"))
        has_ptid = not is_blank(row.get("Product Targeting Id"))
        
        if has_kwid and has_ptid:
            issues.append({"row": idx, "code": "GEN002", "msg": "Row has both Keyword Id and PT Id (auto-corrected)", "severity": "warning"})
            # Keep the one matching the entity type
            entity = str(row.get("Entity", "")).lower()
            if "keyword" in entity:
                df.at[idx, "Product Targeting Id"] = ""
            else:
                df.at[idx, "Keyword Id"] = ""
    
    # --- GEN003: Missing Entity-Specific IDs (Only for Updates) ---
    missing_kwid_count = 0
    missing_ptid_count = 0
    for idx, row in df.iterrows():
        entity = str(row.get("Entity", "")).lower()
        operation = str(row.get("Operation", "")).lower()
        
        # We only care about missing IDs if we are UPDATING an existing entity
        # If we are CREATING a new negative, it won't have an ID yet
        if operation == "create":
            continue
            
        if "keyword" in entity:
            if is_blank(row.get("Keyword Id")):
                missing_kwid_count += 1
        elif "product" in entity or "targeting" in entity:
            if is_blank(row.get("Product Targeting Id")):
                missing_ptid_count += 1
    
    if missing_kwid_count > 0:
        issues.append({"row": -1, "code": "GEN003", "msg": f"{missing_kwid_count} Keyword rows missing Keyword Id", "severity": "warning"})
    if missing_ptid_count > 0:
        issues.append({"row": -1, "code": "GEN003", "msg": f"{missing_ptid_count} PT rows missing Product Targeting Id", "severity": "warning"})
    
    # --- NEG004: Duplicate Detection ---
    # For KW: Campaign + AdGroup + Keyword Text
    # For PT: Campaign + AdGroup + PT Expression
    kw_mask = df["Entity"].str.contains("Keyword", case=False, na=False)
    pt_mask = ~kw_mask
    
    if kw_mask.any():
        kw_df = df[kw_mask]
        dup_kw = kw_df.duplicated(subset=["Campaign Name", "Ad Group Name", "Keyword Text"], keep="first")
        if dup_kw.any():
            dup_count = dup_kw.sum()
            issues.append({"row": -1, "code": "NEG004", "msg": f"Removed {dup_count} duplicate Keyword negatives (kept first)", "severity": "warning"})
            # Mark duplicates for removal
            df = df[~(kw_mask & df.duplicated(subset=["Campaign Name", "Ad Group Name", "Keyword Text"], keep="first"))]
    
    if pt_mask.any():
        pt_df = df[pt_mask]
        dup_pt = pt_df.duplicated(subset=["Campaign Name", "Ad Group Name", "Product Targeting Expression"], keep="first")
        if dup_pt.any():
            dup_count = dup_pt.sum()
            issues.append({"row": -1, "code": "NEG004", "msg": f"Removed {dup_count} duplicate PT negatives (kept first)", "severity": "warning"})
            df = df[~(pt_mask & df.duplicated(subset=["Campaign Name", "Ad Group Name", "Product Targeting Expression"], keep="first"))]
    
    # --- FINAL SORT: Complete rows first, problematic rows at bottom ---
    # Complete = has Campaign Id + Ad Group Id + Entity-specific ID (Keyword Id OR PT Id)
    df = df.reset_index(drop=True)
    
    has_campaign = df["Campaign Id"].notna() & (df["Campaign Id"].astype(str).str.strip() != "")
    has_adgroup = df["Ad Group Id"].notna() & (df["Ad Group Id"].astype(str).str.strip() != "")
    has_kwid = df["Keyword Id"].notna() & (df["Keyword Id"].astype(str).str.strip() != "")
    has_ptid = df["Product Targeting Id"].notna() & (df["Product Targeting Id"].astype(str).str.strip() != "")
    has_entity_id = has_kwid | has_ptid
    
    df["_is_complete"] = has_campaign & has_adgroup & has_entity_id
    df = df.sort_values("_is_complete", ascending=False).reset_index(drop=True)
    df.drop(columns=["_is_complete"], inplace=True)
    
    return df, issues



# ==========================================
# HELPER FUNCTIONS
# ==========================================

def strip_targeting_prefix(term: str) -> str:
    """
    Strip targeting prefixes (asin=, asin-expanded=, category=, keyword-group=) 
    for clean bulk export.
    
    Examples:
        'asin="B01ABC123"' -> 'B01ABC123'
        'asin-expanded="B01ABC123"' -> 'B01ABC123'  
        'category="Sports Water Bottles"' -> 'category="Sports Water Bottles"' (keep for PT expression)
        'keyword-group="..."' -> '...'
        'water bottle' -> 'water bottle' (unchanged)
    """
    term = str(term).strip()
    term_lower = term.lower()
    
    # Handle asin= and asin-expanded= - extract just the ASIN
    if term_lower.startswith('asin="') or term_lower.startswith("asin='"):
        # Extract ASIN from asin="B01ABC123"
        return term.split('=', 1)[1].strip('"\'')
    if term_lower.startswith('asin-expanded="') or term_lower.startswith("asin-expanded='"):
        # Extract ASIN from asin-expanded="B01ABC123"
        return term.split('=', 1)[1].strip('"\'')
    
    # Keep category= expressions as-is for Product Targeting Expression column
    if term_lower.startswith('category='):
        return term
    
    # Strip keyword-group= prefix
    if term_lower.startswith('keyword-group='):
        return term.split('=', 1)[1].strip('"\'')
    
    return term


def is_asin(term: str) -> bool:
    """Check if term looks like an ASIN (B0 followed by alphanumeric)."""
    import re
    term = str(term).strip().upper()
    return bool(re.match(r'^B0[A-Z0-9]{8,}$', term))


def is_product_targeting(term: str) -> bool:
    """Check if term is a product targeting expression."""
    term_lower = str(term).lower().strip()
    AUTO_TARGETS = {'close-match', 'substitutes', 'loose-match', 'complements', 'auto'}
    return (term_lower.startswith('asin=') or 
            term_lower.startswith('asin-expanded=') or
            term_lower.startswith('category=') or
            term_lower.startswith('keyword-group=') or
            term_lower in AUTO_TARGETS or
            is_asin(term))

def clean_id(val) -> str:
    """Format ID as clean integer string, removing .0 decimals."""
    if pd.isna(val) or val is None or str(val).strip() == "" or str(val).lower() == "none":
        return ""
    try:
        # Convert to float first to handle potential strings like "1.23e+10"
        # then to int, then to str
        f_val = float(val)
        if f_val == 0:
            return ""
        return str(int(f_val))
    except (ValueError, TypeError):
        # Fallback to string stripping .0 if it exists
        s_val = str(val).strip()
        if s_val.endswith(".0"):
            return s_val[:-2]
        return s_val


# ==========================================
# BULK FILE GENERATORS
# ==========================================

def generate_negatives_bulk(neg_kw: pd.DataFrame, neg_pt: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Amazon bulk upload file for negatives using standard schema.
    
    Args:
        neg_kw: DataFrame of negative keyword candidates
        neg_pt: DataFrame of negative product targeting candidates
        
    Returns:
        DataFrame formatted for Amazon bulk upload
    """
    frames = []
    
    if neg_kw is not None and not neg_kw.empty:
        # VALIDATE-AT-SOURCE: Filter to only executable recommendations
        if 'recommendation' in neg_kw.columns:
            neg_kw = neg_kw[neg_kw['recommendation'].apply(lambda x: x.can_execute if isinstance(x, OptimizationRecommendation) else True)]
            
        neg_kw = neg_kw.reset_index(drop=True)
        # Initialize with index so scalar assignments work
        df = pd.DataFrame(index=neg_kw.index, columns=EXPORT_COLUMNS)
        
        df["Product"] = "Sponsored Products"
        df["Entity"] = "Negative Keyword"
        df["Operation"] = "Create"
        df["Campaign Id"] = (neg_kw["CampaignId"] if "CampaignId" in neg_kw.columns else pd.Series([""] * len(neg_kw))).apply(clean_id)
        df["Ad Group Id"] = (neg_kw["AdGroupId"] if "AdGroupId" in neg_kw.columns else pd.Series([""] * len(neg_kw))).apply(clean_id)
        df["Campaign Name"] = neg_kw["Campaign Name"]
        df["Ad Group Name"] = neg_kw["Ad Group Name"]
        # Strip any targeting prefixes from the term
        df["Keyword Text"] = neg_kw["Term"].apply(strip_targeting_prefix)
        df["Match Type"] = "negativeExact"
        df["Keyword Id"] = (neg_kw["KeywordId"] if "KeywordId" in neg_kw.columns else pd.Series([""] * len(neg_kw))).apply(clean_id)
        df["Product Targeting Id"] = "" # STRICT EXCLUSIVITY
        df["State"] = "enabled"
        
        # FINAL CAST: Ensure all ID columns are strings
        for col in ["Campaign Id", "Ad Group Id", "Keyword Id"]:
            df[col] = df[col].astype(str).replace("nan", "").replace("None", "").str.strip()
            
        frames.append(df)
        
    if neg_pt is not None and not neg_pt.empty:
        # VALIDATE-AT-SOURCE: Filter to only executable recommendations
        if 'recommendation' in neg_pt.columns:
            neg_pt = neg_pt[neg_pt['recommendation'].apply(lambda x: x.can_execute if isinstance(x, OptimizationRecommendation) else True)]
            
        neg_pt = neg_pt.reset_index(drop=True)
        # Initialize with index
        df = pd.DataFrame(index=neg_pt.index, columns=EXPORT_COLUMNS)
        
        df["Product"] = "Sponsored Products"
        df["Entity"] = "Negative Product Targeting"
        df["Operation"] = "Create"
        df["Campaign Id"] = (neg_pt["CampaignId"] if "CampaignId" in neg_pt.columns else pd.Series([""] * len(neg_pt))).apply(clean_id)
        df["Ad Group Id"] = (neg_pt["AdGroupId"] if "AdGroupId" in neg_pt.columns else pd.Series([""] * len(neg_pt))).apply(clean_id)
        df["Campaign Name"] = neg_pt["Campaign Name"]
        df["Ad Group Name"] = neg_pt["Ad Group Name"]
        # For PT, format as asin="ASIN" expression
        df["Product Targeting Expression"] = neg_pt["Term"].apply(
            lambda x: f'asin="{strip_targeting_prefix(x)}"' if is_asin(strip_targeting_prefix(x)) else x
        )
        df["Match Type"] = ""  # PT should have blank match type per R2
        df["Keyword Id"] = "" # STRICT EXCLUSIVITY
        df["Product Targeting Id"] = (neg_pt["TargetingId"] if "TargetingId" in neg_pt.columns else pd.Series([""] * len(neg_pt))).apply(clean_id)
        df["State"] = "enabled"
        
        # FINAL CAST: Ensure all ID columns are strings
        for col in ["Campaign Id", "Ad Group Id", "Product Targeting Id"]:
            df[col] = df[col].astype(str).replace("nan", "").replace("None", "").str.strip()
            
        frames.append(df)
        
    raw_df = pd.concat(frames) if frames else pd.DataFrame(columns=EXPORT_COLUMNS)
    
    # Validate and clean
    validated_df, issues = validate_negatives_bulk(raw_df)
    
    return validated_df, issues


def generate_bids_bulk(bids_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Generate Amazon bulk upload file for bid updates using standard schema.
    
    Args:
        bids_df: DataFrame of bid optimization results
        
    Returns:
        Tuple of (bulk DataFrame, count of changes)
    """
    if bids_df is None or bids_df.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS), []
        
    # Filter for actually changed bids (exclude holds)
    changes = bids_df[~bids_df["Reason"].str.contains("Hold", case=False, na=False)].copy()
    if changes is None or changes.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS), []
    
    # VALIDATE-AT-SOURCE: Filter to only executable recommendations
    if 'recommendation' in changes.columns:
        changes = changes[changes['recommendation'].apply(lambda x: x.can_execute if isinstance(x, OptimizationRecommendation) else True)]
        
    if changes.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS), []
        
    changes = changes.reset_index(drop=True)
    # Initialize with index to allow scalar assignment
    df = pd.DataFrame(index=changes.index, columns=EXPORT_COLUMNS)
    
    df["Product"] = "Sponsored Products"
    df["Operation"] = "Update"
    df["Campaign Id"] = (changes["CampaignId"] if "CampaignId" in changes.columns else pd.Series([""] * len(df))).apply(clean_id)
    df["Ad Group Id"] = (changes["AdGroupId"] if "AdGroupId" in changes.columns else pd.Series([""] * len(df))).apply(clean_id)
    df["Campaign Name"] = changes["Campaign Name"]
    df["Ad Group Name"] = changes["Ad Group Name"]
    df["Bid"] = changes["New Bid"]
    df["State"] = "enabled"
    
    def get_entity_details(row):
        """Determine entity type and populate appropriate columns."""
        m_type = str(row.get("Match Type", "")).lower()
        bucket = str(row.get("Bucket", ""))
        targeting = str(row.get("Targeting", ""))
        
        # Auto targets -> Product Targeting entity
        if m_type in ["auto", "-"] or bucket == "Auto":
            # Clean the targeting expression
            clean_target = strip_targeting_prefix(targeting)
            return "Product Targeting", "", targeting
            
        # Explicit PT expression
        if bucket == "Product Targeting" or is_product_targeting(targeting):
            return "Product Targeting", "", targeting
            
        # Category targets
        if targeting.lower().startswith("category="):
            return "Product Targeting", "", targeting
            
        # Regular keywords
        clean_kw = strip_targeting_prefix(targeting)
        return "Keyword", clean_kw, row.get("Match Type", "")

    # Apply entity detection
    details = changes.apply(get_entity_details, axis=1, result_type='expand')
    df["Entity"] = details[0]
    df["Keyword Text"] = details[1]
    df["Product Targeting Expression"] = details[2]
    
    # For keywords, put Match Type. For PT, leave blank
    df["Match Type"] = np.where(df["Entity"] == "Keyword", details[2], "")
    
    # CRITICAL: If Match Type is provided (Keyword), Product Targeting Expression MUST be blank
    df["Product Targeting Expression"] = np.where(df["Match Type"] != "", "", df["Product Targeting Expression"])
    
    df["Keyword Id"] = (changes["KeywordId"] if "KeywordId" in changes.columns else pd.Series([""] * len(df))).apply(clean_id)
    df["Product Targeting Id"] = (changes["TargetingId"] if "TargetingId" in changes.columns else pd.Series([""] * len(df))).apply(clean_id)
    
    # Enforce Mutual Exclusivity based on entity
    df["Keyword Id"] = np.where(df["Entity"] == "Keyword", df["Keyword Id"], "")
    df["Product Targeting Id"] = np.where(df["Entity"] == "Product Targeting", df["Product Targeting Id"], "")
    
    # Run bid validation
    validated_df, result = validate_bulk_export(df, export_type="bids", currency="AED")
    issues = result.to_dict_list()
    
    # Additional bid-specific validations
    for idx, row in df.iterrows():
        row_num = idx + 2
        
        # Check for missing IDs
        entity = str(row.get("Entity", "")).lower()
        if "keyword" in entity and is_blank(row.get("Keyword Id")):
            issues.append({"row": row_num, "code": "BID004", "msg": "Keyword row missing Keyword Id", "severity": "warning"})
        elif "targeting" in entity and is_blank(row.get("Product Targeting Id")):
            issues.append({"row": row_num, "code": "BID004", "msg": "PT row missing Product Targeting Id", "severity": "warning"})
        
        # Check for dual IDs
        has_kwid = not is_blank(row.get("Keyword Id"))
        has_ptid = not is_blank(row.get("Product Targeting Id"))
        if has_kwid and has_ptid:
            issues.append({"row": row_num, "code": "BID005", "msg": "Row has both Keyword Id and PT Id", "severity": "error"})

    # --- BID006: Duplicate Detection ---
    # Deduplicate based on Entity logic to prevent conflicting updates
    
    kw_mask = df["Entity"] == "Keyword"
    pt_mask = df["Entity"] == "Product Targeting"
    
    if kw_mask.any():
        # For Keywords: Campaign + AdGroup + Text + Match Type
        # (Same keyword text can exist with different match types)
        dup_subset = ["Campaign Name", "Ad Group Name", "Keyword Text", "Match Type"]
        kw_rows = df[kw_mask]
        dup_kw = kw_rows.duplicated(subset=dup_subset, keep="first")
        
        if dup_kw.any():
            dup_count = dup_kw.sum()
            issues.append({"row": -1, "code": "BID006", "msg": f"Removed {dup_count} duplicate Bid updates for Keywords (kept first)", "severity": "warning"})
            # Identify indices to drop
            drop_indices = kw_rows[dup_kw].index
            df = df.drop(drop_indices)
    
    if pt_mask.any():
        # For PT: Campaign + AdGroup + Expression
        # (PT expression is unique per ad group usually)
        dup_subset = ["Campaign Name", "Ad Group Name", "Product Targeting Expression"]
        pt_rows = df[pt_mask]
        dup_pt = pt_rows.duplicated(subset=dup_subset, keep="first")
        
        if dup_pt.any():
            dup_count = dup_pt.sum()
            issues.append({"row": -1, "code": "BID006", "msg": f"Removed {dup_count} duplicate Bid updates for Targets (kept first)", "severity": "warning"})
            # Identify indices to drop
            drop_indices = pt_rows[dup_pt].index
            df = df.drop(drop_indices)
    
    # --- FINAL SORT: Complete rows first, problematic rows at bottom ---
    # For bids: need Campaign Id + (Keyword Id OR PT Id)
    has_campaign = df["Campaign Id"].notna() & (df["Campaign Id"].astype(str).str.strip() != "")
    has_kwid_col = df["Keyword Id"].notna() & (df["Keyword Id"].astype(str).str.strip() != "")
    has_ptid_col = df["Product Targeting Id"].notna() & (df["Product Targeting Id"].astype(str).str.strip() != "")
    has_entity_id = has_kwid_col | has_ptid_col
    
    df["_is_complete"] = has_campaign & has_entity_id
    df = df.sort_values("_is_complete", ascending=False).reset_index(drop=True)
    df.drop(columns=["_is_complete"], inplace=True)
    
    return df, issues


def generate_harvest_bulk(harvest_df: pd.DataFrame, 
                          target_campaign: str = None,
                          target_ad_group: str = None) -> pd.DataFrame:
    """
    Generate Amazon bulk upload file for harvest keywords (new exact match keywords).
    
    Args:
        harvest_df: DataFrame of harvest candidates
        target_campaign: Optional target campaign name for new keywords
        target_ad_group: Optional target ad group name for new keywords
        
    Returns:
        DataFrame formatted for Amazon bulk upload
    """
    if harvest_df is None or harvest_df.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS)
    
    harvest_df = harvest_df.reset_index(drop=True)
    df = pd.DataFrame(index=harvest_df.index, columns=EXPORT_COLUMNS)
    
    df["Product"] = "Sponsored Products"
    df["Entity"] = "Keyword"
    df["Operation"] = "Create"
    
    # Use target campaign/ad group if specified, otherwise use source
    if target_campaign:
        df["Campaign Name"] = target_campaign
    else:
        df["Campaign Name"] = harvest_df.get("Campaign Name", "")
        
    if target_ad_group:
        df["Ad Group Name"] = target_ad_group
    else:
        df["Ad Group Name"] = harvest_df.get("Ad Group Name", "")
    
    # Get harvest term - clean any targeting prefixes
    term_col = "Customer Search Term" if "Customer Search Term" in harvest_df.columns else "Harvest_Term"
    df["Keyword Text"] = harvest_df[term_col].apply(strip_targeting_prefix)
    df["Match Type"] = "exact"
    df["State"] = "enabled"
    
    # Use suggested bid if available
    if "Suggested Bid" in harvest_df.columns:
        df["Bid"] = harvest_df["Suggested Bid"]
    elif "CPC" in harvest_df.columns:
        df["Bid"] = harvest_df["CPC"]
    
    return df
