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
from typing import Tuple, Optional

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
    return (term_lower.startswith('asin=') or 
            term_lower.startswith('asin-expanded=') or
            term_lower.startswith('category=') or
            is_asin(term))


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
        neg_kw = neg_kw.reset_index(drop=True)
        # Initialize with index so scalar assignments work
        df = pd.DataFrame(index=neg_kw.index, columns=EXPORT_COLUMNS)
        
        df["Product"] = "Sponsored Products"
        df["Entity"] = "Negative Keyword"
        df["Operation"] = "Create"
        df["Campaign Id"] = neg_kw.get("CampaignId", "")
        df["Ad Group Id"] = neg_kw.get("AdGroupId", "")
        df["Campaign Name"] = neg_kw["Campaign Name"]
        df["Ad Group Name"] = neg_kw["Ad Group Name"]
        # Strip any targeting prefixes from the term
        df["Keyword Text"] = neg_kw["Term"].apply(strip_targeting_prefix)
        df["Match Type"] = "negativeExact"
        df["Keyword Id"] = neg_kw.get("KeywordId", "")
        df["State"] = "enabled"
        frames.append(df)
        
    if neg_pt is not None and not neg_pt.empty:
        neg_pt = neg_pt.reset_index(drop=True)
        # Initialize with index
        df = pd.DataFrame(index=neg_pt.index, columns=EXPORT_COLUMNS)
        
        df["Product"] = "Sponsored Products"
        df["Entity"] = "Negative Product Targeting"
        df["Operation"] = "Create"
        df["Campaign Id"] = neg_pt.get("CampaignId", "")
        df["Ad Group Id"] = neg_pt.get("AdGroupId", "")
        df["Campaign Name"] = neg_pt["Campaign Name"]
        df["Ad Group Name"] = neg_pt["Ad Group Name"]
        # For PT, format as asin="ASIN" expression
        df["Product Targeting Expression"] = neg_pt["Term"].apply(
            lambda x: f'asin="{strip_targeting_prefix(x)}"' if is_asin(strip_targeting_prefix(x)) else x
        )
        df["Match Type"] = "negativeExact"
        df["Product Targeting Id"] = neg_pt.get("TargetingId", "")
        df["State"] = "enabled"
        frames.append(df)
        
    return pd.concat(frames) if frames else pd.DataFrame(columns=EXPORT_COLUMNS)


def generate_bids_bulk(bids_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Generate Amazon bulk upload file for bid updates using standard schema.
    
    Args:
        bids_df: DataFrame of bid optimization results
        
    Returns:
        Tuple of (bulk DataFrame, count of changes)
    """
    if bids_df is None or bids_df.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS), 0
        
    # Filter for actually changed bids (exclude holds)
    changes = bids_df[~bids_df["Reason"].str.contains("Hold", case=False, na=False)].copy()
    if changes.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS), 0
    
    changes = changes.reset_index(drop=True)
    # Initialize with index to allow scalar assignment
    df = pd.DataFrame(index=changes.index, columns=EXPORT_COLUMNS)
    
    df["Product"] = "Sponsored Products"
    df["Operation"] = "Update"
    df["Campaign Id"] = changes.get("CampaignId", "")
    df["Ad Group Id"] = changes.get("AdGroupId", "")
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
    
    # ID Mapping
    df["Keyword Id"] = changes.get("KeywordId", "")
    df["Product Targeting Id"] = changes.get("TargetingId", "")
    
    return df, len(changes)


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
