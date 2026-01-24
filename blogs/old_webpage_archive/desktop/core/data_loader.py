"""
Core Data Loading Utilities

Centralized CSV/Excel reading and column mapping.
All data ingestion goes through here.
"""

import pandas as pd
import re
from typing import Dict, Optional
import streamlit as st

class SmartMapper:
    """Smart column mapper with alias support for Amazon PPC reports."""
    
    ALIAS_MAP = {
        "Impressions": ["impressions", "impr"],
        "Clicks": ["clicks"],
        "Spend": ["spend", "cost", "total spend"],
        "Sales": [
            "7 day total sales", "14 day total sales", "sales", "total sales", 
            "ordered product sales", "attributed sales 7d", "7 day sales",
            "attributed sales 14d", "14 day sales", "sales 7d"
        ],
        "Orders": [
            "7 day total orders", "14 day total orders", "orders", "total orders", 
            "units ordered", "7 day total units", "attributed units ordered 7d", 
            "7 day orders", "attributed units ordered 14d", "14 day orders",
            "units ordered 7d", "units ordered 14d", "attributed units ordered"
        ],
        "Sales14": ["14 day total sales", "attributed sales 14d"],
        "Orders14": ["14 day total orders", "attributed units ordered 14d"],
        "CPC": ["cpc", "cost per click"],
        "Campaign Name": ["campaign name", "campaign", "campaign name (informational only)"],
        "Ad Group Name": ["ad group name", "ad group", "ad group name (informational only)"],
        "Customer Search Term": ["customer search term", "search term", "query", "keyword text"],
        "Targeting": ["keyword text", "targeting", "keyword"],
        "TargetingExpression": ["product targeting expression", "targeting expression", "resolved product targeting expression"], 
        "Match Type": ["match type"],
        "Date": ["date", "day", "start date"],
        "CampaignId": ["campaign id"],
        "AdGroupId": ["ad group id"],
        "KeywordId": ["keyword id", "target id"],
        "TargetingId": ["product targeting id", "targeting id"],
        "SKU": ["advertised sku"],  # EXACT match only - don't match "7 Day Advertised SKU Sales"
        "ASIN": ["advertised asin"],  # EXACT match only
        "Entity": ["entity"], 
        "AdGroupDefaultBid": ["ad group default bid"]
    }
    
    # Columns that require EXACT matching (no fuzzy/partial matching)
    # This prevents "7 Day Advertised SKU Sales" from matching "SKU"
    EXACT_MATCH_ONLY = {"SKU", "ASIN"}

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for fuzzy matching."""
        if not isinstance(text, str):
            return ""
        return re.sub(r"[^a-z0-9]", "", text.lower())

    @classmethod
    def map_columns(cls, df: pd.DataFrame) -> dict:
        """Map DataFrame columns to standard names using aliases."""
        mapping = {}
        df_cols = list(df.columns)
        normalized_cols = {cls.normalize(c): c for c in df_cols}

        for standard, aliases in cls.ALIAS_MAP.items():
            found = None
            
            # For SKU/ASIN, use EXACT matching only
            if standard in cls.EXACT_MATCH_ONLY:
                for alias in aliases:
                    norm_alias = cls.normalize(alias)
                    if norm_alias in normalized_cols:
                        found = normalized_cols[norm_alias]
                        break
            else:
                # Normal matching with fuzzy fallback
                for alias in aliases:
                    norm_alias = cls.normalize(alias)
                    if norm_alias in normalized_cols:
                        found = normalized_cols[norm_alias]
                        break
                if not found:
                    for alias in aliases:
                        norm_alias = cls.normalize(alias)
                        for col_norm, col_orig in normalized_cols.items():
                            if norm_alias in col_norm:
                                found = col_orig
                                break
                        if found:
                            break
            if found:
                mapping[standard] = found
        return mapping

def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load uploaded CSV or Excel file into DataFrame.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        DataFrame or None if error
    """
    if uploaded_file is None:
        return None
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        else:
            df = pd.read_excel(uploaded_file)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        return df
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert series to numeric, handling currency symbols and errors."""
    return pd.to_numeric(
        series.astype(str).str.replace(r'[^0-9.-]', '', regex=True), 
        errors='coerce'
    ).fillna(0.0)

def normalize_text(s: str) -> str:
    """Normalize text for string matching."""
    if not isinstance(s, str):
        return ""
    return re.sub(r'[^a-zA-Z0-9\s]', '', s.lower())

def get_tokens(s: str) -> set:
    """Extract meaningful tokens from text."""
    return set(normalize_text(s).split())

def is_asin(s: str) -> bool:
    """
    Check if string is or contains an ASIN (10 chars, starts with B0).
    Handles asin= and asin-expanded= prefixes commonly found in PT reports.
    """
    if not isinstance(s, str) or not s:
        return False
    clean = s.strip().upper()
    
    # Strip PT prefixes
    for prefix in ['ASIN="', "ASIN='", 'ASIN-EXPANDED="', "ASIN-EXPANDED='"]:
        if clean.startswith(prefix):
            clean = clean[len(prefix):].rstrip("\"'")
            break
    # Also handle without quotes: asin=B0...
    if clean.startswith('ASIN='):
        clean = clean[5:].strip('"\'')
    elif clean.startswith('ASIN-EXPANDED='):
        clean = clean[14:].strip('"\'')
    
    if len(clean) == 10 and clean.startswith("B0") and clean.isalnum():
        return True
    return False
