"""
Column Mapping Utilities

Centralized column normalization and lookup functions.
Consolidates duplicated column mapping logic across multiple feature modules.
"""

import pandas as pd
from typing import Optional, List, Dict
from core.data_loader import SmartMapper


def find_column_by_candidates(
    df: pd.DataFrame,
    candidates: List[str],
    case_insensitive: bool = True
) -> Optional[str]:
    """
    Find first matching column from a list of candidate names.

    Args:
        df: DataFrame to search
        candidates: List of candidate column names in priority order
        case_insensitive: If True, match ignoring case

    Returns:
        Actual column name from DataFrame, or None if not found

    Example:
        sku_col = find_column_by_candidates(
            df,
            ['SKU_advertised', 'Advertised SKU_advertised', 'SKU', 'Advertised SKU']
        )
    """
    if case_insensitive:
        col_lower_map = {c.lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate.lower() in col_lower_map:
                return col_lower_map[candidate.lower()]
    else:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate

    return None


def get_sku_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find SKU column using standard candidate list.

    Priority order:
    1. SKU_advertised
    2. Advertised SKU_advertised
    3. SKU
    4. Advertised SKU

    Returns:
        Column name or None
    """
    candidates = ['SKU_advertised', 'Advertised SKU_advertised', 'SKU', 'Advertised SKU']
    return find_column_by_candidates(df, candidates)


def get_asin_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find ASIN column using standard candidate list.

    Priority order:
    1. ASIN_advertised
    2. Advertised ASIN_advertised
    3. ASIN
    4. Advertised ASIN

    Returns:
        Column name or None
    """
    candidates = ['ASIN_advertised', 'Advertised ASIN_advertised', 'ASIN', 'Advertised ASIN']
    return find_column_by_candidates(df, candidates)


def get_search_term_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find search term column using standard candidate list.

    Priority order:
    1. Customer Search Term
    2. Search Term
    3. Query

    Returns:
        Column name or None
    """
    candidates = ['Customer Search Term', 'Search Term', 'Query']
    return find_column_by_candidates(df, candidates)


def ensure_search_term_column(df: pd.DataFrame, use_smartmapper: bool = True) -> pd.DataFrame:
    """
    Ensure 'Customer Search Term' column exists, creating from alternatives if needed.

    Args:
        df: DataFrame to process
        use_smartmapper: If True, fallback to SmartMapper for column mapping

    Returns:
        DataFrame with 'Customer Search Term' column (may be empty if not found)

    Usage:
        Consolidates pattern from assistant.py, kw_cluster.py, creator.py
    """
    # If column already exists, return as-is
    if 'Customer Search Term' in df.columns:
        return df

    # Try 'Search Term' as direct fallback
    if 'Search Term' in df.columns:
        df['Customer Search Term'] = df['Search Term']
        return df

    # Try SmartMapper if enabled
    if use_smartmapper:
        try:
            col_map = SmartMapper.map_columns(df)
            if 'Customer Search Term' in col_map:
                term_col = col_map['Customer Search Term']
                if term_col in df.columns:
                    df['Customer Search Term'] = df[term_col]
                    return df
        except Exception:
            pass

    # If all else fails, create empty column
    df['Customer Search Term'] = pd.Series(dtype=str)
    return df


def normalize_column_name(name: str, normalization: str = 'lower') -> str:
    """
    Normalize column name or value for matching.

    Args:
        name: Column name to normalize
        normalization: Type of normalization ('lower', 'campaign', 'adgroup')

    Returns:
        Normalized string

    Example:
        df['_camp_norm'] = df['Campaign Name'].apply(
            lambda x: normalize_column_name(str(x), 'campaign')
        )
    """
    name_str = str(name).strip()

    if normalization == 'lower':
        return name_str.lower()
    elif normalization == 'campaign' or normalization == 'adgroup':
        # Campaign/AdGroup normalization: lowercase + strip
        return name_str.lower().strip()
    else:
        return name_str


def create_normalized_key_columns(
    df: pd.DataFrame,
    campaign_col: str = 'Campaign Name',
    adgroup_col: str = 'Ad Group Name',
    prefix: str = '_'
) -> pd.DataFrame:
    """
    Create normalized key columns for matching.

    Args:
        df: DataFrame to process
        campaign_col: Name of campaign column
        adgroup_col: Name of ad group column
        prefix: Prefix for normalized column names (default: '_')

    Returns:
        DataFrame with added columns: _camp_norm, _ag_norm

    Usage:
        Consolidates pattern from creator.py:471, 475
    """
    if campaign_col in df.columns:
        df[f'{prefix}camp_norm'] = df[campaign_col].astype(str).str.strip().str.lower()

    if adgroup_col in df.columns:
        df[f'{prefix}ag_norm'] = df[adgroup_col].astype(str).str.strip().str.lower()

    return df


def get_metric_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Find standard metric columns with fallback patterns.

    Returns dict with keys: 'spend', 'sales', 'clicks', 'impressions', 'orders'

    Example:
        cols = get_metric_columns(df)
        if cols['spend']:
            total = df[cols['spend']].sum()
    """
    col_lower_map = {c.lower(): c for c in df.columns}

    metrics = {}

    # Spend
    for pattern in ['spend', 'cost', 'total_spend']:
        if pattern in col_lower_map:
            metrics['spend'] = col_lower_map[pattern]
            break
    else:
        metrics['spend'] = None

    # Sales
    for pattern in ['sales', 'revenue', 'total_sales']:
        if pattern in col_lower_map:
            metrics['sales'] = col_lower_map[pattern]
            break
    else:
        metrics['sales'] = None

    # Clicks
    for pattern in ['clicks', 'total_clicks']:
        if pattern in col_lower_map:
            metrics['clicks'] = col_lower_map[pattern]
            break
    else:
        metrics['clicks'] = None

    # Impressions
    for pattern in ['impressions', 'impr', 'total_impressions']:
        if pattern in col_lower_map:
            metrics['impressions'] = col_lower_map[pattern]
            break
    else:
        metrics['impressions'] = None

    # Orders
    for pattern in ['orders', 'total_orders', 'conversions']:
        if pattern in col_lower_map:
            metrics['orders'] = col_lower_map[pattern]
            break
    else:
        metrics['orders'] = None

    return metrics
