"""
Data Validation Utilities

Common validation functions for all features.
"""

import pandas as pd
from typing import Tuple, List

def validate_search_term_report(df: pd.DataFrame, col_map: dict) -> Tuple[bool, str]:
    """
    Validate search term report structure.
    
    Args:
        df: DataFrame to validate
        col_map: Column mapping from SmartMapper
        
    Returns:
        (is_valid, error_message)
    """
    required = ['Term', 'Impressions', 'Clicks', 'Spend', 'Orders']
    missing = [col for col in required if col not in col_map]
    
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    
    # Check for data
    if df.empty:
        return False, "File is empty"
    
    if len(df) < 10:
        return False, "Not enough data (need at least 10 rows)"
    
    return True, ""

def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> Tuple[bool, str]:
    """
    Validate DataFrame has required columns.
    
    Args:
        df: DataFrame to check
        required_cols: List of required column names
        
    Returns:
        (is_valid, error_message)
    """
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    
    return True, ""

def validate_numeric_column(df: pd.DataFrame, col_name: str) -> Tuple[bool, str]:
    """
    Validate column contains numeric data.
    
    Args:
        df: DataFrame to check
        col_name: Column name to validate
        
    Returns:
        (is_valid, error_message)
    """
    if col_name not in df.columns:
        return False, f"Column '{col_name}' not found"
    
    try:
        pd.to_numeric(df[col_name], errors='coerce')
        return True, ""
    except Exception as e:
        return False, f"Column '{col_name}' is not numeric: {str(e)}"

def validate_minimum_rows(df: pd.DataFrame, min_rows: int) -> Tuple[bool, str]:
    """
    Validate DataFrame has minimum number of rows.
    
    Args:
        df: DataFrame to check
        min_rows: Minimum required rows
        
    Returns:
        (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"Not enough data. Need at least {min_rows} rows, found {len(df)}"
    
    return True, ""
