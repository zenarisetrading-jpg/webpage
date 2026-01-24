"""
Performance Metrics Calculation

All ROAS, ACOS, CTR, CVR, weekly normalization logic lives here.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional

def detect_date_range(df: pd.DataFrame, col_map: dict = None) -> dict:
    """
    Detect date range in uploaded data and calculate weeks for normalization.
    
    Args:
        df: DataFrame with uploaded data
        col_map: Optional column mapping from SmartMapper
        
    Returns:
        dict with: start, end, days, weeks, label
    """
    # Try multiple common date columns if map not provided
    date_cols = [col_map.get("Date")] if col_map and col_map.get("Date") else ["Date", "Start Date", "day", "date"]
    
    found_col = None
    for c in date_cols:
        if c and c in df.columns:
            found_col = c
            break
            
    if not found_col:
        # Fallback: Assume 30 days if no date found (standard monthly report)
        return {
            "start": None,
            "end": None,
            "days": 30,
            "weeks": 4.28,
            "label": "Unknown Range (Assumed 30 days)"
        }

    try:
        dates = pd.to_datetime(df[found_col], errors='coerce').dropna()
        if dates.empty:
            return {
                "start": None,
                "end": None,
                "days": 30,
                "weeks": 4.28,
                "label": "Invalid Dates (Assumed 30 days)"
            }
            
        start_date = dates.min()
        end_date = dates.max()
        days = (end_date - start_date).days + 1
        
        # Avoid division by zero
        days = max(1, days)
        weeks = max(0.14, days / 7.0) # Minimum 1 day
        
        return {
            "start": start_date,
            "end": end_date,
            "days": days,
            "weeks": weeks,
            "label": f"{days} days ({start_date.strftime('%b %d')} - {end_date.strftime('%b %d')})"
        }
    except Exception as e:
        print(f"Date detection error: {e}")
        return {
            "start": None,
            "end": None,
            "days": 30,
            "weeks": 4.28,
            "label": "Error Detecting Dates"
        }

def normalize_to_weekly(metrics: dict, weeks: float) -> dict:
    """
    Normalize volume metrics to weekly averages.
    Ratio metrics (ROAS, ACOS, CVR, CTR) remain unchanged.
    
    Args:
        metrics: Dict with keys like spend, sales, clicks, orders, impressions
        weeks: Number of weeks to divide by
        
    Returns:
        dict with weekly-normalized metrics
    """
    volume_keys = ['spend', 'sales', 'clicks', 'orders', 'impressions']
    
    # Avoid zero division
    if weeks <= 0:
        return metrics
    
    normalized = metrics.copy()
    for key in volume_keys:
        if key in normalized:
            val = normalized[key]
            # FIX: Handle numpy types (np.int64, np.float64)
            try:
                numeric_val = float(val)
                normalized[key] = numeric_val / weeks
            except (TypeError, ValueError):
                pass  # Skip non-numeric values
    
    # Recalculate ratio metrics from normalized volumes
    if normalized.get('spend', 0) > 0:
        normalized['roas'] = normalized['sales'] / normalized['spend']
        normalized['cpc'] = normalized['spend'] / normalized['clicks'] if normalized.get('clicks', 0) > 0 else 0
    else:
        normalized['roas'] = 0
        normalized['cpc'] = 0
            
    if normalized.get('clicks', 0) > 0:
        normalized['cvr'] = (normalized['orders'] / normalized['clicks']) * 100
    else:
        normalized['cvr'] = 0
        
    return normalized

def calculate_metrics(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Calculate all performance metrics (ROAS, ACOS, CTR, CVR, CPC).
    
    Args:
        df: DataFrame with performance data
        col_map: Column mapping from SmartMapper
        
    Returns:
        DataFrame with calculated metrics added
    """
    result = df.copy()
    
    # Extract column names
    spend_col = col_map.get('Spend')
    sales_col = col_map.get('Sales')
    clicks_col = col_map.get('Clicks')
    orders_col = col_map.get('Orders')
    impr_col = col_map.get('Impressions')
    
    # Calculate ROAS
    if spend_col and sales_col:
        result['ROAS'] = np.where(
            result[spend_col] > 0,
            result[sales_col] / result[spend_col],
            0
        )
    
    # Calculate ACOS
    if spend_col and sales_col:
        result['ACOS'] = np.where(
            result[sales_col] > 0,
            (result[spend_col] / result[sales_col]) * 100,
            0
        )
    
    # Calculate CTR
    if clicks_col and impr_col:
        result['CTR'] = np.where(
            result[impr_col] > 0,
            (result[clicks_col] / result[impr_col]) * 100,
            0
        )
    
    # Calculate CVR
    if orders_col and clicks_col:
        result['CVR'] = np.where(
            result[clicks_col] > 0,
            (result[orders_col] / result[clicks_col]) * 100,
            0
        )
    
    # Calculate CPC
    if spend_col and clicks_col:
        result['CPC'] = np.where(
            result[clicks_col] > 0,
            result[spend_col] / result[clicks_col],
            0
        )
    
    return result
