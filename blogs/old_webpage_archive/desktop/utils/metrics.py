"""
Shared PPC Metrics Calculation Utilities

This module provides standardized metric calculations used across multiple features
to ensure consistency and reduce code duplication.
"""

import pandas as pd
import numpy as np
from typing import Literal


def calculate_ppc_metrics(
    df: pd.DataFrame,
    percentage_format: Literal['decimal', 'percentage'] = 'decimal',
    inplace: bool = True
) -> pd.DataFrame:
    """
    Calculate standard PPC metrics: ROAS, CPC, CTR, CVR, ACOS.

    Args:
        df: DataFrame with Spend, Sales, Clicks, Impressions, Orders columns
        percentage_format: How to store CTR/CVR/ACOS
            - 'decimal': 0.05 = 5% (used by optimizer.py)
            - 'percentage': 5.0 = 5% (used by performance_snapshot.py)
        inplace: If False, returns a copy. If True, modifies df directly.

    Returns:
        DataFrame with added metric columns

    Note:
        - ROAS is always stored as a multiplier (2.5 = 2.5x)
        - CPC is always stored in currency units
        - CTR, CVR, ACOS format depends on percentage_format parameter
    """
    if not inplace:
        df = df.copy()

    # Multiplier for percentage metrics
    multiplier = 100 if percentage_format == 'percentage' else 1

    # ROAS: Sales / Spend (always as multiplier)
    df['ROAS'] = np.where(df['Spend'] > 0, df['Sales'] / df['Spend'], 0)

    # CPC: Spend / Clicks (always in currency)
    df['CPC'] = np.where(df['Clicks'] > 0, df['Spend'] / df['Clicks'], 0)

    # CTR: Clicks / Impressions
    df['CTR'] = np.where(
        df['Impressions'] > 0,
        (df['Clicks'] / df['Impressions']) * multiplier,
        0
    )

    # CVR: Orders / Clicks (requires Orders column)
    if 'Orders' in df.columns:
        df['CVR'] = np.where(
            df['Clicks'] > 0,
            (df['Orders'] / df['Clicks']) * multiplier,
            0
        )

    # ACOS: Spend / Sales (ALWAYS in percentage format, regardless of parameter)
    # Both optimizer.py and performance_snapshot.py use percentage for ACOS
    df['ACOS'] = np.where(
        df['Sales'] > 0,
        (df['Spend'] / df['Sales']) * 100,
        0
    )

    return df


def ensure_numeric_columns(
    df: pd.DataFrame,
    columns: list = None,
    default_value: float = 0.0,
    inplace: bool = True
) -> pd.DataFrame:
    """
    Ensure specified columns exist and are numeric.

    Args:
        df: DataFrame to process
        columns: List of column names. If None, uses standard PPC columns.
        default_value: Value to use for missing columns
        inplace: If False, returns a copy. If True, modifies df directly.

    Returns:
        DataFrame with numeric columns ensured
    """
    if not inplace:
        df = df.copy()

    if columns is None:
        columns = ['Impressions', 'Clicks', 'Spend', 'Sales', 'Orders']

    # Import safe_numeric from core.data_loader
    from core.data_loader import safe_numeric

    for col in columns:
        if col not in df.columns:
            df[col] = default_value
        else:
            df[col] = safe_numeric(df[col])

    return df
