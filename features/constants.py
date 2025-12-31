"""
Shared Constants for PPC Features

This module contains constants used across multiple feature modules
to ensure consistency and reduce duplication.
"""

# ==========================================
# AUTO TARGETING TYPES
# ==========================================

# Canonical auto targeting type keywords
AUTO_TARGETING_TYPES = {
    'close-match',
    'loose-match',
    'substitutes',
    'complements'
}

# Pattern for detecting auto/PT targeting in regex
AUTO_TARGETING_PATTERN = r'close-match|loose-match|substitutes|complements|category=|asin|b0'


# ==========================================
# STANDARD COLUMN MAPPINGS
# ==========================================

# Map common column name variations to standard names
STANDARD_COLUMN_MAPPING = {
    "Campaign": "Campaign Name",
    "AdGroup": "Ad Group Name",
    "Term": "Customer Search Term",
    "Match": "Match Type"
}

# Standard numeric columns required for PPC analysis
REQUIRED_NUMERIC_COLUMNS = [
    'Impressions',
    'Clicks',
    'Spend',
    'Sales',
    'Orders'
]

# Date column name variations (in order of preference)
DATE_COLUMN_CANDIDATES = [
    'Date',
    'Start Date',
    'date',
    'Report Date',
    'start_date'
]


# ==========================================
# TARGETING NORMALIZATION
# ==========================================

def normalize_auto_targeting(val) -> str:
    """
    Normalize auto targeting types to canonical lowercase-hyphen form.

    Examples:
        "Close-Match" -> "close-match"
        "Close Match" -> "close-match"
        "LOOSE_MATCH" -> "loose-match"

    Args:
        val: Targeting value to normalize

    Returns:
        Normalized targeting value
    """
    val_norm = str(val).strip().lower().replace(" ", "-").replace("_", "-")
    if val_norm in AUTO_TARGETING_TYPES:
        return val_norm
    return val  # Keep original for non-auto types


def classify_match_type(row) -> str:
    """
    Classify match type based on targeting patterns.

    Used by performance_snapshot.py for refining match types.

    Args:
        row: DataFrame row with 'Refined Match Type' and optional 'Targeting'

    Returns:
        Classified match type string
    """
    mt = str(row.get('Refined Match Type', '')).upper()
    targeting = str(row.get('Targeting', '')).lower()

    # 1. Trust Strong Types
    if mt in ['EXACT', 'BROAD', 'PHRASE', 'PT', 'CATEGORY', 'AUTO']:
        return mt

    # 2. Heuristics on Targeting Text
    if 'asin=' in targeting or (len(targeting) == 10 and targeting.startswith('b0')):
        return 'PT'
    if 'category=' in targeting:
        return 'CATEGORY'

    # 3. Auto targeting keywords
    auto_keywords = ['close-match', 'loose-match', 'substitutes', 'complements', '*']
    if any(k in targeting for k in auto_keywords):
        return 'AUTO'

    # 4. Fallback
    return 'OTHER' if mt in ['-', 'NAN', 'NONE', ''] else mt
