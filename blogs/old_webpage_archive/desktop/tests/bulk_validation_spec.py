"""
Amazon Advertising Bulk Upload Validation Specification
========================================================
Based on official Amazon Bulksheets 2.0 documentation
For use in S2C LaunchPad / SADDLE AdPulse

Author: Generated for Aslam
Date: December 2025
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Callable, Any
from enum import Enum
import re
from datetime import datetime

# =============================================================================
# ENUMS - All Valid Values from Amazon Documentation
# =============================================================================

class RecordType(Enum):
    CAMPAIGN = "Campaign"
    CAMPAIGN_BY_PLACEMENT = "Campaign By Placement"
    AD_GROUP = "Ad Group"
    AD = "Ad"
    KEYWORD = "Keyword"
    PRODUCT_TARGETING = "Product Targeting"


class CampaignStatus(Enum):
    ENABLED = "enabled"
    PAUSED = "paused"
    ARCHIVED = "archived"
    DRAFT = "draft"  # Sponsored Brands only


class AdGroupStatus(Enum):
    ENABLED = "enabled"
    PAUSED = "paused"
    ARCHIVED = "archived"


class KeywordStatus(Enum):
    ENABLED = "enabled"
    PAUSED = "paused"
    ARCHIVED = "archived"
    DELETED = "deleted"  # Campaign negative keywords only


class TargetingType(Enum):
    AUTO = "Auto"
    MANUAL = "Manual"


class BiddingStrategy(Enum):
    FIXED = "Fixed Bids"
    DYNAMIC_DOWN = "Dynamic bidding (down only)"
    DYNAMIC_UP_DOWN = "Dynamic bidding (up and down)"


class KeywordMatchType(Enum):
    BROAD = "broad"
    PHRASE = "phrase"
    EXACT = "exact"
    NEGATIVE_PHRASE = "negative phrase"
    NEGATIVE_EXACT = "negative exact"
    CAMPAIGN_NEGATIVE_PHRASE = "campaign negative phrase"
    CAMPAIGN_NEGATIVE_EXACT = "campaign negative exact"


class ProductTargetMatchType(Enum):
    TARGETING_EXPRESSION = "Targeting Expression"
    NEGATIVE_TARGETING = "negativetargetingexpression"
    PREDEFINED = "Targeting Expression Predefined"


class PlacementType(Enum):
    ALL = "All"
    TOP_OF_SEARCH = "Top of search (page 1)"
    PRODUCT_PAGES = "Product Pages"
    REST_OF_SEARCH = "Rest of search"


class AdFormat(Enum):
    VIDEO = "Video"
    PRODUCT_COLLECTION = "Product Collection"


class BudgetType(Enum):
    DAILY = "daily"
    LIFETIME = "lifetime"


class CampaignTactic(Enum):
    VIEWS_CPC = "Views (CPC)"


# =============================================================================
# CURRENCY-SPECIFIC LIMITS (from Amazon documentation)
# =============================================================================

CURRENCY_LIMITS = {
    # Sponsored Products
    "SP": {
        "USD": {"min_budget": 1, "max_budget": 1_000_000, "min_bid": 0.02, "max_bid": 1000},
        "GBP": {"min_budget": 1, "max_budget": 1_000_000, "min_bid": 0.02, "max_bid": 1000},
        "EUR": {"min_budget": 1, "max_budget": 1_000_000, "min_bid": 0.02, "max_bid": 1000},
        "CAD": {"min_budget": 1, "max_budget": 1_000_000, "min_bid": 0.02, "max_bid": 1000},
        "AUD": {"min_budget": 1.4, "max_budget": 1_500_000, "min_bid": 0.1, "max_bid": 1410},
        "JPY": {"min_budget": 100, "max_budget": 21_000_000, "min_bid": 2, "max_bid": 100_000},
        "INR": {"min_budget": 500, "max_budget": 21_000_000, "min_bid": 1, "max_bid": 5000},
        "AED": {"min_budget": 4, "max_budget": 3_700_000, "min_bid": 0.24, "max_bid": 3670},
        "MXN": {"min_budget": 1, "max_budget": 21_000_000, "min_bid": 0.1, "max_bid": 20_000},
        "CNY": {"min_budget": 1, "max_budget": 21_000_000, "min_bid": 0.1, "max_bid": 1000},
    },
    # Sponsored Brands
    "SB": {
        "USD": {"min_daily": 1, "max_daily": 1_000_000, "min_lifetime": 100, "max_lifetime": 20_000_000, "min_bid": 0.1, "max_bid": 49},
        "GBP": {"min_daily": 1, "max_daily": 1_000_000, "min_lifetime": 100, "max_lifetime": 20_000_000, "min_bid": 0.1, "max_bid": 31},
        "EUR": {"min_daily": 1, "max_daily": 1_000_000, "min_lifetime": 100, "max_lifetime": 20_000_000, "min_bid": 0.1, "max_bid": 39},
        "CAD": {"min_daily": 1, "max_daily": 1_000_000, "min_lifetime": 100, "max_lifetime": 20_000_000, "min_bid": 0.1, "max_bid": 49},
        "AUD": {"min_daily": 1.4, "max_daily": 1_500_000, "min_lifetime": 141, "max_lifetime": 28_000_000, "min_bid": 0.1, "max_bid": 70},
        "JPY": {"min_daily": 100, "max_daily": 21_000_000, "min_lifetime": 10_000, "max_lifetime": 2_000_000_000, "min_bid": 10, "max_bid": 7760},
        "INR": {"min_daily": 100, "max_daily": 21_000_000, "min_lifetime": 5_000, "max_lifetime": 200_000_000, "min_bid": 2, "max_bid": 500},
        "AED": {"min_daily": 4, "max_daily": 3_700_000, "min_lifetime": 367, "max_lifetime": 74_000_000, "min_bid": 0.4, "max_bid": 184},
        "MXN": {"min_daily": 1, "max_daily": 21_000_000, "min_lifetime": 100, "max_lifetime": 200_000_000, "min_bid": 0.1, "max_bid": 20_000},
        "CNY": {"min_daily": 1, "max_daily": 21_000_000, "min_lifetime": 100, "max_lifetime": 200_000_000, "min_bid": 1, "max_bid": 50},
    },
    # Sponsored Display
    "SD": {
        "USD": {"min_budget": 1, "max_budget": 1_000_000, "min_bid": 0.02, "max_bid": 1000},
    }
}

# Bid Multiplier limits (all currencies)
BID_MULTIPLIER_MIN = -99.00
BID_MULTIPLIER_MAX = 99.99
PLACEMENT_BID_MIN = 0
PLACEMENT_BID_MAX = 900


# =============================================================================
# VALIDATION FLAGS / ERROR CODES
# =============================================================================

class ValidationFlag(Enum):
    # File-level errors
    MISSING_HEADER_ROW = "F001"
    INVALID_FILE_FORMAT = "F002"
    ENCODING_ERROR = "F003"
    MAX_ROWS_EXCEEDED = "F004"
    EMPTY_FILE = "F005"
    
    # Column-level errors
    MISSING_REQUIRED_COLUMN = "C001"
    UNKNOWN_COLUMN = "C002"
    
    # Row-level errors
    EMPTY_ROW = "R001"
    DUPLICATE_ROW = "R002"
    
    # Record ID errors
    RECORD_ID_MODIFIED = "ID001"
    RECORD_ID_REQUIRED_FOR_UPDATE = "ID002"
    RECORD_ID_NOT_FOUND = "ID003"
    
    # Record Type errors
    INVALID_RECORD_TYPE = "RT001"
    MISSING_RECORD_TYPE = "RT002"
    
    # Campaign errors
    CAMPAIGN_NAME_MISSING = "CAM001"
    CAMPAIGN_NAME_TOO_LONG = "CAM002"
    CAMPAIGN_NAME_INVALID_CHARS = "CAM003"
    CAMPAIGN_NOT_FOUND = "CAM004"
    CAMPAIGN_ID_REQUIRED = "CAM005"
    CAMPAIGN_DUPLICATE_NAME = "CAM006"
    
    # Budget errors
    BUDGET_MISSING = "BUD001"
    BUDGET_TOO_LOW = "BUD002"
    BUDGET_TOO_HIGH = "BUD003"
    BUDGET_INVALID_FORMAT = "BUD004"
    BUDGET_TYPE_MISSING = "BUD005"
    BUDGET_TYPE_INVALID = "BUD006"
    LIFETIME_BUDGET_REQUIRES_DATES = "BUD007"
    
    # Date errors
    START_DATE_MISSING = "DAT001"
    START_DATE_INVALID_FORMAT = "DAT002"
    START_DATE_IN_PAST = "DAT003"
    END_DATE_INVALID_FORMAT = "DAT004"
    END_DATE_BEFORE_START = "DAT005"
    
    # Targeting errors
    TARGETING_TYPE_MISSING = "TAR001"
    TARGETING_TYPE_INVALID = "TAR002"
    
    # Status errors
    STATUS_MISSING = "STA001"
    STATUS_INVALID = "STA002"
    ARCHIVED_CANNOT_UPDATE = "STA003"
    
    # Bidding Strategy errors
    BIDDING_STRATEGY_MISSING = "BID001"
    BIDDING_STRATEGY_INVALID = "BID002"
    
    # Ad Group errors
    AD_GROUP_NAME_MISSING = "ADG001"
    AD_GROUP_NAME_TOO_LONG = "ADG002"
    AD_GROUP_NOT_FOUND = "ADG003"
    AD_GROUP_MISSING_FOR_KEYWORD = "ADG004"
    
    # Bid errors
    BID_MISSING = "MAX001"
    BID_TOO_LOW = "MAX002"
    BID_TOO_HIGH = "MAX003"
    BID_INVALID_FORMAT = "MAX004"
    BID_CHANGE_EXTREME = "MAX005"  # Warning: >300% change
    
    # Keyword errors
    KEYWORD_MISSING = "KEY001"
    KEYWORD_TOO_LONG = "KEY002"
    KEYWORD_INVALID_CHARS = "KEY003"
    KEYWORD_TOO_MANY_WORDS = "KEY004"
    KEYWORD_DUPLICATE = "KEY005"
    KEYWORD_NEGATIVE_CONFLICT = "KEY006"  # Warning
    MATCH_TYPE_MISSING = "KEY007"
    MATCH_TYPE_INVALID = "KEY008"
    KEYWORD_IN_AUTO_CAMPAIGN = "KEY009"
    
    # Product Targeting errors
    ASIN_INVALID_FORMAT = "PT001"
    TARGETING_EXPRESSION_INVALID = "PT002"
    PRODUCT_TARGETING_ID_INVALID = "PT003"
    
    # Ad errors
    SKU_MISSING = "AD001"
    SKU_INVALID = "AD002"
    ASIN_MISSING = "AD003"
    ASIN_NOT_IN_CATALOG = "AD004"
    
    # Placement errors
    PLACEMENT_TYPE_INVALID = "PLC001"
    PLACEMENT_BID_INVALID = "PLC002"
    PLACEMENT_BID_TOO_HIGH = "PLC003"
    
    # Sponsored Brands specific
    LANDING_PAGE_MISSING = "SB001"
    LANDING_PAGE_URL_INVALID = "SB002"
    LANDING_PAGE_ASINS_INVALID = "SB003"
    BRAND_NAME_MISSING = "SB004"
    BRAND_NAME_TOO_LONG = "SB005"
    BRAND_LOGO_MISSING = "SB006"
    HEADLINE_MISSING = "SB007"
    HEADLINE_TOO_LONG = "SB008"
    CREATIVE_ASINS_MISSING = "SB009"
    CREATIVE_ASINS_TOO_MANY = "SB010"
    AUTOMATED_BIDDING_INVALID = "SB011"
    BID_MULTIPLIER_INVALID = "SB012"
    BID_MULTIPLIER_OUT_OF_RANGE = "SB013"
    
    # Portfolio errors
    PORTFOLIO_NOT_FOUND = "POR001"
    PORTFOLIO_BUDGET_INVALID = "POR002"
    
    # Hierarchy errors
    PARENT_ENTITY_MISSING = "HIE001"
    PARENT_ENTITY_FAILED = "HIE002"
    AD_GROUP_WITHOUT_CAMPAIGN = "HIE003"
    KEYWORD_WITHOUT_AD_GROUP = "HIE004"
    AD_WITHOUT_AD_GROUP = "HIE005"
    
    # Warnings (non-blocking)
    TARGET_PAUSED = "W001"
    CAMPAIGN_PAUSED = "W002"
    LOW_BID_WARNING = "W003"
    HIGH_BUDGET_WARNING = "W004"
    NEGATIVE_MATCHES_ACTIVE = "W005"


# =============================================================================
# VALIDATION RULES DATA STRUCTURES
# =============================================================================

@dataclass
class ColumnRule:
    """Definition of a single column's validation rules"""
    name: str
    required: bool = False
    required_for_create: bool = False
    required_for_update: bool = False
    data_type: str = "string"  # string, decimal, integer, date, enum
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[str]] = None
    pattern: Optional[str] = None  # Regex pattern
    mutable: bool = True
    depends_on: Optional[Dict[str, Any]] = None  # Conditional requirements
    error_code: str = ""
    description: str = ""


@dataclass
class RecordTypeRules:
    """Validation rules for a specific record type"""
    record_type: str
    columns: List[ColumnRule]
    parent_type: Optional[str] = None
    max_per_parent: Optional[int] = None


# =============================================================================
# SPONSORED PRODUCTS VALIDATION RULES
# =============================================================================

SP_CAMPAIGN_RULES = RecordTypeRules(
    record_type="Campaign",
    columns=[
        ColumnRule(
            name="Record ID",
            required_for_update=True,
            mutable=False,
            error_code="ID002",
            description="Internal Amazon ID. Leave blank for create, required for update."
        ),
        ColumnRule(
            name="Record Type",
            required=True,
            allowed_values=["Campaign"],
            mutable=False,
            error_code="RT001"
        ),
        ColumnRule(
            name="Campaign ID",
            required_for_update=True,
            mutable=False,
            error_code="CAM005"
        ),
        ColumnRule(
            name="Campaign",
            required=True,
            max_length=128,
            pattern=r"^[a-zA-Z0-9\s\-_\.\,\!\?\'\"\&\(\)]+$",
            mutable=True,
            error_code="CAM001",
            description="Unique case-sensitive campaign name"
        ),
        ColumnRule(
            name="Campaign Daily Budget",
            required=True,
            data_type="decimal",
            min_value=1,  # Currency-dependent, will be validated dynamically
            max_value=1_000_000,
            error_code="BUD001"
        ),
        ColumnRule(
            name="Portfolio ID",
            required=False,
            mutable=True,
            error_code="POR001"
        ),
        ColumnRule(
            name="Campaign Start Date",
            required=True,
            data_type="date",
            pattern=r"^\d{1,2}/\d{1,2}/\d{4}$",  # mm/dd/yyyy
            error_code="DAT001",
            description="Format: mm/dd/yyyy. Cannot be in past."
        ),
        ColumnRule(
            name="Campaign End Date",
            required=False,
            data_type="date",
            pattern=r"^\d{1,2}/\d{1,2}/\d{4}$",
            mutable=True,
            error_code="DAT004",
            description="Format: mm/dd/yyyy. Must be after start date."
        ),
        ColumnRule(
            name="Campaign Targeting Type",
            required=True,
            allowed_values=["Auto", "Manual"],
            mutable=False,
            error_code="TAR001"
        ),
        ColumnRule(
            name="Campaign Status",
            required=True,
            allowed_values=["enabled", "paused", "archived", "Enabled", "Paused", "Archived"],
            mutable=True,
            error_code="STA001"
        ),
        ColumnRule(
            name="Bidding Strategy",
            required=True,
            allowed_values=[
                "Fixed Bids",
                "Dynamic bidding (up and down)",
                "Dynamic bidding (down only)"
            ],
            mutable=False,
            error_code="BID001"
        ),
    ]
)

SP_AD_GROUP_RULES = RecordTypeRules(
    record_type="Ad Group",
    parent_type="Campaign",
    columns=[
        ColumnRule(
            name="Record ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Record Type",
            required=True,
            allowed_values=["Ad Group"],
            mutable=False
        ),
        ColumnRule(
            name="Campaign ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Campaign",
            required=True,
            mutable=False,
            description="Must match parent campaign name exactly (case-sensitive)"
        ),
        ColumnRule(
            name="Ad Group",
            required=True,
            max_length=255,
            mutable=False,
            error_code="ADG001"
        ),
        ColumnRule(
            name="Max Bid",
            required=True,
            data_type="decimal",
            min_value=0.02,
            max_value=1000,
            mutable=True,
            error_code="MAX001"
        ),
        ColumnRule(
            name="Ad Group Status",
            required=True,
            allowed_values=["enabled", "paused", "archived", "Enabled", "Paused", "Archived"],
            mutable=True
        ),
    ]
)

SP_KEYWORD_RULES = RecordTypeRules(
    record_type="Keyword",
    parent_type="Ad Group",
    max_per_parent=1000,
    columns=[
        ColumnRule(
            name="Record ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Record Type",
            required=True,
            allowed_values=["Keyword"],
            mutable=False
        ),
        ColumnRule(
            name="Campaign ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Campaign",
            required=True,
            mutable=False
        ),
        ColumnRule(
            name="Ad Group",
            required=False,  # Not required for campaign-level negatives
            mutable=False,
            depends_on={"Match Type": ["broad", "phrase", "exact", "negative phrase", "negative exact"]},
            description="Required except for campaign-level negative keywords"
        ),
        ColumnRule(
            name="Max Bid",
            required=False,  # Not required for negatives
            data_type="decimal",
            min_value=0.02,
            max_value=1000,
            mutable=True,
            depends_on={"Match Type": ["broad", "phrase", "exact"]},
            description="Required for biddable keywords, leave blank for negatives"
        ),
        ColumnRule(
            name="Keyword or Product Targeting",
            required=True,
            max_length=80,
            pattern=r"^[a-zA-Z0-9\s\-]+$",  # No special chars except hyphens
            mutable=False,
            error_code="KEY001",
            description="1-80 chars, no special characters except hyphens"
        ),
        ColumnRule(
            name="Match Type",
            required=True,
            allowed_values=[
                "broad", "phrase", "exact",
                "negative phrase", "negative exact",
                "campaign negative phrase", "campaign negative exact"
            ],
            mutable=False,
            error_code="KEY007"
        ),
        ColumnRule(
            name="Status",
            required=True,
            allowed_values=["enabled", "paused", "archived", "deleted"],
            mutable=True,
            description="Use 'deleted' only for campaign negative keywords"
        ),
    ]
)

SP_PRODUCT_TARGETING_RULES = RecordTypeRules(
    record_type="Product Targeting",
    parent_type="Ad Group",
    columns=[
        ColumnRule(
            name="Record ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Record Type",
            required=True,
            allowed_values=["Product Targeting"],
            mutable=False
        ),
        ColumnRule(
            name="Campaign ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Campaign",
            required=True,
            mutable=False
        ),
        ColumnRule(
            name="Ad Group",
            required=True,
            mutable=False
        ),
        ColumnRule(
            name="Max Bid",
            required=True,
            data_type="decimal",
            min_value=0.02,
            max_value=1000,
            mutable=True
        ),
        ColumnRule(
            name="Keyword or Product Targeting",
            required=True,
            mutable=False,
            description='Format: asin="B11AAA777K" or category="CategoryName" or brand="BrandName"'
        ),
        ColumnRule(
            name="Product Targeting ID",
            required=False,
            allowed_values=["close-match", "substitutes", "loose-match", "complements"],
            mutable=False,
            description="For auto targeting: close-match, substitutes, loose-match, complements"
        ),
        ColumnRule(
            name="Match Type",
            required=True,
            allowed_values=["Targeting Expression", "negativetargetingexpression", "Targeting Expression Predefined"],
            mutable=False
        ),
        ColumnRule(
            name="Status",
            required=True,
            allowed_values=["enabled", "paused", "archived"],
            mutable=True
        ),
    ]
)

SP_AD_RULES = RecordTypeRules(
    record_type="Ad",
    parent_type="Ad Group",
    columns=[
        ColumnRule(
            name="Record ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Record Type",
            required=True,
            allowed_values=["Ad"],
            mutable=False
        ),
        ColumnRule(
            name="Campaign ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Campaign",
            required=True,
            mutable=False
        ),
        ColumnRule(
            name="Ad Group",
            required=True,
            mutable=False
        ),
        ColumnRule(
            name="SKU",
            required=True,  # For sellers
            mutable=False,
            error_code="AD001",
            description="Sellers use SKU, vendors use ASIN"
        ),
        ColumnRule(
            name="Status",
            required=True,
            allowed_values=["enabled", "paused", "archived"],
            mutable=True
        ),
    ]
)

SP_PLACEMENT_RULES = RecordTypeRules(
    record_type="Campaign By Placement",
    parent_type="Campaign",
    columns=[
        ColumnRule(
            name="Record ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Record Type",
            required=True,
            allowed_values=["Campaign By Placement"],
            mutable=False
        ),
        ColumnRule(
            name="Campaign ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Campaign",
            required=True,
            mutable=False
        ),
        ColumnRule(
            name="Placement Type",
            required=True,
            allowed_values=["Top of search (page 1)", "Product Pages"],
            mutable=False,
            error_code="PLC001",
            description="Note: 'Rest of search' not supported for bid adjustments"
        ),
        ColumnRule(
            name="Increase Bids By Placement",
            required=True,
            data_type="integer",
            min_value=0,
            max_value=900,
            mutable=True,
            error_code="PLC002",
            description="0-900%, decimals not allowed, % sign optional"
        ),
    ]
)


# =============================================================================
# NEGATIVE KEYWORD SPECIFIC RULES - TWO BUSINESS FLOWS
# =============================================================================

# -----------------------------------------------------------------------------
# ISOLATION / HARVEST NEGATIVES (Campaign-Level)
# Purpose: After promoting a keyword to its own campaign, add as campaign 
#          negative to prevent cannibalization from parent campaign
# -----------------------------------------------------------------------------
ISOLATION_NEGATIVE_RULES = RecordTypeRules(
    record_type="Keyword",
    parent_type="Campaign",  # Campaign-level, no ad group
    columns=[
        ColumnRule(
            name="Record ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Record Type",
            required=True,
            allowed_values=["Keyword"],
            mutable=False
        ),
        ColumnRule(
            name="Campaign ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Campaign",
            required=True,
            mutable=False,
            description="The campaign to ADD the negative to (usually the parent/source campaign)"
        ),
        ColumnRule(
            name="Ad Group",
            required=False,
            mutable=False,
            allowed_values=["", None],  # MUST BE BLANK
            error_code="ISO001",
            description="MUST BE BLANK for isolation/harvest negatives (campaign-level)"
        ),
        ColumnRule(
            name="Max Bid",
            required=False,
            mutable=False,
            allowed_values=["", None],  # MUST BE BLANK - negatives don't have bids
            description="Leave blank - negative keywords are not biddable"
        ),
        ColumnRule(
            name="Keyword or Product Targeting",
            required=True,
            max_length=80,
            pattern=r"^[a-zA-Z0-9\s\-]+$",
            mutable=False,
            description="The harvested keyword to block in source campaign"
        ),
        ColumnRule(
            name="Match Type",
            required=True,
            allowed_values=["campaign negative exact", "campaign negative phrase"],
            mutable=False,
            error_code="ISO002",
            description="Use 'campaign negative exact' (preferred) or 'campaign negative phrase'"
        ),
        ColumnRule(
            name="Status",
            required=True,
            allowed_values=["enabled", "deleted"],  # NOT paused/archived!
            mutable=True,
            error_code="ISO003",
            description="Only 'enabled' or 'deleted' for campaign-level negatives"
        ),
    ]
)

# -----------------------------------------------------------------------------
# BLEEDER NEGATIVES (Ad Group-Level)
# Purpose: Block poor-performing search terms within a specific ad group
# -----------------------------------------------------------------------------
BLEEDER_NEGATIVE_RULES = RecordTypeRules(
    record_type="Keyword",
    parent_type="Ad Group",  # Ad group-level
    columns=[
        ColumnRule(
            name="Record ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Record Type",
            required=True,
            allowed_values=["Keyword"],
            mutable=False
        ),
        ColumnRule(
            name="Campaign ID",
            required_for_update=True,
            mutable=False
        ),
        ColumnRule(
            name="Campaign",
            required=True,
            mutable=False
        ),
        ColumnRule(
            name="Ad Group",
            required=True,  # REQUIRED for bleeders
            mutable=False,
            error_code="BLD001",
            description="REQUIRED - must specify which ad group to add negative to"
        ),
        ColumnRule(
            name="Max Bid",
            required=False,
            mutable=False,
            allowed_values=["", None],
            description="Leave blank - negative keywords are not biddable"
        ),
        ColumnRule(
            name="Keyword or Product Targeting",
            required=True,
            max_length=80,
            pattern=r"^[a-zA-Z0-9\s\-]+$",
            mutable=False,
            description="The bleeding search term to block"
        ),
        ColumnRule(
            name="Match Type",
            required=True,
            allowed_values=["negative exact", "negative phrase"],
            mutable=False,
            error_code="BLD002",
            description="Use 'negative exact' or 'negative phrase' (NOT campaign negative)"
        ),
        ColumnRule(
            name="Status",
            required=True,
            allowed_values=["enabled", "paused", "archived"],  # Full status options
            mutable=True,
            description="Can use enabled/paused/archived for ad group negatives"
        ),
    ]
)

# Legacy alias for backward compatibility
NEGATIVE_KEYWORD_RULES = ISOLATION_NEGATIVE_RULES


# =============================================================================
# BID UPDATE RULES
# =============================================================================

BID_UPDATE_KEYWORD_RULES = RecordTypeRules(
    record_type="Keyword",
    parent_type="Ad Group",
    columns=[
        ColumnRule(
            name="Record ID",
            required=True,  # REQUIRED - this is an update
            mutable=False,
            error_code="BID_UPD001",
            description="Required for bid updates - identifies existing keyword"
        ),
        ColumnRule(
            name="Record Type",
            required=True,
            allowed_values=["Keyword"],
            mutable=False
        ),
        ColumnRule(
            name="Campaign ID",
            required=True,  # REQUIRED for update
            mutable=False
        ),
        ColumnRule(
            name="Campaign",
            required=True,
            mutable=False
        ),
        ColumnRule(
            name="Ad Group",
            required=True,
            mutable=False
        ),
        ColumnRule(
            name="Max Bid",
            required=True,  # This is what we're updating
            data_type="decimal",
            min_value=0.02,  # Currency-dependent
            max_value=1000,  # Currency-dependent
            mutable=True,
            error_code="BID_UPD002",
            description="New bid value - will be validated against currency limits"
        ),
        ColumnRule(
            name="Keyword or Product Targeting",
            required=False,  # Not changing this
            mutable=False
        ),
        ColumnRule(
            name="Match Type",
            required=False,  # Not changing this
            mutable=False
        ),
        ColumnRule(
            name="Status",
            required=False,  # Optional - can update status along with bid
            allowed_values=["enabled", "paused", "archived"],
            mutable=True
        ),
    ]
)

BID_UPDATE_AD_GROUP_RULES = RecordTypeRules(
    record_type="Ad Group",
    parent_type="Campaign",
    columns=[
        ColumnRule(
            name="Record ID",
            required=True,  # REQUIRED - this is an update
            mutable=False,
            error_code="BID_UPD001"
        ),
        ColumnRule(
            name="Record Type",
            required=True,
            allowed_values=["Ad Group"],
            mutable=False
        ),
        ColumnRule(
            name="Campaign ID",
            required=True,
            mutable=False
        ),
        ColumnRule(
            name="Campaign",
            required=True,
            mutable=False
        ),
        ColumnRule(
            name="Ad Group",
            required=True,
            mutable=False
        ),
        ColumnRule(
            name="Max Bid",
            required=True,
            data_type="decimal",
            min_value=0.02,
            max_value=1000,
            mutable=True,
            error_code="BID_UPD002"
        ),
        ColumnRule(
            name="Ad Group Status",
            required=False,
            allowed_values=["enabled", "paused", "archived"],
            mutable=True
        ),
    ]
)


# =============================================================================
# AUTO-CAMPAIGN RESTRICTION RULES
# =============================================================================

class AutoCampaignRestrictions:
    """
    Rules for what can/cannot be added to Auto-targeting campaigns
    """
    
    # CANNOT add these to Auto campaigns
    FORBIDDEN_MATCH_TYPES = [
        "broad",
        "phrase", 
        "exact",
    ]
    
    # CAN add these to Auto campaigns
    ALLOWED_MATCH_TYPES = [
        "campaign negative exact",
        "campaign negative phrase",
        "negative exact",
        "negative phrase",
    ]
    
    @staticmethod
    def validate_keyword_for_auto_campaign(match_type: str) -> tuple[bool, str]:
        """
        Check if a keyword with given match type can be added to an Auto campaign
        Returns: (is_valid, error_message)
        """
        match_type_lower = match_type.lower().strip()
        
        if match_type_lower in [m.lower() for m in AutoCampaignRestrictions.FORBIDDEN_MATCH_TYPES]:
            return False, f"Cannot add '{match_type}' keywords to Auto-targeting campaigns. Only negative keywords allowed."
        
        if match_type_lower in [m.lower() for m in AutoCampaignRestrictions.ALLOWED_MATCH_TYPES]:
            return True, ""
        
        return False, f"Unknown match type: {match_type}"


# =============================================================================
# BUSINESS LOGIC VALIDATION FLAGS
# =============================================================================

class BusinessValidationFlag(Enum):
    """Additional validation flags for business-specific rules"""
    
    # Isolation/Harvest negatives
    ISO001 = "ISO001"  # Ad Group must be blank for campaign-level negative
    ISO002 = "ISO002"  # Wrong match type for isolation negative
    ISO003 = "ISO003"  # Invalid status for campaign-level negative
    
    # Bleeder negatives
    BLD001 = "BLD001"  # Ad Group required for ad group-level negative
    BLD002 = "BLD002"  # Wrong match type for bleeder negative
    
    # Bid updates
    BID_UPD001 = "BID_UPD001"  # Record ID required for bid update
    BID_UPD002 = "BID_UPD002"  # Invalid bid value
    BID_UPD003 = "BID_UPD003"  # Bid change exceeds 300% (warning)
    
    # Auto campaign restrictions
    AUTO001 = "AUTO001"  # Cannot add positive keywords to Auto campaign
    AUTO002 = "AUTO002"  # Invalid operation for Auto campaign


BUSINESS_ERROR_MESSAGES = {
    # Isolation negatives
    "ISO001": "Ad Group must be BLANK for isolation/harvest negatives (campaign-level). Leave the Ad Group field empty.",
    "ISO002": "Isolation negatives must use 'campaign negative exact' or 'campaign negative phrase'. You used '{match_type}'.",
    "ISO003": "Campaign-level negatives only support 'enabled' or 'deleted' status. Cannot use 'paused' or 'archived'.",
    
    # Bleeder negatives
    "BLD001": "Ad Group is REQUIRED for bleeder negatives (ad group-level). Specify which ad group to add the negative to.",
    "BLD002": "Bleeder negatives must use 'negative exact' or 'negative phrase' (not 'campaign negative'). You used '{match_type}'.",
    
    # Bid updates
    "BID_UPD001": "Record ID is required for bid updates. Cannot update bid without identifying the existing keyword/ad group.",
    "BID_UPD002": "Bid value {bid} is invalid. Must be between {min_bid} and {max_bid} for {currency}.",
    "BID_UPD003": "Warning: Bid change of {change_pct}% exceeds 300%. Current: {current_bid}, New: {new_bid}. Proceed with caution.",
    
    # Auto campaign
    "AUTO001": "Cannot add '{match_type}' keywords to Auto-targeting campaign '{campaign}'. Only negative keywords are allowed in Auto campaigns.",
    "AUTO002": "This operation is not supported for Auto-targeting campaigns.",
}


# =============================================================================
# HELPER VALIDATION FUNCTIONS
# =============================================================================

def validate_asin(asin: str) -> bool:
    """Validate ASIN format (10 alphanumeric, starts with B0)"""
    pattern = r'^B0[A-Z0-9]{8}$'
    return bool(re.match(pattern, asin.upper()))


def validate_date_format(date_str: str) -> bool:
    """Validate date is in mm/dd/yyyy format"""
    try:
        datetime.strptime(date_str, "%m/%d/%Y")
        return True
    except ValueError:
        return False


def validate_date_not_past(date_str: str) -> bool:
    """Validate date is not in the past"""
    try:
        date = datetime.strptime(date_str, "%m/%d/%Y")
        return date.date() >= datetime.now().date()
    except ValueError:
        return False


def validate_keyword_words(keyword: str, max_words: int = 10) -> bool:
    """Validate keyword doesn't exceed max word count"""
    words = keyword.strip().split()
    return len(words) <= max_words


def validate_campaign_name_chars(name: str) -> bool:
    """Validate campaign name contains only allowed characters"""
    # From Amazon docs: https://advertising.amazon.com/API/docs/v2/guides/supported_features#Entity-name-character-constraints
    pattern = r'^[a-zA-Z0-9\s\-_\.\,\!\?\'\"\&\(\)]+$'
    return bool(re.match(pattern, name))


def validate_targeting_expression(expr: str) -> bool:
    """Validate product targeting expression format"""
    patterns = [
        r'^asin="[A-Z0-9]{10}"$',
        r'^category=".+"$',
        r'^brand=".+"$'
    ]
    return any(re.match(p, expr, re.IGNORECASE) for p in patterns)


def validate_bid_multiplier(value: str) -> tuple[bool, Optional[float]]:
    """Validate bid multiplier format and range"""
    # Must include +/- and %
    pattern = r'^[+-]\d+(\.\d{1,2})?%$'
    if not re.match(pattern, value):
        return False, None
    
    numeric = float(value.replace('%', '').replace('+', ''))
    if BID_MULTIPLIER_MIN <= numeric <= BID_MULTIPLIER_MAX:
        return True, numeric
    return False, None


# =============================================================================
# COMPLETE VALIDATION SCHEMA
# =============================================================================

VALIDATION_SCHEMA = {
    "Sponsored Products": {
        "Campaign": SP_CAMPAIGN_RULES,
        "Campaign By Placement": SP_PLACEMENT_RULES,
        "Ad Group": SP_AD_GROUP_RULES,
        "Keyword": SP_KEYWORD_RULES,
        "Product Targeting": SP_PRODUCT_TARGETING_RULES,
        "Ad": SP_AD_RULES,
    },
    
    # Business Logic Specific Schemas
    "Negative Keywords": {
        "Isolation": ISOLATION_NEGATIVE_RULES,      # Harvest/campaign-level
        "Bleeder": BLEEDER_NEGATIVE_RULES,          # Ad group-level
        "Campaign Level": ISOLATION_NEGATIVE_RULES, # Alias
        "Ad Group Level": BLEEDER_NEGATIVE_RULES,   # Alias
    },
    
    "Bid Updates": {
        "Keyword": BID_UPDATE_KEYWORD_RULES,
        "Ad Group": BID_UPDATE_AD_GROUP_RULES,
    },
}


# =============================================================================
# BUSINESS LOGIC HELPER FUNCTIONS
# =============================================================================

def determine_negative_type(row: dict) -> str:
    """
    Determine if a negative keyword row is Isolation or Bleeder type
    based on match type and ad group presence
    """
    match_type = row.get("Match Type", "").lower().strip()
    ad_group = row.get("Ad Group", "").strip()
    
    if "campaign negative" in match_type:
        return "Isolation"
    elif match_type in ["negative exact", "negative phrase"]:
        return "Bleeder"
    else:
        return "Unknown"


def validate_negative_keyword_row(row: dict, campaign_targeting_type: str = "Manual") -> List[dict]:
    """
    Validate a negative keyword row based on business rules
    Returns list of error/warning dicts
    """
    errors = []
    match_type = row.get("Match Type", "").lower().strip()
    ad_group = row.get("Ad Group", "").strip()
    status = row.get("Status", "").lower().strip()
    
    # Determine negative type
    is_campaign_level = "campaign negative" in match_type
    
    if is_campaign_level:
        # ISOLATION RULES
        if ad_group:
            errors.append({
                "code": "ISO001",
                "severity": "error",
                "message": BUSINESS_ERROR_MESSAGES["ISO001"],
                "field": "Ad Group"
            })
        
        if match_type not in ["campaign negative exact", "campaign negative phrase"]:
            errors.append({
                "code": "ISO002",
                "severity": "error", 
                "message": BUSINESS_ERROR_MESSAGES["ISO002"].format(match_type=match_type),
                "field": "Match Type"
            })
        
        if status and status not in ["enabled", "deleted"]:
            errors.append({
                "code": "ISO003",
                "severity": "error",
                "message": BUSINESS_ERROR_MESSAGES["ISO003"],
                "field": "Status"
            })
    else:
        # BLEEDER RULES
        if not ad_group:
            errors.append({
                "code": "BLD001",
                "severity": "error",
                "message": BUSINESS_ERROR_MESSAGES["BLD001"],
                "field": "Ad Group"
            })
        
        if match_type not in ["negative exact", "negative phrase"]:
            errors.append({
                "code": "BLD002",
                "severity": "error",
                "message": BUSINESS_ERROR_MESSAGES["BLD002"].format(match_type=match_type),
                "field": "Match Type"
            })
    
    return errors


def validate_bid_update_row(
    row: dict, 
    current_bid: Optional[float] = None,
    currency: str = "USD"
) -> List[dict]:
    """
    Validate a bid update row
    Returns list of error/warning dicts
    """
    errors = []
    
    # Check Record ID present
    record_id = row.get("Record ID", "").strip()
    if not record_id:
        errors.append({
            "code": "BID_UPD001",
            "severity": "error",
            "message": BUSINESS_ERROR_MESSAGES["BID_UPD001"],
            "field": "Record ID"
        })
    
    # Check bid value
    new_bid_str = row.get("Max Bid", "").strip()
    if new_bid_str:
        try:
            new_bid = float(new_bid_str)
            limits = get_currency_limits("Sponsored Products", currency)
            
            if limits:
                min_bid = limits.get("min_bid", 0.02)
                max_bid = limits.get("max_bid", 1000)
                
                if new_bid < min_bid or new_bid > max_bid:
                    errors.append({
                        "code": "BID_UPD002",
                        "severity": "error",
                        "message": BUSINESS_ERROR_MESSAGES["BID_UPD002"].format(
                            bid=new_bid, min_bid=min_bid, max_bid=max_bid, currency=currency
                        ),
                        "field": "Max Bid"
                    })
                
                # Check for extreme bid change (warning only)
                if current_bid and current_bid > 0:
                    change_pct = abs((new_bid - current_bid) / current_bid) * 100
                    if change_pct > 300:
                        errors.append({
                            "code": "BID_UPD003",
                            "severity": "warning",
                            "message": BUSINESS_ERROR_MESSAGES["BID_UPD003"].format(
                                change_pct=round(change_pct, 1),
                                current_bid=current_bid,
                                new_bid=new_bid
                            ),
                            "field": "Max Bid"
                        })
        except ValueError:
            errors.append({
                "code": "MAX004",
                "severity": "error",
                "message": "Invalid bid format. Must be a decimal number.",
                "field": "Max Bid"
            })
    
    return errors


def validate_keyword_for_campaign_type(
    row: dict,
    campaign_targeting_type: str
) -> List[dict]:
    """
    Validate that keyword operations are allowed for the campaign type
    Returns list of error dicts
    """
    errors = []
    
    if campaign_targeting_type.lower() == "auto":
        match_type = row.get("Match Type", "").lower().strip()
        campaign = row.get("Campaign", "")
        
        is_valid, error_msg = AutoCampaignRestrictions.validate_keyword_for_auto_campaign(match_type)
        
        if not is_valid:
            errors.append({
                "code": "AUTO001",
                "severity": "error",
                "message": BUSINESS_ERROR_MESSAGES["AUTO001"].format(
                    match_type=match_type,
                    campaign=campaign
                ),
                "field": "Match Type"
            })
    
    return errors


# =============================================================================
# VALIDATION ERROR MESSAGES
# =============================================================================

ERROR_MESSAGES = {
    # File-level
    "F001": "Missing header row. First row must contain column names.",
    "F002": "Invalid file format. Supported: .xlsx, .xls, .csv",
    "F003": "File encoding error. Use UTF-8 encoding.",
    "F004": "Maximum rows exceeded. Limit: 10,000 rows per upload.",
    "F005": "File is empty or contains no data rows.",
    
    # Column-level
    "C001": "Required column '{column}' is missing.",
    "C002": "Unknown column '{column}' will be ignored.",
    
    # Row-level
    "R001": "Empty row detected at line {line}.",
    "R002": "Duplicate row detected at line {line}.",
    
    # Record ID
    "ID001": "Record ID has been modified. Do not change Record IDs.",
    "ID002": "Record ID is required when updating existing entities.",
    "ID003": "Record ID not found. Entity may have been deleted.",
    
    # Campaign
    "CAM001": "Campaign name is required.",
    "CAM002": "Campaign name exceeds 128 characters.",
    "CAM003": "Campaign name contains invalid characters.",
    "CAM004": "Campaign '{name}' not found in account.",
    "CAM005": "Campaign ID is required for updates.",
    "CAM006": "Duplicate campaign name '{name}' in file.",
    
    # Budget
    "BUD001": "Campaign daily budget is required.",
    "BUD002": "Budget {value} is below minimum ({min}) for {currency}.",
    "BUD003": "Budget {value} exceeds maximum ({max}) for {currency}.",
    "BUD004": "Invalid budget format. Must be a number.",
    "BUD005": "Budget type is required for Sponsored Brands.",
    "BUD006": "Invalid budget type. Use 'daily' or 'lifetime'.",
    "BUD007": "Lifetime budget requires start and end dates.",
    
    # Date
    "DAT001": "Campaign start date is required.",
    "DAT002": "Invalid start date format. Use mm/dd/yyyy.",
    "DAT003": "Start date cannot be in the past.",
    "DAT004": "Invalid end date format. Use mm/dd/yyyy.",
    "DAT005": "End date must be after start date.",
    
    # Targeting
    "TAR001": "Campaign targeting type is required.",
    "TAR002": "Invalid targeting type. Use 'Auto' or 'Manual'.",
    "TAR003": "Product Targeting Expression must be blank for Keyword entities (Exact/Broad/Phrase).",
    
    # Status
    "STA001": "Status is required.",
    "STA002": "Invalid status. Use 'enabled', 'paused', or 'archived'.",
    "STA003": "Cannot update archived entities.",
    
    # Bidding
    "BID001": "Bidding strategy is required.",
    "BID002": "Invalid bidding strategy.",
    
    # Ad Group
    "ADG001": "Ad group name is required.",
    "ADG002": "Ad group name exceeds 255 characters.",
    "ADG003": "Ad group '{name}' not found.",
    "ADG004": "Ad group is required for this keyword type.",
    
    # Bid
    "MAX001": "Max bid is required.",
    "MAX002": "Bid {value} is below minimum ({min}) for {currency}.",
    "MAX003": "Bid {value} exceeds maximum ({max}) for {currency}.",
    "MAX004": "Invalid bid format. Must be a decimal number.",
    "MAX005": "Warning: Bid change exceeds 300% of current value.",
    
    # Keyword
    "KEY001": "Keyword or product targeting is required.",
    "KEY002": "Keyword exceeds 80 characters.",
    "KEY003": "Keyword contains invalid characters.",
    "KEY004": "Keyword exceeds maximum word count ({max} words).",
    "KEY005": "Duplicate keyword '{keyword}' in same ad group.",
    "KEY006": "Warning: Keyword matches existing negative keyword.",
    "KEY007": "Match type is required.",
    "KEY008": "Invalid match type.",
    "KEY009": "Cannot add keywords to auto-targeting campaign.",
    
    # Product Targeting
    "PT001": "Invalid ASIN format. Must be 10 characters starting with B0.",
    "PT002": "Invalid targeting expression format.",
    "PT003": "Invalid Product Targeting ID.",
    
    # Ad
    "AD001": "SKU is required for sellers.",
    "AD002": "Invalid SKU.",
    "AD003": "ASIN is required for vendors.",
    "AD004": "ASIN not found in product catalog.",
    
    # Placement
    "PLC001": "Invalid placement type. Use 'Top of search (page 1)' or 'Product Pages'.",
    "PLC002": "Invalid placement bid adjustment format.",
    "PLC003": "Placement bid adjustment exceeds 900%.",
    
    # Sponsored Brands
    "SB001": "Landing page URL or ASINs required.",
    "SB002": "Invalid landing page URL.",
    "SB003": "Landing page requires 3-100 ASINs.",
    "SB004": "Brand name is required.",
    "SB005": "Brand name exceeds 30 characters.",
    "SB006": "Brand logo asset ID is required.",
    "SB007": "Headline is required.",
    "SB008": "Headline exceeds 50 characters (35 for Japan).",
    "SB009": "Creative ASINs required (up to 3).",
    "SB010": "Too many creative ASINs. Maximum 3.",
    "SB011": "Invalid automated bidding value. Use 'enabled' or 'disabled'.",
    "SB012": "Invalid bid multiplier format. Use +/-XX.XX%.",
    "SB013": "Bid multiplier out of range (-99% to +99.99%).",
    
    # Portfolio
    "POR001": "Portfolio ID not found.",
    "POR002": "Invalid portfolio budget.",
    
    # Hierarchy
    "HIE001": "Parent entity missing.",
    "HIE002": "Parent entity creation failed, child skipped.",
    "HIE003": "Ad group requires existing campaign.",
    "HIE004": "Keyword requires existing ad group.",
    "HIE005": "Ad requires existing ad group.",
    
    # Warnings
    "W001": "Warning: Target is currently paused.",
    "W002": "Warning: Campaign is currently paused.",
    "W003": "Warning: Bid is unusually low.",
    "W004": "Warning: Budget increase exceeds 200%.",
    "W005": "Warning: Negative keyword matches an active target.",
}


# =============================================================================
# EXPORT FOR USE IN VALIDATION ENGINE
# =============================================================================

def get_validation_rules(ad_type: str, record_type: str) -> Optional[RecordTypeRules]:
    """Get validation rules for specific ad type and record type"""
    if ad_type in VALIDATION_SCHEMA:
        return VALIDATION_SCHEMA[ad_type].get(record_type)
    return None


def get_error_message(code: str, **kwargs) -> str:
    """Get formatted error message for a validation code"""
    template = ERROR_MESSAGES.get(code, f"Unknown error: {code}")
    return template.format(**kwargs) if kwargs else template


def get_currency_limits(ad_type: str, currency: str) -> Optional[Dict]:
    """Get currency-specific limits for an ad type"""
    ad_type_key = {"Sponsored Products": "SP", "Sponsored Brands": "SB", "Sponsored Display": "SD"}.get(ad_type)
    if ad_type_key and ad_type_key in CURRENCY_LIMITS:
        return CURRENCY_LIMITS[ad_type_key].get(currency)
    return None


if __name__ == "__main__":
    # Test/demo
    print("Amazon Bulk Upload Validation Specification")
    print("=" * 50)
    print(f"Supported Ad Types: {list(VALIDATION_SCHEMA.keys())}")
    print(f"Error Codes Defined: {len(ERROR_MESSAGES) + len(BUSINESS_ERROR_MESSAGES)}")
    print(f"Currencies Supported: {list(CURRENCY_LIMITS['SP'].keys())}")
    print()
    print("Business Logic Schemas:")
    print("  - Isolation/Harvest Negatives: ISOLATION_NEGATIVE_RULES")
    print("  - Bleeder Negatives: BLEEDER_NEGATIVE_RULES")
    print("  - Bid Updates (Keyword): BID_UPDATE_KEYWORD_RULES")
    print("  - Bid Updates (Ad Group): BID_UPDATE_AD_GROUP_RULES")
    print("  - Auto Campaign Restrictions: AutoCampaignRestrictions")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "RecordType",
    "CampaignStatus",
    "AdGroupStatus",
    "KeywordStatus",
    "TargetingType",
    "BiddingStrategy",
    "KeywordMatchType",
    "ProductTargetMatchType",
    "PlacementType",
    "AdFormat",
    "BudgetType",
    "ValidationFlag",
    "BusinessValidationFlag",
    "RecommendationType",
    
    # Data structures
    "ColumnRule",
    "RecordTypeRules",
    "ValidationResult",
    "OptimizationRecommendation",
    
    # Standard validation rules
    "SP_CAMPAIGN_RULES",
    "SP_AD_GROUP_RULES",
    "SP_KEYWORD_RULES",
    "SP_PRODUCT_TARGETING_RULES",
    "SP_AD_RULES",
    "SP_PLACEMENT_RULES",
    
    # Business logic rules
    "ISOLATION_NEGATIVE_RULES",
    "BLEEDER_NEGATIVE_RULES",
    "BID_UPDATE_KEYWORD_RULES",
    "BID_UPDATE_AD_GROUP_RULES",
    "AutoCampaignRestrictions",
    
    # Limits
    "CURRENCY_LIMITS",
    "BID_MULTIPLIER_MIN",
    "BID_MULTIPLIER_MAX",
    "PLACEMENT_BID_MIN",
    "PLACEMENT_BID_MAX",
    
    # Schema
    "VALIDATION_SCHEMA",
    
    # Error messages
    "ERROR_MESSAGES",
    "BUSINESS_ERROR_MESSAGES",
    
    # Helper functions
    "validate_asin",
    "validate_date_format",
    "validate_date_not_past",
    "validate_keyword_words",
    "validate_campaign_name_chars",
    "validate_targeting_expression",
    "validate_bid_multiplier",
    "get_validation_rules",
    "get_error_message",
    "get_currency_limits",
    
    # Business logic functions
    "determine_negative_type",
    "validate_negative_keyword_row",
    "validate_bid_update_row",
    "validate_keyword_for_campaign_type",
    
    # Recommendation validation (NEW)
    "validate_recommendation",
    "validate_recommendations_batch",
    "RecommendationValidator",
]


# =============================================================================
# RECOMMENDATION-LEVEL VALIDATION (Validate at Source)
# =============================================================================

class RecommendationType(Enum):
    """Types of optimization recommendations"""
    
    # Negative keywords
    NEGATIVE_ISOLATION = "negative_isolation"      # Harvest → campaign negative
    NEGATIVE_BLEEDER = "negative_bleeder"          # Block bad performer in ad group
    
    # Bid adjustments
    BID_INCREASE = "bid_increase"
    BID_DECREASE = "bid_decrease"
    VISIBILITY_BOOST = "visibility_boost"          # Low visibility boost (+30%)
    
    # Keyword promotion
    KEYWORD_HARVEST = "keyword_harvest"            # Promote to own campaign
    
    # Status changes
    PAUSE_TARGET = "pause_target"
    ENABLE_TARGET = "enable_target"
    
    # Campaign creation
    CREATE_CAMPAIGN = "create_campaign"
    
    # Product targeting
    ADD_PRODUCT_TARGET = "add_product_target"
    REMOVE_PRODUCT_TARGET = "remove_product_target"


@dataclass
class ValidationResult:
    """Result of validating a recommendation or bulk row"""
    is_valid: bool
    can_execute: bool  # False if blocking errors exist
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    def add_error(self, code: str, message: str, field: str = None):
        self.errors.append({
            "code": code,
            "message": message,
            "field": field,
            "severity": "error"
        })
        self.is_valid = False
        self.can_execute = False
    
    def add_warning(self, code: str, message: str, field: str = None):
        self.warnings.append({
            "code": code,
            "message": message,
            "field": field,
            "severity": "warning"
        })


@dataclass
class OptimizationRecommendation:
    """
    A single optimization recommendation from the optimizer.
    Validation happens at creation time, not at export.
    """
    
    # Core recommendation data
    recommendation_id: str
    recommendation_type: RecommendationType
    
    # Target details
    campaign_name: str
    campaign_id: Optional[str] = None
    campaign_targeting_type: str = "Manual"  # "Auto" or "Manual"
    ad_group_name: Optional[str] = None
    ad_group_id: Optional[str] = None
    keyword_id: Optional[str] = None
    product_targeting_id: Optional[str] = None
    
    # Keyword/target details
    keyword_text: Optional[str] = None
    match_type: Optional[str] = None
    current_bid: Optional[float] = None
    new_bid: Optional[float] = None
    
    # ASIN for product targeting
    asin: Optional[str] = None
    product_targeting_expression: Optional[str] = None
    
    # Currency for validation
    currency: str = "USD"
    
    # Validation state (populated at creation)
    validation_result: ValidationResult = field(default_factory=lambda: ValidationResult(True, True))
    
    # Execution state
    is_selected: bool = True  # User can deselect
    
    @property
    def is_valid(self) -> bool:
        return self.validation_result.is_valid
    
    @property
    def can_execute(self) -> bool:
        return self.validation_result.can_execute and self.is_selected
    
    @property
    def errors(self) -> List[Dict]:
        return self.validation_result.errors
    
    @property
    def warnings(self) -> List[Dict]:
        return self.validation_result.warnings
    
    def get_status_icon(self) -> str:
        """Return UI status icon"""
        if not self.is_valid:
            return "❌"
        elif self.validation_result.has_warnings:
            return "⚠️"
        else:
            return "✅"
    
    def get_status_color(self) -> str:
        """Return UI status color"""
        if not self.is_valid:
            return "red"
        elif self.validation_result.has_warnings:
            return "yellow"
        else:
            return "green"


class RecommendationValidator:
    """
    Validates optimization recommendations at source (when generated).
    This ensures invalid recommendations never make it to bulk export.
    """
    
    def __init__(self, currency: str = "USD"):
        self.currency = currency
        self.limits = get_currency_limits("Sponsored Products", currency) or {
            "min_bid": 0.02,
            "max_bid": 1000,
            "min_budget": 1,
            "max_budget": 1_000_000
        }
    
    def validate(self, rec: OptimizationRecommendation) -> ValidationResult:
        """
        Validate a single recommendation.
        Call this when creating recommendations from optimizer.
        """
        result = ValidationResult(is_valid=True, can_execute=True)
        
        # Route to appropriate validator
        if rec.recommendation_type == RecommendationType.NEGATIVE_ISOLATION:
            self._validate_isolation_negative(rec, result)
        
        elif rec.recommendation_type == RecommendationType.NEGATIVE_BLEEDER:
            self._validate_bleeder_negative(rec, result)
        
        elif rec.recommendation_type in [RecommendationType.BID_INCREASE, RecommendationType.BID_DECREASE]:
            self._validate_bid_change(rec, result)
        
        elif rec.recommendation_type == RecommendationType.KEYWORD_HARVEST:
            self._validate_keyword_harvest(rec, result)
        
        elif rec.recommendation_type in [RecommendationType.PAUSE_TARGET, RecommendationType.ENABLE_TARGET]:
            self._validate_status_change(rec, result)
        
        elif rec.recommendation_type == RecommendationType.CREATE_CAMPAIGN:
            self._validate_campaign_creation(rec, result)
        
        # Common validations
        self._validate_common(rec, result)
        
        return result
    
    def _validate_common(self, rec: OptimizationRecommendation, result: ValidationResult):
        """Common validations for all recommendation types"""
        
        # Campaign name required
        if not rec.campaign_name or not rec.campaign_name.strip():
            result.add_error("CAM001", "Campaign name is required", "campaign_name")
        
        # Keyword length check
        if rec.keyword_text and len(rec.keyword_text) > 80:
            result.add_error("KEY002", f"Keyword exceeds 80 characters ({len(rec.keyword_text)} chars)", "keyword_text")
        
        # Keyword character check
        if rec.keyword_text and not re.match(r'^[a-zA-Z0-9\s\-]+$', rec.keyword_text):
            result.add_error("KEY003", "Keyword contains invalid characters (only alphanumeric, spaces, hyphens allowed)", "keyword_text")
    
    def _validate_isolation_negative(self, rec: OptimizationRecommendation, result: ValidationResult):
        """Validate Isolation/Harvest negative (campaign-level)"""
        
        # Ad group must NOT be specified
        if rec.ad_group_name and rec.ad_group_name.strip():
            result.add_error(
                "ISO001",
                "Ad Group must be BLANK for isolation negatives. This is a campaign-level negative.",
                "ad_group_name"
            )
        
        # Match type must be campaign negative
        if rec.match_type:
            mt = rec.match_type.lower().strip()
            if mt not in ["campaign negative exact", "campaign negative phrase"]:
                result.add_error(
                    "ISO002",
                    f"Isolation negatives must use 'campaign negative exact' or 'campaign negative phrase'. Got: '{rec.match_type}'",
                    "match_type"
                )
        
        # Works with both Auto and Manual campaigns (no restriction)
    
    def _validate_bleeder_negative(self, rec: OptimizationRecommendation, result: ValidationResult):
        """Validate Bleeder negative (ad group-level)"""
        
        # Ad group MUST be specified
        if not rec.ad_group_name or not rec.ad_group_name.strip():
            result.add_error(
                "BLD001",
                "Ad Group is REQUIRED for bleeder negatives. Specify which ad group to block the term in.",
                "ad_group_name"
            )
        
        # Match type must be ad-group level negative
        if rec.match_type:
            mt = rec.match_type.lower().strip()
            if mt not in ["negative exact", "negative phrase"]:
                result.add_error(
                    "BLD002",
                    f"Bleeder negatives must use 'negative exact' or 'negative phrase'. Got: '{rec.match_type}'",
                    "match_type"
                )
        
        # Works with both Auto and Manual campaigns (no restriction)
    
    def _validate_bid_change(self, rec: OptimizationRecommendation, result: ValidationResult):
        """Validate bid increase/decrease"""
        
        # New bid required
        if rec.new_bid is None:
            result.add_error("BID_UPD002", "New bid value is required", "new_bid")
            return
        
        # Check bid limits
        if rec.new_bid < self.limits["min_bid"]:
            result.add_error(
                "MAX002",
                f"Bid {rec.new_bid} is below minimum ({self.limits['min_bid']}) for {self.currency}",
                "new_bid"
            )
        
        if rec.new_bid > self.limits["max_bid"]:
            result.add_error(
                "MAX003",
                f"Bid {rec.new_bid} exceeds maximum ({self.limits['max_bid']}) for {self.currency}",
                "new_bid"
            )
        
        # Check for extreme change (warning only)
        if rec.current_bid and rec.current_bid > 0:
            change_pct = abs((rec.new_bid - rec.current_bid) / rec.current_bid) * 100
            if change_pct > 300:
                result.add_warning(
                    "BID_UPD003",
                    f"Bid change of {change_pct:.0f}% exceeds 300%. Current: {rec.current_bid}, New: {rec.new_bid}",
                    "new_bid"
                )
        
        # Ad group required for keyword bids
        if not rec.ad_group_name or not rec.ad_group_name.strip():
            result.add_error("ADG004", "Ad Group is required for keyword bid changes", "ad_group_name")
            
        # MUTUAL EXCLUSIVITY: Match Type (Keyword) vs PT Expression
        is_keyword = rec.match_type and rec.match_type.lower() in ["exact", "broad", "phrase"]
        if is_keyword and rec.product_targeting_expression and rec.product_targeting_expression.strip():
            result.add_error(
                "TAR003", 
                f"Product Targeting Expression ('{rec.product_targeting_expression}') must be blank for keyword Match Type '{rec.match_type}'.",
                "product_targeting_expression"
            )
    
    def _validate_keyword_harvest(self, rec: OptimizationRecommendation, result: ValidationResult):
        """Validate keyword harvest/promotion"""
        
        # Cannot harvest TO an Auto campaign (new campaign must be Manual)
        # This is typically handled at campaign creation, but flag if destination is Auto
        
        # Keyword required
        if not rec.keyword_text or not rec.keyword_text.strip():
            result.add_error("KEY001", "Keyword text is required for harvest", "keyword_text")
        
        # Match type required
        if not rec.match_type:
            result.add_error("KEY007", "Match type is required for harvested keyword", "match_type")
        elif rec.match_type.lower() not in ["broad", "phrase", "exact"]:
            result.add_error("KEY008", f"Invalid match type for harvest: '{rec.match_type}'", "match_type")
        
        # Check if trying to add positive keyword to Auto campaign
        if rec.campaign_targeting_type and rec.campaign_targeting_type.lower() == "auto":
            result.add_error(
                "AUTO001",
                f"Cannot add positive keywords to Auto-targeting campaign '{rec.campaign_name}'",
                "campaign_targeting_type"
            )
    
    def _validate_status_change(self, rec: OptimizationRecommendation, result: ValidationResult):
        """Validate pause/enable target"""
        
        # Need either keyword or ad group to target
        if not rec.keyword_text and not rec.ad_group_name:
            result.add_error("HIE001", "Must specify keyword or ad group to change status", "keyword_text")
    
    def _validate_campaign_creation(self, rec: OptimizationRecommendation, result: ValidationResult):
        """Validate new campaign creation"""
        
        # Campaign name length
        if rec.campaign_name and len(rec.campaign_name) > 128:
            result.add_error("CAM002", f"Campaign name exceeds 128 characters ({len(rec.campaign_name)} chars)", "campaign_name")
        
        # Campaign name characters
        if rec.campaign_name and not validate_campaign_name_chars(rec.campaign_name):
            result.add_error("CAM003", "Campaign name contains invalid characters", "campaign_name")


def validate_recommendation(
    rec: OptimizationRecommendation,
    currency: str = None
) -> ValidationResult:
    """
    Convenience function to validate a single recommendation.
    
    Usage:
        rec = OptimizationRecommendation(...)
        result = validate_recommendation(rec)
        rec.validation_result = result
    """
    validator = RecommendationValidator(currency=currency or rec.currency)
    return validator.validate(rec)


def validate_recommendations_batch(
    recommendations: List[OptimizationRecommendation],
    currency: str = "USD"
) -> Dict[str, Any]:
    """
    Validate a batch of recommendations.
    Returns summary statistics and updates each recommendation's validation_result.
    
    Usage:
        recs = optimizer.generate_recommendations()
        summary = validate_recommendations_batch(recs, currency="AED")
        
        valid_recs = [r for r in recs if r.can_execute]
        invalid_recs = [r for r in recs if not r.is_valid]
    """
    validator = RecommendationValidator(currency=currency)
    
    total = len(recommendations)
    valid_count = 0
    warning_count = 0
    error_count = 0
    
    for rec in recommendations:
        rec.validation_result = validator.validate(rec)
        
        if rec.is_valid:
            if rec.validation_result.has_warnings:
                warning_count += 1
            else:
                valid_count += 1
        else:
            error_count += 1
    
    return {
        "total": total,
        "valid": valid_count,
        "warnings": warning_count,
        "errors": error_count,
        "can_execute": valid_count + warning_count,
        "blocked": error_count,
        "pass_rate": (valid_count + warning_count) / total * 100 if total > 0 else 0
    }


# =============================================================================
# BULK EXPORT HELPER (generates bulk rows from validated recommendations)
# =============================================================================

def recommendation_to_bulk_row(rec: OptimizationRecommendation) -> Optional[Dict[str, str]]:
    """
    Convert a validated recommendation to a bulk sheet row.
    Returns None if recommendation is not executable.
    
    Only call this on recommendations that have passed validation.
    """
    if not rec.can_execute:
        return None
    
    row = {
        "Record Type": "Keyword",
        "Campaign": rec.campaign_name,
    }
    
    # Add campaign ID if available (for updates)
    if rec.campaign_id:
        row["Campaign ID"] = rec.campaign_id
    
    # Handle by recommendation type
    if rec.recommendation_type == RecommendationType.NEGATIVE_ISOLATION:
        row["Ad Group"] = ""  # Blank for campaign-level
        row["Keyword or Product Targeting"] = rec.keyword_text
        row["Match Type"] = rec.match_type or "campaign negative exact"
        row["Status"] = "enabled"
        row["Max Bid"] = ""
    
    elif rec.recommendation_type == RecommendationType.NEGATIVE_BLEEDER:
        row["Ad Group"] = rec.ad_group_name
        row["Keyword or Product Targeting"] = rec.keyword_text
        row["Match Type"] = rec.match_type or "negative exact"
        row["Status"] = "enabled"
        row["Max Bid"] = ""
    
    elif rec.recommendation_type in [RecommendationType.BID_INCREASE, RecommendationType.BID_DECREASE]:
        row["Ad Group"] = rec.ad_group_name
        row["Keyword or Product Targeting"] = rec.keyword_text
        row["Match Type"] = rec.match_type
        row["Max Bid"] = str(rec.new_bid)
        # Need Record ID for update - should be on the recommendation
    
    elif rec.recommendation_type in [RecommendationType.PAUSE_TARGET, RecommendationType.ENABLE_TARGET]:
        row["Ad Group"] = rec.ad_group_name
        row["Keyword or Product Targeting"] = rec.keyword_text
        row["Match Type"] = rec.match_type
        row["Status"] = "paused" if rec.recommendation_type == RecommendationType.PAUSE_TARGET else "enabled"
    
    return row


def recommendations_to_bulk_sheet(
    recommendations: List[OptimizationRecommendation]
) -> List[Dict[str, str]]:
    """
    Convert list of validated recommendations to bulk sheet rows.
    Only includes recommendations that can_execute.
    """
    rows = []
    for rec in recommendations:
        if rec.can_execute:
            row = recommendation_to_bulk_row(rec)
            if row:
                rows.append(row)
    return rows
