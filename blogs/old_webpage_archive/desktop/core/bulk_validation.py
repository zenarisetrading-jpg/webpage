"""
Comprehensive Bulk Validation Engine for Amazon Advertising

Validates bulk export files against Amazon's requirements including:
- Isolation Negatives (campaign-level, ad group BLANK)
- Bleeder Negatives (ad group-level, ad group REQUIRED)
- Bid Updates (currency-aware limits)
- Auto Campaign Restrictions
"""

import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional
import re


# ==========================================
# ENUMS
# ==========================================

class ValidationSeverity(Enum):
    """Validation severity levels with export behavior."""
    CRITICAL = "critical"   # Blocks entire export (file-level issues)
    ERROR = "error"         # Blocks this row only
    WARNING = "warning"     # Exportable but flagged for review
    INFO = "info"           # FYI / best practice suggestion

# Alias for backward compatibility
Severity = ValidationSeverity


class NegativeType(Enum):
    ISOLATION = "isolation"  # Campaign-level, blocks promoted KW
    BLEEDER = "bleeder"      # Ad group-level, blocks poor performers



# ==========================================
# CURRENCY LIMITS
# ==========================================

CURRENCY_LIMITS = {
    "USD": {"min_bid": 0.02, "max_bid": 1000.00, "max_budget": 1000000.00},
    "AED": {"min_bid": 0.10, "max_bid": 3500.00, "max_budget": 3500000.00},
    "EUR": {"min_bid": 0.02, "max_bid": 1000.00, "max_budget": 1000000.00},
    "GBP": {"min_bid": 0.02, "max_bid": 850.00, "max_budget": 850000.00},
    "SAR": {"min_bid": 0.10, "max_bid": 3500.00, "max_budget": 3500000.00},
    "INR": {"min_bid": 1.00, "max_bid": 50000.00, "max_budget": 50000000.00},
    "CAD": {"min_bid": 0.02, "max_bid": 1300.00, "max_budget": 1300000.00},
    "AUD": {"min_bid": 0.02, "max_bid": 1500.00, "max_budget": 1500000.00},
    "JPY": {"min_bid": 2, "max_bid": 100000, "max_budget": 100000000},
    "MXN": {"min_bid": 0.50, "max_bid": 18000.00, "max_budget": 18000000.00},
    "DEFAULT": {"min_bid": 0.02, "max_bid": 1000.00, "max_budget": 1000000.00},
}


# ==========================================
# ERROR CODES
# ==========================================

ERROR_MESSAGES = {
    # Negatives - Isolation (campaign-level)
    "ISO001": "Ad Group must be blank for isolation negatives",
    "ISO002": "Match type must be 'campaign negative exact/phrase' for isolation",
    "ISO003": "Status must be 'enabled' or 'deleted' for isolation",
    "ISO004": "Bid must be blank for isolation negatives",
    
    # Negatives - Bleeder (ad group-level)
    "BLD001": "Ad Group is required for bleeder negatives",
    "BLD002": "Match type must NOT be 'campaign negative' for bleeder",
    
    # Negatives - General
    "NEG001": "Entity field mismatch (auto-corrected)",
    "NEG002": "Match type corrected",
    "NEG003": "Bid cleared (must be blank for negatives)",
    "NEG004": "Duplicate removed (kept first)",
    
    # Bid Updates
    "BID002": "Bid outside currency limits",
    "BID003": "Bid change exceeds 300%",  # Warning only
    "BID004": "Missing entity ID",
    "BID005": "Dual IDs detected",
    "BID006": "Duplicate bid update (kept first)",
    
    # Auto Campaign
    "AUTO001": "Cannot add positive keyword to Auto campaign",
    
    # General
    "GEN001": "Missing Campaign/Ad Group IDs",
    "GEN002": "Dual IDs detected (auto-corrected)",
    "GEN003": "Missing entity-specific ID",
}


# ==========================================
# DATACLASSES
# ==========================================

@dataclass
class ValidationIssue:
    """Single validation issue with row context."""
    code: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.WARNING
    row: int = -1
    field: str = ""
    value: str = ""


@dataclass
class ValidationResult:
    """Result of validation run with 4-level severity support."""
    issues: List[ValidationIssue] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """True if no CRITICAL or ERROR level issues."""
        return not any(i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] for i in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """True if any WARNING level issues."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)
    
    @property
    def has_info(self) -> bool:
        """True if any INFO level issues."""
        return any(i.severity == ValidationSeverity.INFO for i in self.issues)
    
    @property
    def highest_severity(self) -> Optional[ValidationSeverity]:
        """Return the most severe issue level."""
        if not self.issues:
            return None
        severity_order = [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR, ValidationSeverity.WARNING, ValidationSeverity.INFO]
        for sev in severity_order:
            if any(i.severity == sev for i in self.issues):
                return sev
        return None
    
    @property
    def error_count(self) -> int:
        return len([i for i in self.issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]])
    
    @property
    def warning_count(self) -> int:
        return len([i for i in self.issues if i.severity == ValidationSeverity.WARNING])
    
    # Legacy properties for backward compatibility
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
    
    @property
    def valid(self) -> bool:
        return self.is_valid
    
    def to_dict_list(self) -> List[dict]:
        """Convert to list of dicts for UI display."""
        return [
            {
                "row": i.row,
                "code": i.code,
                "msg": i.message,
                "severity": i.severity.value
            }
            for i in self.issues
        ]


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def is_blank(val) -> bool:
    """Check if value is effectively blank."""
    return pd.isna(val) or str(val).strip() == ""


def detect_negative_type(match_type: str) -> Optional[NegativeType]:
    """
    Detect if this is an Isolation or Bleeder negative based on match type.
    
    Returns None if not a negative keyword.
    """
    mt = str(match_type).lower().strip()
    
    if "campaign negative" in mt:
        return NegativeType.ISOLATION
    elif "negative" in mt:
        return NegativeType.BLEEDER
    
    return None


def get_currency_limits(currency: str = "USD") -> dict:
    """Get bid/budget limits for a currency."""
    return CURRENCY_LIMITS.get(currency.upper(), CURRENCY_LIMITS["DEFAULT"])


# ==========================================
# VALIDATION FUNCTIONS
# ==========================================

def validate_isolation_negative(row: dict, idx: int) -> List[ValidationIssue]:
    """
    Validate Isolation (campaign-level) negative.
    
    Rules:
    - Ad Group MUST BE BLANK
    - Match Type MUST be 'campaign negative exact' or 'campaign negative phrase'
    - Status ONLY 'enabled' or 'deleted'
    - Max Bid MUST BE BLANK
    """
    issues = []
    
    # ISO001: Ad Group must be blank
    if not is_blank(row.get("Ad Group Name")) or not is_blank(row.get("Ad Group Id")):
        issues.append(ValidationIssue(
            row=idx,
            code="ISO001",
            message=ERROR_MESSAGES["ISO001"],
            field="Ad Group Name",
            value=str(row.get("Ad Group Name", ""))
        ))
    
    # ISO002: Match Type validation
    match_type = str(row.get("Match Type", "")).lower()
    if match_type not in ["campaign negative exact", "campaign negative phrase"]:
        issues.append(ValidationIssue(
            row=idx,
            code="ISO002",
            message=ERROR_MESSAGES["ISO002"],
            field="Match Type",
            value=match_type
        ))
    
    # ISO003: Status validation
    status = str(row.get("State", row.get("Status", ""))).lower()
    if status and status not in ["enabled", "deleted", ""]:
        issues.append(ValidationIssue(
            row=idx,
            code="ISO003",
            message=ERROR_MESSAGES["ISO003"],
            field="State",
            value=status
        ))
    
    # ISO004: Max Bid must be blank
    if not is_blank(row.get("Bid")) or not is_blank(row.get("Max Bid")):
        issues.append(ValidationIssue(
            row=idx,
            code="ISO004",
            message=ERROR_MESSAGES["ISO004"],
            field="Bid",
            value=str(row.get("Bid", row.get("Max Bid", "")))
        ))
    
    return issues


def validate_bleeder_negative(row: dict, idx: int) -> List[ValidationIssue]:
    """
    Validate Bleeder (ad group-level) negative.
    
    Rules:
    - Ad Group REQUIRED
    - Match Type MUST be 'negative exact' or 'negative phrase' (NOT campaign negative)
    """
    issues = []
    
    # BLD001: Ad Group is required
    if is_blank(row.get("Ad Group Name")) and is_blank(row.get("Ad Group Id")):
        issues.append(ValidationIssue(
            row=idx,
            code="BLD001",
            message=ERROR_MESSAGES["BLD001"],
            field="Ad Group Name"
        ))
    
    # BLD002: Match Type validation (should NOT have "campaign" prefix)
    match_type = str(row.get("Match Type", "")).lower()
    if "campaign" in match_type:
        issues.append(ValidationIssue(
            row=idx,
            code="BLD002",
            message=ERROR_MESSAGES["BLD002"],
            field="Match Type",
            value=match_type
        ))
    
    return issues


def validate_bid_update(row: dict, idx: int, currency: str = "USD", current_bid: float = None) -> List[ValidationIssue]:
    """
    Validate bid update row.
    
    Rules:
    - Bid within currency limits
    - Warning if bid change exceeds 300%
    
    Note: Record ID is NOT required - Amazon matches by Campaign+AdGroup+Keyword+MatchType
    """
    issues = []
    limits = get_currency_limits(currency)
    
    bid_val = row.get("Bid") or row.get("Max Bid") or row.get("New Bid")
    
    if bid_val is None or is_blank(bid_val):
        return issues  # No bid to validate
    
    try:
        bid = float(str(bid_val).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError):
        issues.append(ValidationIssue(
            row=idx,
            code="BID002",
            message=f"Invalid bid value: {bid_val}",
            field="Bid",
            value=str(bid_val)
        ))
        return issues
    
    # BID002: Bid within currency limits
    if bid < limits["min_bid"]:
        issues.append(ValidationIssue(
            row=idx,
            code="BID002",
            message=f"Bid {bid} below minimum {limits['min_bid']} {currency}",
            field="Bid",
            value=str(bid)
        ))
    elif bid > limits["max_bid"]:
        issues.append(ValidationIssue(
            row=idx,
            code="BID002",
            message=f"Bid {bid} exceeds maximum {limits['max_bid']} {currency}",
            field="Bid",
            value=str(bid)
        ))
    
    # BID003: Warning if change exceeds 300%
    if current_bid and current_bid > 0:
        change_pct = abs(bid - current_bid) / current_bid * 100
        if change_pct > 300:
            issues.append(ValidationIssue(
                row=idx,
                code="BID003",
                message=f"Bid change {change_pct:.0f}% exceeds 300% threshold",
                severity=Severity.WARNING,
                field="Bid",
                value=f"{current_bid} -> {bid}"
            ))
    
    return issues


def validate_auto_campaign(row: dict, idx: int, campaign_type: str = "Manual") -> List[ValidationIssue]:
    """
    Validate that positive keywords are not added to Auto campaigns.
    
    Auto campaigns CAN have:
    - Negative keywords (all types)
    - Product targeting adjustments
    
    Auto campaigns CANNOT have:
    - broad keywords
    - phrase keywords
    - exact keywords
    """
    issues = []
    
    if campaign_type.lower() != "auto":
        return issues  # Only validate Auto campaigns
    
    match_type = str(row.get("Match Type", "")).lower()
    
    # Positive match types that are NOT allowed in Auto
    blocked_types = ["broad", "phrase", "exact"]
    
    # Check if this is a positive keyword (not negative)
    if "negative" not in match_type and match_type in blocked_types:
        issues.append(ValidationIssue(
            row=idx,
            code="AUTO001",
            message=ERROR_MESSAGES["AUTO001"],
            field="Match Type",
            value=match_type
        ))
    
    return issues


# ==========================================
# MAIN VALIDATION ORCHESTRATOR
# ==========================================

def validate_bulk_export(
    df: pd.DataFrame,
    export_type: str = "negatives",  # "negatives", "bids", "harvest"
    currency: str = "USD",
    campaign_cache: Dict[str, str] = None  # Maps campaign names to targeting types
) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Comprehensive validation for bulk export files.
    
    Args:
        df: DataFrame to validate
        export_type: Type of export ("negatives", "bids", "harvest")
        currency: Account currency code
        campaign_cache: Optional dict mapping campaign names to targeting types
        
    Returns:
        Tuple of (cleaned DataFrame, ValidationResult)
    """
    if df is None or df.empty:
        return df, ValidationResult()
    
    df = df.copy()
    all_issues = []
    campaign_cache = campaign_cache or {}
    
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        row_num = idx + 2  # Excel row number (header is row 1)
        
        # 1. Negative Keyword Validation
        if export_type == "negatives":
            match_type = str(row_dict.get("Match Type", "")).lower()
            neg_type = detect_negative_type(match_type)
            
            if neg_type == NegativeType.ISOLATION:
                issues = validate_isolation_negative(row_dict, row_num)
            elif neg_type == NegativeType.BLEEDER:
                issues = validate_bleeder_negative(row_dict, row_num)
            else:
                issues = []
            
            all_issues.extend(issues)
        
        # 2. Bid Validation
        if export_type == "bids" or row_dict.get("Bid") or row_dict.get("New Bid"):
            issues = validate_bid_update(row_dict, row_num, currency)
            all_issues.extend(issues)
        
        # 3. Auto Campaign Restrictions
        campaign_name = row_dict.get("Campaign Name", "")
        campaign_type = campaign_cache.get(campaign_name, "Manual")
        issues = validate_auto_campaign(row_dict, row_num, campaign_type)
        all_issues.extend(issues)
    
    result = ValidationResult(issues=all_issues)
    
    return df, result
