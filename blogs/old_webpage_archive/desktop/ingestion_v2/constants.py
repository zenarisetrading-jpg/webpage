"""
Ingestion V2 Constants
======================
Configuration constants for the ingestion pipeline.
PRD Reference: EMAIL_INGESTION_PRD.md
"""

# ==============================================================================
# EMAIL FORMAT
# PRD Section 6: Account Identification
# ==============================================================================
EMAIL_DOMAIN = "ingest.saddle.ai"
EMAIL_PREFIX = "str"  # str-{uuid}@ingest.saddle.ai

# ==============================================================================
# PARSING THRESHOLDS
# PRD Section 11: Partial Parse Tolerance
# ==============================================================================
MAX_DROP_RATE = 0.01  # >1% dropped rows -> QUARANTINE

# ==============================================================================
# SCHEDULE DETECTION
# PRD Section 14: Learning Mode
# ==============================================================================
MIN_INGESTIONS_FOR_BASELINE = 3  # First 3 establish cadence
LATE_THRESHOLD_HOURS = 12  # Mark late after expected + 12h

# ==============================================================================
# DATA RETENTION
# PRD Section 18: Purge Policy
# ==============================================================================
RAW_FILE_RETENTION_DAYS = 120  # Raw files kept 120 days
NORMALIZED_DATA_RETENTION = None  # Keep indefinitely

# ==============================================================================
# REQUIRED CSV HEADERS
# PRD Section 11: Field Mapping (Strict Mode)
# ==============================================================================
REQUIRED_HEADER_MAPPING = {
    'report_date': ['Date', 'Day'],
    'campaign_name': ['Campaign Name'],
    'ad_group_name': ['Ad Group Name'],
    'search_term': ['Customer Search Term'],
    'impressions': ['Impressions'],
    'clicks': ['Clicks'],
    'spend': ['Spend'],
}

OPTIONAL_HEADER_MAPPING = {
    'sales_7d': ['7 Day Total Sales', '7-Day Total Sales', 'Total Sales'],
}

# ==============================================================================
# ALERT SEVERITY
# PRD Section 16: Alerting Matrix
# ==============================================================================
class AlertSeverity:
    CRITICAL = "CRITICAL"  # System-wide failure -> Slack + Email
    HIGH = "HIGH"          # Single account failure -> Slack
    MEDIUM = "MEDIUM"      # Data late -> In-App

# Escalation rule: same failure_reason across N accounts -> Critical
ESCALATION_ACCOUNT_THRESHOLD = 3

# ==============================================================================
# STATE MACHINE
# PRD Section 14: Valid Transitions
# ==============================================================================
VALID_STATE_TRANSITIONS = {
    'RECEIVED': {'PROCESSING'},
    'PROCESSING': {'COMPLETED', 'QUARANTINE', 'FAILED'},
    'FAILED': {'PROCESSING'},  # Manual only
    'QUARANTINE': {'PROCESSING'},  # Manual only
    'COMPLETED': set(),  # Terminal
    'DUPLICATE_IGNORED': set(),  # Terminal
}
