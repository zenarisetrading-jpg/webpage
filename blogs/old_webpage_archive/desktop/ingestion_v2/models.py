"""
Ingestion V2 Models
===================
Data models using stdlib dataclasses (no external dependencies).
PRD Reference: EMAIL_INGESTION_PRD.md Section 11, 13
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from uuid import UUID

from .enums import IngestionSource, IngestionStatus


@dataclass
class IngestionPayload:
    """
    Payload passed from Adapter to Validator.
    This is the common interface between all adapters (Email, API, Manual).
    
    PRD Section 7, 8: Adapter responsibilities
    """
    # Identity
    account_uuid: str
    sender_email: str
    
    # File data
    file_content: bytes
    filename: str
    
    # Metadata
    source: IngestionSource
    received_at: datetime = field(default_factory=datetime.utcnow)
    subject: Optional[str] = None


@dataclass
class IngestionEvent:
    """
    Represents a row in ingestion_events_v2 table.
    
    PRD Section 13: Ingestion History & Auditability
    """
    ingestion_id: UUID
    account_id: UUID
    source: IngestionSource
    status: IngestionStatus
    received_at: datetime
    
    raw_file_path: Optional[str] = None
    source_fingerprint: Optional[str] = None
    processed_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedRow:
    """
    Single parsed row from CSV, ready for DB insert.
    
    PRD Section 11: Field Mapping
    """
    report_date: date
    campaign_name: str
    ad_group_name: str
    search_term: str
    impressions: int = 0
    clicks: int = 0
    spend: float = 0.0
    sales_7d: float = 0.0


@dataclass
class ParseResult:
    """
    Result of parsing a CSV file.
    
    PRD Section 11: Partial Parse Tolerance
    """
    success: bool
    rows: List[ParsedRow] = field(default_factory=list)
    total_rows: int = 0
    dropped_rows: int = 0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    @property
    def drop_rate(self) -> float:
        """Calculate percentage of dropped rows."""
        if self.total_rows == 0:
            return 0.0
        return self.dropped_rows / self.total_rows
    
    @property
    def should_quarantine(self) -> bool:
        """
        PRD Rule: >1% malformed rows -> QUARANTINE
        """
        return self.drop_rate > 0.01


@dataclass
class ValidationResult:
    """
    Result of validation layer checks.
    
    PRD Section 9: Validation Layer
    """
    valid: bool
    account_id: Optional[UUID] = None
    errors: List[str] = field(default_factory=list)
    
    # For duplicate detection
    is_duplicate: bool = False
    source_fingerprint: Optional[str] = None
