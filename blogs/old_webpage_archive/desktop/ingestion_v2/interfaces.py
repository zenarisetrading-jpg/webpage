"""
Ingestion V2 Interfaces
=======================
Abstract base classes defining the contracts for each component.
Implementations must follow these interfaces exactly.

PRD Reference: EMAIL_INGESTION_PRD.md
- Section 7, 8: Adapters
- Section 9: Validation Layer
- Section 10: Raw File Storage
- Section 11: Parsing Engine

CRITICAL: These are abstractions only.
Implementations must NOT reference storage providers directly.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from uuid import UUID

from .models import (
    IngestionPayload,
    IngestionEvent,
    ParseResult,
    ValidationResult,
)
from .enums import IngestionSource


class BaseAdapter(ABC):
    """
    Abstract base class for ingestion adapters.
    
    PRD Section 7, 8:
    - Email Adapter: Receives email, extracts attachment
    - API Adapter: Authenticates with Amazon, fetches STR
    
    All adapters must produce identical IngestionPayload.
    """
    
    source: IngestionSource
    
    @abstractmethod
    async def receive(self) -> Optional[IngestionPayload]:
        """
        Receive data from the source and return a standardized payload.
        
        Returns:
            IngestionPayload if successful, None if no data available.
            
        Raises:
            AdapterError: If extraction fails (no attachment, wrong format, etc.)
        """
        pass
    
    @abstractmethod
    async def acknowledge(self, ingestion_id: UUID) -> None:
        """
        Acknowledge successful processing.
        (e.g., mark email as read, update API cursor)
        """
        pass


class BaseValidator(ABC):
    """
    Abstract base class for validation layer.
    
    PRD Section 9: Source-agnostic validation.
    
    Validation Types:
    1. Identity: Does UUID exist? Is sender whitelisted?
    2. Structural: Is file empty? Is it UTF-8 readable?
    3. Duplicate: Has this exact file been processed before?
    """
    
    @abstractmethod
    async def validate(self, payload: IngestionPayload) -> ValidationResult:
        """
        Validate the incoming payload before parsing.
        
        Args:
            payload: Standardized payload from any adapter.
            
        Returns:
            ValidationResult with valid=True/False and any errors.
        """
        pass
    
    @abstractmethod
    async def check_duplicate(
        self, 
        account_id: UUID, 
        fingerprint: str
    ) -> bool:
        """
        Check if this file has already been processed.
        
        PRD Section 14: Duplicate Protection
        fingerprint = hash(sender + filename + date_range)
        
        Returns:
            True if duplicate, False if new.
        """
        pass


class BaseStorage(ABC):
    """
    Abstract base class for raw file storage.
    
    PRD Section 10: Storage Abstraction
    
    CRITICAL: Business logic must NOT reference provider directly.
    Phase 1: Supabase
    Phase 3+: S3
    """
    
    @abstractmethod
    async def put(
        self, 
        file_content: bytes, 
        metadata: Dict[str, Any]
    ) -> str:
        """
        Store a raw file and return its identifier.
        
        Args:
            file_content: Raw bytes of the file.
            metadata: Dict with account_id, filename, received_at, etc.
            
        Returns:
            file_id: Unique identifier for retrieval.
        """
        pass
    
    @abstractmethod
    async def get(self, file_id: str) -> bytes:
        """
        Retrieve a raw file by its identifier.
        
        Args:
            file_id: Identifier returned by put().
            
        Returns:
            Raw file content as bytes.
            
        Raises:
            FileNotFoundError: If file_id doesn't exist.
        """
        pass
    
    @abstractmethod
    async def delete(self, file_id: str) -> bool:
        """
        Delete a raw file (for retention policy).
        
        Returns:
            True if deleted, False if not found.
        """
        pass


class BaseParser(ABC):
    """
    Abstract base class for CSV parsing engine.
    
    PRD Section 11: Parsing Engine & Field Mapping
    
    CRITICAL: Must run on separate worker queue (celery_queue_v2)
    to avoid starving production V1 system.
    """
    
    # Required headers per PRD Section 11
    REQUIRED_HEADERS = {
        'report_date': ['Date', 'Day'],
        'campaign_name': ['Campaign Name'],
        'ad_group_name': ['Ad Group Name'],
        'search_term': ['Customer Search Term'],
        'impressions': ['Impressions'],
        'clicks': ['Clicks'],
        'spend': ['Spend'],
    }
    
    # Optional headers
    OPTIONAL_HEADERS = {
        'sales_7d': ['7 Day Total Sales'],
    }
    
    @abstractmethod
    async def parse(self, file_content: bytes) -> ParseResult:
        """
        Parse CSV content into normalized rows.
        
        PRD Rules:
        - Headers -> snake_case
        - Strip currency symbols
        - Strip percent symbols
        - ≤1% malformed rows -> drop rows, continue
        - >1% malformed rows -> QUARANTINE
        
        Args:
            file_content: Raw bytes of the CSV file.
            
        Returns:
            ParseResult with rows, warnings, and quarantine flag.
        """
        pass
    
    @abstractmethod
    def compute_fingerprint(
        self,
        sender: str,
        filename: str,
        date_range: tuple
    ) -> str:
        """
        Compute source fingerprint for duplicate detection.
        
        PRD Section 14: source_fingerprint = hash(sender + filename + date_range)
        
        Returns:
            SHA256 hash as hex string.
        """
        pass


class BaseEventLogger(ABC):
    """
    Abstract base class for ingestion event logging.
    
    PRD Section 13: Ingestion History & Auditability
    """
    
    @abstractmethod
    async def create_event(
        self,
        account_id: UUID,
        source: IngestionSource,
        metadata: Dict[str, Any]
    ) -> IngestionEvent:
        """
        Create a new ingestion event (status=RECEIVED).
        """
        pass
    
    @abstractmethod
    async def update_status(
        self,
        ingestion_id: UUID,
        status: str,
        failure_reason: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the status of an ingestion event.
        Must validate state transitions per PRD Section 14.
        """
        pass
    
    @abstractmethod
    async def get_event(self, ingestion_id: UUID) -> Optional[IngestionEvent]:
        """
        Retrieve an ingestion event by ID.
        """
        pass
