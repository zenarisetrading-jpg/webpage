"""
PRD Compliance Test Suite
=========================
Validates that ingestion_v2 implementation matches EMAIL_INGESTION_PRD.md v1.2

Run with: python -m pytest tests/test_prd_compliance.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion_v2.enums import IngestionSource, IngestionStatus
from ingestion_v2.models import (
    IngestionPayload, 
    IngestionEvent, 
    ParsedRow, 
    ParseResult,
    ValidationResult
)
from ingestion_v2.interfaces import (
    BaseAdapter,
    BaseValidator,
    BaseStorage,
    BaseParser,
    BaseEventLogger
)
from ingestion_v2.constants import (
    EMAIL_DOMAIN,
    EMAIL_PREFIX,
    MAX_DROP_RATE,
    MIN_INGESTIONS_FOR_BASELINE,
    LATE_THRESHOLD_HOURS,
    RAW_FILE_RETENTION_DAYS,
    REQUIRED_HEADER_MAPPING,
    OPTIONAL_HEADER_MAPPING,
    VALID_STATE_TRANSITIONS,
    AlertSeverity,
)
from ingestion_v2.exceptions import (
    IngestionError,
    AdapterError,
    ValidationError,
    StorageError,
    ParseError,
    DuplicateError,
    StateTransitionError,
)


class TestPRDSection5_IngestionSources:
    """PRD Section 5: Supported Ingestion Sources"""
    
    def test_enum_has_email(self):
        """PRD: EMAIL source must exist"""
        assert hasattr(IngestionSource, 'EMAIL')
        assert IngestionSource.EMAIL.value == "EMAIL"
    
    def test_enum_has_api(self):
        """PRD: API source must exist"""
        assert hasattr(IngestionSource, 'API')
        assert IngestionSource.API.value == "API"
    
    def test_enum_has_manual(self):
        """PRD: MANUAL source must exist"""
        assert hasattr(IngestionSource, 'MANUAL')
        assert IngestionSource.MANUAL.value == "MANUAL"
    
    def test_only_three_sources(self):
        """PRD: Only EMAIL, API, MANUAL allowed"""
        assert len(IngestionSource) == 3


class TestPRDSection6_AccountIdentification:
    """PRD Section 6: Account Identification & Security Model"""
    
    def test_email_domain(self):
        """PRD: Email format must be str-{uuid}@ingest.saddle.ai"""
        assert EMAIL_DOMAIN == "ingest.saddle.ai"
        assert EMAIL_PREFIX == "str"


class TestPRDSection11_FieldMapping:
    """PRD Section 11: Parsing Engine & Field Mapping"""
    
    def test_required_headers_defined(self):
        """PRD: All required headers must be mapped"""
        required = ['report_date', 'campaign_name', 'ad_group_name', 
                    'search_term', 'impressions', 'clicks', 'spend']
        for field in required:
            assert field in REQUIRED_HEADER_MAPPING, f"Missing required field: {field}"
    
    def test_optional_headers_defined(self):
        """PRD: sales_7d is optional with default 0"""
        assert 'sales_7d' in OPTIONAL_HEADER_MAPPING
    
    def test_amazon_header_variations(self):
        """PRD: Must handle 'Date' or 'Day' for report_date"""
        assert 'Date' in REQUIRED_HEADER_MAPPING['report_date']
        assert 'Day' in REQUIRED_HEADER_MAPPING['report_date']
    
    def test_partial_parse_tolerance(self):
        """PRD: >1% malformed rows -> QUARANTINE"""
        assert MAX_DROP_RATE == 0.01


class TestPRDSection12_Schema:
    """PRD Section 12: Normalized Data Load & Schema"""
    
    def test_parsed_row_has_all_fields(self):
        """PRD: ParsedRow must have all required fields"""
        row = ParsedRow(
            report_date="2026-01-01",
            campaign_name="Test",
            ad_group_name="Test",
            search_term="test",
            impressions=100,
            clicks=10,
            spend=5.0,
            sales_7d=20.0
        )
        assert row.report_date
        assert row.campaign_name
        assert row.ad_group_name
        assert row.search_term
        assert row.impressions == 100
        assert row.clicks == 10
        assert row.spend == 5.0
        assert row.sales_7d == 20.0


class TestPRDSection13_Auditability:
    """PRD Section 13: Ingestion History & Auditability"""
    
    def test_ingestion_event_has_all_fields(self):
        """PRD: IngestionEvent must have required audit fields"""
        from uuid import uuid4
        from datetime import datetime
        
        event = IngestionEvent(
            ingestion_id=uuid4(),
            account_id=uuid4(),
            source=IngestionSource.EMAIL,
            status=IngestionStatus.RECEIVED,
            received_at=datetime.utcnow(),
            metadata={"sender": "test@example.com"}
        )
        assert event.ingestion_id
        assert event.account_id
        assert event.source == IngestionSource.EMAIL
        assert event.status == IngestionStatus.RECEIVED
        assert event.received_at
        assert "sender" in event.metadata


class TestPRDSection14_StateMachine:
    """PRD Section 14: Ingestion State Machine & Concurrency Control"""
    
    def test_valid_transitions_defined(self):
        """PRD: Valid state transitions must be defined"""
        assert 'RECEIVED' in VALID_STATE_TRANSITIONS
        assert 'PROCESSING' in VALID_STATE_TRANSITIONS
        assert 'COMPLETED' in VALID_STATE_TRANSITIONS
        assert 'FAILED' in VALID_STATE_TRANSITIONS
        assert 'QUARANTINE' in VALID_STATE_TRANSITIONS
    
    def test_received_to_processing(self):
        """PRD: RECEIVED -> PROCESSING is valid"""
        assert 'PROCESSING' in VALID_STATE_TRANSITIONS['RECEIVED']
    
    def test_processing_to_completed(self):
        """PRD: PROCESSING -> COMPLETED is valid"""
        assert 'COMPLETED' in VALID_STATE_TRANSITIONS['PROCESSING']
    
    def test_processing_to_quarantine(self):
        """PRD: PROCESSING -> QUARANTINE is valid"""
        assert 'QUARANTINE' in VALID_STATE_TRANSITIONS['PROCESSING']
    
    def test_processing_to_failed(self):
        """PRD: PROCESSING -> FAILED is valid"""
        assert 'FAILED' in VALID_STATE_TRANSITIONS['PROCESSING']
    
    def test_failed_to_processing_manual(self):
        """PRD: FAILED -> PROCESSING is valid (manual only)"""
        assert 'PROCESSING' in VALID_STATE_TRANSITIONS['FAILED']
    
    def test_quarantine_to_processing_manual(self):
        """PRD: QUARANTINE -> PROCESSING is valid (manual only)"""
        assert 'PROCESSING' in VALID_STATE_TRANSITIONS['QUARANTINE']
    
    def test_completed_is_terminal(self):
        """PRD: COMPLETED is terminal state"""
        assert len(VALID_STATE_TRANSITIONS['COMPLETED']) == 0
    
    def test_duplicate_ignored_is_terminal(self):
        """PRD: DUPLICATE_IGNORED is terminal state"""
        assert len(VALID_STATE_TRANSITIONS['DUPLICATE_IGNORED']) == 0
    
    def test_enum_transition_validation(self):
        """PRD: Enum has transition validation method"""
        assert IngestionStatus.can_transition(
            IngestionStatus.RECEIVED, 
            IngestionStatus.PROCESSING
        ) == True
        assert IngestionStatus.can_transition(
            IngestionStatus.COMPLETED, 
            IngestionStatus.PROCESSING
        ) == False


class TestPRDSection14_ScheduleDetection:
    """PRD Section 14: Schedule Detection & Monitoring"""
    
    def test_learning_mode_threshold(self):
        """PRD: First 3 successful ingestions establish baseline"""
        assert MIN_INGESTIONS_FOR_BASELINE == 3
    
    def test_late_threshold(self):
        """PRD: Mark late after expected + 12 hours"""
        assert LATE_THRESHOLD_HOURS == 12


class TestPRDSection16_Alerting:
    """PRD Section 16: Alerting & Notifications"""
    
    def test_severity_levels(self):
        """PRD: CRITICAL, HIGH, MEDIUM severity levels"""
        assert AlertSeverity.CRITICAL == "CRITICAL"
        assert AlertSeverity.HIGH == "HIGH"
        assert AlertSeverity.MEDIUM == "MEDIUM"


class TestPRDSection18_Retention:
    """PRD Section 18: Data Retention & Purge Policy"""
    
    def test_raw_file_retention(self):
        """PRD: Raw STR data retained 120 days"""
        assert RAW_FILE_RETENTION_DAYS == 120


class TestPRDInterfaces:
    """Verify abstract interfaces exist per PRD architecture"""
    
    def test_base_adapter_exists(self):
        """PRD Section 7, 8: Adapter interface exists"""
        assert hasattr(BaseAdapter, 'receive')
        assert hasattr(BaseAdapter, 'acknowledge')
    
    def test_base_validator_exists(self):
        """PRD Section 9: Validator interface exists"""
        assert hasattr(BaseValidator, 'validate')
        assert hasattr(BaseValidator, 'check_duplicate')
    
    def test_base_storage_exists(self):
        """PRD Section 10: Storage interface exists"""
        assert hasattr(BaseStorage, 'put')
        assert hasattr(BaseStorage, 'get')
        assert hasattr(BaseStorage, 'delete')
    
    def test_base_parser_exists(self):
        """PRD Section 11: Parser interface exists"""
        assert hasattr(BaseParser, 'parse')
        assert hasattr(BaseParser, 'compute_fingerprint')
        assert hasattr(BaseParser, 'REQUIRED_HEADERS')


class TestPRDExceptions:
    """Verify exceptions exist per PRD failure handling"""
    
    def test_adapter_error(self):
        """PRD Section 7: Adapter failures"""
        err = AdapterError("No attachment")
        assert isinstance(err, IngestionError)
    
    def test_validation_error(self):
        """PRD Section 9: Validation failures"""
        err = ValidationError("Invalid UUID")
        assert isinstance(err, IngestionError)
    
    def test_parse_error(self):
        """PRD Section 11: Parse failures"""
        err = ParseError("Missing header", dropped_rows=5, total_rows=100)
        assert err.should_quarantine == True
    
    def test_duplicate_error(self):
        """PRD Section 14: Duplicate detection"""
        err = DuplicateError("abc123")
        assert err.should_quarantine == False
    
    def test_state_transition_error(self):
        """PRD Section 14: Invalid transitions"""
        err = StateTransitionError("COMPLETED", "PROCESSING")
        assert "COMPLETED" in str(err)


class TestParseResultQuarantine:
    """PRD Section 11: Partial Parse Tolerance"""
    
    def test_under_threshold_no_quarantine(self):
        """≤1% dropped -> no quarantine"""
        result = ParseResult(
            success=True,
            total_rows=1000,
            dropped_rows=10,  # 1%
            rows=[]
        )
        assert result.should_quarantine == False
    
    def test_over_threshold_quarantine(self):
        """>1% dropped -> quarantine"""
        result = ParseResult(
            success=True,
            total_rows=1000,
            dropped_rows=11,  # 1.1%
            rows=[]
        )
        assert result.should_quarantine == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
