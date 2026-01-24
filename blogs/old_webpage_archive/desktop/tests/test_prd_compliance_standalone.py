#!/usr/bin/env python3
"""
PRD Compliance Test Suite (Standalone)
======================================
Validates that ingestion_v2 implementation matches EMAIL_INGESTION_PRD.md v1.2

Run with: python3 tests/test_prd_compliance_standalone.py
"""

import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Track results
PASSED = 0
FAILED = 0
ERRORS = []


def test(name, condition, error_msg=None):
    """Simple test function"""
    global PASSED, FAILED, ERRORS
    try:
        if condition:
            print(f"  ✅ {name}")
            PASSED += 1
        else:
            print(f"  ❌ {name}")
            FAILED += 1
            if error_msg:
                ERRORS.append(f"{name}: {error_msg}")
    except Exception as e:
        print(f"  ❌ {name} (ERROR: {e})")
        FAILED += 1
        ERRORS.append(f"{name}: {str(e)}")


def run_tests():
    global PASSED, FAILED, ERRORS
    
    print("\n" + "="*60)
    print("PRD COMPLIANCE TEST SUITE")
    print("="*60)
    
    # Import modules
    print("\n📦 Importing modules...")
    try:
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
        print("  ✅ All imports successful")
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return
    
    # =========================================================================
    # PRD Section 5: Supported Ingestion Sources
    # =========================================================================
    print("\n📋 PRD Section 5: Supported Ingestion Sources")
    test("IngestionSource.EMAIL exists", 
         hasattr(IngestionSource, 'EMAIL') and IngestionSource.EMAIL.value == "EMAIL")
    test("IngestionSource.API exists", 
         hasattr(IngestionSource, 'API') and IngestionSource.API.value == "API")
    test("IngestionSource.MANUAL exists", 
         hasattr(IngestionSource, 'MANUAL') and IngestionSource.MANUAL.value == "MANUAL")
    test("Only 3 sources defined", 
         len(IngestionSource) == 3)
    
    # =========================================================================
    # PRD Section 6: Account Identification
    # =========================================================================
    print("\n📋 PRD Section 6: Account Identification")
    test("Email domain is ingest.saddle.ai", 
         EMAIL_DOMAIN == "ingest.saddle.ai")
    test("Email prefix is 'str'", 
         EMAIL_PREFIX == "str")
    
    # =========================================================================
    # PRD Section 11: Field Mapping
    # =========================================================================
    print("\n📋 PRD Section 11: Parsing Engine & Field Mapping")
    required_fields = ['report_date', 'campaign_name', 'ad_group_name', 
                       'search_term', 'impressions', 'clicks', 'spend']
    for field in required_fields:
        test(f"Required field '{field}' mapped", 
             field in REQUIRED_HEADER_MAPPING)
    
    test("sales_7d is optional", 
         'sales_7d' in OPTIONAL_HEADER_MAPPING)
    test("Date/Day variations for report_date", 
         'Date' in REQUIRED_HEADER_MAPPING['report_date'] and 
         'Day' in REQUIRED_HEADER_MAPPING['report_date'])
    test("MAX_DROP_RATE is 1% (0.01)", 
         MAX_DROP_RATE == 0.01)
    
    # =========================================================================
    # PRD Section 12: Schema
    # =========================================================================
    print("\n📋 PRD Section 12: Normalized Data Schema")
    try:
        row = ParsedRow(
            report_date="2026-01-01",
            campaign_name="Test Campaign",
            ad_group_name="Test AdGroup",
            search_term="test keyword",
            impressions=100,
            clicks=10,
            spend=5.0,
            sales_7d=20.0
        )
        test("ParsedRow has all required fields", 
             row.report_date and row.campaign_name and row.search_term)
        test("ParsedRow has numeric defaults", 
             row.impressions == 100 and row.clicks == 10)
    except Exception as e:
        test("ParsedRow creation", False, str(e))
    
    # =========================================================================
    # PRD Section 13: Auditability
    # =========================================================================
    print("\n📋 PRD Section 13: Ingestion History & Auditability")
    try:
        event = IngestionEvent(
            ingestion_id=uuid4(),
            account_id=uuid4(),
            source=IngestionSource.EMAIL,
            status=IngestionStatus.RECEIVED,
            received_at=datetime.utcnow(),
            metadata={"sender": "test@example.com"}
        )
        test("IngestionEvent has ingestion_id", bool(event.ingestion_id))
        test("IngestionEvent has account_id", bool(event.account_id))
        test("IngestionEvent has source", event.source == IngestionSource.EMAIL)
        test("IngestionEvent has status", event.status == IngestionStatus.RECEIVED)
        test("IngestionEvent has metadata", "sender" in event.metadata)
    except Exception as e:
        test("IngestionEvent creation", False, str(e))
    
    # =========================================================================
    # PRD Section 14: State Machine
    # =========================================================================
    print("\n📋 PRD Section 14: State Machine & Transitions")
    test("RECEIVED in state machine", 'RECEIVED' in VALID_STATE_TRANSITIONS)
    test("PROCESSING in state machine", 'PROCESSING' in VALID_STATE_TRANSITIONS)
    test("COMPLETED in state machine", 'COMPLETED' in VALID_STATE_TRANSITIONS)
    test("FAILED in state machine", 'FAILED' in VALID_STATE_TRANSITIONS)
    test("QUARANTINE in state machine", 'QUARANTINE' in VALID_STATE_TRANSITIONS)
    
    test("RECEIVED -> PROCESSING valid", 
         'PROCESSING' in VALID_STATE_TRANSITIONS['RECEIVED'])
    test("PROCESSING -> COMPLETED valid", 
         'COMPLETED' in VALID_STATE_TRANSITIONS['PROCESSING'])
    test("PROCESSING -> QUARANTINE valid", 
         'QUARANTINE' in VALID_STATE_TRANSITIONS['PROCESSING'])
    test("PROCESSING -> FAILED valid", 
         'FAILED' in VALID_STATE_TRANSITIONS['PROCESSING'])
    test("FAILED -> PROCESSING (manual)", 
         'PROCESSING' in VALID_STATE_TRANSITIONS['FAILED'])
    test("QUARANTINE -> PROCESSING (manual)", 
         'PROCESSING' in VALID_STATE_TRANSITIONS['QUARANTINE'])
    test("COMPLETED is terminal", 
         len(VALID_STATE_TRANSITIONS['COMPLETED']) == 0)
    test("DUPLICATE_IGNORED is terminal", 
         len(VALID_STATE_TRANSITIONS['DUPLICATE_IGNORED']) == 0)
    
    test("Enum has can_transition() method", 
         hasattr(IngestionStatus, 'can_transition'))
    test("can_transition(RECEIVED, PROCESSING) = True", 
         IngestionStatus.can_transition(IngestionStatus.RECEIVED, IngestionStatus.PROCESSING))
    test("can_transition(COMPLETED, PROCESSING) = False", 
         not IngestionStatus.can_transition(IngestionStatus.COMPLETED, IngestionStatus.PROCESSING))
    
    # =========================================================================
    # PRD Section 14: Schedule Detection
    # =========================================================================
    print("\n📋 PRD Section 14: Schedule Detection")
    test("MIN_INGESTIONS_FOR_BASELINE = 3", 
         MIN_INGESTIONS_FOR_BASELINE == 3)
    test("LATE_THRESHOLD_HOURS = 12", 
         LATE_THRESHOLD_HOURS == 12)
    
    # =========================================================================
    # PRD Section 16: Alerting
    # =========================================================================
    print("\n📋 PRD Section 16: Alerting & Notifications")
    test("AlertSeverity.CRITICAL exists", 
         AlertSeverity.CRITICAL == "CRITICAL")
    test("AlertSeverity.HIGH exists", 
         AlertSeverity.HIGH == "HIGH")
    test("AlertSeverity.MEDIUM exists", 
         AlertSeverity.MEDIUM == "MEDIUM")
    
    # =========================================================================
    # PRD Section 18: Retention
    # =========================================================================
    print("\n📋 PRD Section 18: Data Retention")
    test("RAW_FILE_RETENTION_DAYS = 120", 
         RAW_FILE_RETENTION_DAYS == 120)
    
    # =========================================================================
    # Interfaces
    # =========================================================================
    print("\n📋 PRD Interfaces (Sections 7-11)")
    test("BaseAdapter has receive()", hasattr(BaseAdapter, 'receive'))
    test("BaseAdapter has acknowledge()", hasattr(BaseAdapter, 'acknowledge'))
    test("BaseValidator has validate()", hasattr(BaseValidator, 'validate'))
    test("BaseValidator has check_duplicate()", hasattr(BaseValidator, 'check_duplicate'))
    test("BaseStorage has put()", hasattr(BaseStorage, 'put'))
    test("BaseStorage has get()", hasattr(BaseStorage, 'get'))
    test("BaseStorage has delete()", hasattr(BaseStorage, 'delete'))
    test("BaseParser has parse()", hasattr(BaseParser, 'parse'))
    test("BaseParser has compute_fingerprint()", hasattr(BaseParser, 'compute_fingerprint'))
    test("BaseParser has REQUIRED_HEADERS", hasattr(BaseParser, 'REQUIRED_HEADERS'))
    
    # =========================================================================
    # Exceptions
    # =========================================================================
    print("\n📋 PRD Exceptions (Section 15)")
    test("AdapterError is IngestionError", 
         issubclass(AdapterError, IngestionError))
    test("ValidationError is IngestionError", 
         issubclass(ValidationError, IngestionError))
    test("ParseError has should_quarantine=True", 
         ParseError("test").should_quarantine)
    test("DuplicateError has should_quarantine=False", 
         not DuplicateError("test").should_quarantine)
    test("StateTransitionError exists", 
         StateTransitionError("A", "B") is not None)
    
    # =========================================================================
    # ParseResult Quarantine Logic
    # =========================================================================
    print("\n📋 PRD Section 11: Quarantine Logic")
    result_ok = ParseResult(success=True, total_rows=1000, dropped_rows=10, rows=[])
    result_bad = ParseResult(success=True, total_rows=1000, dropped_rows=11, rows=[])
    test("≤1% dropped -> no quarantine", not result_ok.should_quarantine)
    test(">1% dropped -> quarantine", result_bad.should_quarantine)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print("="*60)
    
    if ERRORS:
        print("\n⚠️  ERRORS:")
        for err in ERRORS:
            print(f"   - {err}")
    
    if FAILED == 0:
        print("\n🎉 ALL TESTS PASSED - Implementation matches PRD 1:1")
    else:
        print(f"\n❌ {FAILED} DEVIATIONS FROM PRD DETECTED")
    
    return FAILED == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
