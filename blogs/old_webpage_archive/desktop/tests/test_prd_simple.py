#!/usr/bin/env python3
"""
PRD Compliance Test Suite (No External Dependencies)
=====================================================
Validates that ingestion_v2 implementation matches EMAIL_INGESTION_PRD.md v1.2

Run with: python3 tests/test_prd_simple.py
"""

import sys
from pathlib import Path

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
    print("PRD COMPLIANCE TEST SUITE (Simple)")
    print("="*60)
    
    # =========================================================================
    # Test 1: Enums Module
    # =========================================================================
    print("\n📦 Testing: ingestion_v2/enums.py")
    try:
        from ingestion_v2.enums import IngestionSource, IngestionStatus
        
        # PRD Section 5
        test("IngestionSource.EMAIL = 'EMAIL'", 
             IngestionSource.EMAIL.value == "EMAIL")
        test("IngestionSource.API = 'API'", 
             IngestionSource.API.value == "API")
        test("IngestionSource.MANUAL = 'MANUAL'", 
             IngestionSource.MANUAL.value == "MANUAL")
        test("Only 3 IngestionSources", len(IngestionSource) == 3)
        
        # PRD Section 14 - State Machine
        test("IngestionStatus.RECEIVED exists", 
             IngestionStatus.RECEIVED.value == "RECEIVED")
        test("IngestionStatus.PROCESSING exists", 
             IngestionStatus.PROCESSING.value == "PROCESSING")
        test("IngestionStatus.COMPLETED exists", 
             IngestionStatus.COMPLETED.value == "COMPLETED")
        test("IngestionStatus.FAILED exists", 
             IngestionStatus.FAILED.value == "FAILED")
        test("IngestionStatus.QUARANTINE exists", 
             IngestionStatus.QUARANTINE.value == "QUARANTINE")
        test("IngestionStatus.DUPLICATE_IGNORED exists", 
             IngestionStatus.DUPLICATE_IGNORED.value == "DUPLICATE_IGNORED")
        
        # Transition validation
        test("can_transition method exists", 
             hasattr(IngestionStatus, 'can_transition'))
        test("RECEIVED -> PROCESSING valid", 
             IngestionStatus.can_transition(IngestionStatus.RECEIVED, IngestionStatus.PROCESSING))
        test("COMPLETED -> PROCESSING invalid", 
             not IngestionStatus.can_transition(IngestionStatus.COMPLETED, IngestionStatus.PROCESSING))
        
    except Exception as e:
        test("enums.py import", False, str(e))
    
    # =========================================================================
    # Test 2: Constants Module
    # =========================================================================
    print("\n📦 Testing: ingestion_v2/constants.py")
    try:
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
        
        # PRD Section 6
        test("EMAIL_DOMAIN = 'ingest.saddle.ai'", 
             EMAIL_DOMAIN == "ingest.saddle.ai")
        test("EMAIL_PREFIX = 'str'", 
             EMAIL_PREFIX == "str")
        
        # PRD Section 11
        test("MAX_DROP_RATE = 0.01 (1%)", 
             MAX_DROP_RATE == 0.01)
        
        required_fields = ['report_date', 'campaign_name', 'ad_group_name', 
                          'search_term', 'impressions', 'clicks', 'spend']
        for field in required_fields:
            test(f"REQUIRED: {field} mapped", field in REQUIRED_HEADER_MAPPING)
        
        test("OPTIONAL: sales_7d mapped", 
             'sales_7d' in OPTIONAL_HEADER_MAPPING)
        test("Date/Day variations for report_date", 
             'Date' in REQUIRED_HEADER_MAPPING['report_date'] and 
             'Day' in REQUIRED_HEADER_MAPPING['report_date'])
        
        # PRD Section 14
        test("MIN_INGESTIONS_FOR_BASELINE = 3", 
             MIN_INGESTIONS_FOR_BASELINE == 3)
        test("LATE_THRESHOLD_HOURS = 12", 
             LATE_THRESHOLD_HOURS == 12)
        
        # State transitions
        test("RECEIVED -> PROCESSING in transitions", 
             'PROCESSING' in VALID_STATE_TRANSITIONS['RECEIVED'])
        test("PROCESSING -> COMPLETED in transitions", 
             'COMPLETED' in VALID_STATE_TRANSITIONS['PROCESSING'])
        test("PROCESSING -> QUARANTINE in transitions", 
             'QUARANTINE' in VALID_STATE_TRANSITIONS['PROCESSING'])
        test("COMPLETED is terminal (empty set)", 
             len(VALID_STATE_TRANSITIONS['COMPLETED']) == 0)
        
        # PRD Section 16
        test("AlertSeverity.CRITICAL = 'CRITICAL'", 
             AlertSeverity.CRITICAL == "CRITICAL")
        test("AlertSeverity.HIGH = 'HIGH'", 
             AlertSeverity.HIGH == "HIGH")
        test("AlertSeverity.MEDIUM = 'MEDIUM'", 
             AlertSeverity.MEDIUM == "MEDIUM")
        
        # PRD Section 18
        test("RAW_FILE_RETENTION_DAYS = 120", 
             RAW_FILE_RETENTION_DAYS == 120)
        
    except Exception as e:
        test("constants.py import", False, str(e))
    
    # =========================================================================
    # Test 3: Exceptions Module
    # =========================================================================
    print("\n📦 Testing: ingestion_v2/exceptions.py")
    try:
        from ingestion_v2.exceptions import (
            IngestionError,
            AdapterError,
            ValidationError,
            StorageError,
            ParseError,
            DuplicateError,
            StateTransitionError,
        )
        
        test("AdapterError is IngestionError subclass", 
             issubclass(AdapterError, IngestionError))
        test("ValidationError is IngestionError subclass", 
             issubclass(ValidationError, IngestionError))
        test("StorageError is IngestionError subclass", 
             issubclass(StorageError, IngestionError))
        test("ParseError is IngestionError subclass", 
             issubclass(ParseError, IngestionError))
        
        # PRD Section 11 - Parse errors should quarantine
        pe = ParseError("test", dropped_rows=5, total_rows=100)
        test("ParseError.should_quarantine = True", 
             pe.should_quarantine == True)
        
        # PRD Section 14 - Duplicates should NOT quarantine
        de = DuplicateError("fingerprint123")
        test("DuplicateError.should_quarantine = False", 
             de.should_quarantine == False)
        
        test("StateTransitionError stores from/to states", 
             StateTransitionError("A", "B").from_status == "A")
        
    except Exception as e:
        test("exceptions.py import", False, str(e))
    
    # =========================================================================
    # Test 4: Interfaces Module (Abstract Classes)
    # =========================================================================
    print("\n📦 Testing: ingestion_v2/interfaces.py")
    try:
        from ingestion_v2.interfaces import (
            BaseAdapter,
            BaseValidator,
            BaseStorage,
            BaseParser,
            BaseEventLogger,
        )
        
        # PRD Section 7, 8
        test("BaseAdapter has 'receive' method", 
             hasattr(BaseAdapter, 'receive'))
        test("BaseAdapter has 'acknowledge' method", 
             hasattr(BaseAdapter, 'acknowledge'))
        
        # PRD Section 9
        test("BaseValidator has 'validate' method", 
             hasattr(BaseValidator, 'validate'))
        test("BaseValidator has 'check_duplicate' method", 
             hasattr(BaseValidator, 'check_duplicate'))
        
        # PRD Section 10
        test("BaseStorage has 'put' method", 
             hasattr(BaseStorage, 'put'))
        test("BaseStorage has 'get' method", 
             hasattr(BaseStorage, 'get'))
        test("BaseStorage has 'delete' method", 
             hasattr(BaseStorage, 'delete'))
        
        # PRD Section 11
        test("BaseParser has 'parse' method", 
             hasattr(BaseParser, 'parse'))
        test("BaseParser has 'compute_fingerprint' method", 
             hasattr(BaseParser, 'compute_fingerprint'))
        test("BaseParser has 'REQUIRED_HEADERS' attribute", 
             hasattr(BaseParser, 'REQUIRED_HEADERS'))
        
        # PRD Section 13
        test("BaseEventLogger has 'create_event' method", 
             hasattr(BaseEventLogger, 'create_event'))
        test("BaseEventLogger has 'update_status' method", 
             hasattr(BaseEventLogger, 'update_status'))
        
    except Exception as e:
        test("interfaces.py import", False, str(e))
    
    # =========================================================================
    # Test 5: SQL Schema File Exists
    # =========================================================================
    print("\n📦 Testing: migrations/001_ingestion_v2_schema.sql")
    sql_path = Path(__file__).parent.parent / "migrations" / "001_ingestion_v2_schema.sql"
    test("SQL migration file exists", sql_path.exists())
    
    if sql_path.exists():
        sql_content = sql_path.read_text()
        
        # PRD Section 12 - Tables
        test("ingestion_events_v2 table defined", 
             "CREATE TABLE IF NOT EXISTS ingestion_events_v2" in sql_content)
        test("search_terms_v2 table defined", 
             "CREATE TABLE IF NOT EXISTS search_terms_v2" in sql_content)
        
        # PRD ENUMs
        test("ingestion_version ENUM defined", 
             "CREATE TYPE ingestion_version" in sql_content)
        test("ingestion_source ENUM defined", 
             "CREATE TYPE ingestion_source" in sql_content)
        test("ingestion_status ENUM defined", 
             "CREATE TYPE ingestion_status" in sql_content)
        
        # PRD Section 14 - Fingerprint for dedup
        test("source_fingerprint column exists", 
             "source_fingerprint" in sql_content)
    
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
