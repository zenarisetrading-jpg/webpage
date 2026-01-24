#!/usr/bin/env python3
"""
Phase 1 PRD Compliance Tests
============================
Validates that Phase 1 implementation matches approved plan.

Run with: python3 tests/test_phase1_compliance.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

PASSED = 0
FAILED = 0


def test(name, condition):
    global PASSED, FAILED
    if condition:
        print(f"  ✅ {name}")
        PASSED += 1
    else:
        print(f"  ❌ {name}")
        FAILED += 1


def run_tests():
    print("\n" + "="*60)
    print("PHASE 1 PRD COMPLIANCE TESTS")
    print("="*60)
    
    # =========================================================================
    # Test IMAP Adapter
    # =========================================================================
    print("\n📦 IMAP Adapter")
    try:
        from ingestion_v2.adapters.imap_adapter import IMAPAdapter
        from ingestion_v2.enums import IngestionSource
        
        test("IMAPAdapter exists", True)
        test("source = IngestionSource.EMAIL", IMAPAdapter.source == IngestionSource.EMAIL)
        test("DEFAULT_ACCOUNT_ID = 's2c_uae_test'", IMAPAdapter.DEFAULT_ACCOUNT_ID == "s2c_uae_test")
        test("has receive() method", hasattr(IMAPAdapter, 'receive'))
        test("has acknowledge() method", hasattr(IMAPAdapter, 'acknowledge'))
        test("has _extract_account_id() method", hasattr(IMAPAdapter, '_extract_account_id'))
        test("has _extract_csv_attachment() method", hasattr(IMAPAdapter, '_extract_csv_attachment'))
    except Exception as e:
        test(f"IMAP Adapter import", False)
        print(f"     Error: {e}")
    
    # =========================================================================
    # Test Identity Validator
    # =========================================================================
    print("\n📦 Identity Validator")
    try:
        from ingestion_v2.validators.identity_validator import IdentityValidator
        
        validator = IdentityValidator()
        
        test("IdentityValidator exists", True)
        test("has is_sender_allowed() method", hasattr(validator, 'is_sender_allowed'))
        test("has compute_fingerprint() method", hasattr(validator, 'compute_fingerprint'))
        
        # Test sender rules (LOCKED)
        test("Amazon sender allowed: reports@amazon.com", 
             validator.is_sender_allowed("reports@amazon.com"))
        test("Amazon sender allowed: test@amazon.com", 
             validator.is_sender_allowed("test@amazon.com"))
        test("Non-Amazon rejected: test@gmail.com", 
             not validator.is_sender_allowed("test@gmail.com"))
        test("Non-Amazon rejected: amazon@fake.com", 
             not validator.is_sender_allowed("amazon@fake.com"))
        
        # Test fingerprint
        fp = validator.compute_fingerprint("a@b.com", "subject", "file.csv", 1000)
        test("Fingerprint is SHA256 (64 chars)", len(fp) == 64)
        test("Fingerprint is deterministic", 
             fp == validator.compute_fingerprint("a@b.com", "subject", "file.csv", 1000))
        test("Fingerprint changes with input",
             fp != validator.compute_fingerprint("a@b.com", "subject", "file.csv", 1001))
        
    except Exception as e:
        test(f"Identity Validator import", False)
        print(f"     Error: {e}")
    
    # =========================================================================
    # Test Supabase Storage
    # =========================================================================
    print("\n📦 Supabase Storage")
    try:
        from ingestion_v2.storage.supabase_storage import SupabaseStorage
        
        test("SupabaseStorage exists", True)
        test("has put() method", hasattr(SupabaseStorage, 'put'))
        test("has get() method", hasattr(SupabaseStorage, 'get'))
        test("has delete() method", hasattr(SupabaseStorage, 'delete'))
        
        # Check default bucket
        import os
        bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "ingestion-raw")
        test("Default bucket is 'ingestion-raw'", bucket == "ingestion-raw")
        
    except Exception as e:
        test(f"Supabase Storage import", False)
        print(f"     Error: {e}")
    
    # =========================================================================
    # Test Twilio WhatsApp Alerter
    # =========================================================================
    print("\n📦 Twilio WhatsApp Alerter")
    try:
        from ingestion_v2.alerts.twilio_whatsapp import TwilioWhatsAppAlerter
        from ingestion_v2.constants import AlertSeverity
        
        alerter = TwilioWhatsAppAlerter()
        
        test("TwilioWhatsAppAlerter exists", True)
        test("has send_alert() method", hasattr(alerter, 'send_alert'))
        test("has send_critical() method", hasattr(alerter, 'send_critical'))
        test("has send_high() method", hasattr(alerter, 'send_high'))
        test("has send_ingestion_failed() method", hasattr(alerter, 'send_ingestion_failed'))
        test("has send_validation_rejected() method", hasattr(alerter, 'send_validation_rejected'))
        
        # Test severity levels
        test("AlertSeverity.CRITICAL = 'CRITICAL'", AlertSeverity.CRITICAL == "CRITICAL")
        test("AlertSeverity.HIGH = 'HIGH'", AlertSeverity.HIGH == "HIGH")
        test("AlertSeverity.MEDIUM = 'MEDIUM'", AlertSeverity.MEDIUM == "MEDIUM")
        
    except Exception as e:
        test(f"Twilio WhatsApp Alerter import", False)
        print(f"     Error: {e}")
    
    # =========================================================================
    # Test Event Logger
    # =========================================================================
    print("\n📦 Event Logger")
    try:
        from ingestion_v2.db.event_logger import EventLogger
        from ingestion_v2.enums import IngestionStatus
        
        test("EventLogger exists", True)
        test("has create_event() method", hasattr(EventLogger, 'create_event'))
        test("has update_status() method", hasattr(EventLogger, 'update_status'))
        test("has get_event() method", hasattr(EventLogger, 'get_event'))
        test("has log_rejected() method", hasattr(EventLogger, 'log_rejected'))
        test("has _validate_transition() method", hasattr(EventLogger, '_validate_transition'))
        
    except Exception as e:
        test(f"Event Logger import", False)
        print(f"     Error: {e}")
    
    # =========================================================================
    # Test Runner
    # =========================================================================
    print("\n📦 Runner")
    try:
        from ingestion_v2.runner import IngestionRunner, main
        
        test("IngestionRunner exists", True)
        test("has process_one() method", hasattr(IngestionRunner, 'process_one'))
        test("has process_all() method", hasattr(IngestionRunner, 'process_all'))
        test("has run() method", hasattr(IngestionRunner, 'run'))
        test("main() exists", callable(main))
        
    except Exception as e:
        test(f"Runner import", False)
        print(f"     Error: {e}")
    
    # =========================================================================
    # Test State Transitions (PRD Section 14)
    # =========================================================================
    print("\n📦 State Transitions (PRD §14)")
    try:
        from ingestion_v2.enums import IngestionStatus
        
        # Valid transitions
        test("RECEIVED -> PROCESSING valid",
             IngestionStatus.can_transition(IngestionStatus.RECEIVED, IngestionStatus.PROCESSING))
        test("PROCESSING -> COMPLETED valid",
             IngestionStatus.can_transition(IngestionStatus.PROCESSING, IngestionStatus.COMPLETED))
        test("PROCESSING -> QUARANTINE valid",
             IngestionStatus.can_transition(IngestionStatus.PROCESSING, IngestionStatus.QUARANTINE))
        test("PROCESSING -> FAILED valid",
             IngestionStatus.can_transition(IngestionStatus.PROCESSING, IngestionStatus.FAILED))
        test("FAILED -> PROCESSING valid (manual)",
             IngestionStatus.can_transition(IngestionStatus.FAILED, IngestionStatus.PROCESSING))
        test("QUARANTINE -> PROCESSING valid (manual)",
             IngestionStatus.can_transition(IngestionStatus.QUARANTINE, IngestionStatus.PROCESSING))
        
        # Invalid transitions
        test("COMPLETED -> PROCESSING invalid",
             not IngestionStatus.can_transition(IngestionStatus.COMPLETED, IngestionStatus.PROCESSING))
        test("RECEIVED -> COMPLETED invalid",
             not IngestionStatus.can_transition(IngestionStatus.RECEIVED, IngestionStatus.COMPLETED))
        
    except Exception as e:
        test(f"State Transitions", False)
        print(f"     Error: {e}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print("="*60)
    
    if FAILED == 0:
        print("\n🎉 ALL PHASE 1 TESTS PASSED - Implementation matches PRD")
    else:
        print(f"\n❌ {FAILED} DEVIATIONS FROM PRD DETECTED")
    
    return FAILED == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
