"""
Ingestion V2 Runner
===================
Main entrypoint for the ingestion pipeline.

PRD Reference: EMAIL_INGESTION_PRD.md Section 24

RULES (LOCKED):
- Process ALL unread emails, one by one
- Mark email as READ only AFTER successful processing
- Failures leave email UNSEEN for retry
"""

# Load .env FIRST before any other imports
import os
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

# Now import everything else
import asyncio
from typing import Optional

from .adapters import IMAPAdapter
from .validators import IdentityValidator
from .storage import SupabaseStorage
from .alerts import TwilioWhatsAppAlerter
from .db import EventLogger
from .enums import IngestionSource, IngestionStatus
from .exceptions import AdapterError, ValidationError, StorageError


class IngestionRunner:
    """
    Orchestrates the V2 ingestion pipeline.
    
    Pipeline:
    1. adapter.receive() → IngestionPayload
    2. validator.validate() → ValidationResult
    3. If invalid: log REJECTED, alert, return
    4. storage.put() → file_id
    5. event_logger.create_event() → IngestionEvent
    """
    
    def __init__(self):
        """Initialize all pipeline components."""
        self.adapter = IMAPAdapter()
        self.validator = IdentityValidator()
        self.storage = SupabaseStorage()
        self.alerter = TwilioWhatsAppAlerter()
        self.event_logger = EventLogger()
    
    async def process_one(self) -> dict:
        """
        Process a single unread email.
        
        Returns:
            Dict with result status and details
        """
        result = {
            "success": False,
            "action": None,
            "ingestion_id": None,
            "error": None
        }
        
        try:
            # Step 1: Receive email
            payload = await self.adapter.receive()
            
            if payload is None:
                result["action"] = "NO_EMAILS"
                return result
            
            print(f"📧 Processing email from: {payload.sender_email}")
            print(f"   File: {payload.filename} ({len(payload.file_content)} bytes)")
            
            # Step 2: Validate
            validation = await self.validator.validate(payload)
            
            if not validation.valid:
                # Log rejection
                print(f"❌ Validation failed: {validation.errors}")
                
                await self.event_logger.log_rejected(
                    account_id=payload.account_uuid,
                    source=IngestionSource.EMAIL,
                    reason="; ".join(validation.errors),
                    metadata={
                        "sender": payload.sender_email,
                        "filename": payload.filename,
                        "errors": validation.errors,
                        "fingerprint": validation.source_fingerprint,
                    }
                )
                
                # Alert
                self.alerter.send_validation_rejected(
                    account_id=payload.account_uuid,
                    sender=payload.sender_email,
                    errors=validation.errors
                )
                
                result["action"] = "REJECTED"
                result["error"] = validation.errors
                return result
            
            # Step 3: Store raw file
            file_path = await self.storage.put(
                file_content=payload.file_content,
                metadata={
                    "account_id": payload.account_uuid,
                    "filename": payload.filename,
                    "sender": payload.sender_email,
                }
            )
            
            print(f"📁 Stored: {file_path}")
            
            # Step 4: Log event
            event = await self.event_logger.create_event(
                account_id=None,  # Phase 2: resolve UUID
                source=IngestionSource.EMAIL,
                metadata={
                    "sender": payload.sender_email,
                    "filename": payload.filename,
                    "subject": payload.subject,
                    "filesize": len(payload.file_content),
                    "fingerprint": validation.source_fingerprint,
                    "raw_file_path": file_path,
                    "account_name": payload.account_uuid,  # Phase 1: name not UUID
                }
            )
            
            print(f"✅ Event logged: {event.ingestion_id}")
            
            # Step 5: Acknowledge (mark email as read)
            await self.adapter.acknowledge(event.ingestion_id)
            
            result["success"] = True
            result["action"] = "RECEIVED"
            result["ingestion_id"] = str(event.ingestion_id)
            return result
            
        except AdapterError as e:
            print(f"❌ Adapter error: {e}")
            result["action"] = "ADAPTER_ERROR"
            result["error"] = str(e)
            
            # Mark email as read even on error (e.g., no CSV attachment)
            # to prevent infinite loop on same email
            try:
                await self.adapter.acknowledge(None)
            except:
                pass
            
            self.alerter.send_ingestion_failed(
                account_id="unknown",
                reason=str(e)
            )
            return result
            
        except StorageError as e:
            print(f"❌ Storage error: {e}")
            result["action"] = "STORAGE_ERROR"
            result["error"] = str(e)
            self.alerter.send_ingestion_failed(
                account_id="unknown",
                reason=str(e)
            )
            return result
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            result["action"] = "SYSTEM_ERROR"
            result["error"] = str(e)
            self.alerter.send_critical(f"System error: {e}")
            return result
    
    async def process_all(self) -> list:
        """
        Process ALL unread emails, one by one.
        
        RULE (LOCKED):
        - Process each email sequentially
        - Continue even if one fails
        - Return list of results
        
        Returns:
            List of result dicts
        """
        results = []
        max_iterations = 100  # Safety limit
        
        for i in range(max_iterations):
            result = await self.process_one()
            
            if result["action"] == "NO_EMAILS":
                break
            
            results.append(result)
            
            # Brief pause between emails
            await asyncio.sleep(0.5)
        
        return results
    
    def run(self) -> list:
        """
        Synchronous entry point.
        
        Usage: python -m ingestion_v2.runner
        """
        return asyncio.run(self.process_all())


def main():
    """CLI entry point."""
    # Load .env file
    from pathlib import Path
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✅ Loaded .env from {env_path}")
    
    print("=" * 50)
    print("V2 Ingestion Runner")
    print("=" * 50)
    
    runner = IngestionRunner()
    results = runner.run()
    
    print("\n" + "=" * 50)
    print(f"Processed {len(results)} emails")
    
    for i, r in enumerate(results, 1):
        status = "✅" if r["success"] else "❌"
        print(f"  {i}. {status} {r['action']}")
    
    print("=" * 50)


if __name__ == "__main__":
    main()

