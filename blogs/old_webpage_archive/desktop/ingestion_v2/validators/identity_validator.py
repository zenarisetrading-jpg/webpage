"""
Identity Validator
==================
Implements BaseValidator for account and sender validation.

PRD Reference: EMAIL_INGESTION_PRD.md Section 9

RULES (LOCKED for Phase 1):
- Sender: Must end with @amazon.com OR be in exact allowlist
- Fingerprint: hash(sender|subject|filename|size) - stored, not enforced
- DO NOT wildcard beyond @amazon.com
"""

import hashlib
import os
from typing import Optional
from uuid import UUID

from ..interfaces import BaseValidator
from ..models import IngestionPayload, ValidationResult
from ..exceptions import ValidationError, DuplicateError


class IdentityValidator(BaseValidator):
    """
    Validates account existence and sender authorization.
    
    Phase 1 Rules:
    - Sender must be @amazon.com or in exact allowlist
    - Fingerprint computed but not enforced for uniqueness
    """
    
    # RULE (LOCKED): Exact sender allowlist
    # Only @amazon.com suffix OR these exact addresses
    EXACT_ALLOWLIST = [
        "no-reply@amazon.com",
        "advertising-reports@amazon.com",
        "seller-reports@amazon.com",
        "reports@amazon.com",
        "noreply@amazon.com",
    ]
    
    def __init__(self, db_connection=None):
        """
        Initialize validator with optional database connection.
        
        Args:
            db_connection: Database connection for account lookup
                          (None for testing/Phase 1 hardcoded mode)
        """
        self._db = db_connection
    
    def is_sender_allowed(self, sender: str) -> bool:
        """
        Check if sender is allowed per PRD rules.
        
        RULE (LOCKED):
        - Allow if sender.endswith("@amazon.com")
        - OR sender in EXACT_ALLOWLIST
        - DO NOT wildcard beyond @amazon.com
        
        Args:
            sender: Email address of sender
            
        Returns:
            True if allowed, False otherwise
        """
        sender = sender.lower().strip()
        
        # Rule 1: Ends with @amazon.com
        if sender.endswith("@amazon.com"):
            return True
        
        # Rule 2: In exact allowlist
        if sender in self.EXACT_ALLOWLIST:
            return True
        
        return False
    
    def compute_fingerprint(
        self, 
        sender: str, 
        subject: str, 
        filename: str, 
        filesize: int
    ) -> str:
        """
        Compute source fingerprint for duplicate detection.
        
        RULE (LOCKED):
        - fingerprint = hash(sender|subject|filename|size)
        - Phase 1: Store only, do not enforce uniqueness
        - Phase 2: Will add uniqueness constraint
        
        Args:
            sender: Email sender address
            subject: Email subject line
            filename: Attachment filename
            filesize: Attachment size in bytes
            
        Returns:
            SHA256 hash as hex string
        """
        data = f"{sender.lower()}|{subject}|{filename}|{filesize}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _validate_file_structure(self, content: bytes) -> tuple:
        """
        Validate file is not empty and is UTF-8 readable.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not content or len(content) == 0:
            return False, "File is empty"
        
        try:
            # Try to decode as UTF-8
            content.decode('utf-8')
        except UnicodeDecodeError:
            return False, "File is not valid UTF-8"
        
        return True, None
    
    async def validate(self, payload: IngestionPayload) -> ValidationResult:
        """
        Validate the incoming payload.
        
        Checks:
        1. Sender is allowed (Amazon domain or allowlist)
        2. File is not empty and UTF-8 readable
        3. Compute fingerprint (store, don't enforce)
        
        Args:
            payload: Ingestion payload from adapter
            
        Returns:
            ValidationResult with valid=True/False
        """
        errors = []
        account_id = None
        
        # Check 1: Sender validation
        if not self.is_sender_allowed(payload.sender_email):
            errors.append(
                f"Sender not allowed: {payload.sender_email}. "
                f"Must be @amazon.com or in allowlist."
            )
        
        # Check 2: File structure
        is_valid_file, file_error = self._validate_file_structure(payload.file_content)
        if not is_valid_file:
            errors.append(file_error)
        
        # Compute fingerprint (always, for storage)
        fingerprint = self.compute_fingerprint(
            sender=payload.sender_email,
            subject=payload.subject or "",
            filename=payload.filename,
            filesize=len(payload.file_content)
        )
        
        # Phase 1: Account lookup is hardcoded
        # TODO Phase 2: Query database for account by UUID
        # account = await self._db.get_account(payload.account_uuid)
        # if not account:
        #     errors.append(f"Account not found: {payload.account_uuid}")
        
        # For now, trust the account_uuid from adapter (hardcoded to s2c_uae_test)
        # In Phase 2, we'll look up the actual UUID from database
        
        return ValidationResult(
            valid=len(errors) == 0,
            account_id=None,  # Phase 2: actual UUID from lookup
            errors=errors,
            is_duplicate=False,  # Phase 2: check fingerprint in DB
            source_fingerprint=fingerprint
        )
    
    async def check_duplicate(self, account_id: UUID, fingerprint: str) -> bool:
        """
        Check if this file has already been processed.
        
        Phase 1: Always returns False (no enforcement)
        Phase 2: Query ingestion_events_v2 for fingerprint
        
        Args:
            account_id: Account UUID
            fingerprint: Source fingerprint hash
            
        Returns:
            True if duplicate, False if new
        """
        # Phase 1: No duplicate enforcement
        # TODO Phase 2: Query database
        # result = await self._db.query(
        #     "SELECT 1 FROM ingestion_events_v2 "
        #     "WHERE account_id = $1 AND source_fingerprint = $2",
        #     account_id, fingerprint
        # )
        # return len(result) > 0
        
        return False
