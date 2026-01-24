"""
IMAP Email Adapter
==================
Implements BaseAdapter for IMAP email ingestion.

PRD Reference: EMAIL_INGESTION_PRD.md Section 7

RULES (LOCKED for Phase 1):
- Account ID: Email alias only (thasbihak+{id}@zenarise.org), hardcoded for now
- Runner: Process ALL unread, mark read only after success
- DO NOT parse account from subject
- DO NOT infer from file contents
"""

import imaplib
import email
import os
import re
from email.header import decode_header
from typing import Optional, List, Tuple
from datetime import datetime

from ..interfaces import BaseAdapter
from ..models import IngestionPayload
from ..enums import IngestionSource
from ..exceptions import AdapterError


class IMAPAdapter(BaseAdapter):
    """
    IMAP-based email adapter for ingesting Amazon advertising reports.
    
    Phase 1: Hardcoded to account 's2c_uae_test'
    Phase 2+: Parse account_id from email alias (thasbihak+{id}@zenarise.org)
    """
    
    source = IngestionSource.EMAIL
    
    # Phase 1: Hardcoded account ID
    # TODO Phase 2: Extract from email alias
    DEFAULT_ACCOUNT_ID = "s2c_uae_test"
    
    def __init__(
        self,
        host: str = None,
        user: str = None,
        password: str = None,
        folder: str = "INBOX"
    ):
        """
        Initialize IMAP adapter with credentials.
        
        Args:
            host: IMAP server host (default from IMAP_HOST env)
            user: IMAP username (default from IMAP_USER env)
            password: IMAP password (default from IMAP_PASSWORD env)
            folder: Mailbox folder to monitor (default: INBOX)
        """
        self.host = host or os.getenv("IMAP_HOST")
        self.user = user or os.getenv("IMAP_USER")
        self.password = password or os.getenv("IMAP_PASSWORD")
        self.folder = folder
        
        if not all([self.host, self.user, self.password]):
            raise AdapterError(
                "IMAP credentials not configured. "
                "Set IMAP_HOST, IMAP_USER, IMAP_PASSWORD environment variables."
            )
        
        self._connection: Optional[imaplib.IMAP4_SSL] = None
    
    def _connect(self) -> imaplib.IMAP4_SSL:
        """Establish IMAP connection."""
        try:
            conn = imaplib.IMAP4_SSL(self.host)
            conn.login(self.user, self.password)
            conn.select(self.folder)
            return conn
        except Exception as e:
            raise AdapterError(f"IMAP connection failed: {str(e)}")
    
    def _disconnect(self):
        """Close IMAP connection."""
        if self._connection:
            try:
                self._connection.close()
                self._connection.logout()
            except:
                pass
            self._connection = None
    
    def _decode_header_value(self, value: str) -> str:
        """Decode email header value."""
        if not value:
            return ""
        decoded_parts = decode_header(value)
        result = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                result.append(part.decode(charset or 'utf-8', errors='replace'))
            else:
                result.append(part)
        return ''.join(result)
    
    def _extract_account_id(self, to_addresses: List[str]) -> str:
        """
        Extract account ID from recipient email alias.
        
        RULE (LOCKED):
        - Format: thasbihak+{account_id}@zenarise.org
        - Phase 1: Return hardcoded 's2c_uae_test'
        - Phase 2: Parse from alias
        
        Args:
            to_addresses: List of recipient email addresses
            
        Returns:
            account_id string
        """
        # Phase 1: Hardcoded
        # TODO Phase 2: Uncomment and use this logic
        #
        # alias_pattern = re.compile(r'thasbihak\+([^@]+)@zenarise\.org', re.IGNORECASE)
        # for addr in to_addresses:
        #     match = alias_pattern.search(addr)
        #     if match:
        #         return match.group(1)
        # raise AdapterError("No account ID found in recipient email alias")
        
        return self.DEFAULT_ACCOUNT_ID
    
    def _extract_csv_attachment(self, msg) -> Tuple[bytes, str]:
        """
        Extract CSV attachment from email.
        
        Returns:
            Tuple of (file_content, filename)
            
        Raises:
            AdapterError: If no CSV attachment or multiple attachments
        """
        csv_attachments = []
        
        for part in msg.walk():
            content_disposition = part.get("Content-Disposition", "")
            if "attachment" in content_disposition:
                filename = part.get_filename()
                if filename:
                    filename = self._decode_header_value(filename)
                    if filename.lower().endswith('.csv'):
                        content = part.get_payload(decode=True)
                        csv_attachments.append((content, filename))
        
        if len(csv_attachments) == 0:
            raise AdapterError("No CSV attachment found in email")
        
        if len(csv_attachments) > 1:
            raise AdapterError(
                f"Multiple CSV attachments found ({len(csv_attachments)}). "
                "Expected exactly one."
            )
        
        return csv_attachments[0]
    
    async def receive(self) -> Optional[IngestionPayload]:
        """
        Fetch the next unread email with CSV attachment.
        
        RULE (LOCKED):
        - Process ALL unread emails, one by one
        - Email is NOT marked as read here (done in acknowledge())
        
        Returns:
            IngestionPayload if email found, None if no unread emails
        """
        try:
            self._connection = self._connect()
            
            # Search for unread emails
            status, messages = self._connection.search(None, "UNSEEN")
            if status != "OK":
                raise AdapterError(f"IMAP search failed: {status}")
            
            message_ids = messages[0].split()
            if not message_ids:
                self._disconnect()
                return None
            
            # Process first unread (oldest first)
            msg_id = message_ids[0]
            
            # Fetch email without marking as read (using PEEK)
            status, data = self._connection.fetch(msg_id, "(BODY.PEEK[])")
            if status != "OK":
                raise AdapterError(f"IMAP fetch failed: {status}")
            
            raw_email = data[0][1]
            msg = email.message_from_bytes(raw_email)
            
            # Extract metadata
            sender = self._decode_header_value(msg.get("From", ""))
            subject = self._decode_header_value(msg.get("Subject", ""))
            to_raw = msg.get("To", "")
            to_addresses = [addr.strip() for addr in to_raw.split(",")]
            
            # Extract sender email only (strip name)
            sender_match = re.search(r'<([^>]+)>', sender)
            sender_email = sender_match.group(1) if sender_match else sender
            
            # Extract account ID (Phase 1: hardcoded)
            account_id = self._extract_account_id(to_addresses)
            
            # Extract CSV attachment
            file_content, filename = self._extract_csv_attachment(msg)
            
            # Store message ID for acknowledge
            self._current_msg_id = msg_id
            
            return IngestionPayload(
                account_uuid=account_id,
                sender_email=sender_email.lower().strip(),
                file_content=file_content,
                filename=filename,
                source=IngestionSource.EMAIL,
                received_at=datetime.utcnow(),
                subject=subject
            )
            
        except AdapterError:
            self._disconnect()
            raise
        except Exception as e:
            self._disconnect()
            raise AdapterError(f"Failed to receive email: {str(e)}")
    
    async def acknowledge(self, ingestion_id) -> None:
        """
        Mark email as read after successful processing.
        
        RULE (LOCKED):
        - Only mark as read AFTER success
        - Failures leave email UNSEEN for retry
        """
        try:
            if self._connection and hasattr(self, '_current_msg_id'):
                self._connection.store(self._current_msg_id, '+FLAGS', '\\Seen')
        finally:
            self._disconnect()
    
    def get_unread_count(self) -> int:
        """Get count of unread emails (for monitoring)."""
        try:
            conn = self._connect()
            status, messages = conn.search(None, "UNSEEN")
            count = len(messages[0].split()) if messages[0] else 0
            conn.close()
            conn.logout()
            return count
        except:
            return -1
