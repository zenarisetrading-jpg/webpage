"""
Event Logger
============
Implements BaseEventLogger for ingestion_events_v2 table.

PRD Reference: EMAIL_INGESTION_PRD.md Section 13, 14

RULES:
- Logs to ingestion_events_v2 (V2 table isolation)
- Validates state transitions per PRD Section 14
- Stores fingerprint but does not enforce uniqueness (Phase 1)
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID, uuid4

from ..interfaces import BaseEventLogger
from ..models import IngestionEvent
from ..enums import IngestionSource, IngestionStatus
from ..exceptions import StateTransitionError
from ..constants import VALID_STATE_TRANSITIONS


class EventLogger(BaseEventLogger):
    """
    Logs ingestion events to ingestion_events_v2 table.
    
    Uses Supabase for database operations.
    """
    
    def __init__(self, url: str = None, key: str = None):
        """
        Initialize event logger with Supabase connection.
        
        Args:
            url: Supabase project URL (default from SUPABASE_URL env)
            key: Supabase service key (default from SUPABASE_SERVICE_KEY env)
        """
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_SERVICE_KEY")
        self._client = None
    
    def _get_client(self):
        """Get or create Supabase client."""
        if self._client is None:
            try:
                from supabase import create_client
                self._client = create_client(self.url, self.key)
            except ImportError:
                raise RuntimeError("supabase-py not installed")
        return self._client
    
    def _validate_transition(
        self, 
        from_status: IngestionStatus, 
        to_status: IngestionStatus
    ) -> bool:
        """
        Validate state transition per PRD Section 14.
        
        Valid transitions:
            RECEIVED -> PROCESSING
            PROCESSING -> COMPLETED | QUARANTINE | FAILED
            FAILED -> PROCESSING (manual only)
            QUARANTINE -> PROCESSING (manual only)
        
        Returns:
            True if valid, raises StateTransitionError otherwise
        """
        if not IngestionStatus.can_transition(from_status, to_status):
            raise StateTransitionError(from_status.value, to_status.value)
        return True
    
    async def create_event(
        self,
        account_id: UUID,
        source: IngestionSource,
        metadata: Dict[str, Any]
    ) -> IngestionEvent:
        """
        Create a new ingestion event with status=RECEIVED.
        
        Args:
            account_id: Account UUID
            source: IngestionSource (EMAIL, API, MANUAL)
            metadata: Dict with sender, filename, etc.
            
        Returns:
            Created IngestionEvent
        """
        ingestion_id = uuid4()
        now = datetime.utcnow()
        
        event_data = {
            "ingestion_id": str(ingestion_id),
            "account_id": str(account_id) if account_id else None,
            "source": source.value,
            "status": IngestionStatus.RECEIVED.value,
            "received_at": now.isoformat(),
            "metadata": metadata,
            "source_fingerprint": metadata.get("fingerprint"),
            "raw_file_path": metadata.get("raw_file_path"),
        }
        
        try:
            client = self._get_client()
            result = client.table("ingestion_events_v2").insert(event_data).execute()
            
            return IngestionEvent(
                ingestion_id=ingestion_id,
                account_id=account_id,
                source=source,
                status=IngestionStatus.RECEIVED,
                received_at=now,
                metadata=metadata,
                source_fingerprint=metadata.get("fingerprint"),
                raw_file_path=metadata.get("raw_file_path"),
            )
        except Exception as e:
            # If DB fails, return event anyway for logging
            print(f"Warning: Failed to log event to DB: {e}")
            return IngestionEvent(
                ingestion_id=ingestion_id,
                account_id=account_id,
                source=source,
                status=IngestionStatus.RECEIVED,
                received_at=now,
                metadata=metadata,
            )
    
    async def update_status(
        self,
        ingestion_id: UUID,
        status: IngestionStatus,
        failure_reason: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the status of an ingestion event.
        
        Validates state transition per PRD Section 14.
        
        Args:
            ingestion_id: Event UUID
            status: New status
            failure_reason: Reason for failure (if applicable)
            metadata_updates: Additional metadata to merge
        """
        try:
            client = self._get_client()
            
            # Get current event to validate transition
            current = await self.get_event(ingestion_id)
            if current:
                self._validate_transition(current.status, status)
            
            # Prepare update data
            update_data = {
                "status": status.value,
                "processed_at": datetime.utcnow().isoformat(),
            }
            
            if failure_reason:
                update_data["failure_reason"] = failure_reason
            
            # Execute update
            client.table("ingestion_events_v2").update(update_data).eq(
                "ingestion_id", str(ingestion_id)
            ).execute()
            
        except StateTransitionError:
            raise
        except Exception as e:
            print(f"Warning: Failed to update event status: {e}")
    
    async def get_event(self, ingestion_id: UUID) -> Optional[IngestionEvent]:
        """
        Retrieve an ingestion event by ID.
        
        Args:
            ingestion_id: Event UUID
            
        Returns:
            IngestionEvent or None if not found
        """
        try:
            client = self._get_client()
            result = client.table("ingestion_events_v2").select("*").eq(
                "ingestion_id", str(ingestion_id)
            ).execute()
            
            if not result.data:
                return None
            
            row = result.data[0]
            return IngestionEvent(
                ingestion_id=UUID(row["ingestion_id"]),
                account_id=UUID(row["account_id"]) if row.get("account_id") else None,
                source=IngestionSource(row["source"]),
                status=IngestionStatus(row["status"]),
                received_at=datetime.fromisoformat(row["received_at"]),
                processed_at=datetime.fromisoformat(row["processed_at"]) if row.get("processed_at") else None,
                failure_reason=row.get("failure_reason"),
                metadata=row.get("metadata", {}),
                source_fingerprint=row.get("source_fingerprint"),
                raw_file_path=row.get("raw_file_path"),
            )
        except Exception as e:
            print(f"Warning: Failed to get event: {e}")
            return None
    
    async def log_rejected(
        self,
        account_id: str,
        source: IngestionSource,
        reason: str,
        metadata: Dict[str, Any]
    ) -> IngestionEvent:
        """
        Convenience method to log a REJECTED event.
        
        Args:
            account_id: Account identifier
            source: IngestionSource
            reason: Rejection reason
            metadata: Event metadata
            
        Returns:
            Created event
        """
        metadata["failure_reason"] = reason
        event = await self.create_event(
            account_id=None,  # May not have valid UUID
            source=source,
            metadata=metadata
        )
        
        # Update to FAILED status
        try:
            await self.update_status(
                event.ingestion_id,
                IngestionStatus.FAILED,
                failure_reason=reason
            )
        except:
            pass
        
        return event
