"""
Ingestion V2 Enums
==================
PRD Reference: EMAIL_INGESTION_PRD.md Section 5, 13
"""

from enum import Enum


class IngestionSource(str, Enum):
    """
    Source of the ingestion event.
    PRD Section 5: Supported Ingestion Sources
    """
    EMAIL = "EMAIL"
    API = "API"
    MANUAL = "MANUAL"


class IngestionStatus(str, Enum):
    """
    Status of an ingestion event through the pipeline.
    PRD Section 14: State Machine
    
    Valid transitions:
        RECEIVED -> PROCESSING
        PROCESSING -> COMPLETED
        PROCESSING -> QUARANTINE
        PROCESSING -> FAILED
        FAILED -> PROCESSING (manual only)
        QUARANTINE -> PROCESSING (manual only)
    """
    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    QUARANTINE = "QUARANTINE"
    DUPLICATE_IGNORED = "DUPLICATE_IGNORED"
    
    @classmethod
    def can_transition(cls, from_status: "IngestionStatus", to_status: "IngestionStatus") -> bool:
        """Check if a state transition is valid per PRD."""
        valid_transitions = {
            cls.RECEIVED: {cls.PROCESSING},
            cls.PROCESSING: {cls.COMPLETED, cls.QUARANTINE, cls.FAILED},
            cls.FAILED: {cls.PROCESSING},  # Manual only
            cls.QUARANTINE: {cls.PROCESSING},  # Manual only
            cls.COMPLETED: set(),  # Terminal state
            cls.DUPLICATE_IGNORED: set(),  # Terminal state
        }
        return to_status in valid_transitions.get(from_status, set())
