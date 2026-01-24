"""
Ingestion V2 Exceptions
=======================
Custom exceptions for error handling and alerting.
PRD Reference: EMAIL_INGESTION_PRD.md Section 15, 16
"""


class IngestionError(Exception):
    """Base exception for all ingestion errors."""
    
    def __init__(self, message: str, should_quarantine: bool = False):
        super().__init__(message)
        self.should_quarantine = should_quarantine


class AdapterError(IngestionError):
    """
    Error during adapter phase (email extraction, API fetch).
    PRD Section 7: Failure Conditions
    """
    pass


class ValidationError(IngestionError):
    """
    Error during validation phase.
    PRD Section 9: Identity or structural validation failed.
    """
    pass


class StorageError(IngestionError):
    """
    Error during raw file storage.
    PRD Section 10: Storage layer failure.
    """
    pass


class ParseError(IngestionError):
    """
    Error during CSV parsing.
    PRD Section 11: Missing headers, type errors, etc.
    """
    
    def __init__(
        self, 
        message: str, 
        dropped_rows: int = 0, 
        total_rows: int = 0,
        warnings: list = None
    ):
        super().__init__(message, should_quarantine=True)
        self.dropped_rows = dropped_rows
        self.total_rows = total_rows
        self.warnings = warnings or []


class DuplicateError(IngestionError):
    """
    Duplicate file detected.
    PRD Section 14: Logged as DUPLICATE_IGNORED, no alerts.
    """
    
    def __init__(self, fingerprint: str):
        super().__init__(f"Duplicate file detected: {fingerprint}")
        self.fingerprint = fingerprint
        self.should_quarantine = False  # Not an error, just skip


class StateTransitionError(IngestionError):
    """
    Invalid state transition attempted.
    PRD Section 14: State Machine rules violated.
    """
    
    def __init__(self, from_status: str, to_status: str):
        super().__init__(
            f"Invalid state transition: {from_status} -> {to_status}"
        )
        self.from_status = from_status
        self.to_status = to_status
