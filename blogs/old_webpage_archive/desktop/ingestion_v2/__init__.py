"""
Ingestion V2 Module
===================
New ingestion infrastructure for Saddle AdPulse.
PRD Reference: EMAIL_INGESTION_PRD.md

This module is ISOLATED from the V1 codebase.
No imports from or modifications to existing code.
"""

# Load .env FIRST before anything else
import os
from pathlib import Path

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

__version__ = "0.1.0"
__all__ = [
    "IngestionSource",
    "IngestionStatus", 
    "IngestionPayload",
    "IngestionEvent",
    "BaseAdapter",
    "BaseValidator",
    "BaseStorage",
    "BaseParser",
]

from .enums import IngestionSource, IngestionStatus
from .models import IngestionPayload, IngestionEvent
from .interfaces import BaseAdapter, BaseValidator, BaseStorage, BaseParser

