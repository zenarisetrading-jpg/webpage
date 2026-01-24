"""
Supabase Storage Adapter
========================
Implements BaseStorage for raw file storage in Supabase.

PRD Reference: EMAIL_INGESTION_PRD.md Section 10

RULES:
- Bucket: ingestion-raw
- Path format: {account_id}/{YYYY-MM-DD}/{uuid}.csv
- Business logic must NOT reference provider directly (uses interface)
"""

import os
from datetime import datetime
from typing import Dict, Any
from uuid import uuid4

from ..interfaces import BaseStorage
from ..exceptions import StorageError


class SupabaseStorage(BaseStorage):
    """
    Supabase Storage implementation for raw CSV files.
    
    Bucket: ingestion-raw
    """
    
    def __init__(
        self,
        url: str = None,
        key: str = None,
        bucket: str = None
    ):
        """
        Initialize Supabase storage client.
        
        Args:
            url: Supabase project URL (default from SUPABASE_URL env)
            key: Supabase service key (default from SUPABASE_SERVICE_KEY env)
            bucket: Storage bucket name (default from SUPABASE_STORAGE_BUCKET env)
        """
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_SERVICE_KEY")
        self.bucket = bucket or os.getenv("SUPABASE_STORAGE_BUCKET", "ingestion-raw")
        
        if not all([self.url, self.key]):
            raise StorageError(
                "Supabase credentials not configured. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables."
            )
        
        self._client = None
    
    def _get_client(self):
        """Get or create Supabase client."""
        if self._client is None:
            try:
                from supabase import create_client
                self._client = create_client(self.url, self.key)
            except ImportError:
                raise StorageError(
                    "supabase-py not installed. "
                    "Run: pip install supabase"
                )
            except Exception as e:
                raise StorageError(f"Failed to create Supabase client: {str(e)}")
        return self._client
    
    def _generate_path(self, account_id: str, filename: str) -> str:
        """
        Generate storage path for file.
        
        Format: {account_id}/{YYYY-MM-DD}/{uuid}.csv
        
        Args:
            account_id: Account identifier
            filename: Original filename (for extension)
            
        Returns:
            Storage path string
        """
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        file_id = str(uuid4())
        extension = filename.split('.')[-1] if '.' in filename else 'csv'
        return f"{account_id}/{date_str}/{file_id}.{extension}"
    
    async def put(
        self, 
        file_content: bytes, 
        metadata: Dict[str, Any]
    ) -> str:
        """
        Store a raw file to Supabase Storage.
        
        Args:
            file_content: Raw bytes of the CSV file
            metadata: Dict with account_id, filename, etc.
            
        Returns:
            file_id: Storage path for retrieval
            
        Raises:
            StorageError: If upload fails
        """
        try:
            account_id = metadata.get("account_id", "unknown")
            filename = metadata.get("filename", "report.csv")
            
            # Generate storage path
            file_path = self._generate_path(account_id, filename)
            
            # Upload to Supabase Storage
            client = self._get_client()
            result = client.storage.from_(self.bucket).upload(
                file_path,
                file_content,
                file_options={"content-type": "text/csv"}
            )
            
            # Check for errors
            if hasattr(result, 'error') and result.error:
                raise StorageError(f"Upload failed: {result.error}")
            
            return file_path
            
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to store file: {str(e)}")
    
    async def get(self, file_id: str) -> bytes:
        """
        Retrieve a raw file from Supabase Storage.
        
        Args:
            file_id: Storage path returned by put()
            
        Returns:
            Raw file content as bytes
            
        Raises:
            StorageError: If download fails
        """
        try:
            client = self._get_client()
            result = client.storage.from_(self.bucket).download(file_id)
            
            if result is None:
                raise StorageError(f"File not found: {file_id}")
            
            return result
            
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to retrieve file: {str(e)}")
    
    async def delete(self, file_id: str) -> bool:
        """
        Delete a raw file from Supabase Storage.
        
        Used for retention policy cleanup.
        
        Args:
            file_id: Storage path to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            client = self._get_client()
            result = client.storage.from_(self.bucket).remove([file_id])
            
            return True
            
        except Exception as e:
            # Log but don't fail on delete errors
            return False
    
    def get_public_url(self, file_id: str) -> str:
        """
        Get public URL for a file (if bucket is public).
        
        Note: For internal use only, users should NOT see raw files.
        
        Args:
            file_id: Storage path
            
        Returns:
            Public URL string
        """
        try:
            client = self._get_client()
            result = client.storage.from_(self.bucket).get_public_url(file_id)
            return result
        except:
            return None
