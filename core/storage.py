"""
Storage handler for Vercel Blob Storage.
Handles image uploads with local fallback.
"""

import os
import requests
from typing import Optional


class StorageHandler:
    """Handles file uploads to Vercel Blob Storage."""
    
    def __init__(self):
        self.token = os.getenv("BLOB_READ_WRITE_TOKEN")
        self.api_url = "https://blob.vercel-storage.com"
        self.uploads_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
        
        # Create uploads directory for local fallback
        os.makedirs(self.uploads_dir, exist_ok=True)
        
        if self.token:
            print("✓ Vercel Blob Storage configured")
        else:
            print("⚠ BLOB_READ_WRITE_TOKEN not set - using local storage")
    
    @property
    def is_configured(self) -> bool:
        """Check if Vercel Blob is configured."""
        return self.token is not None

    def upload_file(self, file_bytes: bytes, filename: str) -> Optional[str]:
        """
        Upload a file to Vercel Blob Storage.
        Falls back to local storage if not configured.
        
        Args:
            file_bytes: Raw file bytes
            filename: Desired filename
            
        Returns:
            URL of the uploaded file or local path
        """
        if not self.is_configured:
            return self._save_locally(file_bytes, filename)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/octet-stream",
                "x-api-version": "7",
            }
            
            # Vercel Blob PUT API
            response = requests.put(
                f"{self.api_url}/{filename}",
                headers=headers,
                data=file_bytes,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("url")
            else:
                print(f"⚠ Blob upload failed ({response.status_code}): {response.text}")
                return self._save_locally(file_bytes, filename)
                
        except Exception as e:
            print(f"⚠ Blob upload error: {e}")
            return self._save_locally(file_bytes, filename)
    
    def _save_locally(self, file_bytes: bytes, filename: str) -> str:
        """Save file to local uploads directory."""
        local_path = os.path.join(self.uploads_dir, filename)
        with open(local_path, "wb") as f:
            f.write(file_bytes)
        return f"/uploads/{filename}"
    
    def delete_file(self, url: str) -> bool:
        """
        Delete a file from Vercel Blob Storage.
        
        Args:
            url: URL of the file to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.is_configured:
            # For local files, try to delete
            if url.startswith("/uploads/"):
                local_path = os.path.join(self.uploads_dir, url.replace("/uploads/", ""))
                try:
                    os.remove(local_path)
                    return True
                except:
                    return False
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "x-api-version": "7",
            }
            
            response = requests.delete(
                f"{self.api_url}",
                headers=headers,
                json={"urls": [url]},
                timeout=30
            )
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"⚠ Blob delete error: {e}")
            return False
