"""
Secure buffer implementation with encryption and access control.
"""

import os
import time
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import pickle


class EncryptionLevel(Enum):
    """Encryption levels for secure buffer."""
    NONE = "none"
    METADATA_ONLY = "metadata_only"
    FULL = "full"


@dataclass
class SecureEntry:
    """Secure buffer entry with encryption."""
    data: bytes  # Encrypted or plain data
    metadata: Dict[str, Any]
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    
    def mark_accessed(self) -> None:
        """Mark entry as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()


class SecureBuffer:
    """
    Secure buffer with encryption and access control for sensitive BCI data.
    """
    
    def __init__(self, 
                 encryption_level: EncryptionLevel = EncryptionLevel.FULL,
                 password: Optional[str] = None,
                 max_size: int = 1000):
        self.encryption_level = encryption_level
        self.max_size = max_size
        self.buffer: List[SecureEntry] = []
        self._lock = threading.RLock()
        
        # Setup encryption
        if encryption_level != EncryptionLevel.NONE:
            self._setup_encryption(password)
        else:
            self.cipher = None
    
    def _setup_encryption(self, password: Optional[str] = None) -> None:
        """Setup encryption cipher."""
        if password is None:
            # Generate random key
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
        else:
            # Derive key from password
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            self.cipher = Fernet(key)
    
    def _encrypt_data(self, data: Any) -> bytes:
        """Encrypt data if encryption is enabled."""
        if self.cipher is None:
            return pickle.dumps(data)
        
        serialized = pickle.dumps(data)
        return self.cipher.encrypt(serialized)
    
    def _decrypt_data(self, encrypted_data: bytes) -> Any:
        """Decrypt data if encryption is enabled."""
        if self.cipher is None:
            return pickle.loads(encrypted_data)
        
        decrypted = self.cipher.decrypt(encrypted_data)
        return pickle.loads(decrypted)
    
    def put(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add data to secure buffer."""
        with self._lock:
            try:
                # Handle encryption based on level
                if self.encryption_level == EncryptionLevel.FULL:
                    encrypted_data = self._encrypt_data(data)
                    safe_metadata = self._encrypt_data(metadata or {})
                elif self.encryption_level == EncryptionLevel.METADATA_ONLY:
                    encrypted_data = pickle.dumps(data)  # No encryption for data
                    safe_metadata = self._encrypt_data(metadata or {})
                else:
                    encrypted_data = pickle.dumps(data)
                    safe_metadata = metadata or {}
                
                entry = SecureEntry(
                    data=encrypted_data,
                    metadata=safe_metadata,
                    timestamp=time.time()
                )
                
                self.buffer.append(entry)
                
                # Maintain size limit
                if len(self.buffer) > self.max_size:
                    self.buffer.pop(0)
                
                return True
                
            except Exception as e:
                print(f"Error adding to secure buffer: {e}")
                return False
    
    def get(self, index: int = -1) -> Optional[Any]:
        """Get data from secure buffer."""
        with self._lock:
            try:
                if not self.buffer or index >= len(self.buffer):
                    return None
                
                entry = self.buffer[index]
                entry.mark_accessed()
                
                # Decrypt data
                data = self._decrypt_data(entry.data)
                
                return data
                
            except Exception as e:
                print(f"Error retrieving from secure buffer: {e}")
                return None
    
    def get_metadata(self, index: int = -1) -> Optional[Dict[str, Any]]:
        """Get metadata for entry."""
        with self._lock:
            try:
                if not self.buffer or index >= len(self.buffer):
                    return None
                
                entry = self.buffer[index]
                
                if self.encryption_level in [EncryptionLevel.METADATA_ONLY, EncryptionLevel.FULL]:
                    metadata = self._decrypt_data(entry.metadata)
                else:
                    metadata = entry.metadata
                
                return metadata
                
            except Exception as e:
                print(f"Error retrieving metadata: {e}")
                return None
    
    def get_recent(self, count: int = 10) -> List[Any]:
        """Get recent entries."""
        with self._lock:
            recent_data = []
            start_index = max(0, len(self.buffer) - count)
            
            for i in range(start_index, len(self.buffer)):
                data = self.get(i)
                if data is not None:
                    recent_data.append(data)
            
            return recent_data
    
    def clear(self) -> None:
        """Clear all entries from buffer."""
        with self._lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """Get number of entries in buffer."""
        return len(self.buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            if not self.buffer:
                return {
                    "size": 0,
                    "encryption_level": self.encryption_level.value,
                    "max_size": self.max_size
                }
            
            total_accesses = sum(entry.access_count for entry in self.buffer)
            avg_accesses = total_accesses / len(self.buffer) if self.buffer else 0
            
            oldest_timestamp = min(entry.timestamp for entry in self.buffer)
            newest_timestamp = max(entry.timestamp for entry in self.buffer)
            
            return {
                "size": len(self.buffer),
                "encryption_level": self.encryption_level.value,
                "max_size": self.max_size,
                "total_accesses": total_accesses,
                "avg_accesses_per_entry": round(avg_accesses, 2),
                "oldest_entry_age_seconds": time.time() - oldest_timestamp,
                "newest_entry_age_seconds": time.time() - newest_timestamp,
                "utilization_pct": round((len(self.buffer) / self.max_size) * 100, 1)
            }