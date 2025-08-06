"""
Data Protection Manager for BCI-Agent-Bridge.
Handles encryption, anonymization, and secure data handling.
"""

import logging
import hashlib
import secrets
import json
import time
import base64
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import numpy as np


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class EncryptionMethod(Enum):
    """Encryption methods available."""
    AES_256 = "aes_256"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    FERNET = "fernet"


@dataclass
class DataRecord:
    """Data record with protection metadata."""
    record_id: str
    data_type: str
    classification: DataClassification
    created_timestamp: float
    last_accessed: float
    access_count: int
    encrypted: bool
    anonymized: bool
    retention_period: int  # days
    owner_id: str
    authorized_users: List[str]
    encryption_method: Optional[EncryptionMethod] = None
    anonymization_method: Optional[str] = None
    tags: Dict[str, str] = None


@dataclass
class AnonymizationResult:
    """Result of data anonymization."""
    original_record_id: str
    anonymized_record_id: str
    method_used: str
    anonymization_timestamp: float
    reversible: bool
    mapping_key: Optional[str] = None  # For reversible anonymization


class DataProtectionManager:
    """
    Comprehensive data protection manager for BCI systems.
    """
    
    def __init__(self, storage_path: Optional[Path] = None, master_key: Optional[bytes] = None):
        self.storage_path = storage_path or Path("data_protection")
        self.logger = logging.getLogger(__name__)
        
        # Encryption setup
        if master_key:
            self.master_key = master_key
        else:
            self.master_key = self._generate_or_load_master_key()
        
        self.fernet = Fernet(base64.urlsafe_b64encode(self.master_key[:32]))
        
        # Storage
        self.data_records: Dict[str, DataRecord] = {}
        self.encryption_keys: Dict[str, bytes] = {}
        self.anonymization_mappings: Dict[str, Dict[str, str]] = {}
        self.access_logs: List[Dict[str, Any]] = []
        
        # Configuration
        self.default_retention_days = 2555  # 7 years
        self.neural_data_classification = DataClassification.RESTRICTED
        self.auto_anonymize_threshold_days = 90
        
        # Initialize storage
        self._initialize_storage()
        self._load_data()
    
    def _generate_or_load_master_key(self) -> bytes:
        """Generate or load master encryption key."""
        key_file = self.storage_path / "master.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new master key
            key = secrets.token_bytes(32)
            key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            key_file.chmod(0o600)  # Read-only for owner
            self.logger.info("Generated new master encryption key")
            return key
    
    def _initialize_storage(self) -> None:
        """Initialize storage directories."""
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.storage_path / "records").mkdir(exist_ok=True)
        (self.storage_path / "encrypted_data").mkdir(exist_ok=True)
        (self.storage_path / "anonymized_data").mkdir(exist_ok=True)
        (self.storage_path / "keys").mkdir(exist_ok=True)
        (self.storage_path / "access_logs").mkdir(exist_ok=True)
    
    def _load_data(self) -> None:
        """Load existing data protection records."""
        try:
            records_file = self.storage_path / "records" / "data_records.json"
            if records_file.exists():
                with open(records_file, 'r') as f:
                    data = json.load(f)
                    for record_id, record_data in data.items():
                        record_data['classification'] = DataClassification(record_data['classification'])
                        if record_data['encryption_method']:
                            record_data['encryption_method'] = EncryptionMethod(record_data['encryption_method'])
                        self.data_records[record_id] = DataRecord(**record_data)
            
            self.logger.info("Loaded data protection records")
            
        except Exception as e:
            self.logger.error(f"Error loading data protection data: {e}")
    
    def _save_data(self) -> None:
        """Save data protection records."""
        try:
            # Save data records
            records_data = {}
            for record_id, record in self.data_records.items():
                record_dict = asdict(record)
                record_dict['classification'] = record.classification.value
                if record.encryption_method:
                    record_dict['encryption_method'] = record.encryption_method.value
                records_data[record_id] = record_dict
            
            records_file = self.storage_path / "records" / "data_records.json"
            with open(records_file, 'w') as f:
                json.dump(records_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving data protection data: {e}")
    
    def classify_data(self, data: Any, data_type: str) -> DataClassification:
        """Automatically classify data based on type and content."""
        
        # Neural data is always restricted
        if data_type in ["neural_data", "eeg_data", "bci_signals"]:
            return DataClassification.RESTRICTED
        
        # Medical data is confidential
        if data_type in ["medical_records", "patient_info", "clinical_data"]:
            return DataClassification.CONFIDENTIAL
        
        # Personal identifiers are confidential
        if data_type in ["personal_info", "demographics", "identifiers"]:
            return DataClassification.CONFIDENTIAL
        
        # Session data is internal
        if data_type in ["session_data", "system_logs", "performance_metrics"]:
            return DataClassification.INTERNAL
        
        # Default to confidential for unknown types
        return DataClassification.CONFIDENTIAL
    
    def register_data(self, data: Any, data_type: str, owner_id: str,
                     classification: Optional[DataClassification] = None,
                     authorized_users: Optional[List[str]] = None,
                     retention_days: Optional[int] = None) -> str:
        """Register data with protection metadata."""
        record_id = str(uuid.uuid4())
        
        if classification is None:
            classification = self.classify_data(data, data_type)
        
        data_record = DataRecord(
            record_id=record_id,
            data_type=data_type,
            classification=classification,
            created_timestamp=time.time(),
            last_accessed=time.time(),
            access_count=0,
            encrypted=False,
            anonymized=False,
            retention_period=retention_days or self.default_retention_days,
            owner_id=owner_id,
            authorized_users=authorized_users or [owner_id],
            tags={}
        )
        
        self.data_records[record_id] = data_record
        self._save_data()
        
        # Auto-encrypt based on classification
        if classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]:
            self.encrypt_data(record_id, data)
        
        self.logger.info(f"Registered data record: {record_id} ({data_type}, {classification.value})")
        return record_id
    
    def encrypt_data(self, record_id: str, data: Any, 
                    method: EncryptionMethod = EncryptionMethod.FERNET) -> bool:
        """Encrypt data record."""
        if record_id not in self.data_records:
            return False
        
        record = self.data_records[record_id]
        
        try:
            # Serialize data
            if isinstance(data, np.ndarray):
                data_bytes = data.tobytes()
                metadata = {
                    "dtype": str(data.dtype),
                    "shape": data.shape,
                    "data_type": "numpy_array"
                }
            elif isinstance(data, dict):
                data_bytes = json.dumps(data).encode()
                metadata = {"data_type": "json"}
            else:
                data_bytes = str(data).encode()
                metadata = {"data_type": "string"}
            
            # Encrypt based on method
            if method == EncryptionMethod.FERNET:
                encrypted_data = self.fernet.encrypt(data_bytes)
            elif method == EncryptionMethod.AES_256:
                # Use custom AES implementation if needed
                encrypted_data = self._encrypt_aes_256(data_bytes)
            else:
                raise ValueError(f"Encryption method {method} not implemented")
            
            # Store encrypted data
            encrypted_file = self.storage_path / "encrypted_data" / f"{record_id}.enc"
            with open(encrypted_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Store metadata
            metadata_file = self.storage_path / "encrypted_data" / f"{record_id}_meta.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Update record
            record.encrypted = True
            record.encryption_method = method
            self._save_data()
            
            self._log_access(record_id, "encrypt", "system")
            self.logger.info(f"Encrypted data record: {record_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error encrypting data record {record_id}: {e}")
            return False
    
    def decrypt_data(self, record_id: str, user_id: str) -> Optional[Any]:
        """Decrypt data record."""
        if record_id not in self.data_records:
            return None
        
        record = self.data_records[record_id]
        
        # Check authorization
        if user_id not in record.authorized_users and user_id != record.owner_id:
            self.logger.warning(f"Unauthorized decrypt attempt by {user_id} for record {record_id}")
            return None
        
        if not record.encrypted:
            self.logger.warning(f"Attempted to decrypt unencrypted record {record_id}")
            return None
        
        try:
            # Load encrypted data
            encrypted_file = self.storage_path / "encrypted_data" / f"{record_id}.enc"
            if not encrypted_file.exists():
                self.logger.error(f"Encrypted file not found for record {record_id}")
                return None
            
            with open(encrypted_file, 'rb') as f:
                encrypted_data = f.read()
            
            # Load metadata
            metadata_file = self.storage_path / "encrypted_data" / f"{record_id}_meta.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Decrypt based on method
            if record.encryption_method == EncryptionMethod.FERNET:
                decrypted_bytes = self.fernet.decrypt(encrypted_data)
            else:
                raise ValueError(f"Decryption method {record.encryption_method} not implemented")
            
            # Deserialize data
            if metadata["data_type"] == "numpy_array":
                data = np.frombuffer(decrypted_bytes, dtype=metadata["dtype"]).reshape(metadata["shape"])
            elif metadata["data_type"] == "json":
                data = json.loads(decrypted_bytes.decode())
            else:
                data = decrypted_bytes.decode()
            
            # Update access record
            record.last_accessed = time.time()
            record.access_count += 1
            self._save_data()
            
            self._log_access(record_id, "decrypt", user_id)
            return data
            
        except Exception as e:
            self.logger.error(f"Error decrypting data record {record_id}: {e}")
            return None
    
    def anonymize_neural_data(self, record_id: str, method: str = "differential_privacy",
                            epsilon: float = 1.0) -> Optional[AnonymizationResult]:
        """Anonymize neural data using various methods."""
        if record_id not in self.data_records:
            return None
        
        record = self.data_records[record_id]
        
        # Only anonymize if not already anonymized
        if record.anonymized:
            return None
        
        try:
            # Get the data (decrypt if necessary)
            if record.encrypted:
                data = self.decrypt_data(record_id, record.owner_id)
            else:
                # For this implementation, we'll assume data is accessible
                # In practice, you'd load from storage
                data = None  # Placeholder
            
            if data is None:
                return None
            
            anonymized_record_id = str(uuid.uuid4())
            mapping_key = None
            reversible = False
            
            if method == "differential_privacy":
                # Add calibrated noise for differential privacy
                if isinstance(data, np.ndarray):
                    noise_scale = 1.0 / epsilon
                    noise = np.random.laplace(0, noise_scale, data.shape)
                    anonymized_data = data + noise
                else:
                    # For non-numeric data, use k-anonymity approach
                    anonymized_data = self._k_anonymize_data(data, k=5)
            
            elif method == "k_anonymity":
                anonymized_data = self._k_anonymize_data(data, k=5)
            
            elif method == "pseudonymization":
                # Reversible pseudonymization
                anonymized_data, mapping_key = self._pseudonymize_data(data, record_id)
                reversible = True
            
            else:
                raise ValueError(f"Unknown anonymization method: {method}")
            
            # Store anonymized data
            anonymized_file = self.storage_path / "anonymized_data" / f"{anonymized_record_id}.anon"
            if isinstance(anonymized_data, np.ndarray):
                np.save(anonymized_file, anonymized_data)
            else:
                with open(anonymized_file, 'w') as f:
                    json.dump(anonymized_data, f)
            
            # Create anonymized record
            anonymized_record = DataRecord(
                record_id=anonymized_record_id,
                data_type=f"{record.data_type}_anonymized",
                classification=DataClassification.INTERNAL,  # Reduced classification
                created_timestamp=time.time(),
                last_accessed=time.time(),
                access_count=0,
                encrypted=False,
                anonymized=True,
                retention_period=record.retention_period,
                owner_id=record.owner_id,
                authorized_users=record.authorized_users.copy(),
                anonymization_method=method
            )
            
            self.data_records[anonymized_record_id] = anonymized_record
            
            # Update original record
            record.anonymized = True
            record.tags = record.tags or {}
            record.tags["anonymized_version"] = anonymized_record_id
            
            # Store mapping if reversible
            if reversible and mapping_key:
                self.anonymization_mappings[record_id] = {
                    "anonymized_id": anonymized_record_id,
                    "mapping_key": mapping_key,
                    "method": method
                }
            
            self._save_data()
            
            result = AnonymizationResult(
                original_record_id=record_id,
                anonymized_record_id=anonymized_record_id,
                method_used=method,
                anonymization_timestamp=time.time(),
                reversible=reversible,
                mapping_key=mapping_key
            )
            
            self._log_access(record_id, "anonymize", "system", {"method": method})
            self.logger.info(f"Anonymized data record {record_id} -> {anonymized_record_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error anonymizing data record {record_id}: {e}")
            return None
    
    def _k_anonymize_data(self, data: Any, k: int = 5) -> Any:
        """Apply k-anonymity to data."""
        # Simplified k-anonymity implementation
        # In practice, this would be much more sophisticated
        if isinstance(data, dict):
            anonymized = data.copy()
            # Remove or generalize identifying fields
            for key in ["id", "user_id", "patient_id", "session_id"]:
                if key in anonymized:
                    anonymized[key] = f"anonymous_group_{hash(anonymized[key]) % k}"
            return anonymized
        elif isinstance(data, np.ndarray):
            # For neural data, add controlled noise and quantize
            quantized = np.round(data, decimals=2)  # Reduce precision
            return quantized
        else:
            return data
    
    def _pseudonymize_data(self, data: Any, record_id: str) -> Tuple[Any, str]:
        """Pseudonymize data with reversible mapping."""
        mapping_key = secrets.token_urlsafe(32)
        
        if isinstance(data, dict):
            pseudonymized = data.copy()
            mapping = {}
            
            for key in ["id", "user_id", "patient_id", "session_id"]:
                if key in pseudonymized:
                    original_value = pseudonymized[key]
                    pseudo_value = hashlib.sha256(f"{mapping_key}_{original_value}".encode()).hexdigest()[:16]
                    pseudonymized[key] = pseudo_value
                    mapping[pseudo_value] = original_value
            
            # Store mapping securely
            mapping_file = self.storage_path / "keys" / f"{record_id}_mapping.json"
            encrypted_mapping = self.fernet.encrypt(json.dumps(mapping).encode())
            with open(mapping_file, 'wb') as f:
                f.write(encrypted_mapping)
            
            return pseudonymized, mapping_key
        
        return data, mapping_key
    
    def _encrypt_aes_256(self, data: bytes) -> bytes:
        """Encrypt data using AES-256 (placeholder implementation)."""
        # This would use a proper AES-256 implementation
        # For now, using Fernet as fallback
        return self.fernet.encrypt(data)
    
    def _log_access(self, record_id: str, action: str, user_id: str, 
                   details: Optional[Dict[str, Any]] = None) -> None:
        """Log data access event."""
        access_log = {
            "log_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "record_id": record_id,
            "action": action,
            "user_id": user_id,
            "details": details or {}
        }
        
        self.access_logs.append(access_log)
        
        # Save to file for audit trail
        log_file = self.storage_path / "access_logs" / f"{int(time.time())}.json"
        with open(log_file, 'w') as f:
            json.dump(access_log, f, indent=2)
    
    def check_retention_compliance(self) -> List[str]:
        """Check which records have exceeded retention period."""
        current_time = time.time()
        expired_records = []
        
        for record_id, record in self.data_records.items():
            retention_deadline = record.created_timestamp + (record.retention_period * 24 * 3600)
            if current_time > retention_deadline:
                expired_records.append(record_id)
        
        return expired_records
    
    def secure_delete_record(self, record_id: str, user_id: str, reason: str) -> bool:
        """Securely delete a data record."""
        if record_id not in self.data_records:
            return False
        
        record = self.data_records[record_id]
        
        # Check authorization (owner or admin)
        if user_id != record.owner_id and user_id not in record.authorized_users:
            self.logger.warning(f"Unauthorized delete attempt by {user_id} for record {record_id}")
            return False
        
        try:
            # Delete encrypted data
            encrypted_file = self.storage_path / "encrypted_data" / f"{record_id}.enc"
            if encrypted_file.exists():
                encrypted_file.unlink()
            
            metadata_file = self.storage_path / "encrypted_data" / f"{record_id}_meta.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Delete anonymized data if exists
            if record.tags and "anonymized_version" in record.tags:
                anon_id = record.tags["anonymized_version"]
                anon_file = self.storage_path / "anonymized_data" / f"{anon_id}.anon"
                if anon_file.exists():
                    anon_file.unlink()
                if anon_id in self.data_records:
                    del self.data_records[anon_id]
            
            # Delete mapping if exists
            mapping_file = self.storage_path / "keys" / f"{record_id}_mapping.json"
            if mapping_file.exists():
                mapping_file.unlink()
            
            # Remove from mappings
            if record_id in self.anonymization_mappings:
                del self.anonymization_mappings[record_id]
            
            # Remove record
            del self.data_records[record_id]
            self._save_data()
            
            self._log_access(record_id, "secure_delete", user_id, {"reason": reason})
            self.logger.info(f"Securely deleted record: {record_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error securely deleting record {record_id}: {e}")
            return False
    
    def get_protection_summary(self) -> Dict[str, Any]:
        """Get data protection summary."""
        current_time = time.time()
        
        # Classification stats
        classification_stats = {}
        for classification in DataClassification:
            count = len([r for r in self.data_records.values() 
                        if r.classification == classification])
            classification_stats[classification.value] = count
        
        # Protection stats
        encrypted_count = len([r for r in self.data_records.values() if r.encrypted])
        anonymized_count = len([r for r in self.data_records.values() if r.anonymized])
        
        # Retention compliance
        expired_records = self.check_retention_compliance()
        
        return {
            "total_records": len(self.data_records),
            "classification_breakdown": classification_stats,
            "protection_status": {
                "encrypted_records": encrypted_count,
                "anonymized_records": anonymized_count,
                "encryption_rate": encrypted_count / len(self.data_records) if self.data_records else 0
            },
            "retention_compliance": {
                "expired_records": len(expired_records),
                "compliance_rate": (len(self.data_records) - len(expired_records)) / len(self.data_records) if self.data_records else 1
            },
            "access_statistics": {
                "total_access_events": len(self.access_logs),
                "recent_access_24h": len([log for log in self.access_logs 
                                        if log["timestamp"] > current_time - 86400])
            },
            "security_score": self._calculate_security_score()
        }
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score (0-100)."""
        if not self.data_records:
            return 100.0
        
        score = 100.0
        
        # Encryption coverage
        encrypted_restricted = len([r for r in self.data_records.values() 
                                  if r.classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET] 
                                  and r.encrypted])
        total_restricted = len([r for r in self.data_records.values() 
                              if r.classification in [DataClassification.RESTRICTED, DataClassification.TOP_SECRET]])
        
        if total_restricted > 0:
            encryption_rate = encrypted_restricted / total_restricted
            score *= encryption_rate
        
        # Retention compliance
        expired_records = self.check_retention_compliance()
        if expired_records:
            score *= (len(self.data_records) - len(expired_records)) / len(self.data_records)
        
        # Classification appropriateness (assume all are appropriately classified for now)
        
        return score