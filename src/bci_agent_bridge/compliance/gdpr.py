"""
GDPR (General Data Protection Regulation) compliance implementation.
"""

import logging
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path


class ConsentType(Enum):
    NECESSARY = "necessary"
    PERFORMANCE = "performance" 
    FUNCTIONAL = "functional"
    MARKETING = "marketing"


class DataProcessingBasis(Enum):
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class ConsentRecord:
    user_id: str
    consent_type: ConsentType
    granted: bool
    timestamp: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    consent_text: Optional[str] = None
    withdrawal_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['consent_type'] = self.consent_type.value
        return data


@dataclass
class DataSubjectRequest:
    request_id: str
    user_id: str
    request_type: str  # access, rectification, erasure, portability, objection
    timestamp: float
    status: str = "pending"  # pending, in_progress, completed, rejected
    completion_timestamp: Optional[float] = None
    details: Dict[str, Any] = None


@dataclass
class DataProcessingRecord:
    processing_id: str
    data_category: str  # neural_data, personal_info, session_data
    processing_purpose: str
    legal_basis: DataProcessingBasis
    data_subjects_count: int
    retention_period: int  # days
    created_at: float
    last_updated: float
    cross_border_transfer: bool = False
    third_parties: List[str] = None


class GDPRCompliance:
    """
    GDPR compliance management for BCI-Agent-Bridge.
    """
    
    def __init__(self, data_controller: str, dpo_contact: str, 
                 storage_path: Optional[Path] = None):
        self.data_controller = data_controller
        self.dpo_contact = dpo_contact
        self.storage_path = storage_path or Path("gdpr_compliance")
        self.logger = logging.getLogger(__name__)
        
        # Storage
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.subject_requests: Dict[str, DataSubjectRequest] = {}
        self.data_breaches: List[Dict[str, Any]] = []
        
        # Initialize storage
        self._initialize_storage()
        self._load_data()
    
    def _initialize_storage(self) -> None:
        """Initialize storage directories."""
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.storage_path / "consent").mkdir(exist_ok=True)
        (self.storage_path / "processing").mkdir(exist_ok=True)
        (self.storage_path / "requests").mkdir(exist_ok=True)
        (self.storage_path / "breaches").mkdir(exist_ok=True)
    
    def _load_data(self) -> None:
        """Load existing compliance data."""
        try:
            # Load consent records
            consent_file = self.storage_path / "consent" / "records.json"
            if consent_file.exists():
                with open(consent_file, 'r') as f:
                    data = json.load(f)
                    for user_id, records in data.items():
                        self.consent_records[user_id] = [
                            ConsentRecord(**record) for record in records
                        ]
            
            # Load processing records
            processing_file = self.storage_path / "processing" / "records.json"
            if processing_file.exists():
                with open(processing_file, 'r') as f:
                    data = json.load(f)
                    for proc_id, record in data.items():
                        record['legal_basis'] = DataProcessingBasis(record['legal_basis'])
                        self.processing_records[proc_id] = DataProcessingRecord(**record)
            
            # Load subject requests
            requests_file = self.storage_path / "requests" / "records.json"
            if requests_file.exists():
                with open(requests_file, 'r') as f:
                    data = json.load(f)
                    for req_id, request in data.items():
                        self.subject_requests[req_id] = DataSubjectRequest(**request)
            
            self.logger.info("Loaded GDPR compliance data")
            
        except Exception as e:
            self.logger.error(f"Error loading GDPR data: {e}")
    
    def _save_data(self) -> None:
        """Save compliance data to storage."""
        try:
            # Save consent records
            consent_data = {}
            for user_id, records in self.consent_records.items():
                consent_data[user_id] = [record.to_dict() for record in records]
            
            consent_file = self.storage_path / "consent" / "records.json"
            with open(consent_file, 'w') as f:
                json.dump(consent_data, f, indent=2)
            
            # Save processing records
            processing_data = {}
            for proc_id, record in self.processing_records.items():
                record_dict = asdict(record)
                record_dict['legal_basis'] = record.legal_basis.value
                processing_data[proc_id] = record_dict
            
            processing_file = self.storage_path / "processing" / "records.json"
            with open(processing_file, 'w') as f:
                json.dump(processing_data, f, indent=2)
            
            # Save subject requests
            requests_data = {}
            for req_id, request in self.subject_requests.items():
                requests_data[req_id] = asdict(request)
            
            requests_file = self.storage_path / "requests" / "records.json"
            with open(requests_file, 'w') as f:
                json.dump(requests_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving GDPR data: {e}")
    
    def record_consent(self, user_id: str, consent_type: ConsentType, 
                      granted: bool, ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None,
                      consent_text: Optional[str] = None) -> str:
        """Record user consent."""
        consent_record = ConsentRecord(
            user_id=user_id,
            consent_type=consent_type,
            granted=granted,
            timestamp=time.time(),
            ip_address=ip_address,
            user_agent=user_agent,
            consent_text=consent_text
        )
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
        
        self.consent_records[user_id].append(consent_record)
        self._save_data()
        
        self.logger.info(f"Recorded consent for user {user_id}: {consent_type.value} = {granted}")
        return f"{user_id}_{consent_type.value}_{int(consent_record.timestamp)}"
    
    def withdraw_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """Withdraw user consent."""
        if user_id not in self.consent_records:
            return False
        
        # Find the latest consent record for this type
        latest_consent = None
        for record in reversed(self.consent_records[user_id]):
            if record.consent_type == consent_type and record.granted:
                latest_consent = record
                break
        
        if not latest_consent:
            return False
        
        # Record withdrawal
        withdrawal_record = ConsentRecord(
            user_id=user_id,
            consent_type=consent_type,
            granted=False,
            timestamp=time.time(),
            consent_text="Consent withdrawn by user"
        )
        
        # Mark original consent as withdrawn
        latest_consent.withdrawal_timestamp = time.time()
        
        self.consent_records[user_id].append(withdrawal_record)
        self._save_data()
        
        self.logger.info(f"Consent withdrawn for user {user_id}: {consent_type.value}")
        return True
    
    def has_valid_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """Check if user has valid consent for given type."""
        if user_id not in self.consent_records:
            return False
        
        # Find the latest consent record for this type
        for record in reversed(self.consent_records[user_id]):
            if record.consent_type == consent_type:
                return record.granted and record.withdrawal_timestamp is None
        
        return False
    
    def get_consent_history(self, user_id: str) -> List[ConsentRecord]:
        """Get consent history for a user."""
        return self.consent_records.get(user_id, [])
    
    def register_processing_activity(self, data_category: str, processing_purpose: str,
                                   legal_basis: DataProcessingBasis,
                                   retention_period: int,
                                   cross_border_transfer: bool = False,
                                   third_parties: List[str] = None) -> str:
        """Register a data processing activity."""
        processing_id = str(uuid.uuid4())
        
        processing_record = DataProcessingRecord(
            processing_id=processing_id,
            data_category=data_category,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            data_subjects_count=0,  # Will be updated as data is processed
            retention_period=retention_period,
            created_at=time.time(),
            last_updated=time.time(),
            cross_border_transfer=cross_border_transfer,
            third_parties=third_parties or []
        )
        
        self.processing_records[processing_id] = processing_record
        self._save_data()
        
        self.logger.info(f"Registered processing activity: {processing_id}")
        return processing_id
    
    def update_processing_activity(self, processing_id: str, **updates) -> bool:
        """Update a processing activity record."""
        if processing_id not in self.processing_records:
            return False
        
        record = self.processing_records[processing_id]
        
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)
        
        record.last_updated = time.time()
        self._save_data()
        
        self.logger.info(f"Updated processing activity: {processing_id}")
        return True
    
    def submit_subject_request(self, user_id: str, request_type: str,
                             details: Dict[str, Any] = None) -> str:
        """Submit a data subject request."""
        request_id = str(uuid.uuid4())
        
        request = DataSubjectRequest(
            request_id=request_id,
            user_id=user_id,
            request_type=request_type,
            timestamp=time.time(),
            details=details or {}
        )
        
        self.subject_requests[request_id] = request
        self._save_data()
        
        self.logger.info(f"Submitted {request_type} request: {request_id} for user {user_id}")
        return request_id
    
    def process_access_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Process a data access request."""
        if request_id not in self.subject_requests:
            return None
        
        request = self.subject_requests[request_id]
        if request.request_type != "access":
            return None
        
        # Update request status
        request.status = "in_progress"
        self._save_data()
        
        # Collect user data
        user_data = {
            "user_id": request.user_id,
            "consent_records": [
                record.to_dict() for record in self.get_consent_history(request.user_id)
            ],
            "processing_activities": [
                asdict(record) for record in self.processing_records.values()
                if request.user_id in str(record.processing_id)  # Simplified matching
            ],
            "data_requests": [
                asdict(req) for req in self.subject_requests.values()
                if req.user_id == request.user_id
            ]
        }
        
        # Mark request as completed
        request.status = "completed"
        request.completion_timestamp = time.time()
        self._save_data()
        
        self.logger.info(f"Completed access request: {request_id}")
        return user_data
    
    def process_erasure_request(self, request_id: str) -> bool:
        """Process a data erasure request (right to be forgotten)."""
        if request_id not in self.subject_requests:
            return False
        
        request = self.subject_requests[request_id]
        if request.request_type != "erasure":
            return False
        
        user_id = request.user_id
        
        # Update request status
        request.status = "in_progress"
        self._save_data()
        
        # Check if erasure is possible (no legal obligations, etc.)
        can_erase = self._can_erase_user_data(user_id)
        
        if can_erase:
            # Erase user data (pseudonymize instead of delete for audit trail)
            self._pseudonymize_user_data(user_id)
            
            request.status = "completed"
            request.completion_timestamp = time.time()
            
            self.logger.info(f"Completed erasure request: {request_id}")
        else:
            request.status = "rejected"
            request.details = request.details or {}
            request.details["rejection_reason"] = "Legal obligations prevent erasure"
            
            self.logger.info(f"Rejected erasure request: {request_id} - legal obligations")
        
        self._save_data()
        return can_erase
    
    def _can_erase_user_data(self, user_id: str) -> bool:
        """Check if user data can be erased."""
        # Check for legal obligations that prevent erasure
        for record in self.processing_records.values():
            if (record.legal_basis == DataProcessingBasis.LEGAL_OBLIGATION and
                user_id in str(record.processing_id)):  # Simplified check
                return False
        
        # Check for ongoing medical/clinical obligations
        # In a real implementation, this would check active clinical trials, etc.
        
        return True
    
    def _pseudonymize_user_data(self, user_id: str) -> None:
        """Pseudonymize user data instead of deletion."""
        # Generate pseudonym
        pseudonym = hashlib.sha256(f"{user_id}_{time.time()}".encode()).hexdigest()[:16]
        
        # Replace user_id with pseudonym in consent records
        if user_id in self.consent_records:
            for record in self.consent_records[user_id]:
                record.user_id = f"pseudonym_{pseudonym}"
                record.ip_address = None  # Remove IP address
                record.user_agent = None  # Remove user agent
            
            # Move records to pseudonym key
            self.consent_records[f"pseudonym_{pseudonym}"] = self.consent_records[user_id]
            del self.consent_records[user_id]
        
        # Update subject requests
        for request in self.subject_requests.values():
            if request.user_id == user_id:
                request.user_id = f"pseudonym_{pseudonym}"
        
        self.logger.info(f"Pseudonymized user data: {user_id} -> pseudonym_{pseudonym}")
    
    def report_data_breach(self, breach_description: str, affected_users: int,
                         data_categories: List[str], risk_level: str = "high") -> str:
        """Report a data breach."""
        breach_id = str(uuid.uuid4())
        
        breach_record = {
            "breach_id": breach_id,
            "description": breach_description,
            "affected_users": affected_users,
            "data_categories": data_categories,
            "risk_level": risk_level,
            "reported_timestamp": time.time(),
            "reported_to_authority": False,
            "users_notified": False
        }
        
        self.data_breaches.append(breach_record)
        
        # Save breach record to file
        breach_file = self.storage_path / "breaches" / f"{breach_id}.json"
        with open(breach_file, 'w') as f:
            json.dump(breach_record, f, indent=2)
        
        self.logger.critical(f"Data breach reported: {breach_id} - {breach_description}")
        
        # Auto-notify if high risk and affects many users
        if risk_level == "high" and affected_users > 10:
            self.logger.critical(f"High-risk breach affecting {affected_users} users - "
                               f"Authority notification may be required within 72 hours")
        
        return breach_id
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get GDPR compliance summary."""
        total_users = len(self.consent_records)
        total_processing_activities = len(self.processing_records)
        pending_requests = len([r for r in self.subject_requests.values() 
                              if r.status == "pending"])
        
        consent_stats = {}
        for consent_type in ConsentType:
            granted = sum(1 for records in self.consent_records.values()
                         for record in records
                         if record.consent_type == consent_type and record.granted)
            consent_stats[consent_type.value] = granted
        
        return {
            "data_controller": self.data_controller,
            "dpo_contact": self.dpo_contact,
            "total_users": total_users,
            "total_processing_activities": total_processing_activities,
            "pending_subject_requests": pending_requests,
            "consent_statistics": consent_stats,
            "data_breaches": len(self.data_breaches),
            "compliance_score": self._calculate_compliance_score()
        }
    
    def _calculate_compliance_score(self) -> float:
        """Calculate a compliance score (0-100)."""
        score = 100.0
        
        # Deduct points for pending requests older than 30 days
        current_time = time.time()
        old_requests = [
            r for r in self.subject_requests.values()
            if r.status == "pending" and (current_time - r.timestamp) > (30 * 24 * 3600)
        ]
        score -= len(old_requests) * 10
        
        # Deduct points for unreported data breaches
        unreported_breaches = [
            b for b in self.data_breaches
            if not b.get("reported_to_authority", False) and b["risk_level"] == "high"
        ]
        score -= len(unreported_breaches) * 15
        
        # Deduct points for processing activities without proper legal basis documentation
        undocumented_processing = [
            p for p in self.processing_records.values()
            if not p.processing_purpose or not p.legal_basis
        ]
        score -= len(undocumented_processing) * 5
        
        return max(0.0, min(100.0, score))
    
    def generate_privacy_notice(self, language: str = "en") -> Dict[str, Any]:
        """Generate privacy notice for users."""
        processing_purposes = list(set(
            record.processing_purpose 
            for record in self.processing_records.values()
            if record.processing_purpose
        ))
        
        data_categories = list(set(
            record.data_category
            for record in self.processing_records.values()
        ))
        
        return {
            "data_controller": self.data_controller,
            "dpo_contact": self.dpo_contact,
            "processing_purposes": processing_purposes,
            "data_categories": data_categories,
            "legal_bases": [basis.value for basis in DataProcessingBasis],
            "user_rights": [
                "access", "rectification", "erasure", "restriction", 
                "data_portability", "objection", "withdraw_consent"
            ],
            "retention_policy": "Data is retained according to specific retention periods for each category",
            "third_party_transfers": any(
                record.cross_border_transfer 
                for record in self.processing_records.values()
            ),
            "automated_decision_making": False,  # Update based on actual usage
            "contact_information": {
                "data_controller": self.data_controller,
                "dpo": self.dpo_contact
            },
            "language": language,
            "last_updated": time.time()
        }