"""
HIPAA (Health Insurance Portability and Accountability Act) compliance implementation.
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


class PHIType(Enum):
    """Types of Protected Health Information."""
    DEMOGRAPHIC = "demographic"
    MEDICAL_RECORDS = "medical_records"
    NEURAL_DATA = "neural_data"
    SESSION_DATA = "session_data"
    CLINICAL_NOTES = "clinical_notes"
    DEVICE_DATA = "device_data"


class AccessReason(Enum):
    """Reasons for PHI access."""
    TREATMENT = "treatment"
    RESEARCH = "research"
    PAYMENT = "payment"
    HEALTHCARE_OPERATIONS = "healthcare_operations"
    LEGAL_REQUIREMENT = "legal_requirement"
    PATIENT_REQUEST = "patient_request"


@dataclass
class PHIAccessLog:
    """PHI access log entry."""
    access_id: str
    user_id: str
    patient_id: str
    phi_type: PHIType
    access_reason: AccessReason
    timestamp: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    data_accessed: Optional[str] = None  # Description, not actual data
    session_id: Optional[str] = None
    authorized_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['phi_type'] = self.phi_type.value
        data['access_reason'] = self.access_reason.value
        return data


@dataclass
class HIPAAIncident:
    """HIPAA security incident record."""
    incident_id: str
    incident_type: str  # unauthorized_access, data_breach, system_failure
    severity: str  # low, medium, high, critical
    description: str
    affected_patients: List[str]
    phi_types_affected: List[PHIType]
    discovered_timestamp: float
    reported_timestamp: Optional[float] = None
    resolved_timestamp: Optional[float] = None
    root_cause: Optional[str] = None
    remediation_actions: List[str] = None
    reporter_id: Optional[str] = None


@dataclass
class BusinessAssociate:
    """Business Associate information."""
    ba_id: str
    name: str
    contact_email: str
    services_provided: List[str]
    baa_signed_date: float  # Business Associate Agreement
    baa_expiry_date: float
    phi_access_level: str  # limited, full, none
    last_security_assessment: Optional[float] = None
    compliance_status: str = "compliant"  # compliant, non_compliant, under_review


class HIPAACompliance:
    """
    HIPAA compliance management for BCI-Agent-Bridge.
    """
    
    def __init__(self, covered_entity: str, privacy_officer: str, 
                 security_officer: str, storage_path: Optional[Path] = None):
        self.covered_entity = covered_entity
        self.privacy_officer = privacy_officer
        self.security_officer = security_officer
        self.storage_path = storage_path or Path("hipaa_compliance")
        self.logger = logging.getLogger(__name__)
        
        # Storage
        self.access_logs: List[PHIAccessLog] = []
        self.incidents: Dict[str, HIPAAIncident] = {}
        self.business_associates: Dict[str, BusinessAssociate] = {}
        self.authorized_users: Dict[str, Dict[str, Any]] = {}
        self.audit_trail: List[Dict[str, Any]] = []
        
        # Configuration
        self.minimum_necessary_standard = True
        self.access_timeout_hours = 8
        self.audit_retention_years = 6
        self.breach_notification_hours = 60
        
        # Initialize storage
        self._initialize_storage()
        self._load_data()
    
    def _initialize_storage(self) -> None:
        """Initialize storage directories."""
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.storage_path / "access_logs").mkdir(exist_ok=True)
        (self.storage_path / "incidents").mkdir(exist_ok=True)
        (self.storage_path / "business_associates").mkdir(exist_ok=True)
        (self.storage_path / "audit_trail").mkdir(exist_ok=True)
    
    def _load_data(self) -> None:
        """Load existing HIPAA compliance data."""
        try:
            # Load access logs
            access_logs_file = self.storage_path / "access_logs" / "logs.json"
            if access_logs_file.exists():
                with open(access_logs_file, 'r') as f:
                    data = json.load(f)
                    self.access_logs = [
                        PHIAccessLog(**log_data) for log_data in data
                    ]
            
            # Load incidents
            incidents_file = self.storage_path / "incidents" / "incidents.json"
            if incidents_file.exists():
                with open(incidents_file, 'r') as f:
                    data = json.load(f)
                    for incident_id, incident_data in data.items():
                        incident_data['phi_types_affected'] = [
                            PHIType(phi_type) for phi_type in incident_data['phi_types_affected']
                        ]
                        self.incidents[incident_id] = HIPAAIncident(**incident_data)
            
            # Load business associates
            ba_file = self.storage_path / "business_associates" / "associates.json"
            if ba_file.exists():
                with open(ba_file, 'r') as f:
                    data = json.load(f)
                    for ba_id, ba_data in data.items():
                        self.business_associates[ba_id] = BusinessAssociate(**ba_data)
            
            self.logger.info("Loaded HIPAA compliance data")
            
        except Exception as e:
            self.logger.error(f"Error loading HIPAA data: {e}")
    
    def _save_data(self) -> None:
        """Save HIPAA compliance data to storage."""
        try:
            # Save access logs
            access_logs_data = [log.to_dict() for log in self.access_logs]
            access_logs_file = self.storage_path / "access_logs" / "logs.json"
            with open(access_logs_file, 'w') as f:
                json.dump(access_logs_data, f, indent=2)
            
            # Save incidents
            incidents_data = {}
            for incident_id, incident in self.incidents.items():
                incident_dict = asdict(incident)
                incident_dict['phi_types_affected'] = [
                    phi_type.value for phi_type in incident.phi_types_affected
                ]
                incidents_data[incident_id] = incident_dict
            
            incidents_file = self.storage_path / "incidents" / "incidents.json"
            with open(incidents_file, 'w') as f:
                json.dump(incidents_data, f, indent=2)
            
            # Save business associates
            ba_data = {}
            for ba_id, ba in self.business_associates.items():
                ba_data[ba_id] = asdict(ba)
            
            ba_file = self.storage_path / "business_associates" / "associates.json"
            with open(ba_file, 'w') as f:
                json.dump(ba_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving HIPAA data: {e}")
    
    def authorize_user(self, user_id: str, name: str, role: str,
                      phi_access_levels: List[PHIType],
                      access_reason: AccessReason,
                      authorized_by: str,
                      expiry_timestamp: Optional[float] = None) -> str:
        """Authorize user for PHI access."""
        authorization_id = str(uuid.uuid4())
        
        # Default 1-year access if no expiry specified
        if expiry_timestamp is None:
            expiry_timestamp = time.time() + (365 * 24 * 3600)
        
        self.authorized_users[user_id] = {
            "authorization_id": authorization_id,
            "name": name,
            "role": role,
            "phi_access_levels": [phi_type.value for phi_type in phi_access_levels],
            "access_reason": access_reason.value,
            "authorized_by": authorized_by,
            "authorized_timestamp": time.time(),
            "expiry_timestamp": expiry_timestamp,
            "last_access": None,
            "access_count": 0,
            "status": "active"
        }
        
        self._add_audit_entry(
            action="user_authorized",
            user_id=user_id,
            details={
                "authorization_id": authorization_id,
                "role": role,
                "phi_access_levels": [phi_type.value for phi_type in phi_access_levels],
                "authorized_by": authorized_by
            }
        )
        
        self.logger.info(f"Authorized user {user_id} for PHI access: {authorization_id}")
        return authorization_id
    
    def revoke_user_access(self, user_id: str, revoked_by: str, reason: str) -> bool:
        """Revoke user's PHI access authorization."""
        if user_id not in self.authorized_users:
            return False
        
        user_info = self.authorized_users[user_id]
        user_info["status"] = "revoked"
        user_info["revoked_timestamp"] = time.time()
        user_info["revoked_by"] = revoked_by
        user_info["revocation_reason"] = reason
        
        self._add_audit_entry(
            action="access_revoked",
            user_id=user_id,
            details={
                "authorization_id": user_info["authorization_id"],
                "revoked_by": revoked_by,
                "reason": reason
            }
        )
        
        self.logger.info(f"Revoked PHI access for user {user_id}: {reason}")
        return True
    
    def is_access_authorized(self, user_id: str, phi_type: PHIType, 
                           patient_id: Optional[str] = None) -> bool:
        """Check if user is authorized to access specific PHI."""
        if user_id not in self.authorized_users:
            return False
        
        user_info = self.authorized_users[user_id]
        
        # Check if authorization is active and not expired
        if user_info["status"] != "active":
            return False
        
        current_time = time.time()
        if current_time > user_info["expiry_timestamp"]:
            # Auto-revoke expired access
            user_info["status"] = "expired"
            self.logger.warning(f"User {user_id} access expired")
            return False
        
        # Check PHI type access level
        if phi_type.value not in user_info["phi_access_levels"]:
            return False
        
        # Additional patient-specific checks could be implemented here
        
        return True
    
    def log_phi_access(self, user_id: str, patient_id: str, phi_type: PHIType,
                      access_reason: AccessReason, ip_address: Optional[str] = None,
                      user_agent: Optional[str] = None,
                      data_description: Optional[str] = None,
                      session_id: Optional[str] = None) -> str:
        """Log PHI access event."""
        
        # Verify authorization
        if not self.is_access_authorized(user_id, phi_type, patient_id):
            self.logger.warning(f"Unauthorized PHI access attempt by user {user_id}")
            self._report_security_incident(
                "unauthorized_access",
                f"User {user_id} attempted unauthorized access to {phi_type.value} for patient {patient_id}",
                [patient_id],
                [phi_type],
                "medium"
            )
            raise PermissionError("Unauthorized PHI access")
        
        access_id = str(uuid.uuid4())
        
        access_log = PHIAccessLog(
            access_id=access_id,
            user_id=user_id,
            patient_id=patient_id,
            phi_type=phi_type,
            access_reason=access_reason,
            timestamp=time.time(),
            ip_address=ip_address,
            user_agent=user_agent,
            data_accessed=data_description,
            session_id=session_id,
            authorized_by=self.authorized_users[user_id]["authorized_by"]
        )
        
        self.access_logs.append(access_log)
        
        # Update user access stats
        user_info = self.authorized_users[user_id]
        user_info["last_access"] = time.time()
        user_info["access_count"] += 1
        
        self._save_data()
        
        self.logger.info(f"Logged PHI access: {access_id} by user {user_id}")
        return access_id
    
    def add_business_associate(self, name: str, contact_email: str,
                             services_provided: List[str],
                             baa_signed_date: float,
                             baa_expiry_date: float,
                             phi_access_level: str = "limited") -> str:
        """Add business associate."""
        ba_id = str(uuid.uuid4())
        
        business_associate = BusinessAssociate(
            ba_id=ba_id,
            name=name,
            contact_email=contact_email,
            services_provided=services_provided,
            baa_signed_date=baa_signed_date,
            baa_expiry_date=baa_expiry_date,
            phi_access_level=phi_access_level
        )
        
        self.business_associates[ba_id] = business_associate
        self._save_data()
        
        self._add_audit_entry(
            action="business_associate_added",
            details={
                "ba_id": ba_id,
                "name": name,
                "services": services_provided,
                "phi_access_level": phi_access_level
            }
        )
        
        self.logger.info(f"Added business associate: {name} ({ba_id})")
        return ba_id
    
    def _report_security_incident(self, incident_type: str, description: str,
                                affected_patients: List[str],
                                phi_types_affected: List[PHIType],
                                severity: str = "medium") -> str:
        """Report a security incident."""
        incident_id = str(uuid.uuid4())
        
        incident = HIPAAIncident(
            incident_id=incident_id,
            incident_type=incident_type,
            severity=severity,
            description=description,
            affected_patients=affected_patients,
            phi_types_affected=phi_types_affected,
            discovered_timestamp=time.time(),
            remediation_actions=[]
        )
        
        self.incidents[incident_id] = incident
        self._save_data()
        
        # Log critical incidents immediately
        if severity in ["high", "critical"]:
            self.logger.critical(f"HIPAA security incident: {incident_id} - {description}")
            
            # Check if breach notification is required
            if len(affected_patients) >= 500 or severity == "critical":
                self.logger.critical(f"Breach notification may be required within {self.breach_notification_hours} hours")
        
        self._add_audit_entry(
            action="security_incident_reported",
            details={
                "incident_id": incident_id,
                "type": incident_type,
                "severity": severity,
                "affected_patients_count": len(affected_patients)
            }
        )
        
        return incident_id
    
    def resolve_incident(self, incident_id: str, root_cause: str,
                        remediation_actions: List[str],
                        resolved_by: str) -> bool:
        """Resolve a security incident."""
        if incident_id not in self.incidents:
            return False
        
        incident = self.incidents[incident_id]
        incident.resolved_timestamp = time.time()
        incident.root_cause = root_cause
        incident.remediation_actions = remediation_actions
        
        self._add_audit_entry(
            action="incident_resolved",
            user_id=resolved_by,
            details={
                "incident_id": incident_id,
                "root_cause": root_cause,
                "remediation_actions": remediation_actions
            }
        )
        
        self._save_data()
        self.logger.info(f"Resolved security incident: {incident_id}")
        return True
    
    def get_access_audit(self, patient_id: Optional[str] = None,
                        user_id: Optional[str] = None,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None) -> List[PHIAccessLog]:
        """Get PHI access audit trail."""
        filtered_logs = []
        
        for log in self.access_logs:
            # Filter by patient_id
            if patient_id and log.patient_id != patient_id:
                continue
            
            # Filter by user_id
            if user_id and log.user_id != user_id:
                continue
            
            # Filter by time range
            if start_time and log.timestamp < start_time:
                continue
            if end_time and log.timestamp > end_time:
                continue
            
            filtered_logs.append(log)
        
        return filtered_logs
    
    def _add_audit_entry(self, action: str, user_id: Optional[str] = None,
                        details: Optional[Dict[str, Any]] = None) -> None:
        """Add entry to audit trail."""
        audit_entry = {
            "audit_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "action": action,
            "user_id": user_id,
            "details": details or {},
            "system_info": {
                "covered_entity": self.covered_entity,
                "component": "BCI-Agent-Bridge"
            }
        }
        
        self.audit_trail.append(audit_entry)
        
        # Save audit trail to separate file for integrity
        audit_file = self.storage_path / "audit_trail" / f"{int(time.time())}.json"
        with open(audit_file, 'w') as f:
            json.dump(audit_entry, f, indent=2)
    
    def generate_hipaa_report(self) -> Dict[str, Any]:
        """Generate HIPAA compliance report."""
        current_time = time.time()
        
        # Access statistics
        access_stats = {
            "total_access_events": len(self.access_logs),
            "unique_patients": len(set(log.patient_id for log in self.access_logs)),
            "unique_users": len(set(log.user_id for log in self.access_logs)),
            "phi_types_accessed": list(set(log.phi_type.value for log in self.access_logs))
        }
        
        # User statistics
        active_users = [user for user in self.authorized_users.values() if user["status"] == "active"]
        expired_users = [user for user in self.authorized_users.values() 
                        if user["expiry_timestamp"] < current_time]
        
        # Incident statistics
        open_incidents = [i for i in self.incidents.values() if i.resolved_timestamp is None]
        critical_incidents = [i for i in self.incidents.values() if i.severity == "critical"]
        
        # Business associate compliance
        ba_compliance = {
            "total_business_associates": len(self.business_associates),
            "compliant_bas": len([ba for ba in self.business_associates.values() 
                                if ba.compliance_status == "compliant"]),
            "expired_baas": len([ba for ba in self.business_associates.values()
                               if ba.baa_expiry_date < current_time])
        }
        
        return {
            "covered_entity": self.covered_entity,
            "privacy_officer": self.privacy_officer,
            "security_officer": self.security_officer,
            "report_generated": current_time,
            "access_statistics": access_stats,
            "user_management": {
                "active_users": len(active_users),
                "expired_users": len(expired_users),
                "total_authorized": len(self.authorized_users)
            },
            "security_incidents": {
                "total_incidents": len(self.incidents),
                "open_incidents": len(open_incidents),
                "critical_incidents": len(critical_incidents)
            },
            "business_associates": ba_compliance,
            "audit_trail_entries": len(self.audit_trail),
            "compliance_score": self._calculate_hipaa_compliance_score()
        }
    
    def _calculate_hipaa_compliance_score(self) -> float:
        """Calculate HIPAA compliance score (0-100)."""
        score = 100.0
        current_time = time.time()
        
        # Deduct for expired user authorizations
        expired_users = [user for user in self.authorized_users.values() 
                        if user["expiry_timestamp"] < current_time and user["status"] == "active"]
        score -= len(expired_users) * 5
        
        # Deduct for unresolved incidents
        open_incidents = [i for i in self.incidents.values() if i.resolved_timestamp is None]
        score -= len(open_incidents) * 10
        
        # Deduct for critical incidents
        critical_incidents = [i for i in self.incidents.values() if i.severity == "critical"]
        score -= len(critical_incidents) * 15
        
        # Deduct for expired business associate agreements
        expired_baas = [ba for ba in self.business_associates.values()
                       if ba.baa_expiry_date < current_time]
        score -= len(expired_baas) * 8
        
        # Deduct for non-compliant business associates
        non_compliant_bas = [ba for ba in self.business_associates.values()
                           if ba.compliance_status == "non_compliant"]
        score -= len(non_compliant_bas) * 12
        
        return max(0.0, min(100.0, score))
    
    def cleanup_old_logs(self, retention_days: int = 2190) -> int:  # 6 years default
        """Clean up old access logs beyond retention period."""
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        old_logs = [log for log in self.access_logs if log.timestamp < cutoff_time]
        self.access_logs = [log for log in self.access_logs if log.timestamp >= cutoff_time]
        
        self._save_data()
        
        if old_logs:
            self.logger.info(f"Cleaned up {len(old_logs)} old access logs beyond retention period")
        
        return len(old_logs)