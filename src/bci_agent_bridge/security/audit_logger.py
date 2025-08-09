"""
Security-focused audit logging for BCI system.
"""

import json
import logging
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import os


class SecurityEvent(Enum):
    """Types of security events to log."""
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_DENIED = "auth_denied"
    INPUT_VALIDATION_FAILURE = "validation_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "config_change"
    SYSTEM_ERROR = "system_error"
    PRIVACY_VIOLATION = "privacy_violation"


@dataclass
class AuditRecord:
    """Structured audit record."""
    event_type: SecurityEvent
    timestamp: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: str = "success"
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: int = 0  # 0-10 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "iso_timestamp": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "details": self.details,
            "risk_score": self.risk_score
        }


class SecurityAuditLogger:
    """
    Comprehensive security audit logger with real-time monitoring.
    """
    
    def __init__(self, log_file: str = "security_audit.jsonl", 
                 max_log_size_mb: int = 100,
                 enable_console: bool = True):
        self.log_file = log_file
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        self.enable_console = enable_console
        
        # Setup logging
        self.logger = logging.getLogger("security_audit")
        self.logger.setLevel(logging.INFO)
        
        # File handler for audit trail  
        log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "."
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, mode='a')
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for immediate alerts
        if enable_console:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics tracking
        self.event_counts = {event.value: 0 for event in SecurityEvent}
        self.high_risk_events = []
        self.start_time = time.time()
        
        # Log rotation tracking
        self._check_log_rotation()
    
    def _check_log_rotation(self) -> None:
        """Check if log file needs rotation."""
        try:
            if os.path.exists(self.log_file):
                size = os.path.getsize(self.log_file)
                if size > self.max_log_size_bytes:
                    self._rotate_log_file()
        except Exception as e:
            self.logger.error(f"Error checking log rotation: {e}")
    
    def _rotate_log_file(self) -> None:
        """Rotate log file when it gets too large."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_file = f"{self.log_file}.{timestamp}"
            os.rename(self.log_file, archived_file)
            self.logger.info(f"Log file rotated to {archived_file}")
        except Exception as e:
            self.logger.error(f"Error rotating log file: {e}")
    
    def log_security_event(self, 
                          event_type: SecurityEvent,
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          resource: Optional[str] = None,
                          action: Optional[str] = None,
                          result: str = "success",
                          details: Optional[Dict[str, Any]] = None,
                          risk_score: int = 0) -> None:
        """Log a security event with structured data."""
        
        with self._lock:
            try:
                # Create audit record
                record = AuditRecord(
                    event_type=event_type,
                    timestamp=time.time(),
                    user_id=user_id,
                    session_id=session_id,
                    ip_address=ip_address,
                    resource=resource,
                    action=action,
                    result=result,
                    details=details or {},
                    risk_score=risk_score
                )
                
                # Update metrics
                self.event_counts[event_type.value] += 1
                
                # Track high-risk events
                if risk_score >= 7:
                    self.high_risk_events.append(record)
                    # Keep only recent high-risk events
                    if len(self.high_risk_events) > 100:
                        self.high_risk_events.pop(0)
                
                # Log to file as JSON line
                log_entry = json.dumps(record.to_dict(), default=str)
                self.logger.info(log_entry)
                
                # Alert on high-risk events
                if risk_score >= 8:
                    self._alert_high_risk_event(record)
                
                # Check for log rotation
                if self.event_counts[event_type.value] % 1000 == 0:
                    self._check_log_rotation()
                
            except Exception as e:
                # Fallback logging - we must not fail on audit logging
                self.logger.error(f"Failed to log security event: {e}")
    
    def _alert_high_risk_event(self, record: AuditRecord) -> None:
        """Send alert for high-risk security events."""
        alert_msg = (f"HIGH RISK SECURITY EVENT: {record.event_type.value} "
                    f"(score: {record.risk_score}) - {record.details}")
        
        # Log with WARNING level for visibility
        console_logger = logging.getLogger("security_alert")
        console_logger.warning(alert_msg)
    
    def log_authentication_attempt(self, user_id: str, success: bool, 
                                 ip_address: str, details: Dict[str, Any] = None) -> None:
        """Log authentication attempt."""
        event_type = SecurityEvent.AUTHENTICATION_SUCCESS if success else SecurityEvent.AUTHENTICATION_FAILURE
        risk_score = 1 if success else 5
        
        self.log_security_event(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            action="login",
            result="success" if success else "failure",
            details=details or {},
            risk_score=risk_score
        )
    
    def log_data_access(self, user_id: str, resource: str, action: str, 
                       success: bool, details: Dict[str, Any] = None) -> None:
        """Log data access attempts."""
        result = "success" if success else "denied"
        risk_score = 1 if success else 6
        
        self.log_security_event(
            event_type=SecurityEvent.DATA_ACCESS,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            risk_score=risk_score
        )
    
    def log_validation_failure(self, input_type: str, error_message: str, 
                             source: str = None, details: Dict[str, Any] = None) -> None:
        """Log input validation failures."""
        self.log_security_event(
            event_type=SecurityEvent.INPUT_VALIDATION_FAILURE,
            resource=input_type,
            action="validate",
            result="failure",
            details={
                "error_message": error_message,
                "source": source,
                **(details or {})
            },
            risk_score=4
        )
    
    def log_suspicious_activity(self, activity_type: str, details: Dict[str, Any], 
                              risk_score: int = 8) -> None:
        """Log suspicious activities that may indicate security threats."""
        self.log_security_event(
            event_type=SecurityEvent.SUSPICIOUS_ACTIVITY,
            action=activity_type,
            result="detected",
            details=details,
            risk_score=risk_score
        )
    
    def log_configuration_change(self, user_id: str, component: str, 
                               old_value: Any, new_value: Any) -> None:
        """Log configuration changes."""
        self.log_security_event(
            event_type=SecurityEvent.CONFIGURATION_CHANGE,
            user_id=user_id,
            resource=component,
            action="modify",
            result="success",
            details={
                "old_value": str(old_value)[:1000],  # Limit size
                "new_value": str(new_value)[:1000],
                "change_hash": hashlib.md5(f"{old_value}{new_value}".encode()).hexdigest()
            },
            risk_score=3
        )
    
    def log_system_error(self, component: str, error_type: str, error_message: str,
                        stack_trace: str = None) -> None:
        """Log system errors that may have security implications."""
        self.log_security_event(
            event_type=SecurityEvent.SYSTEM_ERROR,
            resource=component,
            action=error_type,
            result="error",
            details={
                "error_message": error_message[:1000],
                "stack_trace": stack_trace[:2000] if stack_trace else None,
                "error_hash": hashlib.md5(error_message.encode()).hexdigest()
            },
            risk_score=2
        )
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics."""
        with self._lock:
            uptime = time.time() - self.start_time
            
            return {
                "uptime_seconds": uptime,
                "total_events": sum(self.event_counts.values()),
                "events_by_type": self.event_counts.copy(),
                "high_risk_events_count": len(self.high_risk_events),
                "events_per_hour": sum(self.event_counts.values()) / (uptime / 3600) if uptime > 0 else 0,
                "log_file_size_bytes": os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0,
                "recent_high_risk_events": len([e for e in self.high_risk_events 
                                              if time.time() - e.timestamp < 3600])  # Last hour
            }
    
    def get_recent_events(self, event_type: Optional[SecurityEvent] = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent security events."""
        try:
            events = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    lines = f.readlines()[-limit:]  # Get last N lines
                    
                for line in reversed(lines):  # Most recent first
                    try:
                        event = json.loads(line.strip())
                        if event_type is None or event.get('event_type') == event_type.value:
                            events.append(event)
                    except json.JSONDecodeError:
                        continue
                        
            return events[:limit]
            
        except Exception as e:
            self.logger.error(f"Error reading recent events: {e}")
            return []
    
    def analyze_threat_patterns(self) -> Dict[str, Any]:
        """Analyze recent events for threat patterns."""
        recent_events = self.get_recent_events(limit=1000)
        
        analysis = {
            "failed_auth_attempts": 0,
            "validation_failures": 0,
            "suspicious_activities": 0,
            "unique_ip_addresses": set(),
            "repeated_failures_by_ip": {},
            "threat_indicators": []
        }
        
        for event in recent_events:
            event_type = event.get('event_type')
            ip_addr = event.get('ip_address')
            
            # Count failure types
            if event_type == SecurityEvent.AUTHENTICATION_FAILURE.value:
                analysis["failed_auth_attempts"] += 1
                if ip_addr:
                    analysis["repeated_failures_by_ip"][ip_addr] = analysis["repeated_failures_by_ip"].get(ip_addr, 0) + 1
                    
            elif event_type == SecurityEvent.INPUT_VALIDATION_FAILURE.value:
                analysis["validation_failures"] += 1
                
            elif event_type == SecurityEvent.SUSPICIOUS_ACTIVITY.value:
                analysis["suspicious_activities"] += 1
            
            if ip_addr:
                analysis["unique_ip_addresses"].add(ip_addr)
        
        # Detect threat patterns
        for ip, failure_count in analysis["repeated_failures_by_ip"].items():
            if failure_count >= 5:
                analysis["threat_indicators"].append({
                    "type": "brute_force",
                    "source_ip": ip,
                    "failure_count": failure_count,
                    "severity": "high" if failure_count >= 10 else "medium"
                })
        
        # Convert set to count for JSON serialization
        analysis["unique_ip_count"] = len(analysis["unique_ip_addresses"])
        del analysis["unique_ip_addresses"]
        
        return analysis


# Global security audit logger
security_logger = SecurityAuditLogger()

def log_security_event(**kwargs) -> None:
    """Convenience function for logging security events."""
    security_logger.log_security_event(**kwargs)