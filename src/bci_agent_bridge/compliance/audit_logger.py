"""
Compliance Audit Logger for BCI-Agent-Bridge.
Provides comprehensive audit logging for regulatory compliance.
"""

import logging
import json
import time
import hashlib
import threading
import queue
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from pathlib import Path
import gzip
import datetime


class AuditEventType(Enum):
    """Types of audit events."""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    USER_AUTHENTICATION = "user_authentication"
    SYSTEM_CONFIGURATION = "system_configuration"
    SECURITY_EVENT = "security_event"
    PRIVACY_EVENT = "privacy_event"
    COMPLIANCE_EVENT = "compliance_event"
    SYSTEM_EVENT = "system_event"
    ERROR_EVENT = "error_event"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event record."""
    event_id: str
    timestamp: float
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    session_id: Optional[str]
    source_component: str
    action: str
    resource: Optional[str]
    outcome: str  # success, failure, partial
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        return data


@dataclass
class AuditConfiguration:
    """Audit logging configuration."""
    enabled: bool = True
    log_level: AuditSeverity = AuditSeverity.INFO
    max_log_size_mb: int = 100
    retention_days: int = 2555  # 7 years
    compress_logs: bool = True
    real_time_alerts: bool = True
    include_stack_traces: bool = False
    mask_sensitive_data: bool = True
    buffer_size: int = 1000
    flush_interval_seconds: int = 30


class ComplianceAuditLogger:
    """
    Comprehensive audit logger for compliance requirements.
    """
    
    def __init__(self, storage_path: Optional[Path] = None, 
                 config: Optional[AuditConfiguration] = None):
        self.storage_path = storage_path or Path("audit_logs")
        self.config = config or AuditConfiguration()
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        self._initialize_storage()
        
        # Event buffer for batched writes
        self.event_buffer: queue.Queue = queue.Queue(maxsize=self.config.buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Background processing
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._flush_thread.start()
        
        # Event handlers
        self.event_handlers: Dict[AuditEventType, List[Callable]] = {}
        
        # Metrics
        self.metrics = {
            "events_logged": 0,
            "events_dropped": 0,
            "last_flush": time.time(),
            "buffer_size": 0
        }
        
        # Integrity verification
        self.log_checksums: Dict[str, str] = {}
        
        self.logger.info("Compliance audit logger initialized")
    
    def _initialize_storage(self) -> None:
        """Initialize audit log storage."""
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.storage_path / "events").mkdir(exist_ok=True)
        (self.storage_path / "daily").mkdir(exist_ok=True)
        (self.storage_path / "archived").mkdir(exist_ok=True)
        (self.storage_path / "integrity").mkdir(exist_ok=True)
    
    def log_event(self, event_type: AuditEventType, action: str, 
                  user_id: Optional[str] = None, session_id: Optional[str] = None,
                  resource: Optional[str] = None, outcome: str = "success",
                  severity: AuditSeverity = AuditSeverity.INFO,
                  source_component: str = "bci-agent-bridge",
                  details: Optional[Dict[str, Any]] = None,
                  ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  correlation_id: Optional[str] = None) -> str:
        """Log an audit event."""
        
        if not self.config.enabled:
            return ""
        
        # Check severity filter
        if severity.value == "info" and self.config.log_level in [AuditSeverity.WARNING, AuditSeverity.ERROR, AuditSeverity.CRITICAL]:
            return ""
        
        event_id = str(uuid.uuid4())
        
        # Sanitize details if needed
        sanitized_details = self._sanitize_details(details or {}) if self.config.mask_sensitive_data else details or {}
        
        event = AuditEvent(
            event_id=event_id,
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            source_component=source_component,
            action=action,
            resource=resource,
            outcome=outcome,
            details=sanitized_details,
            ip_address=ip_address,
            user_agent=user_agent,
            correlation_id=correlation_id
        )
        
        try:
            # Add to buffer
            self.event_buffer.put_nowait(event)
            self.metrics["buffer_size"] = self.event_buffer.qsize()
            
            # Trigger event handlers
            self._trigger_event_handlers(event)
            
            # Immediate flush for critical events
            if severity == AuditSeverity.CRITICAL:
                self._flush_buffer()
            
            self.metrics["events_logged"] += 1
            
        except queue.Full:
            # Buffer is full, drop the event
            self.metrics["events_dropped"] += 1
            self.logger.error(f"Audit event buffer full, dropped event: {event_id}")
        
        return event_id
    
    def log_data_access(self, user_id: str, resource: str, action: str,
                       outcome: str = "success", session_id: Optional[str] = None,
                       ip_address: Optional[str] = None, **kwargs) -> str:
        """Log data access event."""
        return self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            action=action,
            user_id=user_id,
            session_id=session_id,
            resource=resource,
            outcome=outcome,
            severity=AuditSeverity.INFO if outcome == "success" else AuditSeverity.WARNING,
            ip_address=ip_address,
            details=kwargs
        )
    
    def log_security_event(self, action: str, severity: AuditSeverity,
                          user_id: Optional[str] = None, 
                          details: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Log security event."""
        return self.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            action=action,
            user_id=user_id,
            severity=severity,
            outcome="detected",
            details=details,
            **kwargs
        )
    
    def log_privacy_event(self, action: str, user_id: Optional[str] = None,
                         resource: Optional[str] = None, 
                         details: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Log privacy-related event."""
        return self.log_event(
            event_type=AuditEventType.PRIVACY_EVENT,
            action=action,
            user_id=user_id,
            resource=resource,
            severity=AuditSeverity.INFO,
            outcome="success",
            details=details,
            **kwargs
        )
    
    def log_compliance_event(self, action: str, outcome: str = "success",
                           details: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Log compliance-related event."""
        severity = AuditSeverity.INFO if outcome == "success" else AuditSeverity.WARNING
        return self.log_event(
            event_type=AuditEventType.COMPLIANCE_EVENT,
            action=action,
            outcome=outcome,
            severity=severity,
            details=details,
            **kwargs
        )
    
    def log_system_event(self, action: str, outcome: str = "success",
                        details: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Log system event."""
        return self.log_event(
            event_type=AuditEventType.SYSTEM_EVENT,
            action=action,
            outcome=outcome,
            severity=AuditSeverity.INFO,
            details=details,
            **kwargs
        )
    
    def log_error_event(self, action: str, error: str, 
                       details: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Log error event."""
        error_details = details or {}
        error_details["error_message"] = error
        
        return self.log_event(
            event_type=AuditEventType.ERROR_EVENT,
            action=action,
            outcome="failure",
            severity=AuditSeverity.ERROR,
            details=error_details,
            **kwargs
        )
    
    def register_event_handler(self, event_type: AuditEventType, 
                              handler: Callable[[AuditEvent], None]) -> None:
        """Register an event handler for specific event types."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def _trigger_event_handlers(self, event: AuditEvent) -> None:
        """Trigger registered event handlers."""
        handlers = self.event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Error in audit event handler: {e}")
    
    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive data from details."""
        sanitized = details.copy()
        
        # List of sensitive keys to mask
        sensitive_keys = [
            "password", "token", "api_key", "secret", "private_key",
            "ssn", "social_security", "credit_card", "bank_account",
            "neural_data", "raw_data", "personal_info"
        ]
        
        for key, value in sanitized.items():
            # Check if key contains sensitive terms
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                if isinstance(value, str) and len(value) > 4:
                    # Show only first and last 2 characters
                    sanitized[key] = f"{value[:2]}***{value[-2:]}"
                elif isinstance(value, (int, float)):
                    sanitized[key] = "***"
                elif isinstance(value, dict):
                    sanitized[key] = {"masked": True, "keys": len(value)}
                elif isinstance(value, list):
                    sanitized[key] = {"masked": True, "count": len(value)}
                else:
                    sanitized[key] = "***"
        
        return sanitized
    
    def _flush_worker(self) -> None:
        """Background worker to flush events to disk."""
        while self._running:
            try:
                # Wait for flush interval or immediate flush signal
                time.sleep(self.config.flush_interval_seconds)
                self._flush_buffer()
            except Exception as e:
                self.logger.error(f"Error in flush worker: {e}")
    
    def _flush_buffer(self) -> None:
        """Flush buffered events to disk."""
        if self.event_buffer.empty():
            return
        
        events_to_write = []
        
        # Drain the buffer
        with self.buffer_lock:
            while not self.event_buffer.empty():
                try:
                    event = self.event_buffer.get_nowait()
                    events_to_write.append(event)
                except queue.Empty:
                    break
        
        if not events_to_write:
            return
        
        # Write events to files
        self._write_events(events_to_write)
        self.metrics["last_flush"] = time.time()
        self.metrics["buffer_size"] = self.event_buffer.qsize()
    
    def _write_events(self, events: List[AuditEvent]) -> None:
        """Write events to log files."""
        # Group events by date for daily log files
        events_by_date = {}
        
        for event in events:
            date = datetime.datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d")
            if date not in events_by_date:
                events_by_date[date] = []
            events_by_date[date].append(event)
        
        # Write to daily log files
        for date, date_events in events_by_date.items():
            log_file = self.storage_path / "daily" / f"{date}.jsonl"
            
            # Append events to daily log
            with open(log_file, 'a') as f:
                for event in date_events:
                    json.dump(event.to_dict(), f)
                    f.write('\n')
            
            # Update integrity checksum
            self._update_log_checksum(log_file)
        
        # Also write to main event log
        main_log = self.storage_path / "events" / "audit_events.jsonl"
        with open(main_log, 'a') as f:
            for event in events:
                json.dump(event.to_dict(), f)
                f.write('\n')
        
        # Compress old logs if enabled
        if self.config.compress_logs:
            self._compress_old_logs()
    
    def _update_log_checksum(self, log_file: Path) -> None:
        """Update integrity checksum for log file."""
        if not log_file.exists():
            return
        
        # Calculate checksum of log file
        hasher = hashlib.sha256()
        with open(log_file, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        checksum = hasher.hexdigest()
        self.log_checksums[str(log_file)] = checksum
        
        # Save checksum to integrity file
        integrity_file = self.storage_path / "integrity" / f"{log_file.name}.sha256"
        with open(integrity_file, 'w') as f:
            f.write(f"{checksum}  {log_file.name}\n")
    
    def _compress_old_logs(self) -> None:
        """Compress log files older than 7 days."""
        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days ago
        daily_log_dir = self.storage_path / "daily"
        
        for log_file in daily_log_dir.glob("*.jsonl"):
            if log_file.stat().st_mtime < cutoff_time:
                # Compress the file
                compressed_file = self.storage_path / "archived" / f"{log_file.name}.gz"
                
                with open(log_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        f_out.writelines(f_in)
                
                # Remove original file
                log_file.unlink()
                
                self.logger.info(f"Compressed old log file: {log_file.name}")
    
    def verify_log_integrity(self) -> Dict[str, bool]:
        """Verify integrity of log files using checksums."""
        results = {}
        
        for log_file_str, stored_checksum in self.log_checksums.items():
            log_file = Path(log_file_str)
            
            if not log_file.exists():
                results[log_file_str] = False
                continue
            
            # Recalculate checksum
            hasher = hashlib.sha256()
            with open(log_file, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            current_checksum = hasher.hexdigest()
            results[log_file_str] = current_checksum == stored_checksum
            
            if not results[log_file_str]:
                self.logger.critical(f"Log integrity verification failed: {log_file_str}")
        
        return results
    
    def search_events(self, start_time: Optional[float] = None,
                     end_time: Optional[float] = None,
                     event_type: Optional[AuditEventType] = None,
                     user_id: Optional[str] = None,
                     action: Optional[str] = None,
                     severity: Optional[AuditSeverity] = None,
                     limit: int = 1000) -> List[Dict[str, Any]]:
        """Search audit events with filters."""
        results = []
        count = 0
        
        # Search in daily log files
        daily_log_dir = self.storage_path / "daily"
        log_files = sorted(daily_log_dir.glob("*.jsonl"), reverse=True)
        
        for log_file in log_files:
            if count >= limit:
                break
            
            with open(log_file, 'r') as f:
                for line in f:
                    if count >= limit:
                        break
                    
                    try:
                        event_data = json.loads(line.strip())
                        
                        # Apply filters
                        if start_time and event_data.get('timestamp', 0) < start_time:
                            continue
                        if end_time and event_data.get('timestamp', 0) > end_time:
                            continue
                        if event_type and event_data.get('event_type') != event_type.value:
                            continue
                        if user_id and event_data.get('user_id') != user_id:
                            continue
                        if action and event_data.get('action') != action:
                            continue
                        if severity and event_data.get('severity') != severity.value:
                            continue
                        
                        results.append(event_data)
                        count += 1
                        
                    except json.JSONDecodeError:
                        continue
        
        return results
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit logging summary."""
        # Count events by type and severity
        event_counts = {event_type.value: 0 for event_type in AuditEventType}
        severity_counts = {severity.value: 0 for severity in AuditSeverity}
        
        # Read from main log file
        main_log = self.storage_path / "events" / "audit_events.jsonl"
        total_events = 0
        
        if main_log.exists():
            with open(main_log, 'r') as f:
                for line in f:
                    try:
                        event_data = json.loads(line.strip())
                        total_events += 1
                        
                        event_type = event_data.get('event_type', 'unknown')
                        if event_type in event_counts:
                            event_counts[event_type] += 1
                        
                        severity = event_data.get('severity', 'info')
                        if severity in severity_counts:
                            severity_counts[severity] += 1
                            
                    except json.JSONDecodeError:
                        continue
        
        # Calculate storage usage
        storage_usage = sum(
            f.stat().st_size for f in self.storage_path.rglob("*") 
            if f.is_file()
        )
        
        return {
            "total_events": total_events,
            "events_by_type": event_counts,
            "events_by_severity": severity_counts,
            "buffer_metrics": self.metrics,
            "storage_usage_bytes": storage_usage,
            "log_files_count": len(list(self.storage_path.glob("**/*.jsonl"))),
            "compressed_files_count": len(list(self.storage_path.glob("**/*.gz"))),
            "integrity_status": len(self.verify_log_integrity()),
            "configuration": asdict(self.config)
        }
    
    def cleanup_old_logs(self) -> int:
        """Clean up logs older than retention period."""
        retention_seconds = self.config.retention_days * 24 * 3600
        cutoff_time = time.time() - retention_seconds
        
        cleaned_count = 0
        
        # Clean up daily logs
        daily_log_dir = self.storage_path / "daily"
        for log_file in daily_log_dir.glob("*.jsonl"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                cleaned_count += 1
        
        # Clean up archived logs
        archived_dir = self.storage_path / "archived"
        for archive_file in archived_dir.glob("*.gz"):
            if archive_file.stat().st_mtime < cutoff_time:
                archive_file.unlink()
                cleaned_count += 1
        
        # Clean up integrity files
        integrity_dir = self.storage_path / "integrity"
        for integrity_file in integrity_dir.glob("*.sha256"):
            if integrity_file.stat().st_mtime < cutoff_time:
                integrity_file.unlink()
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old log files")
        
        return cleaned_count
    
    def shutdown(self) -> None:
        """Shutdown audit logger gracefully."""
        self._running = False
        
        # Flush remaining events
        self._flush_buffer()
        
        # Wait for flush thread to finish
        if self._flush_thread.is_alive():
            self._flush_thread.join(timeout=10)
        
        self.logger.info("Audit logger shutdown complete")