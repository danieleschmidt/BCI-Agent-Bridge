"""
Comprehensive structured logging with correlation IDs and context propagation for BCI system.
Provides medical-grade logging with audit trails, security, and compliance features.
"""

import logging
import json
import time
import uuid
import threading
import asyncio
import sys
import os
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import contextvars
from pathlib import Path
import traceback
import hashlib
import hmac


class LogLevel(Enum):
    """Enhanced log levels for BCI systems."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    MEDICAL_ALERT = 55  # Special level for medical safety alerts
    SECURITY_ALERT = 60  # Special level for security incidents


class LogCategory(Enum):
    """Categories for BCI system logging."""
    SYSTEM = "system"
    NEURAL_PROCESSING = "neural_processing"
    SIGNAL_QUALITY = "signal_quality"
    DECODING = "decoding"
    BCI_DEVICE = "bci_device"
    CLAUDE_INTEGRATION = "claude_integration"
    MEDICAL_SAFETY = "medical_safety"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    USER_INTERACTION = "user_interaction"
    API = "api"
    DATA_FLOW = "data_flow"


@dataclass
class LogContext:
    """Structured context for logging."""
    # Core identifiers
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # BCI-specific context
    neural_session_id: Optional[str] = None
    patient_id: Optional[str] = None  # Anonymized/hashed patient identifier
    device_id: Optional[str] = None
    paradigm: Optional[str] = None  # P300, SSVEP, Motor Imagery, etc.
    
    # Medical context
    clinical_trial_id: Optional[str] = None
    safety_level: Optional[str] = None
    medical_professional_id: Optional[str] = None
    
    # Technical context
    service_name: str = "bci-agent-bridge"
    version: Optional[str] = None
    environment: str = "production"
    hostname: Optional[str] = None
    thread_id: Optional[str] = None
    process_id: Optional[int] = None
    
    # User context (anonymized)
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    
    # Custom fields
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result

    def copy_with_updates(self, **updates) -> 'LogContext':
        """Create a copy with updated fields."""
        data = asdict(self)
        data.update(updates)
        return LogContext(**data)


@dataclass
class StructuredLogRecord:
    """Structured log record for BCI systems."""
    timestamp: float
    level: LogLevel
    category: LogCategory
    message: str
    context: LogContext
    
    # Additional structured data
    fields: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Error information
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Medical/Safety specific
    patient_safety_flag: bool = False
    data_privacy_flag: bool = False
    audit_flag: bool = False
    
    # Performance data
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Security data
    security_event: bool = False
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        data = {
            "timestamp": self.timestamp,
            "timestamp_iso": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ', time.gmtime(self.timestamp)),
            "level": self.level.name,
            "level_value": self.level.value,
            "category": self.category.value,
            "message": self.message,
            "context": self.context.to_dict(),
            "fields": self.fields,
            "tags": self.tags
        }
        
        # Add error information if present
        if self.exception_type:
            data["error"] = {
                "type": self.exception_type,
                "message": self.exception_message,
                "stack_trace": self.stack_trace
            }
        
        # Add flags
        if self.patient_safety_flag:
            data["patient_safety_flag"] = True
        if self.data_privacy_flag:
            data["data_privacy_flag"] = True
        if self.audit_flag:
            data["audit_flag"] = True
        if self.security_event:
            data["security_event"] = True
        
        # Add performance data
        if self.duration_ms is not None:
            data["performance"] = {
                "duration_ms": self.duration_ms,
                "memory_usage_mb": self.memory_usage_mb,
                "cpu_usage_percent": self.cpu_usage_percent
            }
        
        # Add security data
        if self.ip_address or self.user_agent:
            data["request_info"] = {
                "ip_address": self.ip_address,
                "user_agent": self.user_agent
            }
        
        return json.dumps(data, default=str, separators=(',', ':'))


class LogContextManager:
    """Manages log context propagation."""
    
    def __init__(self):
        self._context_var: contextvars.ContextVar[Optional[LogContext]] = (
            contextvars.ContextVar('log_context', default=None)
        )
    
    def get_context(self) -> Optional[LogContext]:
        """Get current log context."""
        return self._context_var.get()
    
    def set_context(self, context: LogContext) -> contextvars.Token:
        """Set log context and return token."""
        return self._context_var.set(context)
    
    def update_context(self, **updates) -> Optional[contextvars.Token]:
        """Update current context with new fields."""
        current = self.get_context()
        if current:
            new_context = current.copy_with_updates(**updates)
            return self.set_context(new_context)
        return None
    
    def clear_context(self) -> None:
        """Clear current context."""
        self._context_var.set(None)


class SecurityLogger:
    """Special security-focused logger with tamper protection."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
    
    def create_security_signature(self, record: StructuredLogRecord) -> str:
        """Create HMAC signature for log record integrity."""
        # Create canonical representation for signing
        canonical_data = {
            "timestamp": record.timestamp,
            "level": record.level.value,
            "message": record.message,
            "correlation_id": record.context.correlation_id
        }
        canonical_str = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'))
        
        signature = hmac.new(
            self.secret_key,
            canonical_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_security_signature(self, record_data: Dict[str, Any], signature: str) -> bool:
        """Verify log record signature."""
        try:
            canonical_data = {
                "timestamp": record_data["timestamp"],
                "level": record_data["level_value"],
                "message": record_data["message"],
                "correlation_id": record_data["context"]["correlation_id"]
            }
            canonical_str = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'))
            
            expected_signature = hmac.new(
                self.secret_key,
                canonical_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False


class StructuredLogger:
    """
    Main structured logger for BCI systems with comprehensive features.
    """
    
    def __init__(self, 
                 name: str = "bci-logger",
                 level: LogLevel = LogLevel.INFO,
                 enable_context_propagation: bool = True,
                 enable_audit_trail: bool = True,
                 enable_security_signing: bool = True,
                 security_key: Optional[str] = None,
                 max_field_length: int = 10000,
                 enable_sanitization: bool = True):
        
        self.name = name
        self.level = level
        self.enable_context_propagation = enable_context_propagation
        self.enable_audit_trail = enable_audit_trail
        self.enable_security_signing = enable_security_signing
        self.max_field_length = max_field_length
        self.enable_sanitization = enable_sanitization
        
        # Context management
        self.context_manager = LogContextManager() if enable_context_propagation else None
        
        # Security features
        if enable_security_signing:
            security_key = security_key or os.environ.get('BCI_LOG_SECRET_KEY', 'default-key-change-in-production')
            self.security_logger = SecurityLogger(security_key)
        else:
            self.security_logger = None
        
        # Handlers and processors
        self.handlers: List[Callable[[StructuredLogRecord], None]] = []
        self.processors: List[Callable[[StructuredLogRecord], StructuredLogRecord]] = []
        
        # Statistics
        self.stats = {
            'logs_processed': 0,
            'logs_by_level': defaultdict(int),
            'logs_by_category': defaultdict(int),
            'security_events': 0,
            'medical_alerts': 0,
            'audit_events': 0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Setup default handlers
        self._setup_default_handlers()
        
        # Setup default processors
        self._setup_default_processors()

    def _setup_default_handlers(self):
        """Setup default log handlers."""
        # Console handler for development
        console_handler = ConsoleLogHandler()
        self.add_handler(console_handler)
        
        # File handler for production logs
        file_handler = FileLogHandler("logs/bci_structured.log")
        self.add_handler(file_handler)
        
        # Audit handler for compliance
        if self.enable_audit_trail:
            audit_handler = AuditLogHandler("logs/bci_audit.log")
            self.add_handler(audit_handler)
        
        # Security handler for security events
        security_handler = SecurityLogHandler("logs/bci_security.log")
        self.add_handler(security_handler)

    def _setup_default_processors(self):
        """Setup default log processors."""
        # Sanitization processor
        if self.enable_sanitization:
            self.add_processor(SanitizationProcessor())
        
        # Context enrichment processor
        self.add_processor(ContextEnrichmentProcessor())
        
        # Performance tracking processor
        self.add_processor(PerformanceProcessor())
        
        # Medical safety processor
        self.add_processor(MedicalSafetyProcessor())

    def add_handler(self, handler: Callable[[StructuredLogRecord], None]):
        """Add log handler."""
        self.handlers.append(handler)

    def add_processor(self, processor: Callable[[StructuredLogRecord], StructuredLogRecord]):
        """Add log processor."""
        self.processors.append(processor)

    def with_context(self, **context_updates) -> 'StructuredLogger':
        """Create logger with additional context."""
        if self.context_manager:
            self.context_manager.update_context(**context_updates)
        return self

    def with_correlation_id(self, correlation_id: str) -> 'StructuredLogger':
        """Create logger with specific correlation ID."""
        return self.with_context(correlation_id=correlation_id)

    def with_neural_session(self, session_id: str, patient_id: str = None, 
                           device_id: str = None, paradigm: str = None) -> 'StructuredLogger':
        """Create logger with neural session context."""
        context_updates = {
            'neural_session_id': session_id,
            'paradigm': paradigm
        }
        if patient_id:
            # Hash patient ID for privacy
            context_updates['patient_id'] = hashlib.sha256(patient_id.encode()).hexdigest()[:16]
        if device_id:
            context_updates['device_id'] = device_id
            
        return self.with_context(**context_updates)

    def _create_base_context(self) -> LogContext:
        """Create base log context."""
        # Get current context if available
        if self.context_manager:
            current_context = self.context_manager.get_context()
            if current_context:
                return current_context
        
        # Create new context
        return LogContext(
            service_name=self.name,
            hostname=os.uname().nodename if hasattr(os, 'uname') else None,
            process_id=os.getpid(),
            thread_id=threading.get_ident()
        )

    def _sanitize_field(self, value: Any) -> Any:
        """Sanitize field value for logging."""
        if not self.enable_sanitization:
            return value
        
        if isinstance(value, str):
            # Truncate long strings
            if len(value) > self.max_field_length:
                value = value[:self.max_field_length] + "...[truncated]"
            
            # Remove sensitive patterns
            sensitive_patterns = [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            ]
            
            import re
            for pattern in sensitive_patterns:
                value = re.sub(pattern, '[REDACTED]', value, flags=re.IGNORECASE)
        
        return value

    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on level."""
        return level.value >= self.level.value

    def log(self, level: LogLevel, category: LogCategory, message: str,
           fields: Dict[str, Any] = None, tags: List[str] = None,
           exception: Exception = None, duration_ms: float = None,
           patient_safety: bool = False, audit: bool = False,
           security: bool = False, **kwargs):
        """Core logging method."""
        
        if not self._should_log(level):
            return
        
        # Create log record
        context = self._create_base_context()
        
        # Update context with any additional fields
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
            else:
                context.custom_fields[key] = value
        
        record = StructuredLogRecord(
            timestamp=time.time(),
            level=level,
            category=category,
            message=message,
            context=context,
            fields=fields or {},
            tags=tags or [],
            duration_ms=duration_ms,
            patient_safety_flag=patient_safety,
            audit_flag=audit,
            security_event=security
        )
        
        # Add exception information if present
        if exception:
            record.exception_type = type(exception).__name__
            record.exception_message = str(exception)
            record.stack_trace = ''.join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
        
        # Process through processors
        for processor in self.processors:
            try:
                record = processor(record)
            except Exception as e:
                # Don't fail logging due to processor errors
                print(f"Log processor error: {e}", file=sys.stderr)
        
        # Update statistics
        with self._lock:
            self.stats['logs_processed'] += 1
            self.stats['logs_by_level'][level.name] += 1
            self.stats['logs_by_category'][category.value] += 1
            
            if security:
                self.stats['security_events'] += 1
            if level == LogLevel.MEDICAL_ALERT:
                self.stats['medical_alerts'] += 1
            if audit:
                self.stats['audit_events'] += 1
        
        # Send to handlers
        for handler in self.handlers:
            try:
                handler(record)
            except Exception as e:
                # Don't fail logging due to handler errors
                print(f"Log handler error: {e}", file=sys.stderr)

    # Convenience methods for different log levels
    def trace(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log trace message."""
        self.log(LogLevel.TRACE, category, message, **kwargs)

    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, category, message, **kwargs)

    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, category, message, **kwargs)

    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, category, message, **kwargs)

    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
             exception: Exception = None, **kwargs):
        """Log error message."""
        self.log(LogLevel.ERROR, category, message, exception=exception, **kwargs)

    def critical(self, message: str, category: LogCategory = LogCategory.SYSTEM,
                exception: Exception = None, **kwargs):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, category, message, exception=exception, **kwargs)

    def medical_alert(self, message: str, **kwargs):
        """Log medical alert."""
        self.log(LogLevel.MEDICAL_ALERT, LogCategory.MEDICAL_SAFETY, message, 
                patient_safety=True, audit=True, **kwargs)

    def security_alert(self, message: str, **kwargs):
        """Log security alert."""
        self.log(LogLevel.SECURITY_ALERT, LogCategory.SECURITY, message,
                security=True, audit=True, **kwargs)

    # BCI-specific logging methods
    def neural_data_received(self, channels: int, sampling_rate: int, quality: float, **kwargs):
        """Log neural data reception."""
        self.info("Neural data received", LogCategory.NEURAL_PROCESSING,
                 fields={"channels": channels, "sampling_rate": sampling_rate, "quality": quality},
                 **kwargs)

    def signal_quality_change(self, old_quality: float, new_quality: float, **kwargs):
        """Log signal quality change."""
        level = LogLevel.WARNING if new_quality < 0.5 else LogLevel.INFO
        self.log(level, LogCategory.SIGNAL_QUALITY, "Signal quality changed",
                fields={"old_quality": old_quality, "new_quality": new_quality, 
                       "change": new_quality - old_quality}, **kwargs)

    def decoding_result(self, paradigm: str, confidence: float, intention: str, 
                       latency_ms: float, **kwargs):
        """Log decoding result."""
        self.info("Intention decoded", LogCategory.DECODING,
                 fields={"paradigm": paradigm, "confidence": confidence, 
                        "intention": intention, "latency_ms": latency_ms},
                 duration_ms=latency_ms, **kwargs)

    def claude_interaction(self, request_type: str, response_time_ms: float, 
                          tokens_used: int, safety_flags: List[str] = None, **kwargs):
        """Log Claude interaction."""
        fields = {
            "request_type": request_type,
            "response_time_ms": response_time_ms,
            "tokens_used": tokens_used
        }
        
        if safety_flags:
            fields["safety_flags"] = safety_flags
            tags = ["claude", "safety_flagged"]
        else:
            tags = ["claude"]
        
        self.info("Claude interaction", LogCategory.CLAUDE_INTEGRATION,
                 fields=fields, tags=tags, duration_ms=response_time_ms, **kwargs)

    def device_status_change(self, device_id: str, old_status: str, new_status: str, **kwargs):
        """Log device status change."""
        level = LogLevel.ERROR if new_status in ["disconnected", "error"] else LogLevel.INFO
        self.log(level, LogCategory.BCI_DEVICE, "Device status changed",
                fields={"device_id": device_id, "old_status": old_status, 
                       "new_status": new_status}, **kwargs)

    def user_action(self, action: str, user_id: str, result: str = "success", **kwargs):
        """Log user action."""
        # Hash user ID for privacy
        hashed_user_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        
        self.info("User action", LogCategory.USER_INTERACTION,
                 fields={"action": action, "user_id": hashed_user_id, "result": result},
                 audit=True, **kwargs)

    def compliance_event(self, event_type: str, details: Dict[str, Any], **kwargs):
        """Log compliance event."""
        self.info(f"Compliance event: {event_type}", LogCategory.COMPLIANCE,
                 fields=details, audit=True, **kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self._lock:
            return {
                "logs_processed": self.stats['logs_processed'],
                "logs_by_level": dict(self.stats['logs_by_level']),
                "logs_by_category": dict(self.stats['logs_by_category']),
                "security_events": self.stats['security_events'],
                "medical_alerts": self.stats['medical_alerts'],
                "audit_events": self.stats['audit_events'],
                "handlers_count": len(self.handlers),
                "processors_count": len(self.processors)
            }


# Log processors
class LogProcessor:
    """Base class for log processors."""
    
    def __call__(self, record: StructuredLogRecord) -> StructuredLogRecord:
        return self.process(record)
    
    def process(self, record: StructuredLogRecord) -> StructuredLogRecord:
        raise NotImplementedError


class SanitizationProcessor(LogProcessor):
    """Processor that sanitizes sensitive data."""
    
    def process(self, record: StructuredLogRecord) -> StructuredLogRecord:
        # Sanitize message
        record.message = self._sanitize_string(record.message)
        
        # Sanitize fields
        sanitized_fields = {}
        for key, value in record.fields.items():
            if isinstance(value, str):
                sanitized_fields[key] = self._sanitize_string(value)
            else:
                sanitized_fields[key] = value
        record.fields = sanitized_fields
        
        return record
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize a string by removing/masking sensitive data."""
        import re
        
        # Mask email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     '[EMAIL]', text, flags=re.IGNORECASE)
        
        # Mask phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Mask credit card numbers
        text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CREDIT_CARD]', text)
        
        # Mask SSNs
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        return text


class ContextEnrichmentProcessor(LogProcessor):
    """Processor that enriches context with additional information."""
    
    def process(self, record: StructuredLogRecord) -> StructuredLogRecord:
        # Add environment information
        record.context.environment = os.environ.get('BCI_ENVIRONMENT', 'production')
        record.context.version = os.environ.get('BCI_VERSION', '1.0.0')
        
        # Add thread and process information
        record.context.thread_id = str(threading.get_ident())
        record.context.process_id = os.getpid()
        
        # Add hostname if not present
        if not record.context.hostname:
            record.context.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        
        return record


class PerformanceProcessor(LogProcessor):
    """Processor that adds performance information."""
    
    def process(self, record: StructuredLogRecord) -> StructuredLogRecord:
        # Add memory usage
        try:
            import psutil
            process = psutil.Process()
            record.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            record.cpu_usage_percent = process.cpu_percent()
        except ImportError:
            pass  # psutil not available
        except Exception:
            pass  # Error getting performance data
        
        return record


class MedicalSafetyProcessor(LogProcessor):
    """Processor that handles medical safety requirements."""
    
    def process(self, record: StructuredLogRecord) -> StructuredLogRecord:
        # Check for medical safety keywords
        safety_keywords = [
            'emergency', 'critical', 'patient', 'medical', 'safety',
            'seizure', 'abnormal', 'alert', 'danger', 'risk'
        ]
        
        message_lower = record.message.lower()
        if any(keyword in message_lower for keyword in safety_keywords):
            record.patient_safety_flag = True
            record.audit_flag = True
        
        # Special handling for neural signal quality
        if record.category == LogCategory.SIGNAL_QUALITY:
            quality = record.fields.get('new_quality', record.fields.get('quality'))
            if quality and quality < 0.3:  # Critical quality threshold
                record.patient_safety_flag = True
                record.level = LogLevel.MEDICAL_ALERT
        
        return record


# Log handlers
class LogHandler:
    """Base class for log handlers."""
    
    def __call__(self, record: StructuredLogRecord) -> None:
        self.handle(record)
    
    def handle(self, record: StructuredLogRecord) -> None:
        raise NotImplementedError


class ConsoleLogHandler(LogHandler):
    """Handler that outputs logs to console with color coding."""
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        self.colors = {
            LogLevel.TRACE: '\033[90m',      # Dark gray
            LogLevel.DEBUG: '\033[36m',      # Cyan
            LogLevel.INFO: '\033[32m',       # Green
            LogLevel.WARNING: '\033[33m',    # Yellow
            LogLevel.ERROR: '\033[31m',      # Red
            LogLevel.CRITICAL: '\033[35m',   # Magenta
            LogLevel.MEDICAL_ALERT: '\033[41m',  # Red background
            LogLevel.SECURITY_ALERT: '\033[45m', # Magenta background
        }
        self.reset_color = '\033[0m'
    
    def handle(self, record: StructuredLogRecord) -> None:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.timestamp))
        
        if self.use_colors and record.level in self.colors:
            color = self.colors[record.level]
            formatted = f"{color}[{timestamp}] {record.level.name} [{record.category.value}] {record.message}{self.reset_color}"
        else:
            formatted = f"[{timestamp}] {record.level.name} [{record.category.value}] {record.message}"
        
        # Add correlation ID if present
        if record.context.correlation_id:
            formatted += f" (correlation_id: {record.context.correlation_id[:8]}...)"
        
        # Add patient safety flag
        if record.patient_safety_flag:
            formatted += " [PATIENT_SAFETY]"
        
        # Add security flag
        if record.security_event:
            formatted += " [SECURITY]"
        
        print(formatted)
        
        # Print fields if present and debug level or lower
        if record.fields and record.level.value <= LogLevel.DEBUG.value:
            print(f"  Fields: {json.dumps(record.fields, default=str, indent=2)}")


class FileLogHandler(LogHandler):
    """Handler that outputs logs to files with rotation."""
    
    def __init__(self, file_path: str, max_size_mb: int = 100, backup_count: int = 5):
        self.file_path = Path(file_path)
        self.max_size_mb = max_size_mb
        self.backup_count = backup_count
        
        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
    
    def handle(self, record: StructuredLogRecord) -> None:
        with self._lock:
            # Check if rotation is needed
            if self.file_path.exists() and self.file_path.stat().st_size > self.max_size_mb * 1024 * 1024:
                self._rotate_file()
            
            # Write log record as JSON
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(record.to_json() + '\n')
    
    def _rotate_file(self):
        """Rotate log file."""
        for i in range(self.backup_count - 1, 0, -1):
            old_file = self.file_path.with_suffix(f'.{i}.log')
            new_file = self.file_path.with_suffix(f'.{i + 1}.log')
            
            if old_file.exists():
                if new_file.exists():
                    new_file.unlink()
                old_file.rename(new_file)
        
        # Move current file to .1
        if self.file_path.exists():
            backup_file = self.file_path.with_suffix('.1.log')
            if backup_file.exists():
                backup_file.unlink()
            self.file_path.rename(backup_file)


class AuditLogHandler(FileLogHandler):
    """Special handler for audit logs with enhanced security."""
    
    def handle(self, record: StructuredLogRecord) -> None:
        # Only handle audit events
        if not record.audit_flag:
            return
        
        super().handle(record)


class SecurityLogHandler(FileLogHandler):
    """Special handler for security events."""
    
    def handle(self, record: StructuredLogRecord) -> None:
        # Only handle security events
        if not record.security_event and record.level != LogLevel.SECURITY_ALERT:
            return
        
        super().handle(record)


# Context managers for logging
class LoggingContext:
    """Context manager for setting logging context."""
    
    def __init__(self, logger: StructuredLogger, **context):
        self.logger = logger
        self.context = context
        self.token = None
    
    def __enter__(self):
        if self.logger.context_manager:
            self.token = self.logger.context_manager.update_context(**self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token and self.logger.context_manager:
            # Context variables automatically restore on exit
            pass


# Global logger instance
_global_logger: Optional[StructuredLogger] = None


def get_logger(name: str = "bci-agent-bridge") -> StructuredLogger:
    """Get or create global structured logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger(name)
    return _global_logger


def configure_logging(level: LogLevel = LogLevel.INFO, 
                     enable_security: bool = True,
                     log_directory: str = "logs",
                     **kwargs) -> StructuredLogger:
    """Configure global logging with BCI-specific settings."""
    global _global_logger
    
    _global_logger = StructuredLogger(
        level=level,
        enable_security_signing=enable_security,
        **kwargs
    )
    
    # Create log directory
    Path(log_directory).mkdir(parents=True, exist_ok=True)
    
    return _global_logger


# Decorators for automatic logging
def log_function_calls(category: LogCategory = LogCategory.SYSTEM,
                      level: LogLevel = LogLevel.DEBUG,
                      log_args: bool = False,
                      log_result: bool = False):
    """Decorator to automatically log function calls."""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            start_time = time.time()
            
            fields = {"function": func.__name__, "module": func.__module__}
            if log_args:
                fields["args"] = str(args)
                fields["kwargs"] = str(kwargs)
            
            logger.log(level, category, f"Function {func.__name__} called", fields=fields)
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                success_fields = {"function": func.__name__, "duration_ms": duration_ms}
                if log_result and result is not None:
                    success_fields["result"] = str(result)
                
                logger.log(level, category, f"Function {func.__name__} completed", 
                          fields=success_fields, duration_ms=duration_ms)
                
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"Function {func.__name__} failed", category, 
                           exception=e, duration_ms=duration_ms,
                           fields={"function": func.__name__})
                raise
        
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def demo_structured_logging():
        print("BCI Structured Logging Demo")
        print("=" * 40)
        
        # Configure logging
        logger = configure_logging(
            level=LogLevel.DEBUG,
            name="bci-demo",
            enable_security=True
        )
        
        # Basic logging
        logger.info("BCI system starting up", LogCategory.SYSTEM)
        logger.debug("Debug information", fields={"component": "main", "version": "1.0.0"})
        
        # BCI-specific logging with context
        with LoggingContext(logger, neural_session_id="session_123", paradigm="P300"):
            logger.neural_data_received(channels=64, sampling_rate=250, quality=0.85)
            logger.signal_quality_change(old_quality=0.9, new_quality=0.7)
            
            # Simulate decoding
            logger.decoding_result(
                paradigm="P300",
                confidence=0.92,
                intention="select_letter_A",
                latency_ms=120
            )
            
            # Claude interaction
            logger.claude_interaction(
                request_type="intention_processing",
                response_time_ms=850,
                tokens_used=150,
                safety_flags=None
            )
        
        # Error handling
        try:
            raise ValueError("Simulated error for testing")
        except Exception as e:
            logger.error("Error in neural processing", LogCategory.NEURAL_PROCESSING, exception=e)
        
        # Medical safety alert
        logger.medical_alert("Signal quality critically low - patient safety concern",
                           fields={"current_quality": 0.15, "threshold": 0.30})
        
        # Security event
        logger.security_alert("Unauthorized access attempt detected",
                            fields={"ip_address": "192.168.1.100", "user_agent": "Unknown"})
        
        # User action
        logger.user_action("calibration_completed", user_id="user123", 
                          result="success", fields={"calibration_accuracy": 0.94})
        
        # Compliance event
        logger.compliance_event("data_retention_check", 
                               {"policy": "30_days", "records_checked": 1500, "violations": 0})
        
        # Function with automatic logging
        @log_function_calls(LogCategory.NEURAL_PROCESSING, LogLevel.INFO, log_args=True)
        def process_neural_signal(signal_data, filter_params):
            """Example function with automatic logging."""
            time.sleep(0.1)  # Simulate processing
            return {"processed": True, "quality": 0.88}
        
        result = process_neural_signal("mock_signal_data", {"lowpass": 30, "highpass": 0.5})
        
        # Show statistics
        stats = logger.get_statistics()
        print(f"\n--- Logging Statistics ---")
        print(f"Total logs processed: {stats['logs_processed']}")
        print(f"Logs by level: {stats['logs_by_level']}")
        print(f"Medical alerts: {stats['medical_alerts']}")
        print(f"Security events: {stats['security_events']}")
        print(f"Audit events: {stats['audit_events']}")
        
        print("\nStructured logging demo completed")
        print("Check 'logs/' directory for output files")
    
    asyncio.run(demo_structured_logging())