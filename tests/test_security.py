"""
Tests for security validation and audit logging.
"""

import pytest
import numpy as np
import time
from unittest.mock import patch

from bci_agent_bridge.security.input_validator import (
    InputValidator, ValidationError, SecurityPolicy
)
from bci_agent_bridge.security.audit_logger import (
    SecurityAuditLogger, SecurityEvent, AuditRecord
)


class TestInputValidator:
    """Test input validation functionality."""
    
    def setUp(self):
        self.validator = InputValidator(SecurityPolicy.STANDARD)
    
    def test_neural_data_validation_success(self):
        """Test successful neural data validation."""
        validator = InputValidator(SecurityPolicy.STANDARD)
        data = np.random.randn(8, 250).astype(np.float32)
        
        # Should not raise exception
        validator.validate_neural_data(data, channels=8, sampling_rate=250)
    
    def test_neural_data_validation_channel_mismatch(self):
        """Test neural data validation with channel mismatch."""
        validator = InputValidator(SecurityPolicy.STANDARD)
        data = np.random.randn(16, 250).astype(np.float32)
        
        with pytest.raises(ValidationError, match="Channel mismatch"):
            validator.validate_neural_data(data, channels=8, sampling_rate=250)
    
    def test_neural_data_validation_invalid_type(self):
        """Test neural data validation with invalid data type."""
        validator = InputValidator(SecurityPolicy.STANDARD)
        data = [[1, 2, 3], [4, 5, 6]]  # List instead of numpy array
        
        with pytest.raises(ValidationError, match="must be numpy array"):
            validator.validate_neural_data(data, channels=2, sampling_rate=250)
    
    def test_neural_data_validation_nan_values(self):
        """Test neural data validation with NaN values."""
        validator = InputValidator(SecurityPolicy.STANDARD)
        data = np.random.randn(8, 250).astype(np.float32)
        data[0, 0] = np.nan
        
        with pytest.raises(ValidationError, match="NaN or infinite"):
            validator.validate_neural_data(data, channels=8, sampling_rate=250)
    
    def test_string_input_validation_success(self):
        """Test successful string input validation."""
        validator = InputValidator(SecurityPolicy.STANDARD)
        safe_text = validator.validate_string_input("Hello World", "test_field")
        assert safe_text == "Hello World"
    
    def test_string_input_validation_blocked_patterns(self):
        """Test string validation with blocked patterns."""
        validator = InputValidator(SecurityPolicy.CLINICAL)
        
        with pytest.raises(ValidationError, match="Blocked pattern"):
            validator.validate_string_input("<script>alert('xss')</script>", "test_field")
    
    def test_string_input_validation_length_limit(self):
        """Test string validation with length limits."""
        validator = InputValidator(SecurityPolicy.CLINICAL)  # Stricter length limits
        long_text = "A" * 300  # Exceeds clinical policy limit
        
        with pytest.raises(ValidationError, match="exceeds max length"):
            validator.validate_string_input(long_text, "test_field")
    
    def test_api_key_validation_success(self):
        """Test successful API key validation."""
        validator = InputValidator(SecurityPolicy.STANDARD)
        api_key = "sk-1234567890abcdef1234567890abcdef"
        
        validated_key = validator.validate_api_key(api_key)
        assert validated_key == api_key
    
    def test_api_key_validation_weak_key(self):
        """Test API key validation with weak keys."""
        validator = InputValidator(SecurityPolicy.STANDARD)
        
        with pytest.raises(ValidationError, match="API key too short"):
            validator.validate_api_key("password")
        
        # Test actual weak key pattern (long enough but weak)  
        with pytest.raises(ValidationError, match="Weak API key"):
            validator.validate_api_key("test123456789012")  # Contains 'test' which is blocked
    
    def test_api_key_validation_too_short(self):
        """Test API key validation with short keys."""
        validator = InputValidator(SecurityPolicy.STANDARD)
        
        with pytest.raises(ValidationError, match="too short"):
            validator.validate_api_key("short")
    
    def test_confidence_score_validation(self):
        """Test confidence score validation."""
        validator = InputValidator(SecurityPolicy.STANDARD)
        
        # Valid scores
        assert validator.validate_confidence_score(0.5) == 0.5
        assert validator.validate_confidence_score(0.0) == 0.0
        assert validator.validate_confidence_score(1.0) == 1.0
        
        # Invalid scores
        with pytest.raises(ValidationError):
            validator.validate_confidence_score(-0.1)
        
        with pytest.raises(ValidationError):
            validator.validate_confidence_score(1.1)
    
    def test_timestamp_validation(self):
        """Test timestamp validation."""
        validator = InputValidator(SecurityPolicy.STANDARD)
        
        current_time = time.time()
        assert validator.validate_timestamp(current_time) == current_time
        
        # Invalid timestamps
        with pytest.raises(ValidationError, match="must be positive"):
            validator.validate_timestamp(-1.0)
        
        # Future timestamps beyond reasonable limit
        future_time = current_time + 40000000  # > 1 year
        with pytest.raises(ValidationError, match="too far in the future"):
            validator.validate_timestamp(future_time)
    
    def test_metadata_validation(self):
        """Test metadata validation."""
        validator = InputValidator(SecurityPolicy.STANDARD)
        
        metadata = {
            "device": "OpenBCI",
            "channels": 8,
            "sampling_rate": 250,
            "experiment_id": "EXP_001"
        }
        
        validated = validator.validate_metadata(metadata)
        assert isinstance(validated, dict)
        assert validated["device"] == "OpenBCI"
        assert validated["channels"] == 8
    
    def test_security_policy_differences(self):
        """Test different security policies have different restrictions."""
        permissive = InputValidator(SecurityPolicy.PERMISSIVE)
        clinical = InputValidator(SecurityPolicy.CLINICAL)
        
        # Clinical should be more restrictive
        assert clinical.rules.max_channels < permissive.rules.max_channels
        assert clinical.rules.max_string_length < permissive.rules.max_string_length
        assert len(clinical.rules.blocked_patterns) > len(permissive.rules.blocked_patterns)


class TestSecurityAuditLogger:
    """Test security audit logging functionality."""
    
    def test_audit_record_creation(self):
        """Test audit record creation."""
        record = AuditRecord(
            event_type=SecurityEvent.AUTHENTICATION_SUCCESS,
            timestamp=time.time(),
            user_id="test_user",
            result="success"
        )
        
        assert record.event_type == SecurityEvent.AUTHENTICATION_SUCCESS
        assert record.user_id == "test_user"
        assert record.result == "success"
        
        # Test serialization
        record_dict = record.to_dict()
        assert isinstance(record_dict, dict)
        assert record_dict["event_type"] == "auth_success"
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_security_logger_initialization(self, mock_makedirs, mock_exists):
        """Test security logger initialization."""
        mock_exists.return_value = False
        
        logger = SecurityAuditLogger(log_file="/tmp/test_security.log")
        
        assert logger.log_file == "/tmp/test_security.log"
        assert isinstance(logger.event_counts, dict)
        mock_makedirs.assert_called_once()
    
    def test_security_event_logging(self):
        """Test security event logging."""
        logger = SecurityAuditLogger(log_file="/tmp/test_audit.log")
        
        # Log a security event
        logger.log_security_event(
            event_type=SecurityEvent.DATA_ACCESS,
            user_id="test_user",
            resource="neural_data",
            action="read",
            result="success",
            risk_score=2
        )
        
        # Verify event was counted
        assert logger.event_counts[SecurityEvent.DATA_ACCESS.value] == 1
    
    def test_authentication_logging(self):
        """Test authentication event logging."""
        logger = SecurityAuditLogger(log_file="/tmp/test_auth.log")
        
        # Log successful authentication
        logger.log_authentication_attempt(
            user_id="test_user",
            success=True,
            ip_address="192.168.1.1"
        )
        
        # Log failed authentication
        logger.log_authentication_attempt(
            user_id="test_user",
            success=False,
            ip_address="192.168.1.2"
        )
        
        assert logger.event_counts[SecurityEvent.AUTHENTICATION_SUCCESS.value] == 1
        assert logger.event_counts[SecurityEvent.AUTHENTICATION_FAILURE.value] == 1
    
    def test_validation_failure_logging(self):
        """Test validation failure logging."""
        logger = SecurityAuditLogger(log_file="/tmp/test_validation.log")
        
        logger.log_validation_failure(
            input_type="neural_data",
            error_message="Invalid channel count",
            source="bci_bridge"
        )
        
        assert logger.event_counts[SecurityEvent.INPUT_VALIDATION_FAILURE.value] == 1
    
    def test_high_risk_event_tracking(self):
        """Test high-risk event tracking."""
        logger = SecurityAuditLogger(log_file="/tmp/test_highrisk.log")
        
        # Log high-risk event
        logger.log_security_event(
            event_type=SecurityEvent.SUSPICIOUS_ACTIVITY,
            action="brute_force_attempt",
            risk_score=9,
            details={"attempts": 20, "ip": "192.168.1.100"}
        )
        
        # Should be tracked as high-risk event
        assert len(logger.high_risk_events) == 1
        assert logger.high_risk_events[0].risk_score == 9
    
    def test_security_metrics(self):
        """Test security metrics generation."""
        logger = SecurityAuditLogger(log_file="/tmp/test_metrics.log")
        
        # Log some events
        logger.log_authentication_attempt("user1", True, "192.168.1.1")
        logger.log_authentication_attempt("user2", False, "192.168.1.2")
        logger.log_data_access("user1", "neural_data", "read", True)
        
        metrics = logger.get_security_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_events" in metrics
        assert "events_by_type" in metrics
        assert "high_risk_events_count" in metrics
        assert metrics["total_events"] == 3
    
    def test_threat_pattern_analysis(self):
        """Test threat pattern analysis."""
        logger = SecurityAuditLogger(log_file="/tmp/test_threats.log")
        
        # Simulate brute force attack
        for i in range(6):
            logger.log_authentication_attempt(
                f"user{i}", False, "192.168.1.100"
            )
        
        # Analyze patterns
        analysis = logger.analyze_threat_patterns()
        
        assert isinstance(analysis, dict)
        assert "failed_auth_attempts" in analysis
        assert "threat_indicators" in analysis
        assert analysis["failed_auth_attempts"] == 6
        
        # Should detect brute force pattern
        indicators = analysis["threat_indicators"]
        assert len(indicators) > 0
        assert indicators[0]["type"] == "brute_force"
        assert indicators[0]["source_ip"] == "192.168.1.100"