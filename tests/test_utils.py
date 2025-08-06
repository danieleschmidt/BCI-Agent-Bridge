"""
Tests for utility components (validation, retry, circuit breaker).
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from bci_agent_bridge.utils.validation import (
    InputValidator, SafetyChecker, ValidationResult, ValidationResponse
)
from bci_agent_bridge.utils.retry import (
    RetryManager, ExponentialBackoff, RetryConfig, RetryStrategy,
    retry_neural_acquisition, retry_claude_api
)
from bci_agent_bridge.utils.circuit_breaker import (
    CircuitBreaker, CircuitState, CircuitBreakerConfig, 
    CircuitBreakerOpenException, CircuitBreakerManager
)


class TestInputValidator:
    """Test suite for InputValidator."""
    
    def test_validator_initialization(self):
        """Test input validator initialization."""
        validator = InputValidator()
        
        assert validator.neural_data_rules["min_channels"] == 1
        assert validator.neural_data_rules["max_channels"] == 256
        assert validator.confidence_rules["min_confidence"] == 0.0
        assert validator.confidence_rules["max_confidence"] == 1.0
    
    def test_validate_neural_data_valid(self):
        """Test neural data validation with valid data."""
        validator = InputValidator()
        
        # Valid neural data
        data = np.random.randn(8, 250)  # 8 channels, 250 samples
        
        result = validator.validate_neural_data(data, sampling_rate=250, channels=8)
        
        assert result.result == ValidationResult.VALID
        assert "validation passed" in result.message.lower()
        assert result.details["channels"] == 8
        assert result.details["samples"] == 250
    
    def test_validate_neural_data_invalid_shape(self):
        """Test neural data validation with invalid shape."""
        validator = InputValidator()
        
        # Invalid shape (1D instead of 2D)
        data = np.random.randn(250)
        
        result = validator.validate_neural_data(data, sampling_rate=250, channels=8)
        
        assert result.result == ValidationResult.INVALID
        assert "2D array" in result.message
    
    def test_validate_neural_data_channel_mismatch(self):
        """Test neural data validation with channel mismatch."""
        validator = InputValidator()
        
        # Channel mismatch
        data = np.random.randn(4, 250)  # 4 channels
        
        result = validator.validate_neural_data(data, sampling_rate=250, channels=8)
        
        assert result.result == ValidationResult.INVALID
        assert "channel mismatch" in result.message.lower()
        assert result.details["expected_channels"] == 8
        assert result.details["actual_channels"] == 4
    
    def test_validate_neural_data_with_nan_values(self):
        """Test neural data validation with NaN values."""
        validator = InputValidator()
        
        # Data with NaN values
        data = np.random.randn(8, 250)
        data[0, 0] = np.nan
        data[1, 1] = np.inf
        
        result = validator.validate_neural_data(data, sampling_rate=250, channels=8)
        
        assert result.result == ValidationResult.INVALID
        assert "NaN or infinite" in result.message
        assert result.details["nan_count"] == 1
        assert result.details["inf_count"] == 1
    
    def test_validate_neural_data_high_voltage(self):
        """Test neural data validation with high voltage."""
        validator = InputValidator()
        
        # Data with very high amplitude
        data = np.random.randn(8, 250) * 2000  # Very high amplitude
        
        result = validator.validate_neural_data(data, sampling_rate=250, channels=8)
        
        assert result.result == ValidationResult.WARNING
        assert "high voltage" in result.message.lower()
        assert "Check electrode contact" in result.suggestions
    
    def test_validate_neural_data_flat_signal(self):
        """Test neural data validation with flat signal."""
        validator = InputValidator()
        
        # Flat signal on one channel
        data = np.random.randn(8, 250)
        data[0, :] = 0  # Flat signal
        
        result = validator.validate_neural_data(data, sampling_rate=250, channels=8)
        
        assert result.result == ValidationResult.WARNING
        assert "flat signal" in result.message.lower()
        assert result.details["channel"] == 1
    
    def test_validate_confidence_score_valid(self):
        """Test confidence score validation with valid values."""
        validator = InputValidator()
        
        # High confidence
        result = validator.validate_confidence_score(0.9, "test")
        assert result.result == ValidationResult.VALID
        assert result.details["confidence"] == 0.9
        
        # Medium confidence  
        result = validator.validate_confidence_score(0.5, "test")
        assert result.result == ValidationResult.WARNING
        assert "moderate" in result.message.lower()
    
    def test_validate_confidence_score_invalid(self):
        """Test confidence score validation with invalid values."""
        validator = InputValidator()
        
        # Out of range
        result = validator.validate_confidence_score(1.5, "test")
        assert result.result == ValidationResult.INVALID
        assert "out of range" in result.message
        
        # Wrong type
        result = validator.validate_confidence_score("invalid", "test")
        assert result.result == ValidationResult.INVALID
        assert "must be numeric" in result.message
    
    def test_validate_confidence_score_low(self):
        """Test confidence score validation with low values."""
        validator = InputValidator()
        
        result = validator.validate_confidence_score(0.2, "test")
        
        assert result.result == ValidationResult.WARNING
        assert "low confidence" in result.message.lower()
        assert "recalibration" in result.suggestions[0].lower()
    
    def test_validate_command_string_valid(self):
        """Test command string validation with valid commands."""
        validator = InputValidator()
        
        result = validator.validate_command_string("Move left")
        
        assert result.result == ValidationResult.VALID
        assert result.details["command"] == "Move left"
        assert result.details["length"] == 9
    
    def test_validate_command_string_empty(self):
        """Test command string validation with empty command."""
        validator = InputValidator()
        
        result = validator.validate_command_string("")
        
        assert result.result == ValidationResult.INVALID
        assert "cannot be empty" in result.message
    
    def test_validate_command_string_dangerous(self):
        """Test command string validation with dangerous patterns."""
        validator = InputValidator()
        
        dangerous_commands = [
            "delete all files",
            "format disk drive", 
            "shutdown system computer",
            "execute dangerous command"
        ]
        
        for cmd in dangerous_commands:
            result = validator.validate_command_string(cmd)
            assert result.result == ValidationResult.DANGEROUS
            assert "dangerous" in result.message.lower()
    
    def test_validate_command_string_long(self):
        """Test command string validation with very long commands."""
        validator = InputValidator()
        
        long_command = "a" * 1500  # Very long command
        
        result = validator.validate_command_string(long_command)
        
        assert result.result == ValidationResult.WARNING
        assert "very long command" in result.message.lower()
        assert result.details["length"] == 1500
    
    def test_validate_device_parameters_valid(self):
        """Test device parameter validation with valid parameters."""
        validator = InputValidator()
        
        result = validator.validate_device_parameters(
            device="OpenBCI",
            channels=8,
            sampling_rate=250,
            paradigm="P300"
        )
        
        assert result.result == ValidationResult.VALID
        assert result.details["device"] == "OpenBCI"
        assert result.details["paradigm"] == "P300"
    
    def test_validate_device_parameters_invalid(self):
        """Test device parameter validation with invalid parameters."""
        validator = InputValidator()
        
        result = validator.validate_device_parameters(
            device="InvalidDevice",
            channels=1000,  # Too many channels
            sampling_rate=50000,  # Too high sampling rate
            paradigm="InvalidParadigm"
        )
        
        assert result.result == ValidationResult.INVALID
        assert "invalid device" in result.message.lower()


class TestSafetyChecker:
    """Test suite for SafetyChecker."""
    
    def test_safety_checker_initialization(self):
        """Test safety checker initialization."""
        checker = SafetyChecker()
        
        assert checker.safety_thresholds["max_session_duration"] == 7200
        assert "emergency" in checker.emergency_keywords
        assert "help" in checker.emergency_keywords
    
    def test_check_session_safety_normal(self):
        """Test session safety check with normal duration."""
        checker = SafetyChecker()
        
        result = checker.check_session_safety(
            session_duration=1800,  # 30 minutes
            last_break=1200         # 20 minutes ago
        )
        
        assert result.result == ValidationResult.VALID
        assert "safety check passed" in result.message.lower()
    
    def test_check_session_safety_too_long(self):
        """Test session safety check with excessive duration."""
        checker = SafetyChecker()
        
        result = checker.check_session_safety(
            session_duration=8000,  # Over 2 hours
            last_break=0
        )
        
        assert result.result == ValidationResult.DANGEROUS
        assert "session too long" in result.message.lower()
        assert "end session immediately" in result.suggestions[0].lower()
    
    def test_check_session_safety_break_needed(self):
        """Test session safety check when break is needed."""
        checker = SafetyChecker()
        
        result = checker.check_session_safety(
            session_duration=3000,
            last_break=1200  # Long time since last break
        )
        
        assert result.result == ValidationResult.WARNING
        assert "break recommended" in result.message.lower()
        assert "take a" in result.suggestions[0].lower()
    
    def test_detect_emergency_signals_keywords(self):
        """Test emergency signal detection with keywords."""
        checker = SafetyChecker()
        
        neural_data = np.random.randn(8, 250)
        
        # Test with emergency keyword
        result = checker.detect_emergency_signals(
            neural_data, "help me emergency", confidence=0.9
        )
        
        assert result.result == ValidationResult.DANGEROUS
        assert "emergency situation detected" in result.message.lower()
        assert "alert medical staff" in result.suggestions[0].lower()
    
    def test_detect_emergency_signals_high_amplitude(self):
        """Test emergency signal detection with high amplitude signals."""
        checker = SafetyChecker()
        
        # Create data with very high amplitude
        neural_data = np.random.randn(8, 250)
        neural_data[0, 0] = 300  # Very high amplitude spike
        
        result = checker.detect_emergency_signals(
            neural_data, "normal command", confidence=0.9
        )
        
        assert result.result == ValidationResult.DANGEROUS
        assert "high amplitude signal" in result.message.lower()
    
    def test_detect_emergency_signals_sudden_change(self):
        """Test emergency signal detection with sudden signal changes."""
        checker = SafetyChecker()
        
        # Create data with sudden large change
        neural_data = np.zeros((8, 250))
        neural_data[:, 100] = 150  # Sudden large change
        
        result = checker.detect_emergency_signals(
            neural_data, "normal command", confidence=0.9
        )
        
        assert result.result == ValidationResult.DANGEROUS
        assert "sudden signal change" in result.message.lower()
    
    def test_detect_emergency_signals_normal(self):
        """Test emergency signal detection with normal signals."""
        checker = SafetyChecker()
        
        neural_data = np.random.randn(8, 250) * 10  # Normal amplitude
        
        result = checker.detect_emergency_signals(
            neural_data, "move left", confidence=0.8
        )
        
        assert result.result == ValidationResult.VALID
        assert "no emergency signals" in result.message.lower()
    
    def test_validate_medical_context_normal(self):
        """Test medical context validation with normal patient."""
        checker = SafetyChecker()
        
        context = {
            "age": 25,
            "medical_conditions": [],
            "medications": []
        }
        
        result = checker.validate_medical_context(context)
        
        assert result.result == ValidationResult.VALID
        assert "validation passed" in result.message.lower()
    
    def test_validate_medical_context_pediatric(self):
        """Test medical context validation with pediatric patient."""
        checker = SafetyChecker()
        
        context = {
            "age": 16,
            "medical_conditions": [],
            "medications": []
        }
        
        result = checker.validate_medical_context(context)
        
        assert result.result == ValidationResult.WARNING
        assert "pediatric patient" in result.message.lower()
    
    def test_validate_medical_context_contraindications(self):
        """Test medical context validation with contraindications."""
        checker = SafetyChecker()
        
        context = {
            "age": 30,
            "medical_conditions": ["epilepsy", "seizure_disorder"],
            "medications": ["anticonvulsants"]
        }
        
        result = checker.validate_medical_context(context)
        
        assert result.result == ValidationResult.DANGEROUS
        assert "contraindications" in result.message.lower()
        assert "consult medical team" in result.suggestions[0].lower()


class TestRetryManager:
    """Test suite for RetryManager."""
    
    def test_retry_manager_initialization(self):
        """Test retry manager initialization."""
        config = RetryConfig(
            max_attempts=5,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=0.5
        )
        
        manager = RetryManager(config)
        
        assert manager.config.max_attempts == 5
        assert manager.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert manager.config.base_delay == 0.5
    
    def test_should_retry_logic(self):
        """Test retry decision logic."""
        config = RetryConfig(max_attempts=3, exceptions=(ValueError, ConnectionError))
        manager = RetryManager(config)
        
        # Should retry on specified exception within attempt limit
        assert manager.should_retry(ValueError("test"), 1)
        assert manager.should_retry(ConnectionError("test"), 2)
        
        # Should not retry after max attempts
        assert not manager.should_retry(ValueError("test"), 3)
        
        # Should not retry on unspecified exception
        assert not manager.should_retry(RuntimeError("test"), 1)
    
    def test_calculate_delay_strategies(self):
        """Test delay calculation for different strategies."""
        # Fixed delay
        config = RetryConfig(strategy=RetryStrategy.FIXED_DELAY, base_delay=1.0)
        manager = RetryManager(config)
        
        assert manager.calculate_delay(1) == 1.0
        assert manager.calculate_delay(5) == 1.0
        
        # Exponential backoff
        config = RetryConfig(strategy=RetryStrategy.EXPONENTIAL_BACKOFF, base_delay=1.0)
        manager = RetryManager(config)
        
        assert manager.calculate_delay(1) == 1.0
        assert manager.calculate_delay(2) == 2.0
        assert manager.calculate_delay(3) == 4.0
        
        # Linear backoff
        config = RetryConfig(strategy=RetryStrategy.LINEAR_BACKOFF, base_delay=1.0)
        manager = RetryManager(config)
        
        assert manager.calculate_delay(1) == 1.0
        assert manager.calculate_delay(2) == 2.0
        assert manager.calculate_delay(3) == 3.0
    
    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        """Test async retry with eventual success."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        manager = RetryManager(config)
        
        call_count = 0
        
        async def unreliable_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Failed")
            return "Success"
        
        result = await manager.retry_async(unreliable_func)
        
        assert result == "Success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_async_max_attempts(self):
        """Test async retry reaching max attempts."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        manager = RetryManager(config)
        
        async def always_fails():
            raise ConnectionError("Always fails")
        
        with pytest.raises(ConnectionError):
            await manager.retry_async(always_fails)
    
    def test_retry_sync_success(self):
        """Test sync retry with eventual success."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        manager = RetryManager(config)
        
        call_count = 0
        
        def unreliable_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Failed")
            return "Success"
        
        result = manager.retry_sync(unreliable_func)
        
        assert result == "Success"
        assert call_count == 3
    
    def test_exponential_backoff_decorator(self):
        """Test exponential backoff decorator."""
        call_count = 0
        
        @ExponentialBackoff(base_delay=0.01, max_attempts=3)
        def unreliable_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Failed")
            return "Success"
        
        result = unreliable_function()
        
        assert result == "Success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_decorators(self):
        """Test retry decorators for specific operations."""
        call_count = 0
        
        @retry_neural_acquisition
        def neural_acquisition():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise IOError("Acquisition failed")
            return "Neural data acquired"
        
        result = neural_acquisition()
        
        assert result == "Neural data acquired"
        assert call_count == 2


class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=5.0
        )
        
        breaker = CircuitBreaker("test_service", config)
        
        assert breaker.name == "test_service"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.config.failure_threshold == 3
        assert breaker.metrics.total_calls == 0
    
    def test_circuit_breaker_success_recording(self):
        """Test recording successful calls."""
        breaker = CircuitBreaker("test_service")
        
        breaker._record_success()
        
        assert breaker.metrics.successful_calls == 1
        assert breaker.metrics.consecutive_successes == 1
        assert breaker.metrics.consecutive_failures == 0
    
    def test_circuit_breaker_failure_recording(self):
        """Test recording failed calls."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test_service", config)
        
        # First failure - should stay closed
        breaker._record_failure(Exception("test error"))
        assert breaker.state == CircuitState.CLOSED
        assert breaker.metrics.consecutive_failures == 1
        
        # Second failure - should open
        breaker._record_failure(Exception("test error"))
        assert breaker.state == CircuitState.OPEN
        assert breaker.metrics.consecutive_failures == 2
    
    def test_circuit_breaker_call_sync_success(self):
        """Test successful sync call through circuit breaker."""
        breaker = CircuitBreaker("test_service")
        
        def success_func(data):
            return f"Processed: {data}"
        
        result = breaker.call_sync(success_func, "test_data")
        
        assert result == "Processed: test_data"
        assert breaker.metrics.successful_calls == 1
    
    def test_circuit_breaker_call_sync_failure(self):
        """Test failed sync call through circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test_service", config)
        
        def failing_func():
            raise ValueError("Test error")
        
        # First call should fail and open circuit
        with pytest.raises(ValueError):
            breaker.call_sync(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.metrics.failed_calls == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_call_async_success(self):
        """Test successful async call through circuit breaker."""
        breaker = CircuitBreaker("test_service")
        
        async def success_func(data):
            return f"Async processed: {data}"
        
        result = await breaker.call_async(success_func, "test_data")
        
        assert result == "Async processed: test_data"
        assert breaker.metrics.successful_calls == 1
    
    def test_circuit_breaker_open_exception(self):
        """Test circuit breaker open exception."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=10.0)
        breaker = CircuitBreaker("test_service", config)
        
        def failing_func():
            raise ConnectionError("Connection failed")
        
        # Trigger circuit opening
        with pytest.raises(ConnectionError):
            breaker.call_sync(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Next call should raise CircuitBreakerOpenException
        def any_func():
            return "success"
        
        with pytest.raises(CircuitBreakerOpenException):
            breaker.call_sync(any_func)
    
    def test_circuit_breaker_half_open_transition(self):
        """Test transition from open to half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.01)
        breaker = CircuitBreaker("test_service", config)
        
        # Open the circuit
        def failing_func():
            raise Exception("Test error")
        
        with pytest.raises(Exception):
            breaker.call_sync(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.02)
        
        # Next call should transition to half-open
        def success_func():
            return "success"
        
        # This should work and close the circuit
        result = breaker.call_sync(success_func)
        
        assert result == "success"
        # After successful call in half-open, should transition to closed
        # (depending on success_threshold configuration)
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker as decorator."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test_service", config)
        
        call_count = 0
        
        @breaker
        def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Failed")
            return "Success"
        
        # First two calls should fail
        with pytest.raises(ConnectionError):
            decorated_func()
        
        with pytest.raises(ConnectionError):
            decorated_func()
        
        # Circuit should be open now
        assert breaker.state == CircuitState.OPEN
        
        # Next call should raise CircuitBreakerOpenException
        with pytest.raises(CircuitBreakerOpenException):
            decorated_func()
    
    def test_circuit_breaker_metrics(self):
        """Test circuit breaker metrics collection."""
        breaker = CircuitBreaker("test_service")
        
        # Successful call
        def success_func():
            return "success"
        
        breaker.call_sync(success_func)
        
        # Failed call
        def failing_func():
            raise ValueError("error")
        
        with pytest.raises(ValueError):
            breaker.call_sync(failing_func)
        
        metrics = breaker.get_metrics()
        
        assert metrics["name"] == "test_service"
        assert metrics["total_calls"] == 2
        assert metrics["successful_calls"] == 1
        assert metrics["failed_calls"] == 1
        assert metrics["failure_rate_pct"] == 50.0
    
    def test_circuit_breaker_reset(self):
        """Test manual circuit breaker reset."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test_service", config)
        
        # Open the circuit
        def failing_func():
            raise Exception("error")
        
        with pytest.raises(Exception):
            breaker.call_sync(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Reset circuit
        breaker.reset()
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.metrics.consecutive_failures == 0


class TestCircuitBreakerManager:
    """Test suite for CircuitBreakerManager."""
    
    def test_manager_initialization(self):
        """Test circuit breaker manager initialization."""
        manager = CircuitBreakerManager()
        
        assert len(manager.breakers) == 0
    
    def test_create_breaker(self):
        """Test creating circuit breakers through manager."""
        manager = CircuitBreakerManager()
        
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = manager.create_breaker("test_service", config)
        
        assert breaker.name == "test_service"
        assert breaker.config.failure_threshold == 3
        assert "test_service" in manager.breakers
    
    def test_get_breaker(self):
        """Test getting circuit breaker by name."""
        manager = CircuitBreakerManager()
        
        created_breaker = manager.create_breaker("test_service")
        retrieved_breaker = manager.get_breaker("test_service")
        
        assert retrieved_breaker is created_breaker
        
        # Non-existent breaker
        none_breaker = manager.get_breaker("non_existent")
        assert none_breaker is None
    
    def test_get_all_metrics(self):
        """Test getting metrics for all circuit breakers."""
        manager = CircuitBreakerManager()
        
        breaker1 = manager.create_breaker("service1")
        breaker2 = manager.create_breaker("service2")
        
        # Add some test calls
        breaker1.call_sync(lambda: "success")
        
        with pytest.raises(Exception):
            breaker2.call_sync(lambda: (_ for _ in ()).throw(Exception("error")))
        
        all_metrics = manager.get_all_metrics()
        
        assert "service1" in all_metrics
        assert "service2" in all_metrics
        assert all_metrics["service1"]["successful_calls"] == 1
        assert all_metrics["service2"]["failed_calls"] == 1
    
    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        manager = CircuitBreakerManager()
        
        # Create breakers and open one
        breaker1 = manager.create_breaker("service1", CircuitBreakerConfig(failure_threshold=1))
        breaker2 = manager.create_breaker("service2")
        
        # Open one circuit
        with pytest.raises(Exception):
            breaker1.call_sync(lambda: (_ for _ in ()).throw(Exception("error")))
        
        assert breaker1.state == CircuitState.OPEN
        
        # Reset all
        manager.reset_all()
        
        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED
    
    def test_health_summary(self):
        """Test getting health summary from manager."""
        manager = CircuitBreakerManager()
        
        breaker1 = manager.create_breaker("service1", CircuitBreakerConfig(failure_threshold=1))
        breaker2 = manager.create_breaker("service2")
        
        # Open one circuit
        with pytest.raises(Exception):
            breaker1.call_sync(lambda: (_ for _ in ()).throw(Exception("error")))
        
        summary = manager.get_health_summary()
        
        assert summary["total_breakers"] == 2
        assert summary["open_breakers"] == 1
        assert summary["healthy_breakers"] == 1
        assert summary["overall_health"] == "degraded"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])