"""
Comprehensive input validation and security checks for BCI system.
"""

import re
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class ValidationError(Exception):
    """Exception raised for input validation failures."""
    pass


class SecurityPolicy(Enum):
    """Security policy levels."""
    PERMISSIVE = "permissive"
    STANDARD = "standard"  
    STRICT = "strict"
    CLINICAL = "clinical"


@dataclass
class ValidationRules:
    """Configuration for input validation rules."""
    max_channels: int = 256
    max_sampling_rate: int = 8000
    max_buffer_size: int = 100000
    max_string_length: int = 1000
    allowed_file_extensions: List[str] = field(default_factory=lambda: ['.edf', '.bdf', '.fif', '.set'])
    blocked_patterns: List[str] = field(default_factory=lambda: ['<script', 'javascript:', 'eval('])
    require_ssl: bool = True


class InputValidator:
    """
    Comprehensive input validator for BCI system with security checks.
    """
    
    def __init__(self, policy: SecurityPolicy = SecurityPolicy.STANDARD):
        self.policy = policy
        self.logger = logging.getLogger(__name__)
        
        # Configure rules based on policy
        if policy == SecurityPolicy.PERMISSIVE:
            self.rules = ValidationRules(
                max_channels=512,
                max_sampling_rate=16000,
                require_ssl=False
            )
        elif policy == SecurityPolicy.STRICT:
            self.rules = ValidationRules(
                max_channels=128,
                max_sampling_rate=4000,
                max_string_length=500,
                require_ssl=True
            )
        elif policy == SecurityPolicy.CLINICAL:
            self.rules = ValidationRules(
                max_channels=64,
                max_sampling_rate=2000,
                max_string_length=200,
                require_ssl=True,
                blocked_patterns=['<script', 'javascript:', 'eval(', 'exec(', 'import ', 'os.', 'sys.']
            )
        else:
            self.rules = ValidationRules()
    
    def validate_neural_data(self, data: np.ndarray, channels: int, sampling_rate: int) -> None:
        """Validate neural data array and parameters."""
        try:
            # Check data type and structure
            if not isinstance(data, np.ndarray):
                raise ValidationError(f"Neural data must be numpy array, got {type(data)}")
            
            if data.dtype not in [np.float32, np.float64, np.int16, np.int32]:
                raise ValidationError(f"Invalid data type: {data.dtype}")
            
            if len(data.shape) != 2:
                raise ValidationError(f"Neural data must be 2D (channels, samples), got shape {data.shape}")
            
            # Validate dimensions
            if data.shape[0] != channels:
                raise ValidationError(f"Channel mismatch: expected {channels}, got {data.shape[0]}")
            
            if channels <= 0 or channels > self.rules.max_channels:
                raise ValidationError(f"Invalid channel count: {channels} (max: {self.rules.max_channels})")
            
            if sampling_rate <= 0 or sampling_rate > self.rules.max_sampling_rate:
                raise ValidationError(f"Invalid sampling rate: {sampling_rate} (max: {self.rules.max_sampling_rate})")
            
            # Check for anomalous values
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                raise ValidationError("Neural data contains NaN or infinite values")
            
            # Check for unrealistic amplitude ranges (typical EEG: ±100μV)
            max_amplitude = np.max(np.abs(data))
            if max_amplitude > 1000:  # 1000μV threshold
                self.logger.warning(f"Unusually high amplitude detected: {max_amplitude}μV")
            
            # Check for DC offset
            dc_offset = np.mean(data, axis=1)
            if np.any(np.abs(dc_offset) > 100):
                self.logger.warning("High DC offset detected in neural data")
                
        except Exception as e:
            raise ValidationError(f"Neural data validation failed: {e}")
    
    def validate_string_input(self, text: str, field_name: str = "input") -> str:
        """Validate and sanitize string inputs."""
        if not isinstance(text, str):
            raise ValidationError(f"{field_name} must be string, got {type(text)}")
        
        if len(text) > self.rules.max_string_length:
            raise ValidationError(f"{field_name} exceeds max length {self.rules.max_string_length}")
        
        # Check for blocked patterns
        text_lower = text.lower()
        for pattern in self.rules.blocked_patterns:
            if pattern in text_lower:
                raise ValidationError(f"Blocked pattern detected in {field_name}: {pattern}")
        
        # Sanitize common injection attempts
        text = text.replace('\x00', '')  # Remove null bytes
        text = re.sub(r'[^\w\s\-\.\@\(\)\[\]\{\},:;!?]', '', text)  # Keep only safe chars
        
        return text
    
    def validate_file_path(self, file_path: str) -> str:
        """Validate file path for security."""
        if not isinstance(file_path, str):
            raise ValidationError(f"File path must be string, got {type(file_path)}")
        
        # Check for path traversal attempts
        if '..' in file_path or '~' in file_path:
            raise ValidationError("Path traversal detected in file path")
        
        # Check file extension
        if self.rules.allowed_file_extensions:
            ext = '.' + file_path.split('.')[-1].lower() if '.' in file_path else ''
            if ext not in self.rules.allowed_file_extensions:
                raise ValidationError(f"File extension {ext} not allowed")
        
        return file_path
    
    def validate_network_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate network configuration."""
        validated_config = {}
        
        # Validate host
        if 'host' in config:
            host = self.validate_string_input(config['host'], 'host')
            # Simple hostname/IP validation
            if not re.match(r'^[a-zA-Z0-9\-\.]+$', host):
                raise ValidationError("Invalid host format")
            validated_config['host'] = host
        
        # Validate port
        if 'port' in config:
            port = config['port']
            if not isinstance(port, int) or port < 1 or port > 65535:
                raise ValidationError(f"Invalid port: {port}")
            validated_config['port'] = port
        
        # SSL validation
        if self.rules.require_ssl and config.get('ssl', False) is False:
            raise ValidationError("SSL required by security policy")
        
        return validated_config
    
    def validate_api_key(self, api_key: str) -> str:
        """Validate API key format and security."""
        if not isinstance(api_key, str):
            raise ValidationError("API key must be string")
        
        if len(api_key) < 16:
            raise ValidationError("API key too short (minimum 16 characters)")
        
        if len(api_key) > 1000:
            raise ValidationError("API key too long (maximum 1000 characters)")
        
        # Check for common weak patterns (check if any weak pattern is contained in the key)
        weak_patterns = ['test', 'demo', 'example', '123456', 'password']
        api_key_lower = api_key.lower()
        for pattern in weak_patterns:
            if pattern in api_key_lower:
                raise ValidationError("Weak API key detected")
        
        return api_key
    
    def validate_buffer_size(self, buffer_size: int, context: str = "buffer") -> int:
        """Validate buffer size parameters."""
        if not isinstance(buffer_size, int):
            raise ValidationError(f"{context} size must be integer")
        
        if buffer_size <= 0:
            raise ValidationError(f"{context} size must be positive")
        
        if buffer_size > self.rules.max_buffer_size:
            raise ValidationError(f"{context} size exceeds maximum {self.rules.max_buffer_size}")
        
        return buffer_size
    
    def validate_confidence_score(self, confidence: float) -> float:
        """Validate confidence score."""
        if not isinstance(confidence, (int, float)):
            raise ValidationError("Confidence must be numeric")
        
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValidationError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        
        return confidence
    
    def validate_timestamp(self, timestamp: float) -> float:
        """Validate timestamp."""
        if not isinstance(timestamp, (int, float)):
            raise ValidationError("Timestamp must be numeric")
        
        timestamp = float(timestamp)
        if timestamp <= 0:
            raise ValidationError("Timestamp must be positive")
        
        # Check for reasonable timestamp range (avoid future dates > 1 year)
        import time
        current_time = time.time()
        if timestamp > current_time + 31536000:  # 1 year in seconds
            raise ValidationError("Timestamp too far in the future")
        
        return timestamp
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata dictionary."""
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be dictionary")
        
        validated_metadata = {}
        
        for key, value in metadata.items():
            # Validate key
            if not isinstance(key, str) or len(key) > 100:
                raise ValidationError(f"Invalid metadata key: {key}")
            
            # Sanitize key
            safe_key = re.sub(r'[^\w\-]', '', key)
            
            # Validate and sanitize value
            if isinstance(value, str):
                safe_value = self.validate_string_input(value, f"metadata[{key}]")
            elif isinstance(value, (int, float, bool)):
                safe_value = value
            else:
                # Convert complex types to string representation
                safe_value = str(value)[:100]  # Limit length
            
            validated_metadata[safe_key] = safe_value
        
        return validated_metadata
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation rules and policy."""
        return {
            "policy": self.policy.value,
            "rules": {
                "max_channels": self.rules.max_channels,
                "max_sampling_rate": self.rules.max_sampling_rate,
                "max_buffer_size": self.rules.max_buffer_size,
                "max_string_length": self.rules.max_string_length,
                "require_ssl": self.rules.require_ssl,
                "allowed_extensions": self.rules.allowed_file_extensions,
                "blocked_patterns_count": len(self.rules.blocked_patterns)
            }
        }


# Global validator instance
_default_validator = InputValidator(SecurityPolicy.STANDARD)

def validate_neural_data(data: np.ndarray, channels: int, sampling_rate: int) -> None:
    """Convenience function for neural data validation."""
    return _default_validator.validate_neural_data(data, channels, sampling_rate)

def validate_string_input(text: str, field_name: str = "input") -> str:
    """Convenience function for string validation."""
    return _default_validator.validate_string_input(text, field_name)