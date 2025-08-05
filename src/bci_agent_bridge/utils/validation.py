"""
Input validation and safety checking utilities.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ValidationResult(Enum):
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    DANGEROUS = "dangerous"


@dataclass
class ValidationResponse:
    result: ValidationResult
    message: str
    details: Dict[str, Any] = None
    suggestions: List[str] = None


class InputValidator:
    """
    Comprehensive input validation for BCI system components.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define validation rules
        self.neural_data_rules = {
            "min_channels": 1,
            "max_channels": 256,
            "min_sampling_rate": 1,
            "max_sampling_rate": 10000,
            "max_voltage_range": 1000,  # microvolts
            "min_data_length": 1
        }
        
        self.confidence_rules = {
            "min_confidence": 0.0,
            "max_confidence": 1.0,
            "warning_threshold": 0.3,
            "good_threshold": 0.7
        }
    
    def validate_neural_data(self, data: np.ndarray, sampling_rate: int, 
                           channels: int) -> ValidationResponse:
        """Validate neural data input."""
        try:
            # Check data shape
            if len(data.shape) != 2:
                return ValidationResponse(
                    ValidationResult.INVALID,
                    "Neural data must be 2D array (channels x samples)",
                    {"shape": data.shape}
                )
            
            actual_channels, samples = data.shape
            
            # Validate channels
            if actual_channels != channels:
                return ValidationResponse(
                    ValidationResult.INVALID,
                    f"Channel mismatch: expected {channels}, got {actual_channels}",
                    {"expected_channels": channels, "actual_channels": actual_channels}
                )
            
            if not (self.neural_data_rules["min_channels"] <= channels <= 
                   self.neural_data_rules["max_channels"]):
                return ValidationResponse(
                    ValidationResult.INVALID,
                    f"Invalid channel count: {channels}",
                    {"valid_range": (self.neural_data_rules["min_channels"], 
                                   self.neural_data_rules["max_channels"])}
                )
            
            # Validate sampling rate
            if not (self.neural_data_rules["min_sampling_rate"] <= sampling_rate <= 
                   self.neural_data_rules["max_sampling_rate"]):
                return ValidationResponse(
                    ValidationResult.INVALID,
                    f"Invalid sampling rate: {sampling_rate} Hz",
                    {"valid_range": (self.neural_data_rules["min_sampling_rate"],
                                   self.neural_data_rules["max_sampling_rate"])}
                )
            
            # Check for data anomalies
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                return ValidationResponse(
                    ValidationResult.INVALID,
                    "Neural data contains NaN or infinite values",
                    {"nan_count": np.sum(np.isnan(data)), 
                     "inf_count": np.sum(np.isinf(data))}
                )
            
            # Check voltage range (assuming microvolts)
            max_voltage = np.max(np.abs(data))
            if max_voltage > self.neural_data_rules["max_voltage_range"]:
                return ValidationResponse(
                    ValidationResult.WARNING,
                    f"High voltage detected: {max_voltage:.1f} μV",
                    {"max_voltage": float(max_voltage),
                     "threshold": self.neural_data_rules["max_voltage_range"]},
                    ["Check electrode contact", "Verify signal conditioning"]
                )
            
            # Check for flat signals (potential electrode issues)
            for ch in range(channels):
                if np.std(data[ch]) < 0.1:  # Very low variance
                    return ValidationResponse(
                        ValidationResult.WARNING,
                        f"Flat signal detected on channel {ch+1}",
                        {"channel": ch+1, "std": float(np.std(data[ch]))},
                        ["Check electrode connection", "Verify impedance"]
                    )
            
            return ValidationResponse(
                ValidationResult.VALID,
                "Neural data validation passed",
                {"channels": channels, "samples": samples, 
                 "sampling_rate": sampling_rate, "max_voltage": float(max_voltage)}
            )
            
        except Exception as e:
            self.logger.error(f"Neural data validation error: {e}")
            return ValidationResponse(
                ValidationResult.INVALID,
                f"Validation error: {str(e)}"
            )
    
    def validate_confidence_score(self, confidence: float, context: str = "") -> ValidationResponse:
        """Validate confidence score."""
        try:
            if not isinstance(confidence, (int, float)):
                return ValidationResponse(
                    ValidationResult.INVALID,
                    f"Confidence must be numeric, got {type(confidence)}"
                )
            
            if not (self.confidence_rules["min_confidence"] <= confidence <= 
                   self.confidence_rules["max_confidence"]):
                return ValidationResponse(
                    ValidationResult.INVALID,
                    f"Confidence out of range: {confidence}",
                    {"valid_range": (self.confidence_rules["min_confidence"],
                                   self.confidence_rules["max_confidence"])}
                )
            
            if confidence < self.confidence_rules["warning_threshold"]:
                return ValidationResponse(
                    ValidationResult.WARNING,
                    f"Low confidence score: {confidence:.3f}",
                    {"confidence": confidence, "context": context},
                    ["Consider recalibration", "Check signal quality"]
                )
            
            if confidence >= self.confidence_rules["good_threshold"]:
                result = ValidationResult.VALID
                message = f"High confidence score: {confidence:.3f}"
            else:
                result = ValidationResult.WARNING
                message = f"Moderate confidence score: {confidence:.3f}"
            
            return ValidationResponse(
                result,
                message,
                {"confidence": confidence, "context": context}
            )
            
        except Exception as e:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"Confidence validation error: {str(e)}"
            )
    
    def validate_command_string(self, command: str) -> ValidationResponse:
        """Validate neural command string."""
        try:
            if not isinstance(command, str):
                return ValidationResponse(
                    ValidationResult.INVALID,
                    f"Command must be string, got {type(command)}"
                )
            
            if len(command.strip()) == 0:
                return ValidationResponse(
                    ValidationResult.INVALID,
                    "Command cannot be empty"
                )
            
            if len(command) > 1000:  # Reasonable limit
                return ValidationResponse(
                    ValidationResult.WARNING,
                    f"Very long command: {len(command)} characters",
                    {"length": len(command)},
                    ["Consider breaking into smaller commands"]
                )
            
            # Check for potentially harmful patterns
            dangerous_patterns = [
                r'\b(delete|remove|destroy)\b.*\b(all|everything)\b',
                r'\b(format|wipe|erase)\b.*\b(disk|drive|system)\b',
                r'\bshutdown\b.*\b(system|computer)\b',
                r'\b(execute|run)\b.*\b(dangerous|harmful)\b'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, command.lower()):
                    return ValidationResponse(
                        ValidationResult.DANGEROUS,
                        f"Potentially dangerous command detected: {command}",
                        {"matched_pattern": pattern},
                        ["Review command intent", "Use safety mode"]
                    )
            
            return ValidationResponse(
                ValidationResult.VALID,
                "Command validation passed",
                {"command": command, "length": len(command)}
            )
            
        except Exception as e:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"Command validation error: {str(e)}"
            )
    
    def validate_device_parameters(self, device: str, channels: int, 
                                 sampling_rate: int, paradigm: str) -> ValidationResponse:
        """Validate BCI device parameters."""
        try:
            valid_devices = ["Simulation", "OpenBCI", "Emotiv", "NeuroSky", "Muse"]
            valid_paradigms = ["P300", "MotorImagery", "SSVEP"]
            
            issues = []
            
            if device not in valid_devices:
                issues.append(f"Invalid device: {device}")
            
            if paradigm not in valid_paradigms:
                issues.append(f"Invalid paradigm: {paradigm}")
            
            if not (1 <= channels <= 64):  # Reasonable channel range
                issues.append(f"Channel count out of range: {channels}")
            
            if not (50 <= sampling_rate <= 2000):  # Reasonable sampling rate range
                issues.append(f"Sampling rate out of range: {sampling_rate}")
            
            if issues:
                return ValidationResponse(
                    ValidationResult.INVALID,
                    f"Device parameter validation failed: {'; '.join(issues)}",
                    {"valid_devices": valid_devices, "valid_paradigms": valid_paradigms}
                )
            
            return ValidationResponse(
                ValidationResult.VALID,
                "Device parameters validation passed",
                {"device": device, "channels": channels, 
                 "sampling_rate": sampling_rate, "paradigm": paradigm}
            )
            
        except Exception as e:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"Device parameter validation error: {str(e)}"
            )


class SafetyChecker:
    """
    Medical and safety validation for BCI operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Medical safety thresholds
        self.safety_thresholds = {
            "max_session_duration": 7200,  # 2 hours in seconds
            "min_break_interval": 1800,    # 30 minutes
            "max_stimulation_intensity": 1.0,
            "max_continuous_use": 3600     # 1 hour
        }
        
        # Emergency detection patterns
        self.emergency_keywords = [
            "emergency", "help", "pain", "stop", "distress", 
            "medical", "urgent", "critical", "danger"
        ]
    
    def check_session_safety(self, session_duration: float, 
                           last_break: float) -> ValidationResponse:
        """Check session duration and break intervals for safety."""
        try:
            current_time = session_duration
            time_since_break = current_time - last_break
            
            # Check maximum session duration
            if session_duration > self.safety_thresholds["max_session_duration"]:
                return ValidationResponse(
                    ValidationResult.DANGEROUS,
                    f"Session too long: {session_duration/3600:.1f} hours",
                    {"session_duration": session_duration, 
                     "max_allowed": self.safety_thresholds["max_session_duration"]},
                    ["End session immediately", "Take extended break"]
                )
            
            # Check break intervals
            if time_since_break > self.safety_thresholds["min_break_interval"]:
                return ValidationResponse(
                    ValidationResult.WARNING,
                    f"Break recommended: {time_since_break/60:.0f} minutes since last break",
                    {"time_since_break": time_since_break,
                     "recommended_interval": self.safety_thresholds["min_break_interval"]},
                    ["Take a 10-15 minute break", "Hydrate and rest eyes"]
                )
            
            return ValidationResponse(
                ValidationResult.VALID,
                "Session safety check passed",
                {"session_duration": session_duration, "time_since_break": time_since_break}
            )
            
        except Exception as e:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"Session safety check error: {str(e)}"
            )
    
    def detect_emergency_signals(self, neural_data: np.ndarray, 
                                decoded_command: str, confidence: float) -> ValidationResponse:
        """Detect potential emergency situations from neural signals."""
        try:
            emergency_detected = False
            emergency_reasons = []
            
            # Check for emergency keywords in decoded command
            command_lower = decoded_command.lower()
            for keyword in self.emergency_keywords:
                if keyword in command_lower:
                    emergency_detected = True
                    emergency_reasons.append(f"Emergency keyword detected: {keyword}")
            
            # Check for anomalous neural patterns
            if neural_data.size > 0:
                # High amplitude spikes might indicate distress
                max_amplitude = np.max(np.abs(neural_data))
                if max_amplitude > 200:  # microvolts
                    emergency_detected = True
                    emergency_reasons.append(f"High amplitude signal: {max_amplitude:.1f}μV")
                
                # Sudden signal changes
                if neural_data.shape[1] > 1:
                    signal_diff = np.diff(neural_data, axis=1)
                    max_change = np.max(np.abs(signal_diff))
                    if max_change > 100:
                        emergency_detected = True
                        emergency_reasons.append(f"Sudden signal change: {max_change:.1f}μV")
            
            # High confidence on emergency-related commands
            if emergency_detected and confidence > 0.8:
                return ValidationResponse(
                    ValidationResult.DANGEROUS,
                    f"Emergency situation detected: {'; '.join(emergency_reasons)}",
                    {"command": decoded_command, "confidence": confidence, 
                     "max_amplitude": float(np.max(np.abs(neural_data))) if neural_data.size > 0 else 0},
                    ["Alert medical staff", "Stop stimulation", "Check patient status"]
                )
            elif emergency_detected:
                return ValidationResponse(
                    ValidationResult.WARNING,
                    f"Possible emergency indicators: {'; '.join(emergency_reasons)}",
                    {"command": decoded_command, "confidence": confidence},
                    ["Monitor closely", "Verify patient status"]
                )
            
            return ValidationResponse(
                ValidationResult.VALID,
                "No emergency signals detected",
                {"command": decoded_command, "confidence": confidence}
            )
            
        except Exception as e:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"Emergency detection error: {str(e)}"
            )
    
    def validate_medical_context(self, patient_context: Dict[str, Any]) -> ValidationResponse:
        """Validate medical context and contraindications."""
        try:
            warnings = []
            contraindications = []
            
            # Check age restrictions
            age = patient_context.get("age")
            if age is not None:
                if age < 18:
                    warnings.append("Pediatric patient - requires special protocols")
                elif age > 75:
                    warnings.append("Elderly patient - monitor for fatigue")
            
            # Check medical conditions
            conditions = patient_context.get("medical_conditions", [])
            high_risk_conditions = [
                "epilepsy", "seizure_disorder", "brain_tumor", "stroke",
                "traumatic_brain_injury", "psychiatric_disorder"
            ]
            
            for condition in conditions:
                if condition.lower() in high_risk_conditions:
                    contraindications.append(f"High-risk condition: {condition}")
            
            # Check medications
            medications = patient_context.get("medications", [])
            interfering_meds = [
                "anticonvulsants", "sedatives", "antipsychotics", 
                "muscle_relaxants", "stimulants"
            ]
            
            for med in medications:
                if any(interfering in med.lower() for interfering in interfering_meds):
                    warnings.append(f"Potentially interfering medication: {med}")
            
            # Determine overall safety
            if contraindications:
                return ValidationResponse(
                    ValidationResult.DANGEROUS,
                    f"Medical contraindications: {'; '.join(contraindications)}",
                    {"contraindications": contraindications, "warnings": warnings},
                    ["Consult medical team", "Review protocols", "Consider alternatives"]
                )
            elif warnings:
                return ValidationResponse(
                    ValidationResult.WARNING,
                    f"Medical considerations: {'; '.join(warnings)}",
                    {"warnings": warnings},
                    ["Monitor closely", "Use conservative settings"]
                )
            
            return ValidationResponse(
                ValidationResult.VALID,
                "Medical context validation passed",
                {"patient_context": patient_context}
            )
            
        except Exception as e:
            return ValidationResponse(
                ValidationResult.INVALID,
                f"Medical context validation error: {str(e)}"
            )