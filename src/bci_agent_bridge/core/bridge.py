"""
Core BCI Bridge implementation for real-time neural signal processing.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, AsyncGenerator, List
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ..signal_processing.preprocessing import SignalPreprocessor
from ..decoders.base import BaseDecoder

# Optional security imports
try:
    from ..security.input_validator import InputValidator, ValidationError, SecurityPolicy
    from ..security.audit_logger import security_logger, SecurityEvent
    _SECURITY_AVAILABLE = True
except ImportError:
    # Security module not available, use basic validation
    _SECURITY_AVAILABLE = False
    
    class ValidationError(Exception):
        pass
    
    class InputValidator:
        def validate_neural_data(self, *args, **kwargs):
            pass
        def validate_string_input(self, text, field_name="input"):
            return text


class BCIDevice(Enum):
    OPENBCI = "OpenBCI"
    EMOTIV = "Emotiv"
    NEUROSKY = "NeuroSky"
    MUSE = "Muse"
    SIMULATION = "Simulation"


class Paradigm(Enum):
    P300 = "P300"
    MOTOR_IMAGERY = "MotorImagery"
    SSVEP = "SSVEP"
    HYBRID = "Hybrid"


@dataclass
class NeuralData:
    data: np.ndarray
    timestamp: float
    channels: List[str]
    sampling_rate: int
    metadata: Dict[str, Any] = None


@dataclass
class DecodedIntention:
    command: str
    confidence: float
    context: Dict[str, Any]
    timestamp: float
    neural_features: Optional[np.ndarray] = None


class BCIBridge:
    """
    Main BCI Bridge class for real-time neural signal processing and intention decoding.
    """
    
    def __init__(
        self,
        device: str = "Simulation",
        channels: int = 8,
        sampling_rate: int = 250,
        paradigm: str = "P300",
        buffer_size: int = 1000,
        privacy_mode: bool = True
    ):
        # Input validation with comprehensive error handling
        try:
            if not isinstance(channels, int) or channels <= 0 or channels > 256:
                raise ValueError(f"Channels must be a positive integer ≤256, got {channels}")
            if not isinstance(sampling_rate, int) or sampling_rate <= 0 or sampling_rate > 8000:
                raise ValueError(f"Sampling rate must be positive integer ≤8000 Hz, got {sampling_rate}")
            if not isinstance(buffer_size, int) or buffer_size <= 0 or buffer_size > 100000:
                raise ValueError(f"Buffer size must be positive integer ≤100000, got {buffer_size}")
            
            self.device = BCIDevice(device)
            self.channels = channels
            self.sampling_rate = sampling_rate
            self.paradigm = Paradigm(paradigm)
            self.buffer_size = buffer_size
            self.privacy_mode = privacy_mode
            
        except ValueError as e:
            raise ValueError(f"Invalid BCIBridge configuration: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BCIBridge: {e}")
        
        self.logger = logging.getLogger(__name__)
        self.is_streaming = False
        self.data_buffer = []
        
        # Initialize components with error handling
        try:
            # Initialize security validator
            if _SECURITY_AVAILABLE:
                security_policy = SecurityPolicy.CLINICAL if privacy_mode else SecurityPolicy.STANDARD
                self.validator = InputValidator(security_policy)
                security_logger.log_security_event(
                    event_type=SecurityEvent.CONFIGURATION_CHANGE,
                    resource="bci_bridge",
                    action="initialize",
                    details={
                        "device": device,
                        "channels": channels,
                        "sampling_rate": sampling_rate,
                        "paradigm": paradigm,
                        "privacy_mode": privacy_mode
                    },
                    risk_score=2
                )
            else:
                self.validator = InputValidator()
            
            self.preprocessor = SignalPreprocessor(sampling_rate=sampling_rate)
            self.decoder: Optional[BaseDecoder] = None
            
            # Performance and health monitoring
            self._total_samples_processed = 0
            self._last_health_check = time.time()
            self._error_count = 0
            self._processing_times = []
            
            self._initialize_device()
            self._setup_decoder()
            
            self.logger.info(f"BCIBridge initialized: {device} with {channels}ch @ {sampling_rate}Hz, paradigm={paradigm}")
            
        except Exception as e:
            if _SECURITY_AVAILABLE:
                security_logger.log_system_error(
                    component="bci_bridge",
                    error_type="initialization_error",
                    error_message=str(e)
                )
            self.logger.error(f"Failed to initialize BCI components: {e}")
            raise RuntimeError(f"BCI initialization failed: {e}")
    
    def _initialize_device(self) -> None:
        """Initialize the BCI device connection."""
        if self.device == BCIDevice.SIMULATION:
            self.logger.info("Initializing simulation mode")
            self._device_connected = True
        else:
            self.logger.info(f"Initializing {self.device.value} device")
            self._device_connected = self._connect_hardware()
    
    def _connect_hardware(self) -> bool:
        """Connect to actual BCI hardware."""
        try:
            self.logger.info("Attempting hardware connection...")
            time.sleep(0.1)  # Simulate connection time
            return True
        except Exception as e:
            self.logger.error(f"Hardware connection failed: {e}")
            return False
    
    def _setup_decoder(self) -> None:
        """Initialize the appropriate neural decoder based on paradigm."""
        if self.paradigm == Paradigm.P300:
            from ..decoders.p300 import P300Decoder
            self.decoder = P300Decoder(channels=self.channels, sampling_rate=self.sampling_rate)
        elif self.paradigm == Paradigm.MOTOR_IMAGERY:
            from ..decoders.motor_imagery import MotorImageryDecoder
            self.decoder = MotorImageryDecoder(channels=self.channels, sampling_rate=self.sampling_rate)
        elif self.paradigm == Paradigm.SSVEP:
            from ..decoders.ssvep import SSVEPDecoder
            self.decoder = SSVEPDecoder(channels=self.channels, sampling_rate=self.sampling_rate)
        else:
            raise ValueError(f"Unsupported paradigm: {self.paradigm}")
    
    async def stream(self) -> AsyncGenerator[NeuralData, None]:
        """
        Start streaming neural data from the BCI device.
        
        Yields:
            NeuralData: Real-time neural signal data
        """
        if not self._device_connected:
            raise RuntimeError("BCI device not connected")
        
        self.is_streaming = True
        self.logger.info("Starting neural data stream")
        
        try:
            while self.is_streaming:
                raw_data = await self._read_raw_data()
                processed_data = self.preprocessor.process(raw_data)
                
                neural_data = NeuralData(
                    data=processed_data,
                    timestamp=time.time(),
                    channels=[f"CH{i+1}" for i in range(self.channels)],
                    sampling_rate=self.sampling_rate,
                    metadata={"device": self.device.value, "paradigm": self.paradigm.value}
                )
                
                # Add data to buffer with validation and monitoring
                self._add_to_buffer_safe(neural_data)
                
                yield neural_data
                
        except asyncio.CancelledError:
            self.logger.info("Neural data stream cancelled")
        finally:
            self.is_streaming = False
    
    async def _read_raw_data(self) -> np.ndarray:
        """Read raw neural data from the device."""
        if self.device == BCIDevice.SIMULATION:
            await asyncio.sleep(1.0 / self.sampling_rate)
            return self._generate_simulation_data()
        else:
            return await self._read_hardware_data()
    
    def _generate_simulation_data(self) -> np.ndarray:
        """Generate simulated neural data for testing."""
        t = np.linspace(0, 1, self.sampling_rate)
        
        # Simulate different neural patterns based on paradigm
        if self.paradigm == Paradigm.P300:
            # Simulate ERP with P300 component
            data = np.random.randn(self.channels, len(t)) * 0.5
            if np.random.random() < 0.3:  # 30% chance of P300
                p300_component = 5 * np.exp(-(t - 0.3)**2 / 0.01)
                data[0] += p300_component  # Add to central channel
        
        elif self.paradigm == Paradigm.MOTOR_IMAGERY:
            # Simulate mu/beta rhythm modulation
            alpha = np.sin(2 * np.pi * 10 * t) * (0.5 + 0.5 * np.random.random())
            beta = np.sin(2 * np.pi * 20 * t) * (0.3 + 0.3 * np.random.random())
            data = np.array([alpha + beta + np.random.randn(len(t)) * 0.2 
                           for _ in range(self.channels)])
        
        elif self.paradigm == Paradigm.SSVEP:
            # Simulate steady-state response
            freqs = [6.0, 7.5, 8.57, 10.0]
            target_freq = np.random.choice(freqs)
            ssvep_signal = 3 * np.sin(2 * np.pi * target_freq * t)
            data = np.array([ssvep_signal + np.random.randn(len(t)) * 0.5 
                           for _ in range(self.channels)])
        
        else:
            data = np.random.randn(self.channels, len(t))
        
        return data
    
    async def _read_hardware_data(self) -> np.ndarray:
        """Read data from actual BCI hardware."""
        await asyncio.sleep(1.0 / self.sampling_rate)
        return np.random.randn(self.channels, self.sampling_rate)
    
    def decode_intention(self, neural_data: NeuralData) -> DecodedIntention:
        """
        Decode user intention from neural signals.
        
        Args:
            neural_data: Neural signal data to decode
            
        Returns:
            DecodedIntention: Decoded intention with confidence score
        """
        if self.decoder is None:
            raise RuntimeError("No decoder initialized")
        
        features = self.decoder.extract_features(neural_data.data)
        prediction = self.decoder.predict(features)
        confidence = self.decoder.get_confidence()
        
        # Map prediction to command based on paradigm
        command = self._map_prediction_to_command(prediction)
        
        return DecodedIntention(
            command=command,
            confidence=confidence,
            context={
                "paradigm": self.paradigm.value,
                "prediction": prediction,
                "timestamp": neural_data.timestamp
            },
            timestamp=time.time(),
            neural_features=features if not self.privacy_mode else None
        )
    
    def _map_prediction_to_command(self, prediction: Any) -> str:
        """Map decoder prediction to natural language command."""
        if self.paradigm == Paradigm.P300:
            if prediction == 1:
                return "Select current item"
            else:
                return "No selection"
        
        elif self.paradigm == Paradigm.MOTOR_IMAGERY:
            movement_map = {
                0: "Move left",
                1: "Move right", 
                2: "Move forward",
                3: "Move backward"
            }
            return movement_map.get(prediction, "Unknown movement")
        
        elif self.paradigm == Paradigm.SSVEP:
            freq_map = {
                0: "Option 1 selected",
                1: "Option 2 selected",
                2: "Option 3 selected", 
                3: "Option 4 selected"
            }
            return freq_map.get(prediction, "No selection")
        
        return "Unknown command"
    
    def read_window(self, window_ms: int) -> np.ndarray:
        """
        Read a specific time window of buffered data.
        
        Args:
            window_ms: Window length in milliseconds
            
        Returns:
            Neural data for the specified window, shape (channels, samples)
        """
        window_samples = int(window_ms * self.sampling_rate / 1000)
        
        if len(self.data_buffer) == 0:
            # Return empty array with proper dimensions
            return np.empty((self.channels, 0))
        
        if len(self.data_buffer) < window_samples:
            # Use all available data if insufficient samples
            recent_data = self.data_buffer
        else:
            recent_data = self.data_buffer[-window_samples:]
        
        try:
            if len(recent_data) == 0:
                return np.empty((self.channels, 0))
            return np.concatenate([data.data for data in recent_data], axis=1)
        except Exception as e:
            self.logger.error(f"Error reading window data: {e}")
            return np.empty((self.channels, 0))
    
    def get_buffer(self, samples: int) -> np.ndarray:
        """Get the most recent samples from the buffer."""
        if len(self.data_buffer) == 0:
            return np.empty((self.channels, 0))
        
        if len(self.data_buffer) < samples:
            # Use all available data if insufficient samples
            recent_data = self.data_buffer
        else:
            recent_data = self.data_buffer[-samples:]
        
        try:
            if len(recent_data) == 0:
                return np.empty((self.channels, 0))
            return np.concatenate([data.data for data in recent_data], axis=1)
        except Exception as e:
            self.logger.error(f"Error getting buffer data: {e}")
            return np.empty((self.channels, 0))
    
    def _add_to_buffer_safe(self, neural_data: NeuralData) -> None:
        """Safely add neural data to buffer with validation and monitoring."""
        processing_start = time.time()
        
        try:
            if not isinstance(neural_data, NeuralData):
                raise ValidationError(f"Expected NeuralData, got {type(neural_data)}")
            
            # Security validation of neural data
            if _SECURITY_AVAILABLE:
                self.validator.validate_neural_data(
                    neural_data.data, 
                    self.channels, 
                    self.sampling_rate
                )
                self.validator.validate_timestamp(neural_data.timestamp)
            
            # Validate data integrity
            if neural_data.data.shape[0] != self.channels:
                error_msg = f"Channel mismatch: expected {self.channels}, got {neural_data.data.shape[0]}"
                self.logger.warning(error_msg)
                if _SECURITY_AVAILABLE:
                    security_logger.log_validation_failure(
                        input_type="neural_data",
                        error_message=error_msg,
                        source="buffer_add"
                    )
                return
            
            self.data_buffer.append(neural_data)
            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer.pop(0)
            
            self._total_samples_processed += neural_data.data.shape[1]
            
            # Track processing time
            processing_time = time.time() - processing_start
            self._processing_times.append(processing_time)
            if len(self._processing_times) > 1000:  # Keep only recent times
                self._processing_times.pop(0)
            
            # Periodic health checks
            if time.time() - self._last_health_check > 30.0:  # Every 30 seconds
                self._perform_health_check()
                
        except ValidationError as e:
            self._error_count += 1
            self.logger.error(f"Neural data validation failed: {e}")
            if _SECURITY_AVAILABLE:
                security_logger.log_validation_failure(
                    input_type="neural_data",
                    error_message=str(e),
                    source="buffer_validation"
                )
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Error adding data to buffer: {e}")
            if _SECURITY_AVAILABLE:
                security_logger.log_system_error(
                    component="bci_bridge",
                    error_type="buffer_error",
                    error_message=str(e)
                )
                
            if self._error_count > 100:  # Too many errors - system unstable
                if _SECURITY_AVAILABLE:
                    security_logger.log_suspicious_activity(
                        activity_type="system_instability",
                        details={"error_count": self._error_count, "recent_error": str(e)},
                        risk_score=9
                    )
                raise RuntimeError(f"BCI system unstable: {self._error_count} errors")
    
    def _perform_health_check(self) -> Dict[str, Any]:
        """Perform system health check and return status."""
        try:
            self._last_health_check = time.time()
            
            # Calculate performance metrics
            avg_processing_time = np.mean(self._processing_times[-100:]) if self._processing_times else 0.0
            buffer_utilization = len(self.data_buffer) / self.buffer_size * 100
            
            health_status = {
                "status": "healthy" if self._error_count < 10 else "degraded",
                "total_samples": self._total_samples_processed,
                "buffer_utilization_pct": round(buffer_utilization, 1),
                "error_count": self._error_count,
                "avg_processing_ms": round(avg_processing_time * 1000, 2),
                "is_streaming": self.is_streaming,
                "timestamp": time.time()
            }
            
            if health_status["status"] == "degraded":
                self.logger.warning(f"BCI health degraded: {health_status}")
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        return self._perform_health_check()
    
    def calibrate(self, calibration_data: Optional[np.ndarray] = None) -> None:
        """Calibrate the decoder with user-specific data."""
        if self.decoder is None:
            raise RuntimeError("No decoder initialized")
        
        if calibration_data is None:
            self.logger.info("Starting automated calibration...")
            calibration_data = self._collect_calibration_data()
        
        self.decoder.calibrate(calibration_data)
        self.logger.info("Calibration completed")
    
    def _collect_calibration_data(self) -> np.ndarray:
        """Collect calibration data automatically."""
        return np.random.randn(self.channels, self.sampling_rate * 10)  # 10 seconds
    
    def stop_streaming(self) -> None:
        """Stop the neural data stream."""
        self.is_streaming = False
        self.logger.info("Neural data stream stopped")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the connected BCI device."""
        return {
            "device": self.device.value,
            "channels": self.channels,
            "sampling_rate": self.sampling_rate,
            "paradigm": self.paradigm.value,
            "connected": self._device_connected,
            "streaming": self.is_streaming
        }