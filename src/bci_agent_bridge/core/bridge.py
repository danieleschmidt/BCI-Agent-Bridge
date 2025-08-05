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
        self.device = BCIDevice(device)
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.paradigm = Paradigm(paradigm)
        self.buffer_size = buffer_size
        self.privacy_mode = privacy_mode
        
        self.logger = logging.getLogger(__name__)
        self.is_streaming = False
        self.data_buffer = []
        self.preprocessor = SignalPreprocessor(sampling_rate=sampling_rate)
        self.decoder: Optional[BaseDecoder] = None
        
        self._initialize_device()
        self._setup_decoder()
    
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
                
                self.data_buffer.append(neural_data)
                if len(self.data_buffer) > self.buffer_size:
                    self.data_buffer.pop(0)
                
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
            Neural data for the specified window
        """
        window_samples = int(window_ms * self.sampling_rate / 1000)
        
        if len(self.data_buffer) < window_samples:
            return np.array([])
        
        recent_data = self.data_buffer[-window_samples:]
        return np.concatenate([data.data for data in recent_data], axis=1)
    
    def get_buffer(self, samples: int) -> np.ndarray:
        """Get the most recent samples from the buffer."""
        if len(self.data_buffer) < samples:
            return np.array([])
        
        recent_data = self.data_buffer[-samples:]
        return np.concatenate([data.data for data in recent_data], axis=1)
    
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