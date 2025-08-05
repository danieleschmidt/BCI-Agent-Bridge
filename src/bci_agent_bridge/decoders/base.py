"""
Base decoder interface for neural signal decoding.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseDecoder(ABC):
    """Abstract base class for neural signal decoders."""
    
    def __init__(self, channels: int, sampling_rate: int):
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.is_calibrated = False
        self.confidence_threshold = 0.7
        self.last_prediction = None
        self.last_confidence = 0.0
    
    @abstractmethod
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from raw neural data."""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> Any:
        """Make prediction from extracted features."""
        pass
    
    @abstractmethod
    def calibrate(self, calibration_data: np.ndarray) -> None:
        """Calibrate the decoder with user-specific data."""
        pass
    
    def get_confidence(self) -> float:
        """Get confidence score of last prediction."""
        return self.last_confidence
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set minimum confidence threshold for predictions."""
        self.confidence_threshold = threshold
    
    def is_prediction_confident(self) -> bool:
        """Check if last prediction meets confidence threshold."""
        return self.last_confidence >= self.confidence_threshold
    
    def get_decoder_info(self) -> Dict[str, Any]:
        """Get information about the decoder."""
        return {
            "type": self.__class__.__name__,
            "channels": self.channels,
            "sampling_rate": self.sampling_rate,
            "calibrated": self.is_calibrated,
            "confidence_threshold": self.confidence_threshold,
            "last_confidence": self.last_confidence
        }