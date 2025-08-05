"""
Signal preprocessing for neural data.
"""

import numpy as np
from scipy import signal
from scipy.stats import zscore
import logging
from typing import Dict, Any, Tuple, Optional


class SignalPreprocessor:
    """
    Preprocessor for neural signals with filtering, artifact removal, and normalization.
    """
    
    def __init__(
        self,
        sampling_rate: int = 250,
        bandpass_low: float = 0.5,
        bandpass_high: float = 50.0,
        notch_freq: float = 60.0,
        artifact_threshold: float = 100.0
    ):
        self.sampling_rate = sampling_rate
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.artifact_threshold = artifact_threshold  # microvolts
        
        self.logger = logging.getLogger(__name__)
        
        # Design filters
        self._design_filters()
        
        # Artifact detection
        self.artifact_channels = set()
        self.artifact_samples = []
        
        self.logger.info("Signal preprocessor initialized")
    
    def _design_filters(self) -> None:
        """Design digital filters for preprocessing."""
        nyquist = self.sampling_rate / 2
        
        try:
            # Bandpass filter
            low = max(self.bandpass_low / nyquist, 0.001)  # Avoid numerical issues
            high = min(self.bandpass_high / nyquist, 0.99)
            
            self.bp_b, self.bp_a = signal.butter(4, [low, high], btype='band')
            
            # Notch filter for line noise
            notch_freq_norm = self.notch_freq / nyquist
            if notch_freq_norm < 0.99:
                self.notch_b, self.notch_a = signal.iirnotch(notch_freq_norm, Q=30)
            else:
                self.notch_b, self.notch_a = None, None
                
        except Exception as e:
            self.logger.error(f"Filter design failed: {e}")
            self.bp_b, self.bp_a = None, None
            self.notch_b, self.notch_a = None, None
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to neural data.
        
        Args:
            data: Raw neural data (channels x samples)
            
        Returns:
            Preprocessed neural data
        """
        if data.size == 0:
            return data
        
        # Ensure correct shape
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        processed_data = data.copy()
        
        # 1. Artifact detection and marking
        self._detect_artifacts(processed_data)
        
        # 2. Bandpass filtering
        processed_data = self._apply_bandpass_filter(processed_data)
        
        # 3. Notch filtering (line noise removal)
        processed_data = self._apply_notch_filter(processed_data)
        
        # 4. Artifact interpolation
        processed_data = self._interpolate_artifacts(processed_data)
        
        # 5. Normalization
        processed_data = self._normalize_data(processed_data)
        
        return processed_data
    
    def _detect_artifacts(self, data: np.ndarray) -> None:
        """Detect artifacts in neural data."""
        self.artifact_channels.clear()
        self.artifact_samples.clear()
        
        for ch in range(data.shape[0]):
            channel_data = data[ch]
            
            # Amplitude-based artifact detection
            amplitude_artifacts = np.abs(channel_data) > self.artifact_threshold
            
            # Gradient-based artifact detection (sudden jumps)
            if len(channel_data) > 1:
                gradient = np.abs(np.diff(channel_data))
                gradient_threshold = 5 * np.std(gradient)
                gradient_artifacts = np.zeros_like(amplitude_artifacts)
                gradient_artifacts[1:] = gradient > gradient_threshold
            else:
                gradient_artifacts = amplitude_artifacts
            
            # Combine artifact types
            artifacts = amplitude_artifacts | gradient_artifacts
            
            if np.any(artifacts):
                self.artifact_channels.add(ch)
                artifact_indices = np.where(artifacts)[0]
                for idx in artifact_indices:
                    self.artifact_samples.append((ch, idx))
        
        if len(self.artifact_samples) > 0:
            self.logger.debug(f"Detected {len(self.artifact_samples)} artifact samples")
    
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to data."""
        if self.bp_b is None or self.bp_a is None:
            return data
        
        try:
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                if data.shape[1] > 10:  # Need minimum samples for filtering
                    filtered_data[ch] = signal.filtfilt(self.bp_b, self.bp_a, data[ch])
                else:
                    filtered_data[ch] = data[ch]
            return filtered_data
        except Exception as e:
            self.logger.warning(f"Bandpass filtering failed: {e}")
            return data
    
    def _apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove line noise."""
        if self.notch_b is None or self.notch_a is None:
            return data
        
        try:
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                if data.shape[1] > 10:
                    filtered_data[ch] = signal.filtfilt(self.notch_b, self.notch_a, data[ch])
                else:
                    filtered_data[ch] = data[ch]
            return filtered_data
        except Exception as e:
            self.logger.warning(f"Notch filtering failed: {e}")
            return data
    
    def _interpolate_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Interpolate artifact samples with neighboring values."""
        if not self.artifact_samples:
            return data
        
        interpolated_data = data.copy()
        
        for ch, sample_idx in self.artifact_samples:
            if 0 < sample_idx < data.shape[1] - 1:
                # Linear interpolation with neighboring samples
                prev_val = data[ch, sample_idx - 1]
                next_val = data[ch, sample_idx + 1]
                interpolated_data[ch, sample_idx] = (prev_val + next_val) / 2
            elif sample_idx == 0:
                # Use next sample
                interpolated_data[ch, sample_idx] = data[ch, 1] if data.shape[1] > 1 else 0
            else:
                # Use previous sample
                interpolated_data[ch, sample_idx] = data[ch, sample_idx - 1]
        
        return interpolated_data
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using z-score normalization."""
        try:
            normalized_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                channel_data = data[ch]
                if np.std(channel_data) > 1e-10:  # Avoid division by zero
                    normalized_data[ch] = zscore(channel_data)
                else:
                    normalized_data[ch] = channel_data
            return normalized_data
        except Exception as e:
            self.logger.warning(f"Normalization failed: {e}")
            return data
    
    def apply_common_average_reference(self, data: np.ndarray) -> np.ndarray:
        """Apply common average reference (CAR) to reduce common noise."""
        if data.shape[0] < 2:
            return data
        
        try:
            # Calculate average across all channels
            common_average = np.mean(data, axis=0)
            
            # Subtract from each channel
            car_data = data - common_average
            return car_data
        except Exception as e:
            self.logger.warning(f"CAR failed: {e}")
            return data
    
    def apply_laplacian_filter(self, data: np.ndarray, channel_layout: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply spatial Laplacian filter for local activity enhancement."""
        if channel_layout is None or data.shape[0] < 4:
            return data
        
        try:
            # Simple 4-neighbor Laplacian (assumes grid layout)
            laplacian_data = data.copy()
            
            for ch in range(1, data.shape[0] - 1):  # Skip edge channels
                # Estimate local average from neighbors
                neighbors = [ch-1, ch+1]  # Simple linear neighbors
                if len(neighbors) >= 2:
                    neighbor_avg = np.mean(data[neighbors], axis=0)
                    laplacian_data[ch] = data[ch] - neighbor_avg
            
            return laplacian_data
        except Exception as e:
            self.logger.warning(f"Laplacian filtering failed: {e}")
            return data
    
    def compute_power_spectral_density(self, data: np.ndarray, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density of the signal."""
        try:
            if data.shape[1] < nperseg:
                nperseg = data.shape[1] // 2
            
            freqs, psd = signal.welch(
                data, 
                fs=self.sampling_rate, 
                nperseg=nperseg,
                axis=1
            )
            return freqs, psd
        except Exception as e:
            self.logger.error(f"PSD computation failed: {e}")
            return np.array([]), np.array([])
    
    def extract_frequency_bands(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract power in standard EEG frequency bands."""
        freqs, psd = self.compute_power_spectral_density(data)
        
        if len(freqs) == 0:
            return {}
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        band_powers = {}
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if np.any(band_mask):
                band_powers[band_name] = np.mean(psd[:, band_mask], axis=1)
            else:
                band_powers[band_name] = np.zeros(psd.shape[0])
        
        return band_powers
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get statistics about the preprocessing."""
        return {
            "sampling_rate": self.sampling_rate,
            "bandpass_range": (self.bandpass_low, self.bandpass_high),
            "notch_frequency": self.notch_freq,
            "artifact_threshold": self.artifact_threshold,
            "artifact_channels": list(self.artifact_channels),
            "n_artifact_samples": len(self.artifact_samples)
        }