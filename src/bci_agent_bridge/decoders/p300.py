"""
P300 Event-Related Potential decoder for BCI spelling and selection tasks.
"""

import numpy as np
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import logging
from typing import Dict, Any, Optional, Tuple

from .base import BaseDecoder


class P300Decoder(BaseDecoder):
    """
    P300 decoder for detecting event-related potentials in EEG signals.
    
    The P300 is a positive deflection occurring ~300ms after a rare or target stimulus.
    Used for BCI spelling (P300 speller) and binary selection tasks.
    """
    
    def __init__(self, channels: int = 8, sampling_rate: int = 250):
        super().__init__(channels, sampling_rate)
        
        self.logger = logging.getLogger(__name__)
        
        # P300 specific parameters
        self.window_start = 0.0  # seconds after stimulus
        self.window_end = 0.8    # seconds after stimulus  
        self.target_latency = 0.3  # P300 peak latency
        
        # Preprocessing parameters
        self.bandpass_low = 0.1   # Hz
        self.bandpass_high = 15.0  # Hz
        self.downsample_factor = 2
        
        # Classification
        self.classifier = LinearDiscriminantAnalysis()
        self.feature_weights = None
        
        # Calibration data
        self.target_epochs = []
        self.nontarget_epochs = []
        
        self.logger.info("P300 decoder initialized")
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract P300 features from EEG epoch.
        
        Args:
            data: EEG data (channels x samples)
            
        Returns:
            Feature vector for classification
        """
        if data.size == 0:
            return np.array([])
        
        # Ensure correct shape
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Apply bandpass filter
        filtered_data = self._apply_bandpass_filter(data)
        
        # Extract time windows around expected P300
        p300_window = self._extract_p300_window(filtered_data)
        
        # Feature extraction methods
        features = []
        
        # 1. Amplitude features - mean amplitude in P300 window
        for ch in range(p300_window.shape[0]):
            channel_features = [
                np.mean(p300_window[ch]),           # Mean amplitude
                np.max(p300_window[ch]),            # Peak amplitude  
                np.std(p300_window[ch]),            # Variability
                np.mean(p300_window[ch, 60:100])    # P300 time window (240-400ms)
            ]
            features.extend(channel_features)
        
        # 2. Temporal features - specific time points
        p300_samples = self._get_p300_samples()
        if p300_samples < p300_window.shape[1]:
            for ch in range(p300_window.shape[0]):
                features.append(p300_window[ch, p300_samples])
        
        # 3. Spatial features - differences between channels
        if p300_window.shape[0] >= 2:
            # Central vs peripheral channels (assuming central channels first)
            central_channels = p300_window[:p300_window.shape[0]//2]
            peripheral_channels = p300_window[p300_window.shape[0]//2:]
            
            central_mean = np.mean(central_channels, axis=0)
            peripheral_mean = np.mean(peripheral_channels, axis=0)
            
            # Add spatial difference features
            features.extend([
                np.mean(central_mean - peripheral_mean),
                np.max(central_mean - peripheral_mean)
            ])
        
        return np.array(features)
    
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to EEG data."""
        try:
            # Design Butterworth bandpass filter
            nyquist = self.sampling_rate / 2
            low = self.bandpass_low / nyquist
            high = min(self.bandpass_high / nyquist, 0.99)  # Prevent filter instability
            
            b, a = signal.butter(4, [low, high], btype='band')
            
            # Apply filter to each channel
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered_data[ch] = signal.filtfilt(b, a, data[ch])
            
            return filtered_data
            
        except Exception as e:
            self.logger.warning(f"Bandpass filtering failed: {e}, returning raw data")
            return data
    
    def _extract_p300_window(self, data: np.ndarray) -> np.ndarray:
        """Extract the time window containing P300 response."""
        start_sample = int(self.window_start * self.sampling_rate)
        end_sample = int(self.window_end * self.sampling_rate)
        
        # Ensure we don't exceed data bounds
        end_sample = min(end_sample, data.shape[1])
        
        if start_sample >= end_sample:
            return data
        
        return data[:, start_sample:end_sample]
    
    def _get_p300_samples(self) -> int:
        """Get sample index corresponding to P300 latency."""
        return int(self.target_latency * self.sampling_rate)
    
    def predict(self, features: np.ndarray) -> int:
        """
        Predict whether epoch contains P300 (target) or not (non-target).
        
        Args:
            features: Extracted feature vector
            
        Returns:
            Prediction: 1 for target (P300), 0 for non-target
        """
        if features.size == 0:
            self.last_confidence = 0.0
            return 0
        
        if not self.is_calibrated:
            self.logger.warning("Decoder not calibrated, returning random prediction")
            prediction = np.random.choice([0, 1])
            self.last_confidence = 0.5
            return prediction
        
        try:
            # Reshape for sklearn
            features = features.reshape(1, -1)
            
            # Get prediction and probability
            prediction = self.classifier.predict(features)[0]
            probabilities = self.classifier.predict_proba(features)[0]
            
            # Confidence is the maximum probability
            self.last_confidence = np.max(probabilities)
            self.last_prediction = prediction
            
            return int(prediction)
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            self.last_confidence = 0.0
            return 0
    
    def calibrate(self, calibration_data: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
        """
        Calibrate P300 decoder with labeled training data.
        
        Args:
            calibration_data: EEG epochs (n_epochs x channels x samples)
            labels: Labels for each epoch (1=target, 0=non-target)
        """
        self.logger.info("Starting P300 calibration...")
        
        if calibration_data.size == 0:
            self.logger.warning("Empty calibration data")
            return
        
        # Generate labels if not provided (for simulation)
        if labels is None:
            labels = self._generate_simulation_labels(calibration_data.shape[0])
        
        try:
            # Extract features from all epochs
            features_list = []
            valid_labels = []
            
            for i, epoch in enumerate(calibration_data):
                if epoch.ndim == 1:
                    epoch = epoch.reshape(self.channels, -1)
                
                features = self.extract_features(epoch)
                if features.size > 0:
                    features_list.append(features)
                    valid_labels.append(labels[i])
            
            if len(features_list) == 0:
                self.logger.error("No valid features extracted for calibration")
                return
            
            X = np.array(features_list)
            y = np.array(valid_labels)
            
            # Train classifier
            self.classifier.fit(X, y)
            
            # Evaluate performance using cross-validation
            cv_scores = cross_val_score(self.classifier, X, y, cv=5)
            accuracy = np.mean(cv_scores)
            
            self.is_calibrated = True
            self.logger.info(f"P300 calibration completed - CV accuracy: {accuracy:.3f}")
            
            # Store feature importance if available
            if hasattr(self.classifier, 'coef_'):
                self.feature_weights = self.classifier.coef_[0]
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            self.is_calibrated = False
    
    def _generate_simulation_labels(self, n_epochs: int) -> np.ndarray:
        """Generate simulated labels for calibration (30% targets)."""
        labels = np.zeros(n_epochs)
        n_targets = int(0.3 * n_epochs)  # 30% target stimuli
        target_indices = np.random.choice(n_epochs, n_targets, replace=False)
        labels[target_indices] = 1
        return labels
    
    def add_training_epoch(self, epoch: np.ndarray, is_target: bool) -> None:
        """Add a single training epoch for incremental learning."""
        if is_target:
            self.target_epochs.append(epoch)
        else:
            self.nontarget_epochs.append(epoch)
    
    def retrain_incremental(self) -> None:
        """Retrain classifier with accumulated epochs."""
        if len(self.target_epochs) == 0 or len(self.nontarget_epochs) == 0:
            self.logger.warning("Insufficient training data for incremental learning")
            return
        
        # Combine epochs and labels
        all_epochs = self.target_epochs + self.nontarget_epochs
        labels = [1] * len(self.target_epochs) + [0] * len(self.nontarget_epochs)
        
        calibration_data = np.array(all_epochs)
        self.calibrate(calibration_data, np.array(labels))
    
    def get_p300_characteristics(self) -> Dict[str, Any]:
        """Get P300-specific decoder characteristics."""
        return {
            "window_start": self.window_start,
            "window_end": self.window_end,
            "target_latency": self.target_latency,
            "bandpass_range": (self.bandpass_low, self.bandpass_high),
            "n_target_epochs": len(self.target_epochs),
            "n_nontarget_epochs": len(self.nontarget_epochs),
            "feature_weights": self.feature_weights.tolist() if self.feature_weights is not None else None
        }
    
    def detect_p300_peak(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Detect P300 peak latency and amplitude.
        
        Returns:
            Tuple of (peak_latency_ms, peak_amplitude_uv)
        """
        if data.size == 0:
            return 0.0, 0.0
        
        # Focus on central channels and P300 time window
        if data.ndim == 2 and data.shape[0] > 0:
            central_data = np.mean(data[:min(4, data.shape[0])], axis=0)
        else:
            central_data = data.flatten()
        
        # Look for peak in 250-450ms window
        start_sample = int(0.25 * self.sampling_rate)
        end_sample = int(0.45 * self.sampling_rate)
        
        if len(central_data) <= end_sample:
            return 0.0, 0.0
        
        search_window = central_data[start_sample:end_sample]
        peak_idx = np.argmax(search_window)
        
        peak_latency_ms = (start_sample + peak_idx) / self.sampling_rate * 1000
        peak_amplitude_uv = search_window[peak_idx]
        
        return peak_latency_ms, peak_amplitude_uv