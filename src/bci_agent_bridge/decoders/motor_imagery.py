"""
Motor Imagery decoder for BCI control applications.
"""

import numpy as np
from scipy import signal
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import logging
from typing import Dict, Any, Optional, Tuple, List

from .base import BaseDecoder


class CommonSpatialPatterns:
    """Common Spatial Patterns (CSP) implementation for motor imagery."""
    
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.filters = None
        self.eigenvalues = None
        self.is_fitted = False
    
    def fit(self, X1: np.ndarray, X2: np.ndarray) -> 'CommonSpatialPatterns':
        """
        Fit CSP filters on two-class data.
        
        Args:
            X1: Data for class 1 (trials x channels x time)
            X2: Data for class 2 (trials x channels x time)
        """
        # Compute covariance matrices
        C1 = self._compute_covariance_matrix(X1)
        C2 = self._compute_covariance_matrix(X2)
        
        # Solve generalized eigenvalue problem
        eigenvals, eigenvecs = eigh(C1, C1 + C2)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Select components (first and last n_components/2)
        n_comp_half = self.n_components // 2
        selected_idx = np.concatenate([
            np.arange(n_comp_half),  # First components
            np.arange(-n_comp_half, 0)  # Last components
        ])
        
        self.filters = eigenvecs[:, selected_idx].T
        self.eigenvalues = eigenvals[selected_idx]
        self.is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply CSP filters to data.
        
        Args:
            X: Input data (trials x channels x time)
            
        Returns:
            CSP features (trials x features)
        """
        if not self.is_fitted:
            raise ValueError("CSP not fitted. Call fit() first.")
        
        n_trials, n_channels, n_times = X.shape
        features = np.zeros((n_trials, self.n_components))
        
        for trial in range(n_trials):
            # Apply spatial filters
            filtered = self.filters @ X[trial]
            
            # Compute log-variance features
            for comp in range(self.n_components):
                var = np.var(filtered[comp])
                features[trial, comp] = np.log(var + 1e-10)  # Add small constant for stability
        
        return features
    
    def _compute_covariance_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute average covariance matrix for trials."""
        n_trials, n_channels, _ = X.shape
        C = np.zeros((n_channels, n_channels))
        
        for trial in range(n_trials):
            trial_data = X[trial]
            trial_cov = np.cov(trial_data)
            C += trial_cov / np.trace(trial_cov)  # Normalize by trace
        
        return C / n_trials


class MotorImageryDecoder(BaseDecoder):
    """
    Motor Imagery decoder using CSP and machine learning classification.
    
    Decodes imagined movements (left hand, right hand, feet, tongue) from
    mu and beta rhythm modulations in sensorimotor cortex.
    """
    
    def __init__(
        self, 
        channels: int = 8, 
        sampling_rate: int = 250,
        frequency_bands: List[Tuple[float, float]] = None,
        n_csp_components: int = 4
    ):
        super().__init__(channels, sampling_rate)
        
        self.logger = logging.getLogger(__name__)
        
        # Motor imagery parameters
        self.frequency_bands = frequency_bands or [(8, 12), (12, 30)]  # Alpha and beta
        self.n_csp_components = n_csp_components
        
        # Time window for motor imagery (typically 2-4 seconds)
        self.mi_window_start = 0.5  # seconds after cue
        self.mi_window_end = 3.5    # seconds after cue
        
        # Feature extraction
        self.csp_filters = {}  # One CSP per frequency band
        self.frequency_features = {}
        
        # Classification
        self.classifier = LinearDiscriminantAnalysis()
        self.classes = ['left_hand', 'right_hand', 'feet', 'tongue']
        
        # Training data storage
        self.training_data = {cls: [] for cls in self.classes}
        
        self.logger.info("Motor Imagery decoder initialized")
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract motor imagery features from EEG data.
        
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
        
        # Extract motor imagery time window
        mi_data = self._extract_mi_window(data)
        
        if mi_data.size == 0:
            return np.array([])
        
        features = []
        
        # 1. Frequency band power features
        for band_name, (low_freq, high_freq) in zip(['alpha', 'beta'], self.frequency_bands):
            band_features = self._extract_frequency_band_features(mi_data, low_freq, high_freq)
            features.extend(band_features)
        
        # 2. CSP features (if available)
        if self.is_calibrated:
            csp_features = self._extract_csp_features(mi_data)
            features.extend(csp_features)
        
        # 3. Temporal features
        temporal_features = self._extract_temporal_features(mi_data)
        features.extend(temporal_features)
        
        # 4. Spatial features
        spatial_features = self._extract_spatial_features(mi_data)
        features.extend(spatial_features)
        
        return np.array(features)
    
    def _extract_mi_window(self, data: np.ndarray) -> np.ndarray:
        """Extract motor imagery time window."""
        start_sample = int(self.mi_window_start * self.sampling_rate)
        end_sample = int(self.mi_window_end * self.sampling_rate)
        
        if end_sample > data.shape[1]:
            end_sample = data.shape[1]
        
        if start_sample >= end_sample:
            return np.array([])
        
        return data[:, start_sample:end_sample]
    
    def _extract_frequency_band_features(
        self, 
        data: np.ndarray, 
        low_freq: float, 
        high_freq: float
    ) -> List[float]:
        """Extract power features from specific frequency band."""
        try:
            # Design bandpass filter
            nyquist = self.sampling_rate / 2
            low = max(low_freq / nyquist, 0.01)
            high = min(high_freq / nyquist, 0.99)
            
            b, a = signal.butter(4, [low, high], btype='band')
            
            features = []
            for ch in range(data.shape[0]):
                if data.shape[1] > 10:  # Minimum samples for filtering
                    filtered = signal.filtfilt(b, a, data[ch])
                    power = np.mean(filtered ** 2)
                    features.append(np.log(power + 1e-10))  # Log power
                else:
                    features.append(0.0)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Frequency band feature extraction failed: {e}")
            return [0.0] * data.shape[0]
    
    def _extract_csp_features(self, data: np.ndarray) -> List[float]:
        """Extract CSP features if filters are available."""
        features = []
        
        try:
            for band_name, csp in self.csp_filters.items():
                if csp.is_fitted:
                    # Reshape for CSP (expects trials x channels x time)
                    data_reshaped = data.reshape(1, data.shape[0], data.shape[1])
                    csp_features = csp.transform(data_reshaped)
                    features.extend(csp_features.flatten())
        
        except Exception as e:
            self.logger.warning(f"CSP feature extraction failed: {e}")
        
        return features
    
    def _extract_temporal_features(self, data: np.ndarray) -> List[float]:
        """Extract temporal dynamics features."""
        features = []
        
        try:
            for ch in range(data.shape[0]):
                channel_data = data[ch]
                
                # Statistical moments
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    self._skewness(channel_data),
                    self._kurtosis(channel_data)
                ])
                
                # Temporal dynamics
                if len(channel_data) > 1:
                    # First-order difference (velocity)
                    velocity = np.diff(channel_data)
                    features.append(np.std(velocity))
                    
                    # Zero-crossing rate
                    zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
                    features.append(zero_crossings / len(channel_data))
                else:
                    features.extend([0.0, 0.0])
        
        except Exception as e:
            self.logger.warning(f"Temporal feature extraction failed: {e}")
        
        return features
    
    def _extract_spatial_features(self, data: np.ndarray) -> List[float]:
        """Extract spatial relationship features between channels."""
        features = []
        
        try:
            if data.shape[0] >= 2:
                # Correlation features
                for i in range(min(4, data.shape[0])):  # Limit to avoid too many features
                    for j in range(i + 1, min(4, data.shape[0])):
                        if data.shape[1] > 1:
                            corr = np.corrcoef(data[i], data[j])[0, 1]
                            features.append(corr if not np.isnan(corr) else 0.0)
                
                # Laterality features (assuming channels are ordered)
                if data.shape[0] >= 4:
                    left_channels = data[:data.shape[0]//2]
                    right_channels = data[data.shape[0]//2:]
                    
                    left_power = np.mean(left_channels ** 2)
                    right_power = np.mean(right_channels ** 2)
                    
                    # Laterality index
                    total_power = left_power + right_power
                    if total_power > 0:
                        laterality = (left_power - right_power) / total_power
                        features.append(laterality)
                    else:
                        features.append(0.0)
        
        except Exception as e:
            self.logger.warning(f"Spatial feature extraction failed: {e}")
        
        return features
    
    def _skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                return np.mean(((data - mean) / std) ** 3)
            return 0.0
        except:
            return 0.0
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                return np.mean(((data - mean) / std) ** 4) - 3
            return 0.0
        except:
            return 0.0
    
    def predict(self, features: np.ndarray) -> int:
        """
        Predict motor imagery class from features.
        
        Args:
            features: Extracted feature vector
            
        Returns:
            Predicted class (0=left_hand, 1=right_hand, 2=feet, 3=tongue)
        """
        if not self.is_calibrated:
            self.logger.warning("Decoder not calibrated, returning random prediction")
            prediction = np.random.choice(len(self.classes))
            self.last_confidence = 0.25  # Random chance for 4 classes
            return prediction
        
        if features.size == 0:
            self.last_confidence = 0.0
            return 0
        
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
        Calibrate motor imagery decoder with training data.
        
        Args:
            calibration_data: EEG trials (n_trials x channels x samples)
            labels: Class labels for each trial
        """
        self.logger.info("Starting Motor Imagery calibration...")
        
        if calibration_data.size == 0:
            self.logger.warning("Empty calibration data")
            return
        
        # Generate labels if not provided
        if labels is None:
            labels = self._generate_simulation_labels(calibration_data.shape[0])
        
        try:
            # Train CSP filters for each frequency band
            self._train_csp_filters(calibration_data, labels)
            
            # Extract features from all trials
            features_list = []
            valid_labels = []
            
            for i, trial in enumerate(calibration_data):
                if trial.ndim == 2:  # channels x samples
                    features = self.extract_features(trial)
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
            
            # Evaluate performance
            if len(np.unique(y)) > 1:  # Need multiple classes for CV
                cv_scores = cross_val_score(self.classifier, X, y, cv=3)
                accuracy = np.mean(cv_scores)
            else:
                accuracy = 0.5
            
            self.is_calibrated = True
            self.logger.info(f"Motor Imagery calibration completed - CV accuracy: {accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            self.is_calibrated = False
    
    def _train_csp_filters(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Train CSP filters for each frequency band."""
        try:
            unique_labels = np.unique(labels)
            
            # For each frequency band, train CSP on binary problems
            for band_idx, (low_freq, high_freq) in enumerate(self.frequency_bands):
                band_name = f"band_{band_idx}"
                
                # Filter data to frequency band
                filtered_data = self._filter_data_to_band(data, low_freq, high_freq)
                
                # Train CSP on first two classes (if available)
                if len(unique_labels) >= 2:
                    class1_data = filtered_data[labels == unique_labels[0]]
                    class2_data = filtered_data[labels == unique_labels[1]]
                    
                    if len(class1_data) > 0 and len(class2_data) > 0:
                        csp = CommonSpatialPatterns(n_components=self.n_csp_components)
                        csp.fit(class1_data, class2_data)
                        self.csp_filters[band_name] = csp
        
        except Exception as e:
            self.logger.warning(f"CSP training failed: {e}")
    
    def _filter_data_to_band(
        self, 
        data: np.ndarray, 
        low_freq: float, 
        high_freq: float
    ) -> np.ndarray:
        """Filter data to specific frequency band."""
        try:
            nyquist = self.sampling_rate / 2
            low = max(low_freq / nyquist, 0.01)
            high = min(high_freq / nyquist, 0.99)
            
            b, a = signal.butter(4, [low, high], btype='band')
            
            filtered_data = np.zeros_like(data)
            for trial in range(data.shape[0]):
                for ch in range(data.shape[1]):
                    if data.shape[2] > 10:
                        filtered_data[trial, ch] = signal.filtfilt(b, a, data[trial, ch])
                    else:
                        filtered_data[trial, ch] = data[trial, ch]
            
            return filtered_data
            
        except Exception as e:
            self.logger.warning(f"Band filtering failed: {e}")
            return data
    
    def _generate_simulation_labels(self, n_trials: int) -> np.ndarray:
        """Generate simulated labels for calibration."""
        return np.random.choice(len(self.classes), n_trials)
    
    def add_training_trial(self, trial_data: np.ndarray, class_label: str) -> None:
        """Add a training trial for incremental learning."""
        if class_label in self.classes:
            self.training_data[class_label].append(trial_data)
        else:
            self.logger.warning(f"Unknown class label: {class_label}")
    
    def retrain_incremental(self) -> None:
        """Retrain with accumulated training data."""
        all_trials = []
        all_labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            trials = self.training_data[class_name]
            all_trials.extend(trials)
            all_labels.extend([class_idx] * len(trials))
        
        if len(all_trials) > 0:
            calibration_data = np.array(all_trials)
            labels = np.array(all_labels)
            self.calibrate(calibration_data, labels)
    
    def get_mi_characteristics(self) -> Dict[str, Any]:
        """Get motor imagery specific characteristics."""
        return {
            "classes": self.classes,
            "frequency_bands": self.frequency_bands,
            "n_csp_components": self.n_csp_components,
            "mi_window": (self.mi_window_start, self.mi_window_end),
            "csp_filters_trained": list(self.csp_filters.keys()),
            "training_trials_per_class": {
                cls: len(trials) for cls, trials in self.training_data.items()
            }
        }
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index."""
        if 0 <= class_idx < len(self.classes):
            return self.classes[class_idx]
        return "unknown"
    
    def compute_feature_importance(self) -> Optional[np.ndarray]:
        """Compute feature importance if available."""
        if self.is_calibrated and hasattr(self.classifier, 'coef_'):
            return np.abs(self.classifier.coef_).mean(axis=0)
        return None