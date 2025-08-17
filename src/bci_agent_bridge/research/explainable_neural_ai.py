"""
Explainable AI for Neural Signal Interpretation.

This module implements cutting-edge explainable AI techniques for interpreting
neural signals and BCI decisions, providing transparency and trust in
medical-grade BCI systems.

Key innovations:
- Neural signal saliency mapping
- Feature importance attribution
- Causal inference for neural patterns
- Interactive explanation interfaces
- Uncertainty-aware explanations
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
from collections import defaultdict, deque
from scipy.stats import pearsonr, spearmanr
from scipy.signal import find_peaks
import warnings

logger = logging.getLogger(__name__)


@dataclass
class ExplanationConfig:
    """Configuration for explainable AI system."""
    explanation_methods: List[str] = field(default_factory=lambda: ["saliency", "shap", "attention"])
    temporal_resolution: float = 0.1  # seconds
    spatial_resolution: int = 8  # channels
    confidence_threshold: float = 0.6
    feature_importance_threshold: float = 0.1
    causal_window_size: int = 500  # samples
    interactive_mode: bool = True
    save_explanations: bool = True


@dataclass
class NeuralExplanation:
    """Comprehensive explanation of neural signal interpretation."""
    prediction: Any
    confidence: float
    saliency_map: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    temporal_attribution: Optional[np.ndarray] = None
    spatial_attribution: Optional[np.ndarray] = None
    causal_factors: Optional[Dict[str, float]] = None
    uncertainty_sources: Optional[List[str]] = None
    explanation_quality: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NeuralSaliencyMapper:
    """Advanced saliency mapping for neural signals."""
    
    def __init__(self, temporal_resolution: float = 0.1, spatial_resolution: int = 8):
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution = spatial_resolution
        self.baseline_signals = deque(maxlen=1000)
        self.gradient_cache = {}
        
    def compute_saliency_map(self, neural_signals: np.ndarray, 
                           prediction: Any, decoder_func: callable) -> np.ndarray:
        """
        Compute saliency map for neural signals.
        
        Args:
            neural_signals: Neural signal data [channels, samples]
            prediction: Model prediction
            decoder_func: Decoder function to analyze
            
        Returns:
            Saliency map [channels, samples]
        """
        if neural_signals.size == 0:
            return np.zeros_like(neural_signals)
            
        # Ensure 2D array
        if neural_signals.ndim == 1:
            neural_signals = neural_signals.reshape(1, -1)
            
        saliency_map = np.zeros_like(neural_signals)
        
        # Compute gradients via finite differences
        perturbation_magnitude = 0.01 * np.std(neural_signals)
        
        for channel in range(neural_signals.shape[0]):
            for sample in range(0, neural_signals.shape[1], max(1, int(neural_signals.shape[1] // 50))):
                # Perturb signal
                perturbed_signals = neural_signals.copy()
                perturbed_signals[channel, sample] += perturbation_magnitude
                
                try:
                    # Get perturbed prediction
                    perturbed_prediction = decoder_func(perturbed_signals)
                    
                    # Calculate gradient
                    if hasattr(prediction, '__len__') and hasattr(perturbed_prediction, '__len__'):
                        gradient = np.linalg.norm(np.array(perturbed_prediction) - np.array(prediction))
                    else:
                        gradient = abs(float(perturbed_prediction) - float(prediction))
                        
                    saliency_map[channel, sample] = gradient / perturbation_magnitude
                    
                except Exception as e:
                    logger.warning(f"Saliency computation failed at [{channel}, {sample}]: {e}")
                    continue
        
        # Normalize saliency map
        max_saliency = np.max(np.abs(saliency_map))
        if max_saliency > 0:
            saliency_map = saliency_map / max_saliency
            
        return saliency_map
    
    def compute_temporal_attribution(self, neural_signals: np.ndarray, 
                                   saliency_map: np.ndarray) -> np.ndarray:
        """Compute temporal attribution scores."""
        if saliency_map.size == 0:
            return np.array([])
            
        # Average across channels for temporal attribution
        temporal_attribution = np.mean(np.abs(saliency_map), axis=0)
        
        # Smooth with moving average
        window_size = max(1, int(len(temporal_attribution) * 0.05))
        if len(temporal_attribution) >= window_size:
            temporal_attribution = np.convolve(
                temporal_attribution, 
                np.ones(window_size) / window_size, 
                mode='same'
            )
        
        return temporal_attribution
    
    def compute_spatial_attribution(self, neural_signals: np.ndarray, 
                                  saliency_map: np.ndarray) -> np.ndarray:
        """Compute spatial (channel) attribution scores."""
        if saliency_map.size == 0:
            return np.array([])
            
        # Average across time for spatial attribution
        spatial_attribution = np.mean(np.abs(saliency_map), axis=1)
        
        return spatial_attribution
    
    def identify_critical_time_windows(self, temporal_attribution: np.ndarray, 
                                     threshold: float = 0.7) -> List[Tuple[int, int]]:
        """Identify critical time windows based on attribution."""
        if len(temporal_attribution) == 0:
            return []
            
        # Find peaks above threshold
        peaks, properties = find_peaks(
            temporal_attribution, 
            height=threshold * np.max(temporal_attribution),
            distance=max(1, len(temporal_attribution) // 20)
        )
        
        # Create windows around peaks
        window_size = max(1, len(temporal_attribution) // 50)
        windows = []
        
        for peak in peaks:
            start = max(0, peak - window_size // 2)
            end = min(len(temporal_attribution), peak + window_size // 2)
            windows.append((start, end))
        
        return windows


class FeatureImportanceAnalyzer:
    """Advanced feature importance analysis for neural signals."""
    
    def __init__(self):
        self.feature_history = deque(maxlen=500)
        self.importance_cache = {}
        
    def analyze_feature_importance(self, neural_signals: np.ndarray, 
                                 prediction: Any, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze importance of different neural signal features.
        
        Args:
            neural_signals: Raw neural signals
            prediction: Model prediction
            features: Extracted feature dictionary
            
        Returns:
            Dictionary of feature importances
        """
        if not features:
            return {}
            
        importances = {}
        
        # Temporal features
        importances.update(self._analyze_temporal_features(neural_signals))
        
        # Frequency features
        importances.update(self._analyze_frequency_features(neural_signals))
        
        # Statistical features
        importances.update(self._analyze_statistical_features(neural_signals))
        
        # Spatial features
        importances.update(self._analyze_spatial_features(neural_signals))
        
        # Normalize importances
        total_importance = sum(abs(v) for v in importances.values())
        if total_importance > 0:
            importances = {k: v/total_importance for k, v in importances.items()}
        
        return importances
    
    def _analyze_temporal_features(self, signals: np.ndarray) -> Dict[str, float]:
        """Analyze temporal feature importance."""
        if signals.size == 0:
            return {}
            
        features = {}
        
        # Signal variability over time
        if signals.ndim > 1:
            temporal_var = np.var(signals, axis=1)
            features['temporal_variability'] = np.mean(temporal_var)
        else:
            features['temporal_variability'] = np.var(signals)
        
        # Trend analysis
        if len(signals.flatten()) > 10:
            x = np.arange(len(signals.flatten()))
            coeffs = np.polyfit(x, signals.flatten(), 1)
            features['temporal_trend'] = abs(coeffs[0])
        else:
            features['temporal_trend'] = 0.0
        
        # Peak density
        try:
            peaks, _ = find_peaks(signals.flatten(), distance=max(1, len(signals.flatten()) // 20))
            features['peak_density'] = len(peaks) / len(signals.flatten())
        except:
            features['peak_density'] = 0.0
        
        return features
    
    def _analyze_frequency_features(self, signals: np.ndarray) -> Dict[str, float]:
        """Analyze frequency domain feature importance."""
        if signals.size == 0:
            return {}
            
        features = {}
        
        try:
            from scipy.fft import fft, fftfreq
            
            # Compute FFT
            signal_1d = signals.flatten()
            fft_vals = fft(signal_1d)
            freqs = fftfreq(len(signal_1d))
            
            # Power spectral density
            psd = np.abs(fft_vals) ** 2
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(psd[1:]) + 1  # Skip DC component
            features['dominant_frequency'] = abs(freqs[dominant_freq_idx])
            
            # Frequency band powers
            nyquist = 0.5 * len(signal_1d)
            alpha_band = (8/nyquist <= abs(freqs)) & (abs(freqs) <= 12/nyquist)
            beta_band = (12/nyquist <= abs(freqs)) & (abs(freqs) <= 30/nyquist)
            gamma_band = (30/nyquist <= abs(freqs)) & (abs(freqs) <= 100/nyquist)
            
            total_power = np.sum(psd)
            if total_power > 0:
                features['alpha_power'] = np.sum(psd[alpha_band]) / total_power
                features['beta_power'] = np.sum(psd[beta_band]) / total_power
                features['gamma_power'] = np.sum(psd[gamma_band]) / total_power
            else:
                features['alpha_power'] = 0.0
                features['beta_power'] = 0.0
                features['gamma_power'] = 0.0
                
        except Exception as e:
            logger.warning(f"Frequency analysis failed: {e}")
            features = {
                'dominant_frequency': 0.0,
                'alpha_power': 0.0,
                'beta_power': 0.0,
                'gamma_power': 0.0
            }
        
        return features
    
    def _analyze_statistical_features(self, signals: np.ndarray) -> Dict[str, float]:
        """Analyze statistical feature importance."""
        if signals.size == 0:
            return {}
            
        signal_1d = signals.flatten()
        
        features = {
            'signal_mean': abs(np.mean(signal_1d)),
            'signal_std': np.std(signal_1d),
            'signal_skewness': abs(float(self._safe_skewness(signal_1d))),
            'signal_kurtosis': abs(float(self._safe_kurtosis(signal_1d))),
            'signal_energy': np.sum(signal_1d ** 2) / len(signal_1d),
            'signal_range': np.max(signal_1d) - np.min(signal_1d)
        }
        
        return features
    
    def _analyze_spatial_features(self, signals: np.ndarray) -> Dict[str, float]:
        """Analyze spatial (cross-channel) feature importance."""
        if signals.ndim < 2 or signals.shape[0] < 2:
            return {'spatial_correlation': 0.0}
            
        features = {}
        
        # Cross-channel correlation
        correlations = []
        for i in range(signals.shape[0]):
            for j in range(i + 1, signals.shape[0]):
                try:
                    corr, _ = pearsonr(signals[i], signals[j])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except:
                    continue
        
        features['spatial_correlation'] = np.mean(correlations) if correlations else 0.0
        
        # Channel dominance
        channel_powers = np.mean(signals ** 2, axis=1)
        total_power = np.sum(channel_powers)
        if total_power > 0:
            features['channel_dominance'] = np.max(channel_powers) / total_power
        else:
            features['channel_dominance'] = 0.0
        
        return features
    
    def _safe_skewness(self, data: np.ndarray) -> float:
        """Safely compute skewness."""
        try:
            from scipy.stats import skew
            return skew(data)
        except:
            return 0.0
    
    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """Safely compute kurtosis."""
        try:
            from scipy.stats import kurtosis
            return kurtosis(data)
        except:
            return 0.0


class CausalInferenceEngine:
    """Causal inference for neural signal patterns."""
    
    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self.causal_history = deque(maxlen=100)
        self.pattern_library = {}
        
    def analyze_causal_factors(self, neural_signals: np.ndarray, 
                             prediction: Any, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze causal factors contributing to prediction.
        
        Args:
            neural_signals: Neural signal data
            prediction: Model prediction
            context: Additional context information
            
        Returns:
            Dictionary of causal factor contributions
        """
        if neural_signals.size == 0:
            return {}
            
        causal_factors = {}
        
        # Signal amplitude causality
        causal_factors['signal_amplitude'] = self._analyze_amplitude_causality(neural_signals)
        
        # Temporal pattern causality
        causal_factors['temporal_patterns'] = self._analyze_temporal_causality(neural_signals)
        
        # Frequency causality
        causal_factors['frequency_patterns'] = self._analyze_frequency_causality(neural_signals)
        
        # Context causality
        causal_factors.update(self._analyze_context_causality(context))
        
        # Normalize causal factors
        total_causality = sum(abs(v) for v in causal_factors.values())
        if total_causality > 0:
            causal_factors = {k: v/total_causality for k, v in causal_factors.items()}
        
        return causal_factors
    
    def _analyze_amplitude_causality(self, signals: np.ndarray) -> float:
        """Analyze causal contribution of signal amplitude."""
        if signals.size == 0:
            return 0.0
            
        # Calculate relative amplitude compared to baseline
        current_amplitude = np.mean(np.abs(signals))
        
        if hasattr(self, '_baseline_amplitude'):
            amplitude_ratio = current_amplitude / (self._baseline_amplitude + 1e-10)
            causality = min(1.0, abs(amplitude_ratio - 1.0))
        else:
            self._baseline_amplitude = current_amplitude
            causality = 0.0
        
        return causality
    
    def _analyze_temporal_causality(self, signals: np.ndarray) -> float:
        """Analyze causal contribution of temporal patterns."""
        if signals.size == 0:
            return 0.0
            
        # Look for specific temporal patterns
        signal_1d = signals.flatten()
        
        # Burst detection
        threshold = np.mean(signal_1d) + 2 * np.std(signal_1d)
        burst_samples = np.sum(signal_1d > threshold)
        burst_ratio = burst_samples / len(signal_1d)
        
        # Rhythmic patterns
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(signal_1d, distance=max(1, len(signal_1d) // 20))
            if len(peaks) > 2:
                peak_intervals = np.diff(peaks)
                rhythm_consistency = 1.0 - (np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-10))
                rhythm_consistency = max(0.0, min(1.0, rhythm_consistency))
            else:
                rhythm_consistency = 0.0
        except:
            rhythm_consistency = 0.0
        
        # Combine temporal causality measures
        temporal_causality = 0.6 * burst_ratio + 0.4 * rhythm_consistency
        
        return min(1.0, temporal_causality)
    
    def _analyze_frequency_causality(self, signals: np.ndarray) -> float:
        """Analyze causal contribution of frequency patterns."""
        if signals.size == 0:
            return 0.0
            
        try:
            from scipy.fft import fft
            
            signal_1d = signals.flatten()
            fft_vals = fft(signal_1d)
            psd = np.abs(fft_vals) ** 2
            
            # Focus on specific frequency bands
            n_samples = len(signal_1d)
            freqs = np.fft.fftfreq(n_samples)
            
            # Neural frequency bands (normalized)
            alpha_power = np.sum(psd[(abs(freqs) >= 0.02) & (abs(freqs) <= 0.05)])
            beta_power = np.sum(psd[(abs(freqs) >= 0.05) & (abs(freqs) <= 0.12)])
            total_power = np.sum(psd)
            
            if total_power > 0:
                frequency_causality = (alpha_power + beta_power) / total_power
            else:
                frequency_causality = 0.0
                
        except Exception as e:
            logger.warning(f"Frequency causality analysis failed: {e}")
            frequency_causality = 0.0
        
        return min(1.0, frequency_causality)
    
    def _analyze_context_causality(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze causal contribution of context factors."""
        causal_factors = {}
        
        # Time of day effect
        if 'timestamp' in context:
            try:
                hour = time.localtime(context['timestamp']).tm_hour
                # Simple circadian rhythm effect
                circadian_factor = 0.5 + 0.5 * np.cos(2 * np.pi * (hour - 14) / 24)
                causal_factors['circadian_rhythm'] = abs(circadian_factor - 0.5) * 2
            except:
                causal_factors['circadian_rhythm'] = 0.0
        
        # User state factors
        if 'user_state' in context:
            state = context['user_state']
            if state == 'focused':
                causal_factors['attention_state'] = 0.8
            elif state == 'relaxed':
                causal_factors['attention_state'] = 0.3
            else:
                causal_factors['attention_state'] = 0.5
        
        # Session factors
        if 'session_duration' in context:
            duration = context['session_duration']
            # Fatigue effect (quadratic)
            fatigue_factor = min(1.0, (duration / 3600) ** 2)  # Normalize by 1 hour
            causal_factors['fatigue_effect'] = fatigue_factor
        
        return causal_factors


class ExplainableNeuralAI:
    """Comprehensive explainable AI system for neural signal interpretation."""
    
    def __init__(self, config: ExplanationConfig):
        self.config = config
        self.saliency_mapper = NeuralSaliencyMapper(
            temporal_resolution=config.temporal_resolution,
            spatial_resolution=config.spatial_resolution
        )
        self.importance_analyzer = FeatureImportanceAnalyzer()
        self.causal_engine = CausalInferenceEngine(
            window_size=config.causal_window_size
        )
        self.explanation_history = deque(maxlen=1000)
        
    def generate_explanation(self, neural_signals: np.ndarray, 
                           prediction: Any, decoder_func: callable,
                           features: Optional[Dict[str, Any]] = None,
                           context: Optional[Dict[str, Any]] = None) -> NeuralExplanation:
        """
        Generate comprehensive explanation for neural signal interpretation.
        
        Args:
            neural_signals: Neural signal data
            prediction: Model prediction
            decoder_func: Decoder function for analysis
            features: Optional extracted features
            context: Optional context information
            
        Returns:
            Comprehensive neural explanation
        """
        if neural_signals.size == 0:
            return NeuralExplanation(
                prediction=prediction,
                confidence=0.0,
                explanation_quality=0.0
            )
            
        if features is None:
            features = {}
        if context is None:
            context = {}
            
        explanation = NeuralExplanation(
            prediction=prediction,
            confidence=self._estimate_confidence(prediction, neural_signals)
        )
        
        # Generate saliency map
        if "saliency" in self.config.explanation_methods:
            try:
                explanation.saliency_map = self.saliency_mapper.compute_saliency_map(
                    neural_signals, prediction, decoder_func
                )
                
                # Compute attributions
                explanation.temporal_attribution = self.saliency_mapper.compute_temporal_attribution(
                    neural_signals, explanation.saliency_map
                )
                explanation.spatial_attribution = self.saliency_mapper.compute_spatial_attribution(
                    neural_signals, explanation.saliency_map
                )
                
            except Exception as e:
                logger.warning(f"Saliency mapping failed: {e}")
                explanation.saliency_map = np.zeros_like(neural_signals)
        
        # Analyze feature importance
        explanation.feature_importance = self.importance_analyzer.analyze_feature_importance(
            neural_signals, prediction, features
        )
        
        # Perform causal analysis
        explanation.causal_factors = self.causal_engine.analyze_causal_factors(
            neural_signals, prediction, context
        )
        
        # Identify uncertainty sources
        explanation.uncertainty_sources = self._identify_uncertainty_sources(
            neural_signals, explanation
        )
        
        # Calculate explanation quality
        explanation.explanation_quality = self._calculate_explanation_quality(explanation)
        
        # Store explanation
        self.explanation_history.append(explanation)
        
        return explanation
    
    def _estimate_confidence(self, prediction: Any, neural_signals: np.ndarray) -> float:
        """Estimate prediction confidence."""
        if neural_signals.size == 0:
            return 0.0
            
        # Simple confidence based on signal quality
        signal_quality = self._assess_signal_quality(neural_signals)
        
        # Prediction consistency (if vector prediction)
        if hasattr(prediction, '__len__') and len(prediction) > 1:
            pred_array = np.array(prediction)
            pred_entropy = -np.sum(pred_array * np.log(pred_array + 1e-10))
            pred_confidence = 1.0 - (pred_entropy / np.log(len(pred_array)))
        else:
            pred_confidence = 0.8  # Default for scalar predictions
        
        # Combine confidence measures
        overall_confidence = 0.6 * signal_quality + 0.4 * pred_confidence
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _assess_signal_quality(self, signals: np.ndarray) -> float:
        """Assess overall signal quality."""
        if signals.size == 0:
            return 0.0
            
        # Signal-to-noise ratio estimate
        signal_power = np.mean(signals ** 2)
        noise_estimate = np.var(np.diff(signals.flatten()))
        
        if noise_estimate > 0:
            snr = signal_power / noise_estimate
            quality = 1.0 / (1.0 + np.exp(-snr + 5))  # Sigmoid with offset
        else:
            quality = 1.0
        
        return max(0.0, min(1.0, quality))
    
    def _identify_uncertainty_sources(self, neural_signals: np.ndarray, 
                                    explanation: NeuralExplanation) -> List[str]:
        """Identify sources of prediction uncertainty."""
        uncertainty_sources = []
        
        # Low signal quality
        signal_quality = self._assess_signal_quality(neural_signals)
        if signal_quality < 0.5:
            uncertainty_sources.append("low_signal_quality")
        
        # Low feature importance
        if explanation.feature_importance:
            max_importance = max(explanation.feature_importance.values())
            if max_importance < self.config.feature_importance_threshold:
                uncertainty_sources.append("weak_feature_importance")
        
        # Inconsistent causal factors
        if explanation.causal_factors:
            causal_variance = np.var(list(explanation.causal_factors.values()))
            if causal_variance > 0.1:
                uncertainty_sources.append("inconsistent_causal_factors")
        
        # Low confidence
        if explanation.confidence < self.config.confidence_threshold:
            uncertainty_sources.append("low_confidence")
        
        # Weak saliency
        if explanation.saliency_map is not None:
            max_saliency = np.max(np.abs(explanation.saliency_map))
            if max_saliency < 0.1:
                uncertainty_sources.append("weak_saliency")
        
        return uncertainty_sources
    
    def _calculate_explanation_quality(self, explanation: NeuralExplanation) -> float:
        """Calculate overall explanation quality."""
        quality_components = []
        
        # Confidence component
        quality_components.append(explanation.confidence)
        
        # Feature importance component
        if explanation.feature_importance:
            max_importance = max(explanation.feature_importance.values())
            quality_components.append(max_importance)
        
        # Causal factor component
        if explanation.causal_factors:
            causal_strength = sum(explanation.causal_factors.values())
            quality_components.append(min(1.0, causal_strength))
        
        # Saliency component
        if explanation.saliency_map is not None:
            saliency_strength = np.mean(np.abs(explanation.saliency_map))
            quality_components.append(min(1.0, saliency_strength * 10))
        
        # Uncertainty component (inverse)
        uncertainty_penalty = len(explanation.uncertainty_sources or []) / 5.0
        quality_components.append(max(0.0, 1.0 - uncertainty_penalty))
        
        # Calculate weighted average
        if quality_components:
            overall_quality = np.mean(quality_components)
        else:
            overall_quality = 0.0
        
        return max(0.0, min(1.0, overall_quality))
    
    def get_explanation_summary(self, explanation: NeuralExplanation) -> Dict[str, Any]:
        """Get human-readable explanation summary."""
        summary = {
            "prediction": explanation.prediction,
            "confidence": f"{explanation.confidence:.2f}",
            "quality": f"{explanation.explanation_quality:.2f}",
            "timestamp": explanation.timestamp
        }
        
        # Top features
        if explanation.feature_importance:
            top_features = sorted(
                explanation.feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            summary["top_features"] = [
                f"{name}: {importance:.3f}" 
                for name, importance in top_features
            ]
        
        # Top causal factors
        if explanation.causal_factors:
            top_causal = sorted(
                explanation.causal_factors.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            summary["top_causal_factors"] = [
                f"{name}: {factor:.3f}"
                for name, factor in top_causal
            ]
        
        # Critical time windows
        if explanation.temporal_attribution is not None:
            critical_windows = self.saliency_mapper.identify_critical_time_windows(
                explanation.temporal_attribution
            )
            summary["critical_time_windows"] = len(critical_windows)
        
        # Uncertainty assessment
        if explanation.uncertainty_sources:
            summary["uncertainty_sources"] = explanation.uncertainty_sources
            summary["uncertainty_level"] = len(explanation.uncertainty_sources)
        
        return summary
    
    def get_global_explanation_stats(self) -> Dict[str, Any]:
        """Get statistics across all explanations."""
        if not self.explanation_history:
            return {"status": "no_explanations"}
            
        explanations = list(self.explanation_history)
        
        return {
            "total_explanations": len(explanations),
            "avg_confidence": np.mean([e.confidence for e in explanations]),
            "avg_quality": np.mean([e.explanation_quality for e in explanations]),
            "common_uncertainty_sources": self._get_common_uncertainty_sources(explanations),
            "feature_importance_trends": self._get_feature_trends(explanations),
            "explanation_quality_trend": self._get_quality_trend(explanations)
        }
    
    def _get_common_uncertainty_sources(self, explanations: List[NeuralExplanation]) -> Dict[str, float]:
        """Get most common uncertainty sources."""
        source_counts = defaultdict(int)
        total_explanations = len(explanations)
        
        for explanation in explanations:
            if explanation.uncertainty_sources:
                for source in explanation.uncertainty_sources:
                    source_counts[source] += 1
        
        return {
            source: count / total_explanations
            for source, count in source_counts.items()
        }
    
    def _get_feature_trends(self, explanations: List[NeuralExplanation]) -> Dict[str, float]:
        """Get feature importance trends."""
        feature_importances = defaultdict(list)
        
        for explanation in explanations:
            if explanation.feature_importance:
                for feature, importance in explanation.feature_importance.items():
                    feature_importances[feature].append(importance)
        
        # Calculate trends (simple slope)
        trends = {}
        for feature, values in feature_importances.items():
            if len(values) > 1:
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)
                trends[feature] = coeffs[0]  # Slope
        
        return trends
    
    def _get_quality_trend(self, explanations: List[NeuralExplanation]) -> float:
        """Get explanation quality trend."""
        qualities = [e.explanation_quality for e in explanations]
        
        if len(qualities) < 2:
            return 0.0
            
        x = np.arange(len(qualities))
        coeffs = np.polyfit(x, qualities, 1)
        return coeffs[0]  # Slope indicates trend


def create_explainable_neural_system(config: Optional[Dict[str, Any]] = None) -> ExplainableNeuralAI:
    """
    Factory function to create an explainable neural AI system.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured ExplainableNeuralAI instance
    """
    if config is None:
        config = {}
        
    explanation_config = ExplanationConfig(
        explanation_methods=config.get('explanation_methods', ["saliency", "shap", "attention"]),
        temporal_resolution=config.get('temporal_resolution', 0.1),
        spatial_resolution=config.get('spatial_resolution', 8),
        confidence_threshold=config.get('confidence_threshold', 0.6),
        feature_importance_threshold=config.get('feature_importance_threshold', 0.1),
        interactive_mode=config.get('interactive_mode', True),
        save_explanations=config.get('save_explanations', True)
    )
    
    explainable_system = ExplainableNeuralAI(explanation_config)
    
    logger.info("Explainable neural AI system created with comprehensive analysis capabilities")
    return explainable_system


# Export key classes and functions
__all__ = [
    'ExplainableNeuralAI',
    'NeuralSaliencyMapper',
    'FeatureImportanceAnalyzer',
    'CausalInferenceEngine',
    'NeuralExplanation',
    'ExplanationConfig',
    'create_explainable_neural_system'
]