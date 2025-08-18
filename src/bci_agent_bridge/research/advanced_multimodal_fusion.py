"""
Advanced Multimodal Fusion for Hybrid BCI Paradigms.

This module implements state-of-the-art multimodal fusion techniques that combine
multiple BCI paradigms (P300, SSVEP, Motor Imagery, etc.) for superior performance
and robustness.

Key innovations:
- Attention-based fusion mechanisms
- Cross-modal consistency learning
- Dynamic paradigm weighting
- Uncertainty-aware ensemble methods
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
from collections import defaultdict
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Configuration for multimodal fusion."""
    fusion_strategy: str = "attention"  # "attention", "weighted", "voting", "cascade"
    attention_heads: int = 4
    temporal_window: int = 500  # ms
    confidence_threshold: float = 0.7
    min_paradigms: int = 2
    max_paradigms: int = 4
    adaptation_rate: float = 0.05
    uncertainty_estimation: bool = True
    cross_modal_validation: bool = True


@dataclass
class ModalityData:
    """Data structure for individual modality information."""
    paradigm: str
    features: np.ndarray
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Result of multimodal fusion."""
    prediction: Any
    confidence: float
    paradigm_weights: Dict[str, float]
    uncertainty_score: float
    fusion_quality: float
    contributing_paradigms: List[str]
    timestamp: float = field(default_factory=time.time)


class AttentionFusionMechanism:
    """Advanced attention-based fusion for multimodal BCI data."""
    
    def __init__(self, n_heads: int = 4, feature_dim: int = 64):
        self.n_heads = n_heads
        self.feature_dim = feature_dim
        self.attention_weights = {}
        self.learned_embeddings = {}
        self.adaptation_history = []
        
    def initialize_attention(self, paradigms: List[str]):
        """Initialize attention mechanisms for each paradigm."""
        for paradigm in paradigms:
            # Initialize random attention weights
            self.attention_weights[paradigm] = np.random.normal(
                0, 0.1, (self.n_heads, self.feature_dim)
            )
            self.learned_embeddings[paradigm] = np.random.normal(
                0, 0.1, self.feature_dim
            )
            
        logger.info(f"Initialized attention mechanisms for {len(paradigms)} paradigms")
    
    def compute_attention(self, modality_data: List[ModalityData]) -> Dict[str, float]:
        """Compute attention weights for each modality."""
        if not modality_data:
            return {}
            
        attention_scores = {}
        
        for data in modality_data:
            paradigm = data.paradigm
            
            if paradigm not in self.attention_weights:
                # Initialize if new paradigm
                self.attention_weights[paradigm] = np.random.normal(
                    0, 0.1, (self.n_heads, self.feature_dim)
                )
                self.learned_embeddings[paradigm] = np.random.normal(
                    0, 0.1, self.feature_dim
                )
            
            # Compute multi-head attention
            features = self._normalize_features(data.features)
            
            head_scores = []
            for head in range(self.n_heads):
                # Simplified attention computation
                attention_vec = self.attention_weights[paradigm][head]
                score = np.dot(features, attention_vec) * data.confidence
                head_scores.append(score)
            
            # Aggregate attention scores
            attention_scores[paradigm] = np.mean(head_scores)
        
        # Normalize attention scores
        total_score = sum(attention_scores.values())
        if total_score > 0:
            attention_scores = {k: v/total_score for k, v in attention_scores.items()}
        
        return attention_scores
    
    def update_attention(self, modality_data: List[ModalityData], 
                        performance_feedback: float):
        """Update attention weights based on performance feedback."""
        if not modality_data or not (-1 <= performance_feedback <= 1):
            return
            
        learning_rate = 0.01
        
        for data in modality_data:
            paradigm = data.paradigm
            
            if paradigm in self.attention_weights:
                # Update based on feedback
                features = self._normalize_features(data.features)
                
                for head in range(self.n_heads):
                    gradient = performance_feedback * features * data.confidence
                    self.attention_weights[paradigm][head] += learning_rate * gradient
                    
                # Update embedding
                embedding_gradient = performance_feedback * features
                self.learned_embeddings[paradigm] += learning_rate * embedding_gradient
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features for attention computation."""
        if features.size == 0:
            return np.zeros(self.feature_dim)
            
        # Pad or truncate to feature_dim
        if len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        elif len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
            
        # Normalize
        norm = np.linalg.norm(features)
        return features / (norm + 1e-10)


class CrossModalValidator:
    """Cross-modal consistency validation for robust fusion."""
    
    def __init__(self, consistency_threshold: float = 0.3):
        self.consistency_threshold = consistency_threshold
        self.paradigm_correlations = defaultdict(dict)
        self.validation_history = []
        
    def validate_consistency(self, modality_data: List[ModalityData]) -> Tuple[bool, float]:
        """Validate cross-modal consistency."""
        if len(modality_data) < 2:
            return True, 1.0  # Single modality is always consistent
            
        consistency_scores = []
        
        # Compare all pairs of modalities
        for i in range(len(modality_data)):
            for j in range(i + 1, len(modality_data)):
                data1, data2 = modality_data[i], modality_data[j]
                
                # Calculate feature similarity
                similarity = self._calculate_similarity(data1.features, data2.features)
                
                # Weight by confidence
                weighted_similarity = similarity * min(data1.confidence, data2.confidence)
                consistency_scores.append(weighted_similarity)
                
                # Update correlation history
                pair_key = f"{data1.paradigm}-{data2.paradigm}"
                if pair_key not in self.paradigm_correlations:
                    self.paradigm_correlations[pair_key] = []
                self.paradigm_correlations[pair_key].append(similarity)
                
                # Keep only recent correlations
                if len(self.paradigm_correlations[pair_key]) > 100:
                    self.paradigm_correlations[pair_key] = self.paradigm_correlations[pair_key][-100:]
        
        # Calculate overall consistency
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        is_consistent = avg_consistency >= self.consistency_threshold
        
        self.validation_history.append({
            'timestamp': time.time(),
            'consistency_score': avg_consistency,
            'is_consistent': is_consistent,
            'n_modalities': len(modality_data)
        })
        
        return is_consistent, avg_consistency
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between feature vectors."""
        if features1.size == 0 or features2.size == 0:
            return 0.0
            
        # Normalize features
        f1 = features1 / (np.linalg.norm(features1) + 1e-10)
        f2 = features2 / (np.linalg.norm(features2) + 1e-10)
        
        # Handle different lengths
        min_len = min(len(f1), len(f2))
        f1_truncated = f1[:min_len]
        f2_truncated = f2[:min_len]
        
        # Calculate cosine similarity
        similarity = np.dot(f1_truncated, f2_truncated)
        return max(0.0, similarity)  # Ensure non-negative


class UncertaintyEstimator:
    """Uncertainty estimation for fusion decisions."""
    
    def __init__(self):
        self.prediction_history = []
        self.uncertainty_calibration = {}
        
    def estimate_uncertainty(self, modality_data: List[ModalityData], 
                           prediction: Any) -> float:
        """Estimate prediction uncertainty."""
        if not modality_data:
            return 1.0  # Maximum uncertainty for no data
            
        uncertainty_components = []
        
        # Confidence variance
        confidences = [data.confidence for data in modality_data]
        confidence_var = np.var(confidences)
        uncertainty_components.append(confidence_var)
        
        # Prediction entropy (for classification)
        if hasattr(prediction, '__len__') and len(prediction) > 1:
            pred_entropy = entropy(np.abs(prediction) + 1e-10)
            uncertainty_components.append(pred_entropy / np.log(len(prediction)))
        
        # Feature diversity
        if len(modality_data) > 1:
            feature_diversity = self._calculate_feature_diversity(modality_data)
            uncertainty_components.append(1.0 - feature_diversity)
        
        # Temporal consistency
        temporal_uncertainty = self._temporal_uncertainty(modality_data)
        uncertainty_components.append(temporal_uncertainty)
        
        # Aggregate uncertainties
        total_uncertainty = np.mean(uncertainty_components)
        
        # Store for calibration
        self.prediction_history.append({
            'timestamp': time.time(),
            'uncertainty': total_uncertainty,
            'n_modalities': len(modality_data),
            'avg_confidence': np.mean(confidences)
        })
        
        return min(1.0, max(0.0, total_uncertainty))
    
    def _calculate_feature_diversity(self, modality_data: List[ModalityData]) -> float:
        """Calculate diversity across modality features."""
        if len(modality_data) < 2:
            return 0.0
            
        feature_matrix = []
        for data in modality_data:
            if data.features.size > 0:
                feature_matrix.append(data.features)
        
        if len(feature_matrix) < 2:
            return 0.0
            
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(feature_matrix)):
            for j in range(i + 1, len(feature_matrix)):
                try:
                    min_len = min(len(feature_matrix[i]), len(feature_matrix[j]))
                    f1 = feature_matrix[i][:min_len]
                    f2 = feature_matrix[j][:min_len]
                    
                    if min_len > 1:
                        corr = np.corrcoef(f1, f2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                except:
                    continue
        
        # Diversity is inverse of average correlation
        avg_correlation = np.mean(correlations) if correlations else 0.0
        return 1.0 - avg_correlation
    
    def _temporal_uncertainty(self, modality_data: List[ModalityData]) -> float:
        """Calculate temporal uncertainty based on timing differences."""
        if len(modality_data) < 2:
            return 0.0
            
        timestamps = [data.timestamp for data in modality_data]
        time_variance = np.var(timestamps)
        
        # Normalize by reasonable time window (1 second)
        normalized_variance = min(1.0, time_variance / 1.0)
        return normalized_variance


class AdvancedMultimodalFusion:
    """Advanced multimodal fusion engine for hybrid BCI systems."""
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.attention_mechanism = AttentionFusionMechanism(
            n_heads=config.attention_heads
        )
        self.cross_modal_validator = CrossModalValidator()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.paradigm_weights = {}
        self.fusion_history = []
        self.performance_tracker = []
        
    def initialize_fusion(self, paradigms: List[str]):
        """Initialize fusion for specified paradigms."""
        self.attention_mechanism.initialize_attention(paradigms)
        
        # Initialize equal weights
        n_paradigms = len(paradigms)
        initial_weight = 1.0 / n_paradigms if n_paradigms > 0 else 0.0
        self.paradigm_weights = {p: initial_weight for p in paradigms}
        
        logger.info(f"Multimodal fusion initialized for paradigms: {paradigms}")
    
    def fuse_modalities(self, modality_data: List[ModalityData]) -> FusionResult:
        """Perform advanced multimodal fusion."""
        if not modality_data:
            return FusionResult(
                prediction=None,
                confidence=0.0,
                paradigm_weights={},
                uncertainty_score=1.0,
                fusion_quality=0.0,
                contributing_paradigms=[]
            )
        
        # Filter by minimum confidence and paradigm requirements
        valid_modalities = [
            data for data in modality_data 
            if data.confidence >= self.config.confidence_threshold
        ]
        
        if len(valid_modalities) < self.config.min_paradigms:
            # Use original data if not enough valid modalities
            valid_modalities = modality_data
        
        # Cross-modal validation
        if self.config.cross_modal_validation:
            is_consistent, consistency_score = self.cross_modal_validator.validate_consistency(valid_modalities)
            if not is_consistent:
                logger.warning(f"Cross-modal inconsistency detected: {consistency_score:.3f}")
        else:
            consistency_score = 1.0
        
        # Compute fusion based on strategy
        if self.config.fusion_strategy == "attention":
            result = self._attention_fusion(valid_modalities)
        elif self.config.fusion_strategy == "weighted":
            result = self._weighted_fusion(valid_modalities)
        elif self.config.fusion_strategy == "voting":
            result = self._voting_fusion(valid_modalities)
        else:
            result = self._cascade_fusion(valid_modalities)
        
        # Estimate uncertainty
        if self.config.uncertainty_estimation:
            uncertainty = self.uncertainty_estimator.estimate_uncertainty(
                valid_modalities, result.prediction
            )
            result.uncertainty_score = uncertainty
        
        # Calculate fusion quality
        result.fusion_quality = self._calculate_fusion_quality(
            valid_modalities, consistency_score, result.uncertainty_score
        )
        
        # Store fusion history
        self.fusion_history.append(result)
        if len(self.fusion_history) > 1000:
            self.fusion_history = self.fusion_history[-1000:]
        
        return result
    
    def _attention_fusion(self, modality_data: List[ModalityData]) -> FusionResult:
        """Attention-based fusion mechanism."""
        attention_weights = self.attention_mechanism.compute_attention(modality_data)
        
        # Weighted prediction combination
        weighted_features = []
        total_weight = 0.0
        
        for data in modality_data:
            paradigm = data.paradigm
            weight = attention_weights.get(paradigm, 0.0)
            
            if weight > 0 and data.features.size > 0:
                weighted_features.append(weight * data.features)
                total_weight += weight
        
        if not weighted_features:
            prediction = np.array([0.0])
            confidence = 0.0
        else:
            # Combine weighted features
            min_len = min(len(f) for f in weighted_features)
            combined_features = np.sum([f[:min_len] for f in weighted_features], axis=0)
            
            # Simple prediction (could be enhanced with trained model)
            prediction = combined_features / (total_weight + 1e-10)
            confidence = total_weight / len(modality_data)
        
        return FusionResult(
            prediction=prediction,
            confidence=confidence,
            paradigm_weights=attention_weights,
            uncertainty_score=0.0,  # Will be calculated later
            fusion_quality=0.0,  # Will be calculated later
            contributing_paradigms=[d.paradigm for d in modality_data]
        )
    
    def _weighted_fusion(self, modality_data: List[ModalityData]) -> FusionResult:
        """Confidence-weighted fusion."""
        weights = {}
        weighted_predictions = []
        total_weight = 0.0
        
        for data in modality_data:
            weight = data.confidence
            weights[data.paradigm] = weight
            
            if data.features.size > 0:
                weighted_predictions.append(weight * data.features)
                total_weight += weight
        
        if not weighted_predictions:
            prediction = np.array([0.0])
            confidence = 0.0
        else:
            min_len = min(len(p) for p in weighted_predictions)
            combined = np.sum([p[:min_len] for p in weighted_predictions], axis=0)
            prediction = combined / (total_weight + 1e-10)
            confidence = total_weight / len(modality_data)
        
        return FusionResult(
            prediction=prediction,
            confidence=confidence,
            paradigm_weights=weights,
            uncertainty_score=0.0,
            fusion_quality=0.0,
            contributing_paradigms=[d.paradigm for d in modality_data]
        )
    
    def _voting_fusion(self, modality_data: List[ModalityData]) -> FusionResult:
        """Majority voting fusion."""
        votes = {}
        weights = {}
        
        for data in modality_data:
            # Simple thresholding for vote
            if data.features.size > 0:
                vote = 1 if np.mean(data.features) > 0 else 0
                votes[data.paradigm] = vote
                weights[data.paradigm] = data.confidence
        
        if not votes:
            prediction = np.array([0.0])
            confidence = 0.0
        else:
            # Weighted majority vote
            weighted_sum = sum(vote * weights[paradigm] for paradigm, vote in votes.items())
            total_weight = sum(weights.values())
            
            prediction = np.array([weighted_sum / (total_weight + 1e-10)])
            confidence = total_weight / len(modality_data)
        
        return FusionResult(
            prediction=prediction,
            confidence=confidence,
            paradigm_weights=weights,
            uncertainty_score=0.0,
            fusion_quality=0.0,
            contributing_paradigms=[d.paradigm for d in modality_data]
        )
    
    def _cascade_fusion(self, modality_data: List[ModalityData]) -> FusionResult:
        """Cascade fusion with hierarchical processing."""
        # Sort by confidence
        sorted_data = sorted(modality_data, key=lambda x: x.confidence, reverse=True)
        
        # Start with highest confidence
        if not sorted_data:
            prediction = np.array([0.0])
            confidence = 0.0
            weights = {}
        else:
            prediction = sorted_data[0].features
            confidence = sorted_data[0].confidence
            weights = {sorted_data[0].paradigm: 1.0}
            
            # Progressively incorporate other modalities
            for data in sorted_data[1:]:
                if data.confidence > 0.5:  # Only use confident predictions
                    # Simple averaging for cascade
                    if prediction.size > 0 and data.features.size > 0:
                        min_len = min(len(prediction), len(data.features))
                        prediction = (prediction[:min_len] + data.features[:min_len]) / 2
                        confidence = (confidence + data.confidence) / 2
                        weights[data.paradigm] = data.confidence
        
        return FusionResult(
            prediction=prediction,
            confidence=confidence,
            paradigm_weights=weights,
            uncertainty_score=0.0,
            fusion_quality=0.0,
            contributing_paradigms=[d.paradigm for d in sorted_data]
        )
    
    def _calculate_fusion_quality(self, modality_data: List[ModalityData], 
                                 consistency_score: float, uncertainty_score: float) -> float:
        """Calculate overall fusion quality metric."""
        if not modality_data:
            return 0.0
            
        # Quality components
        avg_confidence = np.mean([data.confidence for data in modality_data])
        modality_coverage = min(1.0, len(modality_data) / self.config.max_paradigms)
        
        # Combine quality metrics
        quality = (
            0.4 * avg_confidence +
            0.3 * consistency_score +
            0.2 * (1.0 - uncertainty_score) +
            0.1 * modality_coverage
        )
        
        return max(0.0, min(1.0, quality))
    
    def update_fusion(self, performance_feedback: float):
        """Update fusion parameters based on performance feedback."""
        if not (-1 <= performance_feedback <= 1):
            return
            
        # Update attention mechanism
        if len(self.fusion_history) > 0:
            last_fusion = self.fusion_history[-1]
            
            # Reconstruct modality data for attention update
            modality_data = []
            for paradigm in last_fusion.contributing_paradigms:
                # Create dummy modality data for update
                dummy_data = ModalityData(
                    paradigm=paradigm,
                    features=np.array([1.0]),  # Placeholder
                    confidence=last_fusion.paradigm_weights.get(paradigm, 0.0),
                    timestamp=last_fusion.timestamp
                )
                modality_data.append(dummy_data)
            
            self.attention_mechanism.update_attention(modality_data, performance_feedback)
        
        # Track performance
        self.performance_tracker.append({
            'timestamp': time.time(),
            'feedback': performance_feedback
        })
        
        if len(self.performance_tracker) > 100:
            self.performance_tracker = self.performance_tracker[-100:]
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fusion statistics."""
        if not self.fusion_history:
            return {"status": "no_fusions"}
            
        recent_fusions = self.fusion_history[-100:]
        
        return {
            "total_fusions": len(self.fusion_history),
            "avg_confidence": np.mean([f.confidence for f in recent_fusions]),
            "avg_fusion_quality": np.mean([f.fusion_quality for f in recent_fusions]),
            "avg_uncertainty": np.mean([f.uncertainty_score for f in recent_fusions]),
            "paradigm_usage": self._get_paradigm_usage_stats(recent_fusions),
            "performance_trend": self._get_performance_trend(),
            "current_weights": self.paradigm_weights.copy()
        }
    
    def _get_paradigm_usage_stats(self, fusions: List[FusionResult]) -> Dict[str, float]:
        """Get paradigm usage statistics."""
        usage_counts = defaultdict(int)
        total_fusions = len(fusions)
        
        for fusion in fusions:
            for paradigm in fusion.contributing_paradigms:
                usage_counts[paradigm] += 1
        
        return {
            paradigm: count / total_fusions 
            for paradigm, count in usage_counts.items()
        }
    
    def _get_performance_trend(self) -> float:
        """Get recent performance trend."""
        if len(self.performance_tracker) < 10:
            return 0.0
            
        recent_performance = [p['feedback'] for p in self.performance_tracker[-10:]]
        
        # Simple linear trend
        x = np.arange(len(recent_performance))
        coeffs = np.polyfit(x, recent_performance, 1)
        return coeffs[0]  # Slope indicates trend


def create_multimodal_fusion_system(config: Optional[Dict[str, Any]] = None) -> AdvancedMultimodalFusion:
    """
    Factory function to create an advanced multimodal fusion system.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured AdvancedMultimodalFusion instance
    """
    if config is None:
        config = {}
        
    fusion_config = FusionConfig(
        fusion_strategy=config.get('fusion_strategy', 'attention'),
        attention_heads=config.get('attention_heads', 4),
        confidence_threshold=config.get('confidence_threshold', 0.7),
        min_paradigms=config.get('min_paradigms', 2),
        uncertainty_estimation=config.get('uncertainty_estimation', True),
        cross_modal_validation=config.get('cross_modal_validation', True)
    )
    
    fusion_system = AdvancedMultimodalFusion(fusion_config)
    
    logger.info("Advanced multimodal fusion system created with attention mechanisms")
    return fusion_system


# Export key classes and functions
__all__ = [
    'AdvancedMultimodalFusion',
    'AttentionFusionMechanism',
    'CrossModalValidator',
    'UncertaintyEstimator',
    'ModalityData',
    'FusionResult',
    'FusionConfig',
    'create_multimodal_fusion_system'
]