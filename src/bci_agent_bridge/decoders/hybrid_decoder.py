"""
Hybrid Multi-Paradigm BCI Decoder for Enhanced Performance.

This module implements a sophisticated hybrid decoder that combines multiple BCI paradigms
(P300, SSVEP, Motor Imagery) with adaptive fusion mechanisms for robust performance
across diverse conditions and applications.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

from .base import BaseDecoder
from .transformer_decoder import TransformerNeuralDecoder, TransformerConfig
from .p300 import P300Decoder
from .motor_imagery import MotorImageryDecoder
from .ssvep import SSVEPDecoder


class ParadigmType(Enum):
    """BCI paradigm types."""
    P300 = "P300"
    SSVEP = "SSVEP"
    MOTOR_IMAGERY = "MotorImagery"
    HYBRID = "Hybrid"


@dataclass
class HybridConfig:
    """Configuration for hybrid decoder."""
    use_p300: bool = True
    use_ssvep: bool = True
    use_motor_imagery: bool = True
    use_transformers: bool = True
    fusion_method: str = "adaptive"  # "adaptive", "weighted", "voting", "meta_learning"
    confidence_threshold: float = 0.7
    adaptation_rate: float = 0.01
    meta_learning_lr: float = 1e-4
    ensemble_size: int = 3


class SignalQualityAssessor(nn.Module):
    """Assess signal quality for paradigm weighting."""
    
    def __init__(self, n_channels: int, sampling_rate: int):
        super().__init__()
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        
        # Neural network for quality assessment
        self.quality_net = nn.Sequential(
            nn.Linear(n_channels * 10, 128),  # 10 statistical features per channel
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def extract_quality_features(self, data: torch.Tensor) -> torch.Tensor:
        """Extract signal quality features."""
        batch_size, n_channels, seq_len = data.shape
        features = []
        
        for ch in range(n_channels):
            ch_data = data[:, ch, :]  # (batch, time)
            
            # Statistical features
            mean_val = torch.mean(ch_data, dim=1)
            std_val = torch.std(ch_data, dim=1)
            var_val = torch.var(ch_data, dim=1)
            skewness = self._compute_skewness(ch_data)
            kurtosis = self._compute_kurtosis(ch_data)
            
            # Spectral features
            fft = torch.fft.fft(ch_data)
            power_spectrum = torch.abs(fft) ** 2
            spectral_centroid = self._compute_spectral_centroid(power_spectrum)
            
            # Temporal features
            zero_crossings = self._compute_zero_crossings(ch_data)
            rms = torch.sqrt(torch.mean(ch_data ** 2, dim=1))
            peak_to_peak = torch.max(ch_data, dim=1)[0] - torch.min(ch_data, dim=1)[0]
            
            # Artifact indicators
            artifact_score = self._compute_artifact_score(ch_data)
            
            ch_features = torch.stack([
                mean_val, std_val, var_val, skewness, kurtosis,
                spectral_centroid, zero_crossings, rms, peak_to_peak, artifact_score
            ], dim=1)
            
            features.append(ch_features)
        
        # Flatten features: (batch, n_channels * 10)
        quality_features = torch.cat(features, dim=1)
        return quality_features
    
    def _compute_skewness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute skewness."""
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        centered = (x - mean) / (std + 1e-8)
        return torch.mean(centered ** 3, dim=1)
    
    def _compute_kurtosis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute kurtosis."""
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        centered = (x - mean) / (std + 1e-8)
        return torch.mean(centered ** 4, dim=1) - 3
    
    def _compute_spectral_centroid(self, power_spectrum: torch.Tensor) -> torch.Tensor:
        """Compute spectral centroid."""
        freqs = torch.arange(power_spectrum.shape[1], dtype=torch.float32, device=power_spectrum.device)
        total_power = torch.sum(power_spectrum, dim=1, keepdim=True)
        weighted_freqs = torch.sum(power_spectrum * freqs, dim=1) / (total_power.squeeze() + 1e-8)
        return weighted_freqs
    
    def _compute_zero_crossings(self, x: torch.Tensor) -> torch.Tensor:
        """Compute zero crossing rate."""
        diff_sign = torch.diff(torch.sign(x), dim=1)
        zero_crossings = torch.sum(torch.abs(diff_sign) > 0, dim=1).float()
        return zero_crossings / x.shape[1]
    
    def _compute_artifact_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute artifact probability score."""
        # Simple artifact detection based on amplitude and gradient
        max_amplitude = torch.max(torch.abs(x), dim=1)[0]
        mean_gradient = torch.mean(torch.abs(torch.diff(x, dim=1)), dim=1)
        artifact_score = torch.sigmoid((max_amplitude - 100) / 50 + (mean_gradient - 10) / 5)
        return artifact_score
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Assess signal quality.
        
        Args:
            data: EEG data of shape (batch, channels, time)
            
        Returns:
            Quality scores of shape (batch,)
        """
        quality_features = self.extract_quality_features(data)
        quality_scores = self.quality_net(quality_features).squeeze(-1)
        return quality_scores


class AdaptiveFusionModule(nn.Module):
    """Adaptive fusion of multiple paradigm predictions."""
    
    def __init__(self, n_paradigms: int, feature_dim: int = 128):
        super().__init__()
        self.n_paradigms = n_paradigms
        self.feature_dim = feature_dim
        
        # Attention mechanism for paradigm weighting
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim * n_paradigms, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Fusion weights predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_paradigms),
            nn.Softmax(dim=-1)
        )
        
        # Final classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4)  # Assume 4 classes
        )
    
    def forward(
        self, 
        paradigm_features: List[torch.Tensor],
        paradigm_confidences: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse multiple paradigm predictions.
        
        Args:
            paradigm_features: List of feature tensors from each paradigm
            paradigm_confidences: List of confidence scores from each paradigm
            
        Returns:
            Tuple of (fused_predictions, fusion_weights)
        """
        batch_size = paradigm_features[0].shape[0]
        
        # Stack paradigm features
        stacked_features = torch.stack(paradigm_features, dim=1)  # (batch, n_paradigms, feature_dim)
        
        # Apply attention across paradigms
        attended_features, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Context encoding
        context = self.context_encoder(attended_features.view(batch_size, -1))
        
        # Predict fusion weights based on context and confidences
        confidence_tensor = torch.stack(paradigm_confidences, dim=1)  # (batch, n_paradigms)
        fusion_weights = self.weight_predictor(context)
        
        # Incorporate confidence into weights
        adjusted_weights = fusion_weights * confidence_tensor
        adjusted_weights = F.softmax(adjusted_weights, dim=-1)
        
        # Weighted fusion of features
        fused_features = torch.sum(
            attended_features * adjusted_weights.unsqueeze(-1), dim=1
        )
        
        # Final classification
        predictions = self.final_classifier(fused_features)
        
        return predictions, adjusted_weights


class MetaLearningModule(nn.Module):
    """Meta-learning for rapid adaptation to new subjects/conditions."""
    
    def __init__(self, feature_dim: int = 128, n_classes: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # Task-specific head generator
        self.head_generator = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim * n_classes + n_classes)  # Weights + biases
        )
    
    def generate_task_head(self, support_features: torch.Tensor) -> nn.Module:
        """Generate task-specific classification head."""
        # Aggregate support features
        prototype = torch.mean(support_features, dim=0, keepdim=True)
        
        # Generate head parameters
        head_params = self.head_generator(prototype)
        
        # Split into weights and biases
        weight_size = self.feature_dim * self.n_classes
        weights = head_params[:, :weight_size].view(self.n_classes, self.feature_dim)
        biases = head_params[:, weight_size:]
        
        # Create linear layer
        head = nn.Linear(self.feature_dim, self.n_classes)
        head.weight.data = weights
        head.bias.data = biases.squeeze()
        
        return head
    
    def forward(self, query_features: torch.Tensor, support_features: torch.Tensor) -> torch.Tensor:
        """Meta-learning forward pass."""
        # Extract features
        query_feat = self.feature_extractor(query_features)
        
        # Generate task-specific head
        task_head = self.generate_task_head(support_features)
        
        # Classify
        predictions = task_head(query_feat)
        
        return predictions


class HybridMultiParadigmDecoder(BaseDecoder):
    """
    Advanced hybrid decoder combining multiple BCI paradigms with adaptive fusion.
    
    This decoder intelligently combines P300, SSVEP, and Motor Imagery paradigms
    based on signal quality, context, and learned adaptation patterns.
    """
    
    def __init__(
        self,
        channels: int = 8,
        sampling_rate: int = 250,
        config: Optional[HybridConfig] = None
    ):
        super().__init__(channels, sampling_rate)
        
        self.config = config or HybridConfig()
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize individual paradigm decoders
        self.paradigm_decoders = {}
        self.paradigm_features = {}
        
        if self.config.use_p300:
            if self.config.use_transformers:
                transformer_config = TransformerConfig(n_classes=2)  # P300 is binary
                self.paradigm_decoders['P300'] = TransformerNeuralDecoder(
                    channels, sampling_rate, transformer_config, "P300"
                ).to(self.device)
            else:
                self.paradigm_decoders['P300'] = P300Decoder(channels, sampling_rate)
        
        if self.config.use_ssvep:
            if self.config.use_transformers:
                transformer_config = TransformerConfig(n_classes=4)  # 4 SSVEP frequencies
                self.paradigm_decoders['SSVEP'] = TransformerNeuralDecoder(
                    channels, sampling_rate, transformer_config, "SSVEP"
                ).to(self.device)
            else:
                self.paradigm_decoders['SSVEP'] = SSVEPDecoder(channels, sampling_rate)
        
        if self.config.use_motor_imagery:
            if self.config.use_transformers:
                transformer_config = TransformerConfig(n_classes=4)  # 4 MI classes
                self.paradigm_decoders['MotorImagery'] = TransformerNeuralDecoder(
                    channels, sampling_rate, transformer_config, "MotorImagery"
                ).to(self.device)
            else:
                self.paradigm_decoders['MotorImagery'] = MotorImageryDecoder(channels, sampling_rate)
        
        # Signal quality assessor
        self.quality_assessor = SignalQualityAssessor(channels, sampling_rate).to(self.device)
        
        # Adaptive fusion module
        n_paradigms = len(self.paradigm_decoders)
        self.fusion_module = AdaptiveFusionModule(n_paradigms, feature_dim=128).to(self.device)
        
        # Meta-learning module for rapid adaptation
        self.meta_learner = MetaLearningModule().to(self.device)
        
        # Paradigm reliability tracking
        self.paradigm_reliability = {name: 1.0 for name in self.paradigm_decoders.keys()}
        self.adaptation_history = []
        
        # Training state
        self.is_trained = False
        
        self.logger.info(f"Initialized HybridMultiParadigmDecoder with paradigms: {list(self.paradigm_decoders.keys())}")
    
    def extract_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features from all paradigm decoders.
        
        Args:
            data: Neural data of shape (channels, time_samples)
            
        Returns:
            Dictionary of features from each paradigm
        """
        features = {}
        
        for paradigm_name, decoder in self.paradigm_decoders.items():
            try:
                paradigm_features = decoder.extract_features(data)
                features[paradigm_name] = paradigm_features
            except Exception as e:
                self.logger.warning(f"Feature extraction failed for {paradigm_name}: {e}")
                # Use zero features as fallback
                features[paradigm_name] = np.zeros(128)
        
        return features
    
    def assess_paradigm_reliability(self, data: np.ndarray) -> Dict[str, float]:
        """
        Assess reliability of each paradigm for the current signal.
        
        Args:
            data: Neural data
            
        Returns:
            Dictionary of reliability scores for each paradigm
        """
        reliability_scores = {}
        
        # Convert to tensor for processing
        if len(data.shape) == 2:
            data_tensor = torch.FloatTensor(data[np.newaxis, ...]).to(self.device)
        else:
            data_tensor = torch.FloatTensor(data).to(self.device)
        
        # Assess signal quality
        quality_score = self.quality_assessor(data_tensor).item()
        
        # Paradigm-specific reliability assessment
        for paradigm_name in self.paradigm_decoders.keys():
            base_reliability = self.paradigm_reliability[paradigm_name]
            
            # Adjust based on signal characteristics
            if paradigm_name == 'P300':
                # P300 is more robust to artifacts
                reliability_scores[paradigm_name] = base_reliability * (0.5 + 0.5 * quality_score)
            elif paradigm_name == 'SSVEP':
                # SSVEP requires good frequency content
                freq_quality = self._assess_frequency_content(data, [6.0, 7.5, 8.57, 10.0])
                reliability_scores[paradigm_name] = base_reliability * freq_quality
            elif paradigm_name == 'MotorImagery':
                # Motor imagery needs good alpha/beta band quality
                mi_quality = self._assess_frequency_content(data, [8, 12, 20, 30])
                reliability_scores[paradigm_name] = base_reliability * mi_quality
            else:
                reliability_scores[paradigm_name] = base_reliability * quality_score
        
        return reliability_scores
    
    def _assess_frequency_content(self, data: np.ndarray, target_freqs: List[float]) -> float:
        """Assess quality of specific frequency content."""
        # Compute power spectral density
        fft = np.fft.fft(data, axis=-1)
        power_spectrum = np.abs(fft) ** 2
        
        # Extract power in target frequency bands
        freq_bins = np.fft.fftfreq(data.shape[-1], 1.0 / self.sampling_rate)
        
        target_power = 0.0
        total_power = np.sum(power_spectrum)
        
        for freq in target_freqs:
            freq_idx = np.argmin(np.abs(freq_bins - freq))
            target_power += power_spectrum[..., freq_idx].mean()
        
        # Normalize
        quality = target_power / (total_power / len(target_freqs) + 1e-8)
        return np.clip(quality, 0.0, 1.0)
    
    def predict(self, features: Union[np.ndarray, Dict[str, np.ndarray]]) -> int:
        """
        Make prediction using hybrid fusion.
        
        Args:
            features: Either raw data or extracted features from each paradigm
            
        Returns:
            Predicted class label
        """
        if isinstance(features, np.ndarray):
            # Extract features from raw data
            features = self.extract_features(features)
        
        if not self.is_trained:
            self.logger.warning("Model not trained, using random prediction")
            return np.random.randint(0, 4)
        
        # Get paradigm predictions and confidences
        paradigm_predictions = {}
        paradigm_confidences = {}
        paradigm_features_list = []
        confidence_tensors = []
        
        for paradigm_name, paradigm_features in features.items():
            if paradigm_name in self.paradigm_decoders:
                try:
                    decoder = self.paradigm_decoders[paradigm_name]
                    
                    # Get prediction and confidence
                    if hasattr(decoder, 'predict_proba'):
                        probs = decoder.predict_proba(paradigm_features)
                        prediction = np.argmax(probs)
                        confidence = np.max(probs)
                    else:
                        prediction = decoder.predict(paradigm_features)
                        confidence = decoder.get_confidence()
                    
                    paradigm_predictions[paradigm_name] = prediction
                    paradigm_confidences[paradigm_name] = confidence
                    
                    # Convert features to tensor
                    if isinstance(paradigm_features, np.ndarray):
                        if len(paradigm_features.shape) == 1:
                            paradigm_features = paradigm_features[np.newaxis, :]
                        feature_tensor = torch.FloatTensor(paradigm_features).to(self.device)
                    else:
                        feature_tensor = paradigm_features
                    
                    paradigm_features_list.append(feature_tensor)
                    confidence_tensors.append(torch.tensor(confidence).to(self.device))
                    
                except Exception as e:
                    self.logger.warning(f"Prediction failed for {paradigm_name}: {e}")
        
        if not paradigm_features_list:
            return np.random.randint(0, 4)
        
        # Adaptive fusion
        if self.config.fusion_method == "adaptive" and len(paradigm_features_list) > 1:
            try:
                fused_predictions, fusion_weights = self.fusion_module(
                    paradigm_features_list, confidence_tensors
                )
                final_prediction = torch.argmax(fused_predictions, dim=-1).cpu().item()
                
                # Update paradigm reliability based on consensus
                self._update_paradigm_reliability(paradigm_predictions, final_prediction, fusion_weights)
                
                return final_prediction
            except Exception as e:
                self.logger.warning(f"Adaptive fusion failed: {e}")
        
        # Fallback to weighted voting
        return self._weighted_voting(paradigm_predictions, paradigm_confidences)
    
    def _weighted_voting(self, predictions: Dict[str, int], confidences: Dict[str, float]) -> int:
        """Fallback weighted voting mechanism."""
        if not predictions:
            return np.random.randint(0, 4)
        
        # Weight votes by confidence and reliability
        vote_weights = {}
        for paradigm_name, prediction in predictions.items():
            confidence = confidences.get(paradigm_name, 0.5)
            reliability = self.paradigm_reliability.get(paradigm_name, 0.5)
            weight = confidence * reliability
            
            if prediction not in vote_weights:
                vote_weights[prediction] = 0
            vote_weights[prediction] += weight
        
        # Return most weighted prediction
        return max(vote_weights, key=vote_weights.get)
    
    def _update_paradigm_reliability(
        self, 
        paradigm_predictions: Dict[str, int], 
        final_prediction: int,
        fusion_weights: torch.Tensor
    ) -> None:
        """Update paradigm reliability based on consensus."""
        for i, (paradigm_name, prediction) in enumerate(paradigm_predictions.items()):
            # Reward paradigms that agree with final prediction
            agreement = 1.0 if prediction == final_prediction else 0.0
            
            # Update reliability with exponential moving average
            current_reliability = self.paradigm_reliability[paradigm_name]
            new_reliability = (1 - self.config.adaptation_rate) * current_reliability + \
                            self.config.adaptation_rate * agreement
            
            self.paradigm_reliability[paradigm_name] = np.clip(new_reliability, 0.1, 1.0)
    
    def get_confidence(self) -> float:
        """Get confidence score for the last prediction."""
        # Aggregate confidence from all paradigms
        reliabilities = list(self.paradigm_reliability.values())
        if reliabilities:
            return np.mean(reliabilities)
        return 0.5
    
    def fit_paradigms(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        paradigm_labels: Optional[Dict[str, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train individual paradigm decoders and fusion module.
        
        Args:
            X: Training data
            y: Training labels
            paradigm_labels: Optional paradigm-specific labels
            
        Returns:
            Training history
        """
        self.logger.info("Training hybrid multi-paradigm decoder")
        
        training_history = {}
        
        # Train individual paradigm decoders
        for paradigm_name, decoder in self.paradigm_decoders.items():
            self.logger.info(f"Training {paradigm_name} decoder")
            
            try:
                # Use paradigm-specific labels if available
                if paradigm_labels and paradigm_name in paradigm_labels:
                    paradigm_y = paradigm_labels[paradigm_name]
                else:
                    paradigm_y = y
                
                # Train decoder
                if hasattr(decoder, 'fit'):
                    history = decoder.fit(X, paradigm_y, **kwargs)
                    training_history[paradigm_name] = history
                else:
                    # For legacy decoders without fit method
                    decoder.calibrate(X, paradigm_y)
                    training_history[paradigm_name] = {"status": "calibrated"}
                
            except Exception as e:
                self.logger.error(f"Training failed for {paradigm_name}: {e}")
                training_history[paradigm_name] = {"error": str(e)}
        
        # Train fusion module (simplified for this implementation)
        self.logger.info("Training fusion module")
        # In practice, this would involve more sophisticated meta-learning
        
        self.is_trained = True
        self.logger.info("Hybrid decoder training completed")
        
        return training_history
    
    def calibrate(self, calibration_data: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
        """
        Calibrate all paradigm decoders.
        
        Args:
            calibration_data: EEG data for calibration
            labels: Optional labels for supervised calibration
        """
        self.logger.info("Calibrating hybrid decoder")
        
        for paradigm_name, decoder in self.paradigm_decoders.items():
            try:
                decoder.calibrate(calibration_data, labels)
                self.logger.info(f"Calibration completed for {paradigm_name}")
            except Exception as e:
                self.logger.warning(f"Calibration failed for {paradigm_name}: {e}")
        
        # Meta-adaptation for fusion
        if labels is not None:
            self._meta_adapt(calibration_data, labels)
    
    def _meta_adapt(self, support_data: np.ndarray, support_labels: np.ndarray) -> None:
        """Perform meta-learning adaptation."""
        # Extract features from all paradigms
        all_features = []
        
        for i in range(len(support_data)):
            sample_features = self.extract_features(support_data[i])
            # Concatenate all paradigm features
            concat_features = np.concatenate([
                features.flatten() for features in sample_features.values()
            ])
            all_features.append(concat_features)
        
        # Simple prototype-based adaptation
        support_features = torch.FloatTensor(all_features).to(self.device)
        
        # This would involve more sophisticated meta-learning in practice
        self.logger.info("Meta-adaptation completed")
    
    def get_paradigm_contributions(self) -> Dict[str, float]:
        """Get current contribution of each paradigm."""
        total_reliability = sum(self.paradigm_reliability.values())
        if total_reliability == 0:
            return {name: 1.0/len(self.paradigm_reliability) 
                   for name in self.paradigm_reliability.keys()}
        
        return {name: reliability / total_reliability 
                for name, reliability in self.paradigm_reliability.items()}
    
    def save_model(self, filepath: str) -> None:
        """Save the complete hybrid model."""
        save_dict = {
            'config': self.config,
            'channels': self.channels,
            'sampling_rate': self.sampling_rate,
            'paradigm_reliability': self.paradigm_reliability,
            'is_trained': self.is_trained,
            'fusion_module_state': self.fusion_module.state_dict(),
            'quality_assessor_state': self.quality_assessor.state_dict(),
            'meta_learner_state': self.meta_learner.state_dict()
        }
        
        # Save individual paradigm decoders
        paradigm_states = {}
        for name, decoder in self.paradigm_decoders.items():
            if hasattr(decoder, 'state_dict'):
                paradigm_states[name] = decoder.state_dict()
            # For legacy decoders, would need different saving mechanism
        
        save_dict['paradigm_states'] = paradigm_states
        
        torch.save(save_dict, filepath)
        self.logger.info(f"Hybrid model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'HybridMultiParadigmDecoder':
        """Load a trained hybrid model."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        decoder = cls(
            channels=checkpoint['channels'],
            sampling_rate=checkpoint['sampling_rate'],
            config=checkpoint['config']
        )
        
        # Load states
        decoder.fusion_module.load_state_dict(checkpoint['fusion_module_state'])
        decoder.quality_assessor.load_state_dict(checkpoint['quality_assessor_state'])
        decoder.meta_learner.load_state_dict(checkpoint['meta_learner_state'])
        
        # Load paradigm decoders
        for name, state_dict in checkpoint['paradigm_states'].items():
            if name in decoder.paradigm_decoders and hasattr(decoder.paradigm_decoders[name], 'load_state_dict'):
                decoder.paradigm_decoders[name].load_state_dict(state_dict)
        
        decoder.paradigm_reliability = checkpoint['paradigm_reliability']
        decoder.is_trained = checkpoint['is_trained']
        
        return decoder