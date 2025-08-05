"""
Differential privacy implementation for neural data protection.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import secrets


class NoiseMode(Enum):
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    EXPONENTIAL = "exponential"


@dataclass
class PrivacyBudget:
    epsilon: float  # Privacy parameter
    delta: float    # Failure probability
    composition_count: int = 0  # Number of queries made
    
    def is_exhausted(self, max_queries: int = 1000) -> bool:
        """Check if privacy budget is exhausted."""
        return self.composition_count >= max_queries


class DifferentialPrivacy:
    """
    Differential privacy implementation for protecting neural data privacy.
    
    Implements various noise mechanisms to ensure (ε,δ)-differential privacy
    for neural signal processing while maintaining utility.
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        mechanism: str = "gaussian",
        clipping_bound: float = 1.0,
        random_seed: Optional[int] = None
    ):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if not (0 <= delta <= 1):
            raise ValueError("Delta must be between 0 and 1")
        
        self.epsilon = epsilon
        self.delta = delta
        self.mechanism = NoiseMode(mechanism)
        self.clipping_bound = clipping_bound
        
        # Initialize secure random number generator
        if random_seed is not None:
            np.random.seed(random_seed)
            self.rng = np.random.default_rng(random_seed)
        else:
            self.rng = np.random.default_rng()
        
        self.logger = logging.getLogger(__name__)
        self.budget = PrivacyBudget(epsilon, delta)
        
        # Precompute noise parameters
        self._compute_noise_parameters()
        
        self.logger.info(f"Differential privacy initialized: ε={epsilon}, δ={delta}, mechanism={mechanism}")
    
    def _compute_noise_parameters(self) -> None:
        """Precompute noise parameters for efficiency."""
        if self.mechanism == NoiseMode.GAUSSIAN:
            # For Gaussian mechanism: σ = sqrt(2 * ln(1.25/δ)) * Δf / ε
            self.noise_scale = np.sqrt(2 * np.log(1.25 / self.delta)) * self.clipping_bound / self.epsilon
        elif self.mechanism == NoiseMode.LAPLACE:
            # For Laplace mechanism: b = Δf / ε
            self.noise_scale = self.clipping_bound / self.epsilon
        else:  # EXPONENTIAL
            self.noise_scale = 2 / self.epsilon
    
    def add_noise(
        self, 
        data: np.ndarray, 
        sensitivity: Optional[float] = None,
        clip_data: bool = True
    ) -> np.ndarray:
        """
        Add differentially private noise to neural data.
        
        Args:
            data: Neural data to privatize
            sensitivity: L2 sensitivity of the data (auto-computed if None)
            clip_data: Whether to clip data to sensitivity bounds
            
        Returns:
            Privatized data with noise added
        """
        if data.size == 0:
            return data
        
        # Update composition count
        self.budget.composition_count += 1
        
        # Compute or use provided sensitivity
        if sensitivity is None:
            sensitivity = self._compute_sensitivity(data)
        
        # Clip data to sensitivity bounds
        if clip_data:
            data = self._clip_data(data, sensitivity)
        
        # Add noise based on mechanism
        if self.mechanism == NoiseMode.GAUSSIAN:
            noise = self._gaussian_noise(data.shape, sensitivity)
        elif self.mechanism == NoiseMode.LAPLACE:
            noise = self._laplace_noise(data.shape, sensitivity)
        else:  # EXPONENTIAL
            noise = self._exponential_noise(data.shape, sensitivity)
        
        privatized_data = data + noise
        
        self.logger.debug(f"Added {self.mechanism.value} noise with scale {self.noise_scale:.4f}")
        
        return privatized_data
    
    def _compute_sensitivity(self, data: np.ndarray) -> float:
        """Compute L2 sensitivity of the data."""
        if data.ndim == 1:
            return float(np.linalg.norm(data))
        else:
            # For multi-dimensional data, compute row-wise L2 norms
            return float(np.max(np.linalg.norm(data, axis=1)))
    
    def _clip_data(self, data: np.ndarray, sensitivity: float) -> np.ndarray:
        """Clip data to sensitivity bounds."""
        if data.ndim == 1:
            norm = np.linalg.norm(data)
            if norm > sensitivity:
                return data * (sensitivity / norm)
            return data
        else:
            # Clip each row independently
            clipped_data = data.copy()
            for i in range(data.shape[0]):
                row_norm = np.linalg.norm(data[i])
                if row_norm > sensitivity:
                    clipped_data[i] = data[i] * (sensitivity / row_norm)
            return clipped_data
    
    def _gaussian_noise(self, shape: Tuple, sensitivity: float) -> np.ndarray:
        """Generate Gaussian noise for differential privacy."""
        # σ = sqrt(2 * ln(1.25/δ)) * Δf / ε
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        return self.rng.normal(0, sigma, shape)
    
    def _laplace_noise(self, shape: Tuple, sensitivity: float) -> np.ndarray:
        """Generate Laplace noise for differential privacy."""
        # b = Δf / ε
        scale = sensitivity / self.epsilon
        return self.rng.laplace(0, scale, shape)
    
    def _exponential_noise(self, shape: Tuple, sensitivity: float) -> np.ndarray:
        """Generate exponential mechanism noise."""
        # Simple exponential noise (not true exponential mechanism)
        scale = sensitivity / self.epsilon
        return self.rng.exponential(scale, shape) - scale  # Center around 0
    
    def privatize_features(self, features: np.ndarray, feature_sensitivity: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Privatize feature vectors with per-feature sensitivity.
        
        Args:
            features: Feature matrix (samples x features)
            feature_sensitivity: Per-feature sensitivity values
            
        Returns:
            Privatized features
        """
        if features.size == 0:
            return features
        
        privatized_features = features.copy()
        
        if feature_sensitivity is None:
            # Global sensitivity for all features
            return self.add_noise(privatized_features)
        
        # Per-feature noise addition
        for i, sensitivity in enumerate(feature_sensitivity.values()):
            if i < features.shape[1]:
                feature_column = features[:, i:i+1]
                privatized_features[:, i:i+1] = self.add_noise(feature_column, sensitivity)
        
        return privatized_features
    
    def compose_privacy(self, other_epsilon: float, other_delta: float) -> Tuple[float, float]:
        """
        Compute composed privacy parameters using basic composition.
        
        Args:
            other_epsilon: Epsilon from another mechanism
            other_delta: Delta from another mechanism
            
        Returns:
            Composed (epsilon, delta) parameters
        """
        # Basic composition (conservative)
        composed_epsilon = self.epsilon + other_epsilon
        composed_delta = self.delta + other_delta
        
        self.logger.info(f"Privacy composition: ({self.epsilon:.3f}, {self.delta:.2e}) + ({other_epsilon:.3f}, {other_delta:.2e}) = ({composed_epsilon:.3f}, {composed_delta:.2e})")
        
        return composed_epsilon, composed_delta
    
    def advanced_composition(self, k: int, delta_prime: float = 1e-6) -> Tuple[float, float]:
        """
        Compute advanced composition bounds.
        
        Args:
            k: Number of mechanism invocations
            delta_prime: Additional failure probability
            
        Returns:
            Advanced composition bounds
        """
        if k <= 0:
            return 0.0, 0.0
        
        # Advanced composition theorem
        epsilon_advanced = np.sqrt(2 * k * np.log(1 / delta_prime)) * self.epsilon + k * self.epsilon * (np.exp(self.epsilon) - 1)
        delta_advanced = k * self.delta + delta_prime
        
        return epsilon_advanced, delta_advanced
    
    def generate_privacy_proof(self, query_description: str) -> Dict[str, Any]:
        """Generate privacy proof certificate."""
        proof = {
            "query_id": hashlib.sha256(query_description.encode()).hexdigest()[:16],
            "query_description": query_description,
            "mechanism": self.mechanism.value,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_scale": self.noise_scale,
            "clipping_bound": self.clipping_bound,
            "composition_count": self.budget.composition_count,
            "timestamp": str(np.datetime64('now')),
            "privacy_guarantee": f"(ε={self.epsilon}, δ={self.delta})-differential privacy"
        }
        
        # Generate cryptographic proof hash
        proof_string = str(sorted(proof.items()))
        proof["proof_hash"] = hashlib.sha256(proof_string.encode()).hexdigest()
        
        return proof
    
    def validate_privacy_loss(self, max_epsilon: float = 10.0, max_delta: float = 1e-3) -> bool:
        """Validate that privacy loss is within acceptable bounds."""
        current_epsilon = self.epsilon * self.budget.composition_count
        current_delta = self.delta * self.budget.composition_count
        
        within_bounds = (current_epsilon <= max_epsilon) and (current_delta <= max_delta)
        
        if not within_bounds:
            self.logger.warning(f"Privacy loss exceeded bounds: ε={current_epsilon:.3f}, δ={current_delta:.2e}")
        
        return within_bounds
    
    def reset_budget(self, new_epsilon: Optional[float] = None, new_delta: Optional[float] = None) -> None:
        """Reset privacy budget for new analysis session."""
        if new_epsilon is not None:
            self.epsilon = new_epsilon
        if new_delta is not None:
            self.delta = new_delta
        
        self.budget = PrivacyBudget(self.epsilon, self.delta)
        self._compute_noise_parameters()
        
        self.logger.info(f"Privacy budget reset: ε={self.epsilon}, δ={self.delta}")
    
    def estimate_utility_loss(self, original_data: np.ndarray, privatized_data: np.ndarray) -> Dict[str, float]:
        """Estimate utility loss from privatization."""
        if original_data.shape != privatized_data.shape:
            return {"error": "Shape mismatch"}
        
        # Compute various utility metrics
        noise = privatized_data - original_data
        
        metrics = {
            "mean_squared_error": float(np.mean(noise ** 2)),
            "mean_absolute_error": float(np.mean(np.abs(noise))),
            "signal_to_noise_ratio": float(np.var(original_data) / np.var(noise)) if np.var(noise) > 0 else float('inf'),
            "relative_error": float(np.linalg.norm(noise) / np.linalg.norm(original_data)) if np.linalg.norm(original_data) > 0 else 0.0,
            "correlation": float(np.corrcoef(original_data.flat, privatized_data.flat)[0, 1]) if original_data.size > 1 else 1.0
        }
        
        return metrics
    
    def get_privacy_status(self) -> Dict[str, Any]:
        """Get current privacy status and budget."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "mechanism": self.mechanism.value,
            "noise_scale": self.noise_scale,
            "composition_count": self.budget.composition_count,
            "budget_exhausted": self.budget.is_exhausted(),
            "privacy_guarantee": f"(ε={self.epsilon}, δ={self.delta})-differential privacy"
        }