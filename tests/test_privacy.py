"""
Tests for privacy-preserving mechanisms.
"""

import pytest
import numpy as np

from bci_agent_bridge.privacy.differential_privacy import (
    DifferentialPrivacy, 
    PrivacyBudget, 
    NoiseMode
)


class TestDifferentialPrivacy:
    """Test suite for DifferentialPrivacy."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    
    def test_initialization(self):
        """Test differential privacy initialization."""
        assert self.dp.epsilon == 1.0
        assert self.dp.delta == 1e-5
        assert self.dp.mechanism == NoiseMode.GAUSSIAN
        assert isinstance(self.dp.budget, PrivacyBudget)
    
    def test_invalid_parameters(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            DifferentialPrivacy(epsilon=-1.0)
        
        with pytest.raises(ValueError):
            DifferentialPrivacy(delta=1.5)
    
    def test_add_noise_gaussian(self):
        """Test Gaussian noise addition."""
        data = np.random.randn(100)
        privatized = self.dp.add_noise(data)
        
        assert privatized.shape == data.shape
        assert not np.array_equal(privatized, data)
        assert self.dp.budget.composition_count > 0
    
    def test_add_noise_laplace(self):
        """Test Laplace noise addition."""
        dp_laplace = DifferentialPrivacy(epsilon=1.0, mechanism="laplace")
        data = np.random.randn(100)
        privatized = dp_laplace.add_noise(data)
        
        assert privatized.shape == data.shape
        assert not np.array_equal(privatized, data)
    
    def test_add_noise_empty_data(self):
        """Test noise addition with empty data."""
        empty_data = np.array([])
        privatized = self.dp.add_noise(empty_data)
        
        assert privatized.size == 0
    
    def test_sensitivity_computation(self):
        """Test sensitivity computation."""
        data = np.array([1, 2, 3, 4, 5])
        sensitivity = self.dp._compute_sensitivity(data)
        
        assert sensitivity > 0
        assert isinstance(sensitivity, float)
    
    def test_data_clipping(self):
        """Test data clipping to sensitivity bounds."""
        data = np.array([10, 20, 30])
        sensitivity = 5.0
        clipped = self.dp._clip_data(data, sensitivity)
        
        assert np.linalg.norm(clipped) <= sensitivity + 1e-10
    
    def test_privacy_composition(self):
        """Test privacy parameter composition."""
        other_eps, other_delta = 0.5, 1e-6
        composed_eps, composed_delta = self.dp.compose_privacy(other_eps, other_delta)
        
        assert composed_eps == self.dp.epsilon + other_eps
        assert composed_delta == self.dp.delta + other_delta
    
    def test_privacy_budget_exhaustion(self):
        """Test privacy budget exhaustion check."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        
        assert not budget.is_exhausted()
        
        budget.composition_count = 1500
        assert budget.is_exhausted()
    
    def test_privacy_proof_generation(self):
        """Test privacy proof certificate generation."""
        proof = self.dp.generate_privacy_proof("Test query")
        
        expected_keys = [
            "query_id", "query_description", "mechanism", 
            "epsilon", "delta", "privacy_guarantee"
        ]
        
        for key in expected_keys:
            assert key in proof
        
        assert proof["epsilon"] == 1.0
        assert proof["delta"] == 1e-5
    
    def test_privacy_validation(self):
        """Test privacy loss validation."""
        # Should be valid initially
        assert self.dp.validate_privacy_loss()
        
        # Simulate many queries
        self.dp.budget.composition_count = 50
        assert not self.dp.validate_privacy_loss(max_epsilon=1.0)
    
    def test_utility_estimation(self):
        """Test utility loss estimation."""
        original = np.random.randn(100)
        privatized = self.dp.add_noise(original.copy())
        
        metrics = self.dp.estimate_utility_loss(original, privatized)
        
        expected_metrics = [
            "mean_squared_error", "mean_absolute_error", 
            "signal_to_noise_ratio", "relative_error", "correlation"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
    
    def test_budget_reset(self):
        """Test privacy budget reset."""
        # Use budget
        self.dp.add_noise(np.random.randn(10))
        assert self.dp.budget.composition_count > 0
        
        # Reset budget
        self.dp.reset_budget(new_epsilon=2.0)
        assert self.dp.epsilon == 2.0
        assert self.dp.budget.composition_count == 0
    
    def test_privatize_features(self):
        """Test feature privatization."""
        features = np.random.randn(50, 10)  # 50 samples, 10 features
        privatized = self.dp.privatize_features(features)
        
        assert privatized.shape == features.shape
        assert not np.array_equal(privatized, features)
    
    def test_get_privacy_status(self):
        """Test privacy status retrieval."""
        status = self.dp.get_privacy_status()
        
        expected_keys = [
            "epsilon", "delta", "mechanism", "composition_count", 
            "budget_exhausted", "privacy_guarantee"
        ]
        
        for key in expected_keys:
            assert key in status


class TestPrivacyBudget:
    """Test suite for PrivacyBudget."""
    
    def test_budget_creation(self):
        """Test privacy budget creation."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        
        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5
        assert budget.composition_count == 0
    
    def test_exhaustion_check(self):
        """Test budget exhaustion checking."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        
        # Default max queries is 1000
        assert not budget.is_exhausted()
        
        budget.composition_count = 999
        assert not budget.is_exhausted()
        
        budget.composition_count = 1000
        assert budget.is_exhausted()
        
        # Custom max queries
        budget.composition_count = 50
        assert budget.is_exhausted(max_queries=50)


if __name__ == "__main__":
    pytest.main([__file__])