"""
Tests for neural signal decoders.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bci_agent_bridge.decoders.p300 import P300Decoder
from bci_agent_bridge.decoders.base import BaseDecoder


class TestP300Decoder:
    """Test suite for P300Decoder."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.decoder = P300Decoder(channels=8, sampling_rate=250)
    
    def test_initialization(self):
        """Test decoder initialization."""
        assert self.decoder.channels == 8
        assert self.decoder.sampling_rate == 250
        assert not self.decoder.is_calibrated
        assert self.decoder.confidence_threshold == 0.7
        assert self.decoder.window_start == 0.0
        assert self.decoder.window_end == 0.8
        assert self.decoder.target_latency == 0.3
    
    def test_feature_extraction_empty_data(self):
        """Test feature extraction with empty data."""
        empty_data = np.array([])
        features = self.decoder.extract_features(empty_data)
        assert features.size == 0
    
    def test_feature_extraction_valid_data(self):
        """Test feature extraction with valid data."""
        # Create test data (8 channels, 200 samples = 0.8 seconds at 250 Hz)
        test_data = np.random.randn(8, 200)
        
        features = self.decoder.extract_features(test_data)
        
        assert features.size > 0
        assert isinstance(features, np.ndarray)
        # Should have features for each channel plus spatial features
        expected_min_features = 8 * 4  # 4 features per channel
        assert features.size >= expected_min_features
    
    def test_feature_extraction_1d_data(self):
        """Test feature extraction with 1D data."""
        # Single channel data
        test_data = np.random.randn(200)
        
        features = self.decoder.extract_features(test_data)
        
        assert features.size > 0
        assert isinstance(features, np.ndarray)
    
    def test_bandpass_filtering(self):
        """Test bandpass filtering."""
        test_data = np.random.randn(8, 200)
        
        # This should not raise an exception
        filtered_data = self.decoder._apply_bandpass_filter(test_data)
        
        assert filtered_data.shape == test_data.shape
        assert isinstance(filtered_data, np.ndarray)
    
    def test_p300_window_extraction(self):
        """Test P300 time window extraction."""
        test_data = np.random.randn(8, 200)  # 0.8 seconds
        
        window = self.decoder._extract_p300_window(test_data)
        
        # Should return the full window since our test data spans the full window
        assert window.shape[0] == 8  # Same number of channels
        assert window.shape[1] <= 200  # Same or fewer samples
    
    def test_prediction_uncalibrated(self):
        """Test prediction without calibration."""
        features = np.random.randn(32)  # Some features
        
        prediction = self.decoder.predict(features)
        
        # Should return 0 or 1
        assert prediction in [0, 1]
        # Confidence should be low since uncalibrated
        assert self.decoder.get_confidence() <= 1.0
    
    def test_prediction_empty_features(self):
        """Test prediction with empty features."""
        empty_features = np.array([])
        
        prediction = self.decoder.predict(empty_features)
        
        assert prediction == 0
        assert self.decoder.get_confidence() == 0.0
    
    def test_calibration(self):
        """Test decoder calibration."""
        # Create simulated calibration data
        n_epochs = 50
        channels = 8
        samples_per_epoch = 200
        
        calibration_data = np.random.randn(n_epochs, channels, samples_per_epoch)
        
        # Should not raise exception
        self.decoder.calibrate(calibration_data)
        
        # After calibration, decoder should be calibrated
        assert self.decoder.is_calibrated
    
    def test_calibration_with_labels(self):
        """Test calibration with provided labels."""
        n_epochs = 20
        calibration_data = np.random.randn(n_epochs, 8, 200)
        labels = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
                          0, 0, 1, 0, 1, 0, 0, 1, 0, 1])
        
        self.decoder.calibrate(calibration_data, labels)
        
        assert self.decoder.is_calibrated
    
    def test_calibration_empty_data(self):
        """Test calibration with empty data."""
        empty_data = np.array([])
        
        # Should handle gracefully
        self.decoder.calibrate(empty_data)
        
        # Should remain uncalibrated
        assert not self.decoder.is_calibrated
    
    def test_p300_peak_detection(self):
        """Test P300 peak detection."""
        # Create data with simulated P300
        test_data = np.random.randn(8, 250)  # 1 second at 250 Hz
        
        # Add simulated P300 at ~300ms
        p300_sample = int(0.3 * 250)
        test_data[:4, p300_sample:p300_sample+10] += 5.0  # Positive deflection
        
        peak_latency, peak_amplitude = self.decoder.detect_p300_peak(test_data)
        
        assert isinstance(peak_latency, float)
        assert isinstance(peak_amplitude, float)
        assert peak_latency >= 0
    
    def test_p300_peak_detection_empty_data(self):
        """Test P300 peak detection with empty data."""
        empty_data = np.array([])
        
        peak_latency, peak_amplitude = self.decoder.detect_p300_peak(empty_data)
        
        assert peak_latency == 0.0
        assert peak_amplitude == 0.0
    
    def test_incremental_learning(self):
        """Test incremental learning functionality."""
        # Add some training epochs
        for i in range(10):
            epoch = np.random.randn(8, 200)
            is_target = i % 3 == 0  # Every 3rd epoch is target
            self.decoder.add_training_epoch(epoch, is_target)
        
        assert len(self.decoder.target_epochs) > 0
        assert len(self.decoder.nontarget_epochs) > 0
        
        # Retrain with accumulated data
        self.decoder.retrain_incremental()
        
        # Should be calibrated after retraining
        assert self.decoder.is_calibrated
    
    def test_get_p300_characteristics(self):
        """Test P300 characteristics retrieval."""
        characteristics = self.decoder.get_p300_characteristics()
        
        expected_keys = [
            'window_start', 'window_end', 'target_latency',
            'bandpass_range', 'n_target_epochs', 'n_nontarget_epochs'
        ]
        
        for key in expected_keys:
            assert key in characteristics
        
        assert characteristics['window_start'] == 0.0
        assert characteristics['window_end'] == 0.8
        assert characteristics['target_latency'] == 0.3
    
    def test_confidence_threshold(self):
        """Test confidence threshold setting."""
        new_threshold = 0.9
        self.decoder.set_confidence_threshold(new_threshold)
        
        assert self.decoder.confidence_threshold == new_threshold
        
        # Test confidence check
        self.decoder.last_confidence = 0.95
        assert self.decoder.is_prediction_confident()
        
        self.decoder.last_confidence = 0.85
        assert not self.decoder.is_prediction_confident()
    
    def test_decoder_info(self):
        """Test decoder information retrieval."""
        info = self.decoder.get_decoder_info()
        
        expected_keys = [
            'type', 'channels', 'sampling_rate', 'calibrated',
            'confidence_threshold', 'last_confidence'
        ]
        
        for key in expected_keys:
            assert key in info
        
        assert info['type'] == 'P300Decoder'
        assert info['channels'] == 8
        assert info['sampling_rate'] == 250


class TestBaseDecoder:
    """Test suite for BaseDecoder abstract class."""
    
    def test_cannot_instantiate_base_decoder(self):
        """Test that BaseDecoder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDecoder(channels=8, sampling_rate=250)
    
    def test_concrete_decoder_implementation(self):
        """Test that concrete decoders implement required methods."""
        decoder = P300Decoder(channels=8, sampling_rate=250)
        
        # Check that abstract methods are implemented
        assert hasattr(decoder, 'extract_features')
        assert hasattr(decoder, 'predict')
        assert hasattr(decoder, 'calibrate')
        assert callable(decoder.extract_features)
        assert callable(decoder.predict)
        assert callable(decoder.calibrate)


if __name__ == "__main__":
    pytest.main([__file__])