"""
Tests for core BCI Bridge functionality.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch

from bci_agent_bridge.core.bridge import (
    BCIBridge, 
    NeuralData, 
    DecodedIntention,
    BCIDevice,
    Paradigm
)


class TestBCIBridge:
    """Test suite for BCIBridge class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.bridge = BCIBridge(
            device="Simulation",
            channels=8,
            sampling_rate=250,
            paradigm="P300"
        )
    
    def test_initialization(self):
        """Test bridge initialization."""
        assert self.bridge.device == BCIDevice.SIMULATION
        assert self.bridge.channels == 8
        assert self.bridge.sampling_rate == 250
        assert self.bridge.paradigm == Paradigm.P300
        assert not self.bridge.is_streaming
        assert len(self.bridge.data_buffer) == 0
    
    def test_device_info(self):
        """Test device information retrieval."""
        info = self.bridge.get_device_info()
        
        expected_keys = ['device', 'channels', 'sampling_rate', 'paradigm', 'connected', 'streaming']
        assert all(key in info for key in expected_keys)
        assert info['device'] == 'Simulation'
        assert info['channels'] == 8
        assert info['connected'] is True
        assert info['streaming'] is False
    
    def test_simulation_data_generation(self):
        """Test simulated neural data generation."""
        data = self.bridge._generate_simulation_data()
        
        assert data.shape[0] == self.bridge.channels
        assert data.shape[1] == self.bridge.sampling_rate
        assert not np.all(data == 0)  # Should not be all zeros
    
    def test_neural_data_creation(self):
        """Test NeuralData object creation."""
        raw_data = np.random.randn(8, 250)
        
        neural_data = NeuralData(
            data=raw_data,
            timestamp=1234567890.0,
            channels=['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8'],
            sampling_rate=250
        )
        
        assert neural_data.data.shape == (8, 250)
        assert neural_data.timestamp == 1234567890.0
        assert len(neural_data.channels) == 8
        assert neural_data.sampling_rate == 250
    
    @pytest.mark.asyncio
    async def test_streaming(self):
        """Test neural data streaming."""
        stream_count = 0
        max_samples = 3
        
        async for neural_data in self.bridge.stream():
            assert isinstance(neural_data, NeuralData)
            assert neural_data.data.shape[0] == self.bridge.channels
            assert len(neural_data.channels) == self.bridge.channels
            
            stream_count += 1
            if stream_count >= max_samples:
                self.bridge.stop_streaming()
                break
        
        assert stream_count == max_samples
        assert not self.bridge.is_streaming
    
    def test_intention_decoding(self):
        """Test neural intention decoding."""
        # Create test neural data
        test_data = np.random.randn(8, 250)
        neural_data = NeuralData(
            data=test_data,
            timestamp=1234567890.0,
            channels=[f'CH{i+1}' for i in range(8)],
            sampling_rate=250
        )
        
        # Decode intention
        intention = self.bridge.decode_intention(neural_data)
        
        assert isinstance(intention, DecodedIntention)
        assert isinstance(intention.command, str)
        assert 0.0 <= intention.confidence <= 1.0
        assert isinstance(intention.context, dict)
        assert intention.timestamp > 0
    
    def test_command_mapping_p300(self):
        """Test P300 command mapping."""
        # Test target detection
        command = self.bridge._map_prediction_to_command(1)
        assert command == "Select current item"
        
        # Test non-target
        command = self.bridge._map_prediction_to_command(0)
        assert command == "No selection"
    
    def test_motor_imagery_bridge(self):
        """Test motor imagery paradigm setup."""
        mi_bridge = BCIBridge(
            device="Simulation",
            channels=8,
            sampling_rate=250,
            paradigm="MotorImagery"
        )
        
        assert mi_bridge.paradigm == Paradigm.MOTOR_IMAGERY
        
        # Test command mapping
        for i, expected_command in enumerate(["Move left", "Move right", "Move forward", "Move backward"]):
            command = mi_bridge._map_prediction_to_command(i)
            assert command == expected_command
    
    def test_ssvep_bridge(self):
        """Test SSVEP paradigm setup."""
        ssvep_bridge = BCIBridge(
            device="Simulation",
            channels=8,
            sampling_rate=250,
            paradigm="SSVEP"
        )
        
        assert ssvep_bridge.paradigm == Paradigm.SSVEP
        
        # Test command mapping
        for i in range(4):
            command = ssvep_bridge._map_prediction_to_command(i)
            assert f"Option {i+1} selected" in command
    
    def test_buffer_operations(self):
        """Test data buffer operations."""
        # Add some mock data to buffer
        for i in range(5):
            mock_data = NeuralData(
                data=np.random.randn(8, 250),
                timestamp=1234567890.0 + i,
                channels=[f'CH{j+1}' for j in range(8)],
                sampling_rate=250
            )
            self.bridge.data_buffer.append(mock_data)
        
        # Test window reading
        window_data = self.bridge.read_window(1000)  # 1 second
        assert window_data.shape[0] == 8  # Should have 8 channels
        
        # Test buffer reading
        buffer_data = self.bridge.get_buffer(3)
        assert buffer_data.shape[0] == 8
    
    def test_calibration(self):
        """Test decoder calibration."""
        # Should not raise exception
        self.bridge.calibrate()
        
        # Test with custom data
        calibration_data = np.random.randn(8, 2500)  # 10 seconds of data
        self.bridge.calibrate(calibration_data)
    
    def test_invalid_paradigm(self):
        """Test invalid paradigm handling."""
        with pytest.raises(ValueError):
            BCIBridge(paradigm="InvalidParadigm")
    
    def test_privacy_mode(self):
        """Test privacy mode functionality."""
        # Privacy mode enabled
        private_bridge = BCIBridge(privacy_mode=True)
        test_data = NeuralData(
            data=np.random.randn(8, 250),
            timestamp=1234567890.0,
            channels=[f'CH{i+1}' for i in range(8)],
            sampling_rate=250
        )
        
        intention = private_bridge.decode_intention(test_data)
        assert intention.neural_features is None  # Should be None in privacy mode
        
        # Privacy mode disabled
        open_bridge = BCIBridge(privacy_mode=False)
        intention = open_bridge.decode_intention(test_data)
        # neural_features might still be None if decoder is not calibrated, but that's ok


class TestNeuralData:
    """Test suite for NeuralData class."""
    
    def test_neural_data_creation(self):
        """Test NeuralData creation with various inputs."""
        data = np.random.randn(4, 128)
        
        neural_data = NeuralData(
            data=data,
            timestamp=1234567890.0,
            channels=['C1', 'C2', 'C3', 'C4'],
            sampling_rate=128,
            metadata={'experiment': 'test'}
        )
        
        assert np.array_equal(neural_data.data, data)
        assert neural_data.timestamp == 1234567890.0
        assert neural_data.channels == ['C1', 'C2', 'C3', 'C4']
        assert neural_data.sampling_rate == 128
        assert neural_data.metadata['experiment'] == 'test'


class TestDecodedIntention:
    """Test suite for DecodedIntention class."""
    
    def test_decoded_intention_creation(self):
        """Test DecodedIntention creation."""
        intention = DecodedIntention(
            command="Select item",
            confidence=0.85,
            context={'paradigm': 'P300'},
            timestamp=1234567890.0,
            neural_features=np.array([1, 2, 3])
        )
        
        assert intention.command == "Select item"
        assert intention.confidence == 0.85
        assert intention.context['paradigm'] == 'P300'
        assert intention.timestamp == 1234567890.0
        assert np.array_equal(intention.neural_features, np.array([1, 2, 3]))


if __name__ == "__main__":
    pytest.main([__file__])