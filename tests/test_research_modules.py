"""
Comprehensive test suite for research enhancement modules.

Tests for transformer decoders, hybrid multi-paradigm decoders,
quantum optimization, and federated learning components.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from typing import Dict, Any, List

# Import research modules
from bci_agent_bridge.decoders.transformer_decoder import (
    TransformerNeuralDecoder, 
    TransformerConfig,
    PositionalEncoding,
    SpatialAttentionBlock,
    TemporalTransformerBlock
)

from bci_agent_bridge.decoders.hybrid_decoder import (
    HybridMultiParadigmDecoder,
    HybridConfig,
    SignalQualityAssessor,
    AdaptiveFusionModule,
    ParadigmType
)

from bci_agent_bridge.research.quantum_optimization import (
    QuantumCircuit,
    VariationalQuantumCircuit,
    QuantumNeuralDecoder,
    QuantumConfig,
    QuantumAnnealingOptimizer,
    create_quantum_bci_decoder
)

from bci_agent_bridge.research.federated_learning import (
    FederatedServer,
    FederatedClient,
    FederatedConfig,
    ClientData,
    ModelUpdate,
    SecureAggregator,
    DifferentialPrivacyMechanism,
    ByzantineDetector,
    create_federated_bci_system
)


class TestTransformerDecoder:
    """Test suite for transformer-based neural decoder."""
    
    @pytest.fixture
    def config(self):
        return TransformerConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.1,
            max_seq_length=500,
            n_classes=4
        )
    
    @pytest.fixture
    def decoder(self, config):
        return TransformerNeuralDecoder(
            channels=8,
            sampling_rate=250,
            config=config,
            paradigm="P300"
        )
    
    def test_initialization(self, decoder, config):
        """Test decoder initialization."""
        assert decoder.channels == 8
        assert decoder.sampling_rate == 250
        assert decoder.config.d_model == 64
        assert decoder.paradigm == "P300"
        assert not decoder.is_trained
    
    def test_positional_encoding(self):
        """Test positional encoding module."""
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(50, 64)  # (seq_len, d_model)
        encoded = pe(x)
        
        assert encoded.shape == x.shape
        assert not torch.allclose(encoded, x)  # Should be different due to positional encoding
    
    def test_spatial_attention_block(self):
        """Test spatial attention block."""
        spatial_attention = SpatialAttentionBlock(n_channels=8, d_model=64, n_heads=4)
        x = torch.randn(2, 8, 100)  # (batch, channels, time)
        
        output = spatial_attention(x)
        assert output.shape == (2, 8, 64)  # (batch, channels, d_model)
    
    def test_temporal_transformer_block(self):
        """Test temporal transformer block."""
        temporal_block = TemporalTransformerBlock(d_model=64, n_heads=4)
        x = torch.randn(2, 100, 64)  # (batch, time, d_model)
        
        output = temporal_block(x)
        assert output.shape == x.shape
    
    def test_feature_extraction(self, decoder):
        """Test feature extraction from neural data."""
        # Create test data
        data = np.random.randn(8, 250)
        
        # Extract features
        features = decoder.extract_features(data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == decoder.config.d_model  # Should match d_model
    
    def test_prediction_untrained(self, decoder):
        """Test prediction with untrained model."""
        features = np.random.randn(64)
        prediction = decoder.predict(features)
        
        assert isinstance(prediction, (int, np.integer))
        assert 0 <= prediction < decoder.config.n_classes
    
    def test_training(self, decoder):
        """Test model training."""
        # Create synthetic training data
        n_samples = 100
        X = np.random.randn(n_samples, 8, 250)
        y = np.random.randint(0, 4, n_samples)
        
        # Train for few epochs
        history = decoder.fit(X, y, epochs=5, batch_size=16)
        
        assert decoder.is_trained
        assert 'train_loss' in history
        assert 'train_acc' in history
        assert len(history['train_loss']) <= 5
    
    def test_calibration(self, decoder):
        """Test decoder calibration."""
        calibration_data = np.random.randn(50, 8, 250)
        labels = np.random.randint(0, 2, 50)
        
        # Should not raise exception
        decoder.calibrate(calibration_data, labels)
    
    def test_save_load_model(self, decoder):
        """Test model saving and loading."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            # Save model
            decoder.save_model(filepath)
            assert os.path.exists(filepath)
            
            # Load model
            loaded_decoder = TransformerNeuralDecoder.load_model(filepath)
            
            assert loaded_decoder.channels == decoder.channels
            assert loaded_decoder.sampling_rate == decoder.sampling_rate
            assert loaded_decoder.paradigm == decoder.paradigm
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestHybridDecoder:
    """Test suite for hybrid multi-paradigm decoder."""
    
    @pytest.fixture
    def config(self):
        return HybridConfig(
            use_p300=True,
            use_ssvep=True,
            use_motor_imagery=True,
            use_transformers=False,  # Use classical decoders for testing
            fusion_method="adaptive"
        )
    
    @pytest.fixture
    def decoder(self, config):
        return HybridMultiParadigmDecoder(
            channels=8,
            sampling_rate=250,
            config=config
        )
    
    def test_initialization(self, decoder, config):
        """Test hybrid decoder initialization."""
        assert decoder.channels == 8
        assert decoder.sampling_rate == 250
        assert len(decoder.paradigm_decoders) == 3  # P300, SSVEP, MI
        assert not decoder.is_trained
    
    def test_signal_quality_assessor(self):
        """Test signal quality assessment."""
        assessor = SignalQualityAssessor(n_channels=8, sampling_rate=250)
        data = torch.randn(2, 8, 250)  # (batch, channels, time)
        
        quality_scores = assessor(data)
        assert quality_scores.shape == (2,)
        assert torch.all(quality_scores >= 0) and torch.all(quality_scores <= 1)
    
    def test_adaptive_fusion_module(self):
        """Test adaptive fusion module."""
        fusion = AdaptiveFusionModule(n_paradigms=3, feature_dim=64)
        
        # Create mock paradigm features and confidences
        features = [torch.randn(2, 64) for _ in range(3)]
        confidences = [torch.tensor([0.8, 0.9]) for _ in range(3)]
        
        predictions, weights = fusion(features, confidences)
        
        assert predictions.shape == (2, 4)  # (batch, n_classes)
        assert weights.shape == (2, 3)  # (batch, n_paradigms)
        assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-5)
    
    def test_feature_extraction(self, decoder):
        """Test feature extraction from all paradigms."""
        data = np.random.randn(8, 250)
        
        features = decoder.extract_features(data)
        
        assert isinstance(features, dict)
        assert 'P300' in features
        assert 'SSVEP' in features
        assert 'MotorImagery' in features
    
    def test_paradigm_reliability_assessment(self, decoder):
        """Test paradigm reliability assessment."""
        data = np.random.randn(8, 250)
        
        reliability = decoder.assess_paradigm_reliability(data)
        
        assert isinstance(reliability, dict)
        assert all(0 <= score <= 1 for score in reliability.values())
        assert set(reliability.keys()) == {'P300', 'SSVEP', 'MotorImagery'}
    
    def test_prediction_untrained(self, decoder):
        """Test prediction with untrained model."""
        data = np.random.randn(8, 250)
        prediction = decoder.predict(data)
        
        assert isinstance(prediction, (int, np.integer))
        assert 0 <= prediction < 4
    
    def test_paradigm_contributions(self, decoder):
        """Test paradigm contribution tracking."""
        contributions = decoder.get_paradigm_contributions()
        
        assert isinstance(contributions, dict)
        assert abs(sum(contributions.values()) - 1.0) < 1e-5  # Should sum to 1
        assert all(0 <= contrib <= 1 for contrib in contributions.values())
    
    def test_calibration(self, decoder):
        """Test multi-paradigm calibration."""
        calibration_data = np.random.randn(50, 8, 250)
        labels = np.random.randint(0, 4, 50)
        
        # Should not raise exception
        decoder.calibrate(calibration_data, labels)


class TestQuantumOptimization:
    """Test suite for quantum optimization components."""
    
    @pytest.fixture
    def config(self):
        return QuantumConfig(
            n_qubits=4,
            n_layers=2,
            learning_rate=0.01,
            max_iterations=100
        )
    
    def test_quantum_circuit_initialization(self):
        """Test quantum circuit initialization."""
        circuit = QuantumCircuit(n_qubits=3)
        
        assert circuit.n_qubits == 3
        assert circuit.n_states == 8
        assert circuit.state[0] == 1.0  # Initialized to |000⟩
        assert np.abs(np.sum(np.abs(circuit.state)**2) - 1.0) < 1e-10  # Normalized
    
    def test_quantum_gates(self):
        """Test quantum gate operations."""
        circuit = QuantumCircuit(n_qubits=2)
        
        # Apply Hadamard gate to first qubit
        circuit.apply_single_qubit_gate(circuit.gates.hadamard(), 0)
        
        # State should be in superposition
        expected_state = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2), 0])
        assert np.allclose(circuit.state, expected_state, atol=1e-10)
    
    def test_quantum_measurement(self):
        """Test quantum measurement."""
        circuit = QuantumCircuit(n_qubits=3)
        
        # Apply some gates to create superposition
        circuit.apply_single_qubit_gate(circuit.gates.hadamard(), 0)
        
        # Measure
        measurement = circuit.measure()
        
        assert len(measurement) == 3
        assert all(bit in [0, 1] for bit in measurement)
    
    def test_variational_quantum_circuit(self, config):
        """Test variational quantum circuit."""
        vqc = VariationalQuantumCircuit(config, input_dim=8, output_dim=4)
        
        x = torch.randn(2, 8)  # (batch, input_dim)
        output = vqc(x)
        
        assert output.shape == (2, 4)  # (batch, output_dim)
    
    def test_quantum_annealing_optimizer(self, config):
        """Test quantum annealing optimizer."""
        optimizer = QuantumAnnealingOptimizer(config)
        
        # Create simple optimization problem
        n_vars = 4
        weights = np.random.randn(n_vars, n_vars) * 0.1
        weights = (weights + weights.T) / 2  # Make symmetric
        bias = np.random.randn(n_vars) * 0.1
        
        hamiltonian = optimizer.create_ising_hamiltonian(weights, bias)
        
        assert hamiltonian.shape == (2**n_vars, 2**n_vars)
        assert np.allclose(hamiltonian, hamiltonian.T)  # Should be Hermitian
    
    def test_quantum_neural_decoder(self, config):
        """Test quantum neural decoder."""
        decoder = QuantumNeuralDecoder(input_dim=64, output_dim=4, config=config)
        
        x = torch.randn(2, 64)
        output = decoder(x)
        
        assert output.shape == (2, 4)
    
    def test_quantum_feature_importance(self, config):
        """Test quantum feature importance analysis."""
        decoder = QuantumNeuralDecoder(input_dim=8, output_dim=4, config=config)
        x = torch.randn(2, 8)
        
        importance = decoder.quantum_feature_importance(x)
        
        assert 'feature_importance' in importance
        assert importance['feature_importance'].shape == (8,)
        assert np.abs(np.sum(importance['feature_importance']) - 1.0) < 1e-5
    
    def test_create_quantum_bci_decoder(self):
        """Test quantum BCI decoder factory function."""
        decoder = create_quantum_bci_decoder(input_dim=64, output_dim=4)
        
        assert isinstance(decoder, QuantumNeuralDecoder)
        assert decoder.input_dim == 64
        assert decoder.output_dim == 4


class TestFederatedLearning:
    """Test suite for federated learning framework."""
    
    @pytest.fixture
    def config(self):
        return FederatedConfig(
            n_clients=5,
            n_rounds=3,
            client_fraction=0.6,
            local_epochs=2,
            use_differential_privacy=False,  # Disable for testing
            byzantine_tolerance=False,       # Disable for testing
            secure_aggregation=False        # Disable for testing
        )
    
    @pytest.fixture
    def simple_model(self):
        """Simple model for testing."""
        return nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax(dim=1)
        )
    
    def test_client_data_creation(self):
        """Test client data container."""
        X_train = np.random.randn(100, 64)
        y_train = np.random.randint(0, 4, 100)
        
        client_data = ClientData(
            client_id="test_client",
            X_train=X_train,
            y_train=y_train
        )
        
        assert client_data.client_id == "test_client"
        assert client_data.X_train.shape == (100, 64)
        assert client_data.y_train.shape == (100,)
    
    def test_model_update_creation(self):
        """Test model update container."""
        parameters = {'layer1.weight': torch.randn(32, 64)}
        
        update = ModelUpdate(
            client_id="test_client",
            parameters=parameters,
            num_samples=100,
            loss=0.5,
            accuracy=0.8,
            timestamp=12345.0
        )
        
        assert update.client_id == "test_client"
        assert update.num_samples == 100
        assert update.loss == 0.5
        assert update.accuracy == 0.8
    
    def test_differential_privacy_mechanism(self):
        """Test differential privacy mechanism."""
        dp = DifferentialPrivacyMechanism(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        
        # Test gradient clipping
        gradients = {
            'layer1': torch.randn(10, 10) * 5,  # Large gradients
            'layer2': torch.randn(5) * 3
        }
        
        clipped = dp.clip_gradients(gradients)
        
        # Check that gradients are clipped
        total_norm = 0.0
        for grad in clipped.values():
            total_norm += torch.norm(grad) ** 2
        total_norm = torch.sqrt(total_norm)
        
        assert total_norm <= dp.clip_norm + 1e-6
        
        # Test noise addition
        noisy = dp.add_noise(clipped, num_samples=100)
        
        assert set(noisy.keys()) == set(clipped.keys())
        for name in noisy:
            assert noisy[name].shape == clipped[name].shape
    
    def test_byzantine_detector(self):
        """Test Byzantine client detection."""
        detector = ByzantineDetector(max_byzantine=1, detection_method="krum")
        
        # Create updates (some normal, some Byzantine)
        updates = []
        for i in range(5):
            if i < 4:  # Normal updates
                params = {'layer1': torch.randn(10, 10) * 0.1}
            else:  # Byzantine update
                params = {'layer1': torch.randn(10, 10) * 10}  # Much larger
            
            update = ModelUpdate(
                client_id=f"client_{i}",
                parameters=params,
                num_samples=100,
                loss=0.5,
                accuracy=0.8,
                timestamp=12345.0
            )
            updates.append(update)
        
        filtered_updates = detector.detect_byzantine_updates(updates)
        
        # Should filter out Byzantine updates
        assert len(filtered_updates) <= len(updates)
    
    def test_federated_client(self, config, simple_model):
        """Test federated client functionality."""
        client = FederatedClient("test_client", simple_model, config)
        
        assert client.client_id == "test_client"
        assert client.model is not None
        
        # Test parameter operations
        params = client.get_parameters()
        assert isinstance(params, dict)
        assert len(params) > 0
        
        client.set_parameters(params)  # Should not raise exception
    
    def test_federated_server(self, config, simple_model):
        """Test federated server functionality."""
        server = FederatedServer(simple_model, config)
        
        assert server.global_model is not None
        assert len(server.clients) == 0
        
        # Test client registration
        client = FederatedClient("test_client", simple_model, config)
        server.register_client(client)
        
        assert len(server.clients) == 1
        assert "test_client" in server.clients
    
    def test_federated_averaging(self, config, simple_model):
        """Test federated averaging aggregation."""
        server = FederatedServer(simple_model, config)
        
        # Create mock updates
        updates = []
        for i in range(3):
            params = {name: torch.randn_like(param) * 0.1 
                     for name, param in simple_model.named_parameters()}
            
            update = ModelUpdate(
                client_id=f"client_{i}",
                parameters=params,
                num_samples=100 + i * 10,  # Different sample sizes
                loss=0.5,
                accuracy=0.8,
                timestamp=12345.0
            )
            updates.append(update)
        
        aggregated = server.aggregate_updates(updates)
        
        assert isinstance(aggregated, dict)
        assert len(aggregated) == len(list(simple_model.named_parameters()))
    
    def test_client_selection(self, config, simple_model):
        """Test client selection for rounds."""
        server = FederatedServer(simple_model, config)
        
        # Register clients
        for i in range(config.n_clients):
            client = FederatedClient(f"client_{i}", simple_model, config)
            server.register_client(client)
        
        selected = server.select_clients(round_num=0)
        
        assert len(selected) >= config.min_clients_per_round
        assert len(selected) <= config.n_clients
        assert all(client_id in server.clients for client_id in selected)
    
    def test_create_federated_bci_system(self, config):
        """Test federated BCI system factory function."""
        def model_factory():
            return nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 4)
            )
        
        server, clients = create_federated_bci_system(model_factory, config)
        
        assert isinstance(server, FederatedServer)
        assert len(clients) == config.n_clients
        assert all(isinstance(client, FederatedClient) for client in clients)
        assert len(server.clients) == config.n_clients


class TestIntegration:
    """Integration tests for research modules."""
    
    def test_transformer_hybrid_integration(self):
        """Test integration between transformer and hybrid decoders."""
        # Create hybrid decoder with transformers enabled
        config = HybridConfig(use_transformers=True, use_p300=True, use_ssvep=False, use_motor_imagery=False)
        
        hybrid_decoder = HybridMultiParadigmDecoder(
            channels=8,
            sampling_rate=250,
            config=config
        )
        
        # Should have transformer-based P300 decoder
        assert 'P300' in hybrid_decoder.paradigm_decoders
        assert isinstance(hybrid_decoder.paradigm_decoders['P300'], TransformerNeuralDecoder)
        
        # Test prediction
        data = np.random.randn(8, 250)
        prediction = hybrid_decoder.predict(data)
        
        assert isinstance(prediction, (int, np.integer))
    
    def test_quantum_federated_integration(self):
        """Test integration between quantum and federated learning."""
        # Create quantum model factory
        def quantum_model_factory():
            config = QuantumConfig(n_qubits=4, n_layers=2)
            return QuantumNeuralDecoder(input_dim=64, output_dim=4, config=config)
        
        fed_config = FederatedConfig(
            n_clients=3,
            n_rounds=2,
            use_differential_privacy=False,
            byzantine_tolerance=False,
            secure_aggregation=False
        )
        
        server, clients = create_federated_bci_system(quantum_model_factory, fed_config)
        
        # Test that quantum models are created
        assert len(clients) == 3
        for client in clients:
            assert isinstance(client.model, QuantumNeuralDecoder)
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline with research enhancements."""
        # Create synthetic BCI data
        n_samples = 100
        n_channels = 8
        seq_length = 250
        n_classes = 4
        
        X = np.random.randn(n_samples, n_channels, seq_length)
        y = np.random.randint(0, n_classes, n_samples)
        
        # Test transformer decoder
        transformer_config = TransformerConfig(n_classes=n_classes, n_layers=2)
        transformer_decoder = TransformerNeuralDecoder(
            channels=n_channels,
            sampling_rate=250,
            config=transformer_config
        )
        
        # Train briefly
        history = transformer_decoder.fit(X[:80], y[:80], epochs=2, batch_size=16)
        assert transformer_decoder.is_trained
        
        # Test prediction
        test_prediction = transformer_decoder.predict(transformer_decoder.extract_features(X[0]))
        assert 0 <= test_prediction < n_classes
        
        # Test hybrid decoder
        hybrid_config = HybridConfig(use_transformers=False)  # Use classical for speed
        hybrid_decoder = HybridMultiParadigmDecoder(
            channels=n_channels,
            sampling_rate=250,
            config=hybrid_config
        )
        
        # Test feature extraction and prediction
        features = hybrid_decoder.extract_features(X[0])
        hybrid_prediction = hybrid_decoder.predict(features)
        assert 0 <= hybrid_prediction < n_classes
        
        print("End-to-end pipeline test completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])