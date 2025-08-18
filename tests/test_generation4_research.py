"""
Test suite for Generation 4 research modules.

This module tests the advanced research capabilities including:
- Adaptive neural calibration
- Advanced multimodal fusion  
- Explainable neural AI
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

# Import Generation 4 modules directly
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bci_agent_bridge.research.adaptive_neural_calibration import (
    AdaptiveCalibrationEngine,
    NeuralPlasticityDetector,
    PlasticityDetectionConfig,
    create_adaptive_calibration_system
)

from bci_agent_bridge.research.advanced_multimodal_fusion import (
    AdvancedMultimodalFusion,
    AttentionFusionMechanism,
    ModalityData,
    FusionConfig,
    create_multimodal_fusion_system
)

from bci_agent_bridge.research.explainable_neural_ai import (
    ExplainableNeuralAI,
    NeuralSaliencyMapper,
    ExplanationConfig,
    create_explainable_neural_system
)


class TestAdaptiveNeuralCalibration:
    """Test adaptive neural calibration system."""
    
    def test_plasticity_detector_initialization(self):
        """Test plasticity detector initialization."""
        config = PlasticityDetectionConfig()
        detector = NeuralPlasticityDetector(config)
        
        assert detector.config == config
        assert len(detector.signal_history) == 0
        assert detector.baseline_statistics is None
        assert detector.adaptation_count == 0
    
    def test_plasticity_detection_empty_signals(self):
        """Test plasticity detection with empty signals."""
        detector = NeuralPlasticityDetector(PlasticityDetectionConfig())
        
        detected, magnitude = detector.detect_plasticity(np.array([]))
        
        assert detected is False
        assert magnitude == 0.0
    
    def test_plasticity_detection_valid_signals(self):
        """Test plasticity detection with valid signals."""
        detector = NeuralPlasticityDetector(PlasticityDetectionConfig(window_size=100))
        
        # Simulate neural signals
        signals = np.random.normal(0, 1, (8, 100))
        
        detected, magnitude = detector.detect_plasticity(signals)
        
        assert isinstance(detected, bool)
        assert isinstance(magnitude, float)
        assert 0.0 <= magnitude <= 1.0
    
    def test_plasticity_detection_significant_change(self):
        """Test detection of significant plasticity changes."""
        config = PlasticityDetectionConfig(
            window_size=50,
            detection_threshold=0.1,
            min_adaptation_interval=0.1
        )
        detector = NeuralPlasticityDetector(config)
        
        # Initialize with baseline
        baseline_signals = np.random.normal(0, 0.5, (4, 50))
        detector.detect_plasticity(baseline_signals)
        
        # Wait for minimum interval
        time.sleep(0.2)
        
        # Introduce significant change
        changed_signals = np.random.normal(2, 1, (4, 50))
        detected, magnitude = detector.detect_plasticity(changed_signals)
        
        # Should detect significant change
        assert magnitude > 0
    
    def test_adaptive_calibration_engine_initialization(self):
        """Test adaptive calibration engine initialization."""
        engine = AdaptiveCalibrationEngine(n_components=3, adaptation_rate=0.02)
        
        assert engine.n_components == 3
        assert engine.adaptation_rate == 0.02
        assert engine.feature_model is None
        assert len(engine.adaptation_history) == 0
    
    def test_calibration_initialization_empty_data(self):
        """Test calibration initialization with empty data."""
        engine = AdaptiveCalibrationEngine()
        
        with pytest.raises(ValueError):
            engine.initialize_calibration(np.array([]), np.array([]))
    
    def test_calibration_initialization_valid_data(self):
        """Test calibration initialization with valid data."""
        engine = AdaptiveCalibrationEngine()
        
        neural_data = np.random.normal(0, 1, (100, 8))
        labels = np.random.randint(0, 2, 100)
        
        engine.initialize_calibration(neural_data, labels)
        
        assert engine.feature_model is not None
    
    def test_adapt_calibration_empty_data(self):
        """Test adaptation with empty data."""
        engine = AdaptiveCalibrationEngine()
        
        metrics = engine.adapt_calibration(np.array([]))
        
        assert metrics.plasticity_score == 0.0
        assert metrics.signal_stability == 0.0
        assert metrics.calibration_confidence == 0.0
    
    def test_adapt_calibration_valid_data(self):
        """Test adaptation with valid data."""
        engine = AdaptiveCalibrationEngine()
        
        # Initialize first
        init_data = np.random.normal(0, 1, (50, 8))
        labels = np.random.randint(0, 2, 50)
        engine.initialize_calibration(init_data, labels)
        
        # Adapt with new data
        new_data = np.random.normal(0, 1, (20, 8))
        metrics = engine.adapt_calibration(new_data)
        
        assert isinstance(metrics.plasticity_score, float)
        assert isinstance(metrics.signal_stability, float)
        assert isinstance(metrics.calibration_confidence, float)
        assert 0.0 <= metrics.plasticity_score <= 1.0
        assert 0.0 <= metrics.signal_stability <= 1.0
        assert 0.0 <= metrics.calibration_confidence <= 1.0
    
    def test_create_adaptive_calibration_system(self):
        """Test factory function for adaptive calibration."""
        config = {'n_components': 7, 'adaptation_rate': 0.05}
        
        system = create_adaptive_calibration_system(config)
        
        assert isinstance(system, AdaptiveCalibrationEngine)
        assert system.n_components == 7
        assert system.adaptation_rate == 0.05


class TestAdvancedMultimodalFusion:
    """Test advanced multimodal fusion system."""
    
    def test_attention_fusion_mechanism_initialization(self):
        """Test attention fusion mechanism initialization."""
        mechanism = AttentionFusionMechanism(n_heads=6, feature_dim=32)
        
        assert mechanism.n_heads == 6
        assert mechanism.feature_dim == 32
        assert len(mechanism.attention_weights) == 0
    
    def test_attention_initialization(self):
        """Test attention mechanism initialization."""
        mechanism = AttentionFusionMechanism()
        paradigms = ['P300', 'SSVEP', 'MI']
        
        mechanism.initialize_attention(paradigms)
        
        assert len(mechanism.attention_weights) == 3
        assert len(mechanism.learned_embeddings) == 3
        for paradigm in paradigms:
            assert paradigm in mechanism.attention_weights
            assert paradigm in mechanism.learned_embeddings
    
    def test_compute_attention_empty_data(self):
        """Test attention computation with empty data."""
        mechanism = AttentionFusionMechanism()
        
        attention_scores = mechanism.compute_attention([])
        
        assert attention_scores == {}
    
    def test_compute_attention_valid_data(self):
        """Test attention computation with valid data."""
        mechanism = AttentionFusionMechanism()
        
        modality_data = [
            ModalityData(
                paradigm='P300',
                features=np.random.normal(0, 1, 50),
                confidence=0.8,
                timestamp=time.time()
            ),
            ModalityData(
                paradigm='SSVEP',
                features=np.random.normal(0, 1, 50),
                confidence=0.6,
                timestamp=time.time()
            )
        ]
        
        attention_scores = mechanism.compute_attention(modality_data)
        
        assert isinstance(attention_scores, dict)
        assert len(attention_scores) == 2
        assert 'P300' in attention_scores
        assert 'SSVEP' in attention_scores
        
        # Check normalization
        total_attention = sum(attention_scores.values())
        assert abs(total_attention - 1.0) < 1e-6
    
    def test_multimodal_fusion_initialization(self):
        """Test multimodal fusion system initialization."""
        config = FusionConfig(
            fusion_strategy='attention',
            attention_heads=3,
            confidence_threshold=0.5
        )
        fusion_system = AdvancedMultimodalFusion(config)
        
        assert fusion_system.config == config
        assert fusion_system.attention_mechanism.n_heads == 3
    
    def test_fusion_initialization(self):
        """Test fusion system initialization with paradigms."""
        fusion_system = AdvancedMultimodalFusion(FusionConfig())
        paradigms = ['P300', 'SSVEP', 'MI']
        
        fusion_system.initialize_fusion(paradigms)
        
        assert len(fusion_system.paradigm_weights) == 3
        for paradigm in paradigms:
            assert paradigm in fusion_system.paradigm_weights
            assert abs(fusion_system.paradigm_weights[paradigm] - 1/3) < 1e-6
    
    def test_fuse_modalities_empty_data(self):
        """Test fusion with empty modality data."""
        fusion_system = AdvancedMultimodalFusion(FusionConfig())
        
        result = fusion_system.fuse_modalities([])
        
        assert result.prediction is None
        assert result.confidence == 0.0
        assert result.paradigm_weights == {}
        assert result.uncertainty_score == 1.0
        assert result.fusion_quality == 0.0
        assert result.contributing_paradigms == []
    
    def test_fuse_modalities_valid_data(self):
        """Test fusion with valid modality data."""
        config = FusionConfig(
            fusion_strategy='attention',
            confidence_threshold=0.3,
            min_paradigms=1
        )
        fusion_system = AdvancedMultimodalFusion(config)
        
        # Initialize fusion
        paradigms = ['P300', 'SSVEP']
        fusion_system.initialize_fusion(paradigms)
        
        # Create modality data
        modality_data = [
            ModalityData(
                paradigm='P300',
                features=np.array([0.8, 0.6, 0.4]),
                confidence=0.7,
                timestamp=time.time()
            ),
            ModalityData(
                paradigm='SSVEP',
                features=np.array([0.5, 0.9, 0.3]),
                confidence=0.8,
                timestamp=time.time()
            )
        ]
        
        result = fusion_system.fuse_modalities(modality_data)
        
        assert result.prediction is not None
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.paradigm_weights) > 0
        assert 0.0 <= result.uncertainty_score <= 1.0
        assert 0.0 <= result.fusion_quality <= 1.0
        assert len(result.contributing_paradigms) == 2
    
    def test_fusion_strategies(self):
        """Test different fusion strategies."""
        strategies = ['attention', 'weighted', 'voting', 'cascade']
        
        for strategy in strategies:
            config = FusionConfig(fusion_strategy=strategy, min_paradigms=1)
            fusion_system = AdvancedMultimodalFusion(config)
            fusion_system.initialize_fusion(['P300'])
            
            modality_data = [
                ModalityData(
                    paradigm='P300',
                    features=np.array([0.5]),
                    confidence=0.8,
                    timestamp=time.time()
                )
            ]
            
            result = fusion_system.fuse_modalities(modality_data)
            
            assert result.prediction is not None
            assert result.confidence > 0
    
    def test_create_multimodal_fusion_system(self):
        """Test factory function for multimodal fusion."""
        config = {
            'fusion_strategy': 'weighted',
            'confidence_threshold': 0.8,
            'attention_heads': 6
        }
        
        system = create_multimodal_fusion_system(config)
        
        assert isinstance(system, AdvancedMultimodalFusion)
        assert system.config.fusion_strategy == 'weighted'
        assert system.config.confidence_threshold == 0.8
        assert system.config.attention_heads == 6


class TestExplainableNeuralAI:
    """Test explainable neural AI system."""
    
    def test_saliency_mapper_initialization(self):
        """Test saliency mapper initialization."""
        mapper = NeuralSaliencyMapper(temporal_resolution=0.05, spatial_resolution=16)
        
        assert mapper.temporal_resolution == 0.05
        assert mapper.spatial_resolution == 16
        assert len(mapper.baseline_signals) == 0
    
    def test_compute_saliency_map_empty_signals(self):
        """Test saliency map computation with empty signals."""
        mapper = NeuralSaliencyMapper()
        
        # Mock decoder function
        decoder_func = Mock(return_value=0.5)
        
        saliency_map = mapper.compute_saliency_map(
            np.array([]), 
            0.5, 
            decoder_func
        )
        
        assert saliency_map.size == 0
    
    def test_compute_saliency_map_valid_signals(self):
        """Test saliency map computation with valid signals."""
        mapper = NeuralSaliencyMapper()
        
        # Create test signals
        neural_signals = np.random.normal(0, 1, (4, 100))
        
        # Mock decoder function
        def mock_decoder(signals):
            return np.mean(signals)
        
        saliency_map = mapper.compute_saliency_map(
            neural_signals,
            np.mean(neural_signals),
            mock_decoder
        )
        
        assert saliency_map.shape == neural_signals.shape
        assert np.max(np.abs(saliency_map)) <= 1.0  # Should be normalized
    
    def test_temporal_attribution(self):
        """Test temporal attribution computation."""
        mapper = NeuralSaliencyMapper()
        
        # Create test saliency map
        saliency_map = np.random.normal(0, 1, (4, 100))
        
        temporal_attribution = mapper.compute_temporal_attribution(
            np.random.normal(0, 1, (4, 100)),
            saliency_map
        )
        
        assert len(temporal_attribution) == 100
        assert np.all(temporal_attribution >= 0)  # Should be absolute values
    
    def test_spatial_attribution(self):
        """Test spatial attribution computation."""
        mapper = NeuralSaliencyMapper()
        
        # Create test saliency map
        saliency_map = np.random.normal(0, 1, (4, 100))
        
        spatial_attribution = mapper.compute_spatial_attribution(
            np.random.normal(0, 1, (4, 100)),
            saliency_map
        )
        
        assert len(spatial_attribution) == 4
        assert np.all(spatial_attribution >= 0)  # Should be absolute values
    
    def test_explainable_ai_initialization(self):
        """Test explainable AI system initialization."""
        config = ExplanationConfig(
            explanation_methods=['saliency'],
            temporal_resolution=0.2,
            confidence_threshold=0.8
        )
        explainer = ExplainableNeuralAI(config)
        
        assert explainer.config == config
        assert explainer.saliency_mapper.temporal_resolution == 0.2
    
    def test_generate_explanation_empty_signals(self):
        """Test explanation generation with empty signals."""
        explainer = ExplainableNeuralAI(ExplanationConfig())
        
        # Mock decoder function
        decoder_func = Mock(return_value=0.5)
        
        explanation = explainer.generate_explanation(
            np.array([]),
            0.5,
            decoder_func
        )
        
        assert explanation.prediction == 0.5
        assert explanation.confidence == 0.0
        assert explanation.explanation_quality == 0.0
    
    def test_generate_explanation_valid_signals(self):
        """Test explanation generation with valid signals."""
        config = ExplanationConfig(explanation_methods=['saliency'])
        explainer = ExplainableNeuralAI(config)
        
        # Create test data
        neural_signals = np.random.normal(0, 1, (4, 100))
        prediction = 0.8
        
        # Mock decoder function
        def mock_decoder(signals):
            return np.mean(signals) + 0.1
        
        # Mock features and context
        features = {
            'mean_amplitude': np.mean(neural_signals),
            'signal_variance': np.var(neural_signals)
        }
        context = {
            'timestamp': time.time(),
            'user_state': 'focused'
        }
        
        explanation = explainer.generate_explanation(
            neural_signals,
            prediction,
            mock_decoder,
            features,
            context
        )
        
        assert explanation.prediction == prediction
        assert isinstance(explanation.confidence, float)
        assert 0.0 <= explanation.confidence <= 1.0
        assert explanation.saliency_map is not None
        assert explanation.saliency_map.shape == neural_signals.shape
        assert explanation.temporal_attribution is not None
        assert explanation.spatial_attribution is not None
        assert isinstance(explanation.feature_importance, dict)
        assert isinstance(explanation.causal_factors, dict)
        assert isinstance(explanation.uncertainty_sources, list)
        assert 0.0 <= explanation.explanation_quality <= 1.0
    
    def test_explanation_summary(self):
        """Test explanation summary generation."""
        explainer = ExplainableNeuralAI(ExplanationConfig())
        
        # Create mock explanation
        neural_signals = np.random.normal(0, 1, (2, 50))
        prediction = [0.7, 0.3]
        
        def mock_decoder(signals):
            return [0.6, 0.4]
        
        explanation = explainer.generate_explanation(
            neural_signals,
            prediction,
            mock_decoder
        )
        
        summary = explainer.get_explanation_summary(explanation)
        
        assert 'prediction' in summary
        assert 'confidence' in summary
        assert 'quality' in summary
        assert 'timestamp' in summary
        assert isinstance(summary['prediction'], list)
    
    def test_create_explainable_neural_system(self):
        """Test factory function for explainable neural system."""
        config = {
            'explanation_methods': ['saliency', 'attention'],
            'temporal_resolution': 0.15,
            'confidence_threshold': 0.75
        }
        
        system = create_explainable_neural_system(config)
        
        assert isinstance(system, ExplainableNeuralAI)
        assert system.config.explanation_methods == ['saliency', 'attention']
        assert system.config.temporal_resolution == 0.15
        assert system.config.confidence_threshold == 0.75


class TestGeneration4Integration:
    """Test integration between Generation 4 modules."""
    
    def test_adaptive_calibration_with_multimodal_fusion(self):
        """Test integration of adaptive calibration with multimodal fusion."""
        # Create systems
        calibration_system = create_adaptive_calibration_system()
        fusion_system = create_multimodal_fusion_system()
        
        # Initialize calibration
        init_data = np.random.normal(0, 1, (50, 8))
        labels = np.random.randint(0, 2, 50)
        calibration_system.initialize_calibration(init_data, labels)
        
        # Initialize fusion
        fusion_system.initialize_fusion(['P300', 'SSVEP'])
        
        # Test integration
        new_data = np.random.normal(0, 1, (10, 8))
        adaptation_metrics = calibration_system.adapt_calibration(new_data)
        
        # Create modality data from adaptation
        modality_data = [
            ModalityData(
                paradigm='P300',
                features=new_data[0],
                confidence=adaptation_metrics.calibration_confidence,
                timestamp=time.time()
            )
        ]
        
        fusion_result = fusion_system.fuse_modalities(modality_data)
        
        assert fusion_result.confidence > 0
        assert len(fusion_result.contributing_paradigms) > 0
    
    def test_explainable_ai_with_fusion_results(self):
        """Test explainable AI with fusion results."""
        # Create systems
        explainer = create_explainable_neural_system()
        fusion_system = create_multimodal_fusion_system()
        
        # Initialize fusion
        fusion_system.initialize_fusion(['P300'])
        
        # Create test data
        neural_signals = np.random.normal(0, 1, (4, 100))
        
        modality_data = [
            ModalityData(
                paradigm='P300',
                features=neural_signals.flatten()[:50],
                confidence=0.8,
                timestamp=time.time()
            )
        ]
        
        fusion_result = fusion_system.fuse_modalities(modality_data)
        
        # Generate explanation
        def mock_decoder(signals):
            return fusion_result.prediction
        
        explanation = explainer.generate_explanation(
            neural_signals,
            fusion_result.prediction,
            mock_decoder,
            context={'fusion_quality': fusion_result.fusion_quality}
        )
        
        assert explanation.prediction is not None
        assert explanation.confidence > 0
        assert explanation.explanation_quality > 0
    
    def test_end_to_end_generation4_pipeline(self):
        """Test complete Generation 4 pipeline."""
        # Create all systems
        calibration_system = create_adaptive_calibration_system({
            'n_components': 3,
            'adaptation_rate': 0.02
        })
        
        fusion_system = create_multimodal_fusion_system({
            'fusion_strategy': 'attention',
            'confidence_threshold': 0.5
        })
        
        explainer = create_explainable_neural_system({
            'explanation_methods': ['saliency'],
            'confidence_threshold': 0.6
        })
        
        # Initialize systems
        paradigms = ['P300', 'SSVEP']
        fusion_system.initialize_fusion(paradigms)
        
        # Initialize calibration
        init_data = np.random.normal(0, 1, (100, 8))
        labels = np.random.randint(0, 2, 100)
        calibration_system.initialize_calibration(init_data, labels)
        
        # Simulate real-time processing
        test_signals = np.random.normal(0, 1, (8, 250))
        
        # Step 1: Adaptive calibration
        adaptation_metrics = calibration_system.adapt_calibration(test_signals)
        
        # Step 2: Create multimodal data
        modality_data = []
        for i, paradigm in enumerate(paradigms):
            data = ModalityData(
                paradigm=paradigm,
                features=test_signals[i*2:(i+1)*2].flatten(),
                confidence=0.7 + i * 0.1,
                timestamp=time.time()
            )
            modality_data.append(data)
        
        # Step 3: Multimodal fusion
        fusion_result = fusion_system.fuse_modalities(modality_data)
        
        # Step 4: Generate explanation
        def mock_decoder(signals):
            return fusion_result.prediction
        
        explanation = explainer.generate_explanation(
            test_signals,
            fusion_result.prediction,
            mock_decoder,
            context={
                'adaptation_metrics': adaptation_metrics,
                'fusion_quality': fusion_result.fusion_quality
            }
        )
        
        # Verify pipeline results
        assert adaptation_metrics.plasticity_score >= 0
        assert fusion_result.confidence > 0
        assert explanation.explanation_quality > 0
        assert len(explanation.uncertainty_sources) >= 0
        
        # Test system summaries
        calibration_summary = calibration_system.get_adaptation_summary()
        fusion_stats = fusion_system.get_fusion_statistics()
        explanation_stats = explainer.get_global_explanation_stats()
        
        assert isinstance(calibration_summary, dict)
        assert isinstance(fusion_stats, dict)
        assert isinstance(explanation_stats, dict)