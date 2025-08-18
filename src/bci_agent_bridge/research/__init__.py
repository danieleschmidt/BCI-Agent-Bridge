"""
Research Enhancement Modules for BCI-Agent-Bridge.

This package contains state-of-the-art research implementations including:
- Quantum-inspired optimization algorithms
- Federated learning frameworks for privacy-preserving collaboration
- Advanced neural architectures for BCI applications
- Generation 4: Adaptive calibration, multimodal fusion, and explainable AI
"""

__version__ = "0.2.0"  # Updated for Generation 4 enhancements

from .quantum_optimization import (
    QuantumNeuralDecoder,
    QuantumConfig,
    VariationalQuantumCircuit,
    QuantumAnnealingOptimizer,
    create_quantum_bci_decoder
)

from .federated_learning import (
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

# Generation 4: Advanced Research Innovations
from .adaptive_neural_calibration import (
    AdaptiveCalibrationEngine,
    NeuralPlasticityDetector,
    AdaptationMetrics,
    create_adaptive_calibration_system
)

from .advanced_multimodal_fusion import (
    AdvancedMultimodalFusion,
    AttentionFusionMechanism,
    CrossModalValidator,
    UncertaintyEstimator,
    ModalityData,
    FusionResult,
    create_multimodal_fusion_system
)

from .explainable_neural_ai import (
    ExplainableNeuralAI,
    NeuralSaliencyMapper,
    FeatureImportanceAnalyzer,
    CausalInferenceEngine,
    NeuralExplanation,
    create_explainable_neural_system
)

__all__ = [
    # Quantum optimization
    "QuantumNeuralDecoder",
    "QuantumConfig", 
    "VariationalQuantumCircuit",
    "QuantumAnnealingOptimizer",
    "create_quantum_bci_decoder",
    
    # Federated learning
    "FederatedServer",
    "FederatedClient", 
    "FederatedConfig",
    "ClientData",
    "ModelUpdate",
    "SecureAggregator",
    "DifferentialPrivacyMechanism",
    "ByzantineDetector",
    "create_federated_bci_system",
    
    # Generation 4: Adaptive calibration
    "AdaptiveCalibrationEngine",
    "NeuralPlasticityDetector",
    "AdaptationMetrics",
    "create_adaptive_calibration_system",
    
    # Generation 4: Multimodal fusion
    "AdvancedMultimodalFusion",
    "AttentionFusionMechanism",
    "CrossModalValidator", 
    "UncertaintyEstimator",
    "ModalityData",
    "FusionResult",
    "create_multimodal_fusion_system",
    
    # Generation 4: Explainable AI
    "ExplainableNeuralAI",
    "NeuralSaliencyMapper",
    "FeatureImportanceAnalyzer",
    "CausalInferenceEngine",
    "NeuralExplanation",
    "create_explainable_neural_system"
]