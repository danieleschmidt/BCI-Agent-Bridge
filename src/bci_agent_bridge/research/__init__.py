"""
Research Enhancement Modules for BCI-Agent-Bridge.

This package contains state-of-the-art research implementations including:
- Quantum-inspired optimization algorithms
- Federated learning frameworks for privacy-preserving collaboration
- Advanced neural architectures for BCI applications
"""

__version__ = "0.1.0"

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
    "create_federated_bci_system"
]