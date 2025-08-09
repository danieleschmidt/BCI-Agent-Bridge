"""
BCI-Agent-Bridge: Real-time Brain-Computer Interface to LLM bridge.

This package provides tools for translating neural signals into actionable commands
through Claude Flow agents with medical-grade privacy protection.

Enhanced with state-of-the-art research capabilities:
- Transformer-based neural decoders for superior accuracy
- Hybrid multi-paradigm decoders for robust performance
- Quantum-inspired optimization algorithms
- Federated learning for privacy-preserving collaboration
"""

__version__ = "0.2.0"  # Incremented for research enhancements
__author__ = "Daniel Schmidt"
__email__ = "daniel@terraganlabs.com"

# Core components
from .core.bridge import BCIBridge
from .adapters.claude_flow import ClaudeFlowAdapter

# Classical decoders
from .decoders.p300 import P300Decoder
from .decoders.motor_imagery import MotorImageryDecoder
from .decoders.ssvep import SSVEPDecoder

# Advanced research decoders (optional imports)
try:
    from .decoders.transformer_decoder import TransformerNeuralDecoder, TransformerConfig
    from .decoders.hybrid_decoder import HybridMultiParadigmDecoder, HybridConfig
    _ADVANCED_DECODERS_AVAILABLE = True
except ImportError:
    # Define stubs for missing decoders
    class TransformerNeuralDecoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("TransformerNeuralDecoder requires PyTorch. Install with: pip install torch")
    
    class TransformerConfig:
        pass
        
    class HybridMultiParadigmDecoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("HybridMultiParadigmDecoder requires PyTorch. Install with: pip install torch")
    
    class HybridConfig:
        pass
        
    _ADVANCED_DECODERS_AVAILABLE = False

# Privacy and compliance
from .privacy.differential_privacy import DifferentialPrivacy

# Clinical and monitoring
from .clinical.trial_manager import ClinicalTrialManager
from .monitoring.health_monitor import HealthMonitor
from .monitoring.metrics_collector import MetricsCollector, BCIMetricsCollector
from .monitoring.alert_manager import AlertManager

# Research modules (optional imports - may require additional dependencies)
try:
    from .research.quantum_optimization import QuantumNeuralDecoder, create_quantum_bci_decoder
    from .research.federated_learning import FederatedServer, FederatedClient, create_federated_bci_system
    _RESEARCH_AVAILABLE = True
except ImportError:
    _RESEARCH_AVAILABLE = False

__all__ = [
    # Core
    "BCIBridge",
    "ClaudeFlowAdapter",
    
    # Classical decoders
    "P300Decoder",
    "MotorImageryDecoder",
    "SSVEPDecoder",
    
    # Advanced decoders
    "TransformerNeuralDecoder",
    "TransformerConfig",
    "HybridMultiParadigmDecoder", 
    "HybridConfig",
    
    # Privacy
    "DifferentialPrivacy",
    
    # Clinical
    "ClinicalTrialManager",
    
    # Monitoring
    "HealthMonitor",
    "MetricsCollector",
    "BCIMetricsCollector",
    "AlertManager",
]

# Add research components if available
if _RESEARCH_AVAILABLE:
    __all__.extend([
        "QuantumNeuralDecoder",
        "create_quantum_bci_decoder",
        "FederatedServer",
        "FederatedClient", 
        "create_federated_bci_system"
    ])