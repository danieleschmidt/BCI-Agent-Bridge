"""
Neural signal decoders for different BCI paradigms.

Includes both classical and state-of-the-art decoders:
- Classical: P300, Motor Imagery, SSVEP decoders
- Advanced: Transformer-based neural decoders
- Hybrid: Multi-paradigm adaptive fusion decoders
"""

from .base import BaseDecoder
from .p300 import P300Decoder
from .motor_imagery import MotorImageryDecoder
from .ssvep import SSVEPDecoder
# Optional imports for advanced decoders that require PyTorch
try:
    from .transformer_decoder import TransformerNeuralDecoder, TransformerConfig
    from .hybrid_decoder import HybridMultiParadigmDecoder, HybridConfig, ParadigmType
    _ADVANCED_DECODERS_AVAILABLE = True
except ImportError:
    # PyTorch not available, skip advanced decoders
    _ADVANCED_DECODERS_AVAILABLE = False
    
    # Define stub classes for compatibility
    class TransformerNeuralDecoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("TransformerNeuralDecoder requires PyTorch. Install with: pip install torch")
    
    class TransformerConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("TransformerConfig requires PyTorch. Install with: pip install torch")
    
    class HybridMultiParadigmDecoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("HybridMultiParadigmDecoder requires PyTorch. Install with: pip install torch")
    
    class HybridConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("HybridConfig requires PyTorch. Install with: pip install torch")
    
    class ParadigmType:
        def __init__(self, *args, **kwargs):
            raise ImportError("ParadigmType requires PyTorch. Install with: pip install torch")

__all__ = [
    # Base and classical decoders
    "BaseDecoder",
    "P300Decoder",
    "MotorImageryDecoder", 
    "SSVEPDecoder",
    
    # Advanced transformer decoders
    "TransformerNeuralDecoder",
    "TransformerConfig",
    
    # Hybrid multi-paradigm decoders
    "HybridMultiParadigmDecoder",
    "HybridConfig",
    "ParadigmType"
]