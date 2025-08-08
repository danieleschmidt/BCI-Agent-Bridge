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
from .transformer_decoder import TransformerNeuralDecoder, TransformerConfig
from .hybrid_decoder import HybridMultiParadigmDecoder, HybridConfig, ParadigmType

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