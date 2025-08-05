"""
BCI-Agent-Bridge: Real-time Brain-Computer Interface to LLM bridge.

This package provides tools for translating neural signals into actionable commands
through Claude Flow agents with medical-grade privacy protection.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terraganlabs.com"

from .core.bridge import BCIBridge
from .adapters.claude_flow import ClaudeFlowAdapter
from .decoders.p300 import P300Decoder
from .decoders.motor_imagery import MotorImageryDecoder
from .decoders.ssvep import SSVEPDecoder
from .privacy.differential_privacy import DifferentialPrivacy
from .clinical.trial_manager import ClinicalTrialManager

__all__ = [
    "BCIBridge",
    "ClaudeFlowAdapter", 
    "P300Decoder",
    "MotorImageryDecoder",
    "SSVEPDecoder",
    "DifferentialPrivacy",
    "ClinicalTrialManager",
]