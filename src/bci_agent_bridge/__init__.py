"""
BCI-Agent-Bridge: Brain-Computer Interface to LLM Agent Bridge.

Processes simulated EEG/neural signals into agent commands via:
  - Band-pass filtering (alpha/beta/theta bands)
  - FFT-based power spectral feature extraction
  - Linear intent classification (FOCUS, RELAX, LEFT, RIGHT, SELECT)
  - Differential privacy on features
  - Agent command mapping

Usage:
    from bci_agent_bridge import BCIAgentBridge
    bridge = BCIAgentBridge()
    results = bridge.run_demo()
"""

__version__ = "2.0.0"
__author__ = "Daniel Schmidt"

from .eeg_processor import EEGSignalProcessor
from .intent_classifier import IntentClassifier, Intent
from .command_mapper import AgentCommandMapper, AgentCommand
from .bridge import BCIAgentBridge

__all__ = [
    "EEGSignalProcessor",
    "IntentClassifier",
    "Intent",
    "AgentCommandMapper",
    "AgentCommand",
    "BCIAgentBridge",
]
