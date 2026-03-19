"""
AgentCommandMapper: Map classified EEG intents to LLM agent actions.

Intent → Action mapping:
  FOCUS   → search       (initiate a knowledge search)
  RELAX   → idle         (no-op / wait)
  LEFT    → previous     (navigate back)
  RIGHT   → next         (navigate forward)
  SELECT  → execute      (confirm / run current selection)

Each AgentCommand includes the intent, action name, a natural-language
instruction, and the raw classification confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .intent_classifier import ClassificationResult, Intent


@dataclass
class AgentCommand:
    """An agent command produced by the BCI pipeline."""
    intent: Intent
    action: str           # machine-readable action token
    instruction: str      # natural language for the LLM agent
    confidence: float
    metadata: Dict        # passthrough extras

    def __repr__(self) -> str:
        return (
            f"AgentCommand(action={self.action!r}, "
            f"instruction={self.instruction!r}, "
            f"confidence={self.confidence:.2%})"
        )


# ── Default mapping ───────────────────────────────────────────────────────────

_DEFAULT_MAP: Dict[Intent, tuple[str, str]] = {
    Intent.FOCUS:  ("search",   "Search for information on the current topic."),
    Intent.RELAX:  ("idle",     "Stand by and wait for the next signal."),
    Intent.LEFT:   ("previous", "Go back to the previous item or context."),
    Intent.RIGHT:  ("next",     "Advance to the next item or suggestion."),
    Intent.SELECT: ("execute",  "Execute or confirm the currently highlighted action."),
}

# Confidence threshold below which we fall back to RELAX/idle
_DEFAULT_CONFIDENCE_THRESHOLD = 0.35


class AgentCommandMapper:
    """
    Map a ClassificationResult to an AgentCommand.

    Args:
        intent_map:           Custom {Intent: (action, instruction)} overrides.
        confidence_threshold: Minimum confidence to act; below → idle.
    """

    def __init__(
        self,
        intent_map: Optional[Dict[Intent, tuple[str, str]]] = None,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._map = dict(_DEFAULT_MAP)
        if intent_map:
            self._map.update(intent_map)
        self.confidence_threshold = confidence_threshold

    def map(
        self,
        result: ClassificationResult,
        metadata: Optional[Dict] = None,
    ) -> AgentCommand:
        """
        Produce an AgentCommand from a ClassificationResult.

        If confidence is below `confidence_threshold` the command is
        down-graded to an *idle* action regardless of the predicted intent.

        Args:
            result:   ClassificationResult from IntentClassifier.
            metadata: Optional dict passed through to AgentCommand.metadata.

        Returns:
            AgentCommand ready to dispatch to an LLM agent.
        """
        if metadata is None:
            metadata = {}

        if result.confidence >= self.confidence_threshold:
            intent = result.intent
        else:
            intent = Intent.RELAX  # low confidence → safe idle

        action, instruction = self._map[intent]
        return AgentCommand(
            intent=intent,
            action=action,
            instruction=instruction,
            confidence=result.confidence,
            metadata={
                "original_intent": result.intent.value,
                "privacy_applied": result.privacy_applied,
                "all_probs": {
                    i.value: float(p)
                    for i, p in zip(list(Intent), result.probabilities)
                },
                **metadata,
            },
        )
