"""
BCIAgentBridge: Orchestrates the full BCI → Agent pipeline.

Pipeline:
  raw EEG (n_ch × n_t)
      ↓  EEGSignalProcessor.process()
  EEGFeatures  [spectral power per band]
      ↓  IntentClassifier.classify()   ← DP noise injected here
  ClassificationResult  [intent + confidence]
      ↓  AgentCommandMapper.map()
  AgentCommand  [action + instruction]

Demo:
  bridge = BCIAgentBridge()
  results = bridge.run_demo(duration_secs=10, n_frames=10)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from .eeg_processor import EEGSignalProcessor, EEGFeatures
from .intent_classifier import IntentClassifier, ClassificationResult, Intent
from .command_mapper import AgentCommandMapper, AgentCommand


# ── EEG Synthesiser ───────────────────────────────────────────────────────────

def _synthetic_eeg(
    n_channels: int,
    n_samples: int,
    sampling_rate: int,
    dominant_intent: Optional[Intent] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate synthetic EEG with realistic band structure for a given intent.

    Returns:
        (n_channels, n_samples) float array in µV.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, n_samples / sampling_rate, n_samples)
    eeg = np.zeros((n_channels, n_samples))

    # Base signal: pink-ish noise
    for ch in range(n_channels):
        eeg[ch] = rng.normal(0, 5, n_samples)

    # Overlay characteristic oscillations per intent
    if dominant_intent is None:
        dominant_intent = rng.choice(list(Intent))

    if dominant_intent == Intent.FOCUS:
        # Strong beta (20 Hz)
        for ch in range(n_channels):
            eeg[ch] += 8 * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 2 * np.pi))

    elif dominant_intent == Intent.RELAX:
        # Strong alpha (10 Hz)
        for ch in range(n_channels):
            eeg[ch] += 10 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))

    elif dominant_intent == Intent.LEFT:
        # Beta suppressed right hemisphere (upper half channels)
        half = n_channels // 2
        for ch in range(half):          # left channels — strong beta
            eeg[ch] += 8 * np.sin(2 * np.pi * 18 * t + rng.uniform(0, 2 * np.pi))
        for ch in range(half, n_channels):  # right channels — weak beta
            eeg[ch] += 2 * np.sin(2 * np.pi * 18 * t + rng.uniform(0, 2 * np.pi))

    elif dominant_intent == Intent.RIGHT:
        # Beta suppressed left hemisphere
        half = n_channels // 2
        for ch in range(half):          # left channels — weak beta
            eeg[ch] += 2 * np.sin(2 * np.pi * 18 * t + rng.uniform(0, 2 * np.pi))
        for ch in range(half, n_channels):  # right channels — strong beta
            eeg[ch] += 8 * np.sin(2 * np.pi * 18 * t + rng.uniform(0, 2 * np.pi))

    elif dominant_intent == Intent.SELECT:
        # Strong theta (6 Hz)
        for ch in range(n_channels):
            eeg[ch] += 9 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))

    return eeg


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class BridgeResult:
    """Single frame result from the bridge pipeline."""
    frame_idx: int
    dominant_intent: Intent          # ground-truth for demo
    eeg: np.ndarray                  # raw EEG frame
    features: EEGFeatures
    classification: ClassificationResult
    command: AgentCommand

    def summary(self) -> str:
        correct = "✓" if self.classification.intent == self.dominant_intent else "✗"
        return (
            f"Frame {self.frame_idx:02d} | "
            f"GT={self.dominant_intent.value:<6} | "
            f"Pred={self.classification.intent.value:<6} {correct} | "
            f"Conf={self.classification.confidence:.2%} | "
            f"Action={self.command.action!r}"
        )


# ── Bridge ────────────────────────────────────────────────────────────────────

class BCIAgentBridge:
    """
    Orchestrates EEG → features → intent → agent command.

    Args:
        n_channels:    Number of EEG channels (default 8).
        sampling_rate: EEG sampling rate in Hz (default 256).
        epsilon:       Differential privacy ε (0 to disable).
        delta:         Differential privacy δ.
        confidence_threshold: Minimum confidence to issue non-idle command.
        random_seed:   Reproducibility seed.
    """

    def __init__(
        self,
        n_channels: int = 8,
        sampling_rate: int = 256,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        confidence_threshold: float = 0.35,
        random_seed: int = 0,
    ) -> None:
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self._rng = np.random.default_rng(random_seed)

        self.processor = EEGSignalProcessor(sampling_rate=sampling_rate)
        self.classifier = IntentClassifier(
            n_channels=n_channels,
            epsilon=epsilon,
            delta=delta,
            random_seed=random_seed,
        )
        self.mapper = AgentCommandMapper(confidence_threshold=confidence_threshold)

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def process_frame(
        self,
        raw_eeg: np.ndarray,
        metadata: Optional[dict] = None,
        apply_privacy: bool = True,
        frame_idx: int = 0,
        ground_truth_intent: Optional[Intent] = None,
    ) -> BridgeResult:
        """
        Run a single EEG frame through the full pipeline.

        Args:
            raw_eeg:              (n_channels, n_timepoints) array.
            metadata:             Optional passthrough dict.
            apply_privacy:        If True, inject DP noise before classification.
            frame_idx:            Frame counter for display.
            ground_truth_intent:  Optional GT label (demo use only).

        Returns:
            BridgeResult with all intermediate outputs.
        """
        features = self.processor.process(raw_eeg)
        classification = self.classifier.classify(features, apply_privacy=apply_privacy)
        command = self.mapper.map(classification, metadata=metadata)

        return BridgeResult(
            frame_idx=frame_idx,
            dominant_intent=ground_truth_intent or classification.intent,
            eeg=raw_eeg,
            features=features,
            classification=classification,
            command=command,
        )

    # ── Demo ──────────────────────────────────────────────────────────────────

    def run_demo(
        self,
        duration_secs: float = 10.0,
        n_frames: int = 10,
        apply_privacy: bool = True,
    ) -> List[BridgeResult]:
        """
        Generate synthetic EEG for `duration_secs` seconds, split into
        `n_frames` equal windows and classify each.

        Args:
            duration_secs: Total simulated recording length.
            n_frames:      Number of analysis windows.
            apply_privacy: Enable differential privacy.

        Returns:
            List of BridgeResult, one per frame.
        """
        total_samples = int(duration_secs * self.sampling_rate)
        samples_per_frame = total_samples // n_frames
        intents = list(Intent)
        results: List[BridgeResult] = []

        print(f"\n{'─'*70}")
        print(f"  BCIAgentBridge Demo  |  {n_frames} frames × "
              f"{samples_per_frame/self.sampling_rate:.1f}s  |  "
              f"DP ε={self.classifier.epsilon}")
        print(f"{'─'*70}")

        for i in range(n_frames):
            # Round-robin over intents for demo variety
            gt_intent = intents[i % len(intents)]

            raw_eeg = _synthetic_eeg(
                n_channels=self.n_channels,
                n_samples=samples_per_frame,
                sampling_rate=self.sampling_rate,
                dominant_intent=gt_intent,
                rng=self._rng,
            )

            result = self.process_frame(
                raw_eeg,
                apply_privacy=apply_privacy,
                frame_idx=i,
                ground_truth_intent=gt_intent,
            )
            results.append(result)
            print(result.summary())

        # Summary stats
        correct = sum(
            1 for r in results if r.classification.intent == r.dominant_intent
        )
        print(f"{'─'*70}")
        print(f"  Accuracy: {correct}/{n_frames} ({correct/n_frames:.0%})")
        print(f"  Privacy:  {'Gaussian DP active' if apply_privacy else 'disabled'}")
        print(f"{'─'*70}\n")

        return results
