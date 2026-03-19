"""
IntentClassifier: Classify mental state from EEG spectral features.

Mental intents:
  FOCUS   — elevated beta, reduced alpha  (active concentration)
  RELAX   — elevated alpha               (relaxed, idle)
  LEFT    — left motor imagery: beta suppression left hemisphere
  RIGHT   — right motor imagery: beta suppression right hemisphere
  SELECT  — elevated theta               (working memory / decision)

Uses a simple linear (OvR softmax) classifier trained on synthetic labeled data.
Training happens automatically at construction time.

Differential privacy: Gaussian noise (ε, δ)-DP is injected into the feature
vector before classification to protect neural fingerprinting.
"""

from __future__ import annotations

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

from .eeg_processor import EEGFeatures


# ── Intents ───────────────────────────────────────────────────────────────────

class Intent(Enum):
    FOCUS  = "FOCUS"
    RELAX  = "RELAX"
    LEFT   = "LEFT"
    RIGHT  = "RIGHT"
    SELECT = "SELECT"

_INTENT_LIST = list(Intent)
_N_CLASSES = len(_INTENT_LIST)


# ── Classification result ─────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    intent: Intent
    confidence: float          # Softmax probability of winning class
    probabilities: np.ndarray  # (n_classes,) softmax probs
    privacy_applied: bool

    def __repr__(self) -> str:
        return (
            f"ClassificationResult(intent={self.intent.value}, "
            f"confidence={self.confidence:.2%}, "
            f"privacy={'on' if self.privacy_applied else 'off'})"
        )


# ── Synthetic data generation ─────────────────────────────────────────────────

def _generate_synthetic_training_data(
    n_channels: int = 8,
    n_samples_per_class: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (feature, label) training pairs.

    Feature vector layout (from EEGSignalProcessor):
      [theta×n_ch | alpha×n_ch | beta×n_ch]

    Each intent has a characteristic spectral signature:
      - FOCUS:  high beta, moderate alpha, low theta
      - RELAX:  high alpha, low beta, low theta
      - LEFT:   beta suppressed in right-hemisphere channels (upper half)
      - RIGHT:  beta suppressed in left-hemisphere channels (lower half)
      - SELECT: elevated theta, moderate alpha
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_features = 3 * n_channels  # theta + alpha + beta
    ch = n_channels
    half = ch // 2  # split for lateralisation

    def _make_sample(base: np.ndarray, noise_scale: float = 0.15) -> np.ndarray:
        return base + rng.normal(0, noise_scale, n_features)

    # Base power profiles (log-scale, arbitrary units)
    # Index layout: [theta:0..ch-1, alpha:ch..2ch-1, beta:2ch..3ch-1]
    profiles = {
        Intent.FOCUS: lambda: np.concatenate([
            rng.uniform(0.1, 0.3, ch),   # low theta
            rng.uniform(0.2, 0.5, ch),   # moderate alpha
            rng.uniform(0.7, 1.0, ch),   # high beta
        ]),
        Intent.RELAX: lambda: np.concatenate([
            rng.uniform(0.1, 0.3, ch),   # low theta
            rng.uniform(0.7, 1.0, ch),   # high alpha
            rng.uniform(0.1, 0.3, ch),   # low beta
        ]),
        Intent.LEFT: lambda: np.concatenate([
            rng.uniform(0.2, 0.4, ch),
            rng.uniform(0.3, 0.6, ch),
            np.concatenate([             # beta: high left, suppressed right
                rng.uniform(0.6, 1.0, half),
                rng.uniform(0.1, 0.3, ch - half),
            ]),
        ]),
        Intent.RIGHT: lambda: np.concatenate([
            rng.uniform(0.2, 0.4, ch),
            rng.uniform(0.3, 0.6, ch),
            np.concatenate([             # beta: suppressed left, high right
                rng.uniform(0.1, 0.3, half),
                rng.uniform(0.6, 1.0, ch - half),
            ]),
        ]),
        Intent.SELECT: lambda: np.concatenate([
            rng.uniform(0.7, 1.0, ch),   # high theta
            rng.uniform(0.3, 0.6, ch),   # moderate alpha
            rng.uniform(0.2, 0.5, ch),   # moderate beta
        ]),
    }

    X_parts, y_parts = [], []
    for label_idx, intent in enumerate(_INTENT_LIST):
        samples = np.array([profiles[intent]() for _ in range(n_samples_per_class)])
        X_parts.append(samples)
        y_parts.append(np.full(n_samples_per_class, label_idx, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ── Linear softmax classifier ─────────────────────────────────────────────────

class _LinearSoftmax:
    """Minimal OvR linear classifier with gradient-descent softmax training."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        self.W = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_shifted = z - z.max(axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / exp_z.sum(axis=-1, keepdims=True)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X @ self.W.T + self.b)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.05,
        epochs: int = 300,
        l2: float = 1e-3,
    ) -> None:
        n = len(y)
        # One-hot
        Y = np.zeros((n, _N_CLASSES))
        Y[np.arange(n), y] = 1.0

        for _ in range(epochs):
            probs = self._forward(X)          # (n, n_cls)
            err = probs - Y                   # (n, n_cls)
            grad_W = err.T @ X / n + l2 * self.W
            grad_b = err.mean(axis=0)
            self.W -= lr * grad_W
            self.b -= lr * grad_b

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """x: (n_features,) → (n_classes,) probabilities."""
        return self._forward(x[np.newaxis, :])[0]


# ── IntentClassifier ──────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Classify EEG spectral features into discrete intents.

    Args:
        n_channels:           Number of EEG channels expected (default 8).
        epsilon:              DP epsilon (privacy level). 0 disables DP.
        delta:                DP delta parameter.
        n_train_per_class:    Synthetic samples per class for training.
        random_seed:          Reproducibility seed.
    """

    def __init__(
        self,
        n_channels: int = 8,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        n_train_per_class: int = 300,
        random_seed: int = 42,
    ) -> None:
        self.n_channels = n_channels
        self.n_features = 3 * n_channels
        self.epsilon = epsilon
        self.delta = delta
        self._rng = np.random.default_rng(random_seed)

        # Differential privacy noise scale (Gaussian mechanism)
        self._dp_scale = self._gaussian_dp_scale(epsilon, delta, sensitivity=1.0)

        # Train classifier on synthetic data
        self._clf = _LinearSoftmax(self.n_features, _N_CLASSES)
        self._train()

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _gaussian_dp_scale(epsilon: float, delta: float, sensitivity: float) -> float:
        if epsilon <= 0:
            return 0.0
        return float(
            np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        )

    def _train(self) -> None:
        X_train, y_train = _generate_synthetic_training_data(
            n_channels=self.n_channels,
            n_samples_per_class=300,
            rng=self._rng,
        )
        # Standardise features
        self._feat_mean = X_train.mean(axis=0)
        self._feat_std  = X_train.std(axis=0) + 1e-8
        X_norm = (X_train - self._feat_mean) / self._feat_std
        self._clf.fit(X_norm, y_train)

    def _normalise(self, feat: np.ndarray) -> np.ndarray:
        return (feat - self._feat_mean) / self._feat_std

    def _apply_dp_noise(self, feat: np.ndarray) -> np.ndarray:
        if self._dp_scale == 0.0:
            return feat
        noise = self._rng.normal(0.0, self._dp_scale, feat.shape)
        return feat + noise

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(
        self, features: EEGFeatures, apply_privacy: bool = True
    ) -> ClassificationResult:
        """
        Classify EEG features into an intent.

        Args:
            features:      EEGFeatures from EEGSignalProcessor.process().
            apply_privacy: If True, inject DP noise before classification.

        Returns:
            ClassificationResult with intent and confidence.
        """
        feat = features.feature_vector.copy()

        if apply_privacy:
            feat = self._apply_dp_noise(feat)

        feat_norm = self._normalise(feat)
        probs = self._clf.predict_proba(feat_norm)

        best_idx = int(np.argmax(probs))
        return ClassificationResult(
            intent=_INTENT_LIST[best_idx],
            confidence=float(probs[best_idx]),
            probabilities=probs,
            privacy_applied=apply_privacy,
        )

    def classify_raw(
        self, feature_vector: np.ndarray, apply_privacy: bool = True
    ) -> ClassificationResult:
        """Classify from a raw feature vector (bypasses EEGFeatures wrapper)."""
        from .eeg_processor import EEGFeatures as _EF
        dummy = _EF(
            band_powers={},
            feature_vector=feature_vector,
            n_channels=self.n_channels,
            sampling_rate=256,
        )
        return self.classify(dummy, apply_privacy=apply_privacy)
