"""Tests for BCI-Agent-Bridge."""
import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/dschmidt/repos/BCI-Agent-Bridge/src')

from bci_agent_bridge import BCIAgentBridge
from bci_agent_bridge.eeg_processor import EEGSignalProcessor, EEGFeatures
from bci_agent_bridge.intent_classifier import IntentClassifier, Intent
from bci_agent_bridge.command_mapper import AgentCommandMapper


N_CH, SR, N_T = 8, 256, 256


def _eeg(seed=0): return np.random.default_rng(seed).standard_normal((N_CH, N_T))


class TestEEGSignalProcessor:
    def setup_method(self): self.proc = EEGSignalProcessor(sampling_rate=SR)

    def test_returns_eeg_features(self):
        assert isinstance(self.proc.process(_eeg()), EEGFeatures)

    def test_has_band_powers(self):
        feats = self.proc.process(_eeg())
        assert hasattr(feats, 'band_powers')
        assert len(feats.band_powers) > 0

    def test_band_names(self):
        feats = self.proc.process(_eeg())
        assert any(b in feats.band_powers for b in ('alpha', 'beta', 'theta'))

    def test_feature_vector_shape(self):
        feats = self.proc.process(_eeg())
        assert feats.feature_vector.ndim == 1
        assert len(feats.feature_vector) > 0

    def test_feature_vector_finite(self):
        feats = self.proc.process(_eeg())
        assert np.all(np.isfinite(feats.feature_vector))

    def test_band_powers_shape(self):
        feats = self.proc.process(_eeg())
        for arr in feats.band_powers.values():
            assert arr.shape == (N_CH,)

    def test_different_inputs(self):
        f1 = self.proc.process(_eeg(seed=0))
        f2 = self.proc.process(_eeg(seed=1))
        assert not np.allclose(f1.feature_vector, f2.feature_vector)


class TestIntentClassifier:
    def setup_method(self):
        self.proc = EEGSignalProcessor(sampling_rate=SR)
        self.clf = IntentClassifier(n_channels=N_CH, random_seed=42)

    def test_returns_result(self):
        feats = self.proc.process(_eeg())
        assert self.clf.classify(feats) is not None

    def test_intent_valid(self):
        feats = self.proc.process(_eeg())
        assert self.clf.classify(feats).intent in Intent

    def test_confidence_range(self):
        feats = self.proc.process(_eeg())
        r = self.clf.classify(feats)
        assert 0.0 <= r.confidence <= 1.0

    def test_privacy_flag(self):
        feats = self.proc.process(_eeg(seed=7))
        r1 = self.clf.classify(feats, apply_privacy=True)
        r2 = self.clf.classify(feats, apply_privacy=False)
        assert r1.intent in Intent and r2.intent in Intent


class TestAgentCommandMapper:
    def setup_method(self):
        self.mapper = AgentCommandMapper()
        self.proc = EEGSignalProcessor(sampling_rate=SR)
        self.clf = IntentClassifier(n_channels=N_CH, random_seed=0)

    def _classify(self, seed=0):
        return self.clf.classify(self.proc.process(_eeg(seed=seed)))

    def test_returns_command(self):
        cmd = self.mapper.map(self._classify())
        assert cmd is not None

    def test_command_has_action(self):
        cmd = self.mapper.map(self._classify())
        assert isinstance(cmd.action, str)


class TestBCIAgentBridge:
    def setup_method(self):
        self.bridge = BCIAgentBridge(n_channels=N_CH, sampling_rate=SR, random_seed=0)

    def test_process_frame(self):
        result = self.bridge.process_frame(_eeg())
        assert result is not None

    def test_intent_valid(self):
        assert self.bridge.process_frame(_eeg()).dominant_intent in Intent

    def test_run_demo_length(self):
        assert len(self.bridge.run_demo(n_frames=4)) == 4

    def test_frame_indices(self):
        for i, r in enumerate(self.bridge.run_demo(n_frames=5)):
            assert r.frame_idx == i

    def test_no_privacy(self):
        results = self.bridge.run_demo(n_frames=3, apply_privacy=False)
        assert all(r.dominant_intent in Intent for r in results)

    def test_eeg_preserved(self):
        eeg = _eeg(seed=99)
        result = self.bridge.process_frame(eeg)
        np.testing.assert_array_equal(result.eeg, eeg)
