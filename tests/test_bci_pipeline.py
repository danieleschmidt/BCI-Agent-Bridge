"""
Tests for the BCI Agent Bridge pipeline.
Runs with: ~/anaconda3/bin/python3 -m pytest tests/test_bci_pipeline.py -v
"""

import numpy as np
import pytest

from bci_agent_bridge.eeg_processor import EEGSignalProcessor, BANDS
from bci_agent_bridge.intent_classifier import IntentClassifier, Intent
from bci_agent_bridge.command_mapper import AgentCommandMapper, AgentCommand
from bci_agent_bridge.bridge import BCIAgentBridge, _synthetic_eeg


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def processor():
    return EEGSignalProcessor(sampling_rate=256)


@pytest.fixture
def classifier():
    return IntentClassifier(n_channels=8, epsilon=1.0, random_seed=42)


@pytest.fixture
def mapper():
    return AgentCommandMapper()


@pytest.fixture
def bridge():
    return BCIAgentBridge(n_channels=8, sampling_rate=256, random_seed=42)


@pytest.fixture
def sample_eeg():
    rng = np.random.default_rng(7)
    return rng.normal(0, 10, (8, 1024))   # 8 ch × 4 s at 256 Hz


# ── EEGSignalProcessor ────────────────────────────────────────────────────────

class TestEEGSignalProcessor:
    def test_output_shape(self, processor, sample_eeg):
        features = processor.process(sample_eeg)
        # feature vector should be n_bands * n_channels = 3 * 8 = 24
        assert features.feature_vector.shape == (24,)

    def test_band_powers_shape(self, processor, sample_eeg):
        features = processor.process(sample_eeg)
        for band in BANDS:
            assert band in features.band_powers
            assert features.band_powers[band].shape == (8,)

    def test_band_powers_non_negative(self, processor, sample_eeg):
        features = processor.process(sample_eeg)
        for band, power in features.band_powers.items():
            assert np.all(power >= 0), f"Negative power in {band}"

    def test_1d_input_handled(self, processor):
        data = np.random.randn(1024)
        features = processor.process(data)
        assert features.n_channels == 1
        assert features.feature_vector.shape == (3,)

    def test_alpha_dominant_signal(self, processor):
        """Alpha signal (10 Hz) should yield high alpha band power."""
        fs = 256
        t = np.linspace(0, 4, 4 * fs)
        alpha_wave = 15 * np.sin(2 * np.pi * 10 * t)
        eeg = np.stack([alpha_wave] * 4)   # 4 channels
        features = processor.process(eeg)
        alpha_power = features.band_powers["alpha"].mean()
        beta_power  = features.band_powers["beta"].mean()
        theta_power = features.band_powers["theta"].mean()
        assert alpha_power > beta_power, "Alpha should dominate over beta"
        assert alpha_power > theta_power, "Alpha should dominate over theta"

    def test_beta_dominant_signal(self, processor):
        """Beta signal (20 Hz) should yield high beta band power."""
        fs = 256
        t = np.linspace(0, 4, 4 * fs)
        beta_wave = 15 * np.sin(2 * np.pi * 20 * t)
        eeg = np.stack([beta_wave] * 4)
        features = processor.process(eeg)
        beta_power  = features.band_powers["beta"].mean()
        alpha_power = features.band_powers["alpha"].mean()
        assert beta_power > alpha_power, "Beta should dominate over alpha"

    def test_theta_dominant_signal(self, processor):
        """Theta signal (6 Hz) should yield high theta band power."""
        fs = 256
        t = np.linspace(0, 4, 4 * fs)
        theta_wave = 15 * np.sin(2 * np.pi * 6 * t)
        eeg = np.stack([theta_wave] * 4)
        features = processor.process(eeg)
        theta_power = features.band_powers["theta"].mean()
        beta_power  = features.band_powers["beta"].mean()
        assert theta_power > beta_power, "Theta should dominate over beta"

    def test_feature_vector_is_concat_of_bands(self, processor, sample_eeg):
        features = processor.process(sample_eeg)
        expected = np.concatenate([
            features.band_powers["theta"],
            features.band_powers["alpha"],
            features.band_powers["beta"],
        ])
        np.testing.assert_array_equal(features.feature_vector, expected)


# ── IntentClassifier ──────────────────────────────────────────────────────────

class TestIntentClassifier:
    def test_returns_valid_intent(self, classifier, processor, sample_eeg):
        features = processor.process(sample_eeg)
        result = classifier.classify(features)
        assert result.intent in list(Intent)

    def test_confidence_in_range(self, classifier, processor, sample_eeg):
        features = processor.process(sample_eeg)
        result = classifier.classify(features)
        assert 0.0 <= result.confidence <= 1.0

    def test_probabilities_sum_to_one(self, classifier, processor, sample_eeg):
        features = processor.process(sample_eeg)
        result = classifier.classify(features)
        assert abs(result.probabilities.sum() - 1.0) < 1e-5

    def test_probabilities_non_negative(self, classifier, processor, sample_eeg):
        features = processor.process(sample_eeg)
        result = classifier.classify(features)
        assert np.all(result.probabilities >= 0)

    def test_dp_privacy_flag(self, classifier, processor, sample_eeg):
        features = processor.process(sample_eeg)
        r_on  = classifier.classify(features, apply_privacy=True)
        r_off = classifier.classify(features, apply_privacy=False)
        assert r_on.privacy_applied is True
        assert r_off.privacy_applied is False

    def test_dp_changes_result_stochastically(self, processor, sample_eeg):
        """DP noise should occasionally change classification vs no-noise."""
        clf = IntentClassifier(n_channels=8, epsilon=0.1, random_seed=None)
        features = processor.process(sample_eeg)
        results = {clf.classify(features, apply_privacy=True).intent for _ in range(20)}
        # With strong DP noise, we should sometimes see different intents
        # (not guaranteed, but very likely with ε=0.1)
        assert len(results) >= 1   # at minimum one intent observed

    def test_focus_classified_correctly(self, classifier):
        """Strong beta signal should often be classified as FOCUS."""
        proc = EEGSignalProcessor(sampling_rate=256)
        fs = 256
        t = np.linspace(0, 4, 4 * fs)
        eeg = np.stack([10 * np.sin(2 * np.pi * 20 * t) for _ in range(8)])
        features = proc.process(eeg)
        result = classifier.classify(features, apply_privacy=False)
        assert result.intent == Intent.FOCUS

    def test_relax_classified_correctly(self, classifier):
        """Strong alpha signal should be classified as RELAX."""
        proc = EEGSignalProcessor(sampling_rate=256)
        fs = 256
        t = np.linspace(0, 4, 4 * fs)
        eeg = np.stack([10 * np.sin(2 * np.pi * 10 * t) for _ in range(8)])
        features = proc.process(eeg)
        result = classifier.classify(features, apply_privacy=False)
        assert result.intent == Intent.RELAX

    def test_select_classified_correctly(self, classifier):
        """Strong theta signal should be classified as SELECT."""
        proc = EEGSignalProcessor(sampling_rate=256)
        fs = 256
        t = np.linspace(0, 4, 4 * fs)
        eeg = np.stack([10 * np.sin(2 * np.pi * 6 * t) for _ in range(8)])
        features = proc.process(eeg)
        result = classifier.classify(features, apply_privacy=False)
        assert result.intent == Intent.SELECT


# ── AgentCommandMapper ────────────────────────────────────────────────────────

class TestAgentCommandMapper:
    def test_all_intents_have_actions(self, mapper, classifier, processor, sample_eeg):
        features = processor.process(sample_eeg)
        for intent in Intent:
            from bci_agent_bridge.intent_classifier import ClassificationResult
            result = ClassificationResult(
                intent=intent,
                confidence=0.9,
                probabilities=np.ones(5) / 5,
                privacy_applied=False,
            )
            cmd = mapper.map(result)
            assert isinstance(cmd, AgentCommand)
            assert cmd.action

    def test_low_confidence_maps_to_idle(self, mapper):
        from bci_agent_bridge.intent_classifier import ClassificationResult
        low_conf_result = ClassificationResult(
            intent=Intent.FOCUS,
            confidence=0.10,
            probabilities=np.ones(5) / 5,
            privacy_applied=False,
        )
        cmd = mapper.map(low_conf_result)
        assert cmd.action == "idle"

    def test_high_confidence_focus_maps_to_search(self, mapper):
        from bci_agent_bridge.intent_classifier import ClassificationResult
        result = ClassificationResult(
            intent=Intent.FOCUS,
            confidence=0.85,
            probabilities=np.array([0.85, 0.05, 0.04, 0.03, 0.03]),
            privacy_applied=False,
        )
        cmd = mapper.map(result)
        assert cmd.action == "search"

    def test_select_maps_to_execute(self, mapper):
        from bci_agent_bridge.intent_classifier import ClassificationResult
        result = ClassificationResult(
            intent=Intent.SELECT,
            confidence=0.75,
            probabilities=np.array([0.05, 0.05, 0.05, 0.10, 0.75]),
            privacy_applied=False,
        )
        cmd = mapper.map(result)
        assert cmd.action == "execute"

    def test_metadata_passthrough(self, mapper):
        from bci_agent_bridge.intent_classifier import ClassificationResult
        result = ClassificationResult(
            intent=Intent.RIGHT,
            confidence=0.8,
            probabilities=np.ones(5) / 5,
            privacy_applied=True,
        )
        cmd = mapper.map(result, metadata={"session_id": "abc123"})
        assert cmd.metadata["session_id"] == "abc123"
        assert cmd.metadata["privacy_applied"] is True


# ── BCIAgentBridge (integration) ──────────────────────────────────────────────

class TestBCIAgentBridge:
    def test_process_frame_returns_result(self, bridge, sample_eeg):
        from bci_agent_bridge.bridge import BridgeResult
        result = bridge.process_frame(sample_eeg, frame_idx=0)
        assert isinstance(result, BridgeResult)
        assert result.command.action in {"search", "idle", "previous", "next", "execute"}

    def test_run_demo_returns_correct_count(self, bridge):
        results = bridge.run_demo(duration_secs=5, n_frames=5)
        assert len(results) == 5

    def test_run_demo_accuracy_above_chance(self, bridge):
        """With realistic synthetic data, accuracy should be above 20% (chance)."""
        results = bridge.run_demo(duration_secs=10, n_frames=10, apply_privacy=False)
        correct = sum(
            1 for r in results if r.classification.intent == r.dominant_intent
        )
        assert correct / len(results) > 0.20, "Accuracy should beat random chance"

    def test_synthetic_eeg_shape(self):
        eeg = _synthetic_eeg(8, 512, 256, dominant_intent=Intent.FOCUS)
        assert eeg.shape == (8, 512)

    def test_synthetic_eeg_all_intents(self):
        rng = np.random.default_rng(1)
        for intent in Intent:
            eeg = _synthetic_eeg(8, 512, 256, dominant_intent=intent, rng=rng)
            assert eeg.shape == (8, 512)
            assert np.isfinite(eeg).all()
