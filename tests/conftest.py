"""
Minimal shared pytest fixtures for BCI-Agent-Bridge.
"""
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_eeg_8ch(rng):
    """8-channel × 4s EEG at 256 Hz."""
    return rng.normal(0, 10, (8, 1024))
