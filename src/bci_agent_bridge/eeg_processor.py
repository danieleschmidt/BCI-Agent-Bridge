"""
EEGSignalProcessor: Band-pass filter raw EEG and extract power spectral features.

Pipeline:
  1. Apply band-pass filters for alpha (8-13 Hz), beta (13-30 Hz), theta (4-8 Hz)
  2. Compute FFT-based power spectral density per band per channel
  3. Return flat feature vector: [theta_ch0..n, alpha_ch0..n, beta_ch0..n]
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from dataclasses import dataclass, field
from typing import Dict, Optional


# ── Band definitions ──────────────────────────────────────────────────────────
BANDS: Dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}


@dataclass
class EEGFeatures:
    """Container for extracted EEG spectral features."""
    band_powers: Dict[str, np.ndarray]   # band → (n_channels,) power values
    feature_vector: np.ndarray            # flat concatenation, shape (n_bands*n_channels,)
    n_channels: int
    sampling_rate: int
    band_names: list[str] = field(default_factory=lambda: list(BANDS.keys()))

    def __repr__(self) -> str:
        shapes = {b: arr.shape for b, arr in self.band_powers.items()}
        return (
            f"EEGFeatures(n_channels={self.n_channels}, "
            f"fs={self.sampling_rate}, bands={shapes})"
        )


class EEGSignalProcessor:
    """
    Process raw EEG time-series data and extract spectral features.

    Args:
        sampling_rate: Sample rate in Hz (default 256).
        filter_order:  Butterworth filter order (default 4).
    """

    def __init__(self, sampling_rate: int = 256, filter_order: int = 4) -> None:
        self.sampling_rate = sampling_rate
        self.filter_order = filter_order
        self._filters: Dict[str, tuple] = {}
        self._build_filters()

    # ── Filter design ─────────────────────────────────────────────────────────

    def _build_filters(self) -> None:
        nyq = self.sampling_rate / 2.0
        for name, (lo, hi) in BANDS.items():
            lo_n = lo / nyq
            hi_n = min(hi / nyq, 0.99)
            b, a = sp_signal.butter(self.filter_order, [lo_n, hi_n], btype="band")
            self._filters[name] = (b, a)

    def _bandpass(self, data: np.ndarray, band: str) -> np.ndarray:
        """Apply zero-phase band-pass filter for `band`.  data: (n_ch, n_t)."""
        b, a = self._filters[band]
        # Need at least padlen samples — scipy default is 3*(max(len(a),len(b))-1)
        padlen = 3 * (max(len(a), len(b)) - 1)
        if data.shape[1] <= padlen:
            # Too short to filter properly — return raw
            return data.copy()
        return sp_signal.filtfilt(b, a, data, axis=1)

    # ── Power spectral density ────────────────────────────────────────────────

    def _band_power_fft(self, data: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """
        Compute mean power in [lo, hi] Hz per channel using FFT.

        Args:
            data: (n_channels, n_timepoints) after band-pass filtering.
            lo, hi: Frequency bounds in Hz.

        Returns:
            (n_channels,) array of mean power values.
        """
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sampling_rate)
        fft_mag = np.abs(np.fft.rfft(data, axis=1)) ** 2  # (n_ch, n_freq)

        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            return np.zeros(data.shape[0])

        return fft_mag[:, mask].mean(axis=1)

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, raw_eeg: np.ndarray) -> EEGFeatures:
        """
        Full pipeline: filter → FFT power extraction.

        Args:
            raw_eeg: (n_channels, n_timepoints) numpy array, float μV.

        Returns:
            EEGFeatures with per-band power and flat feature vector.
        """
        if raw_eeg.ndim == 1:
            raw_eeg = raw_eeg[np.newaxis, :]

        n_channels = raw_eeg.shape[0]
        band_powers: Dict[str, np.ndarray] = {}

        for band, (lo, hi) in BANDS.items():
            filtered = self._bandpass(raw_eeg, band)
            power = self._band_power_fft(filtered, lo, hi)
            band_powers[band] = power

        # Feature vector: concat bands in fixed order [theta, alpha, beta]
        feature_vector = np.concatenate(
            [band_powers[b] for b in BANDS.keys()], axis=0
        )  # shape: (n_bands * n_channels,)

        return EEGFeatures(
            band_powers=band_powers,
            feature_vector=feature_vector,
            n_channels=n_channels,
            sampling_rate=self.sampling_rate,
        )
