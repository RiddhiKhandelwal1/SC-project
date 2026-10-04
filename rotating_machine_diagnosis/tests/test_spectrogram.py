"""Tests for STFT / spectrogram generation."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import STFTConfig
from preprocessing.spectrogram import (
    compute_stft,
    compute_magnitude_spectrogram,
    compute_power_spectrogram,
    normalize_spectrogram,
    batch_spectrograms,
)
from preprocessing.windowing import VibrationWindow


@pytest.fixture
def window_data():
    """2048-sample sine wave."""
    np.random.seed(42)
    t = np.arange(2048) / 12000.0
    return np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(2048)


@pytest.fixture
def stft_config():
    return STFTConfig(n_fft=256, hop_length=64, win_length=256, window="hann")


class TestSTFT:
    def test_compute_stft(self, window_data, stft_config):
        f, t, Zxx = compute_stft(window_data, stft_config, fs=12000.0)
        assert f.ndim == 1
        assert t.ndim == 1
        assert Zxx.ndim == 2
        assert Zxx.shape[0] == len(f)
        assert Zxx.shape[1] == len(t)
        assert np.iscomplexobj(Zxx)

    def test_frequency_range(self, window_data, stft_config):
        f, _, _ = compute_stft(window_data, stft_config, fs=12000.0)
        assert f[0] >= 0
        assert f[-1] <= 6000.0  # Nyquist

    def test_different_nfft(self, window_data):
        c1 = STFTConfig(n_fft=128, hop_length=32, win_length=128)
        c2 = STFTConfig(n_fft=512, hop_length=128, win_length=512)
        f1, _, Z1 = compute_stft(window_data, c1, fs=12000.0)
        f2, _, Z2 = compute_stft(window_data, c2, fs=12000.0)
        assert Z1.shape != Z2.shape  # Different resolution


class TestMagnitudeSpectrogram:
    def test_shape(self, window_data, stft_config):
        spec = compute_magnitude_spectrogram(window_data, stft_config, fs=12000.0)
        assert spec.ndim == 2
        assert spec.shape[0] > 0
        assert spec.shape[1] > 0

    def test_log_scale(self, window_data, stft_config):
        spec_log = compute_magnitude_spectrogram(
            window_data, stft_config, fs=12000.0, log_scale=True
        )
        spec_lin = compute_magnitude_spectrogram(
            window_data, stft_config, fs=12000.0, log_scale=False
        )
        # Log-scale values should be different
        assert not np.allclose(spec_log, spec_lin)

    def test_no_nan(self, window_data, stft_config):
        spec = compute_magnitude_spectrogram(window_data, stft_config, fs=12000.0)
        assert not np.any(np.isnan(spec))


class TestPowerSpectrogram:
    def test_non_negative(self, window_data, stft_config):
        power = compute_power_spectrogram(window_data, stft_config, fs=12000.0)
        assert np.all(power >= 0)


class TestNormalize:
    def test_normalize_range(self, window_data, stft_config):
        spec = compute_magnitude_spectrogram(window_data, stft_config, fs=12000.0)
        normed = normalize_spectrogram(spec)
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0 + 1e-10

    def test_constant_spectrogram(self):
        const = np.ones((10, 10)) * 5.0
        normed = normalize_spectrogram(const)
        assert np.all(normed == 0.0)  # All same → zeros


class TestBatchSpectrograms:
    def test_batch(self, stft_config):
        windows = [
            VibrationWindow(window_id=i, data=np.random.randn(2048))
            for i in range(5)
        ]
        batch = batch_spectrograms(windows, stft_config, fs=12000.0, normalize=True)
        assert batch.ndim == 4  # (N, F, T, 1)
        assert batch.shape[0] == 5
        assert batch.shape[-1] == 1
        assert batch.min() >= 0.0
        assert batch.max() <= 1.0 + 1e-10
