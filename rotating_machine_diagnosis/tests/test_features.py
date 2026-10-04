"""Tests for traditional vibration feature extraction."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fuzzy.features import (
    compute_rms,
    compute_peak_amplitude,
    compute_kurtosis,
    compute_crest_factor,
    compute_skewness,
    compute_std,
    compute_all_features,
)


@pytest.fixture
def sine_signal():
    """Pure sine wave — predictable features."""
    t = np.arange(10000) / 12000.0
    return np.sin(2 * np.pi * 100 * t)


@pytest.fixture
def impulse_signal():
    """Signal with sharp impulses — high kurtosis."""
    np.random.seed(42)
    sig = np.zeros(10000)
    # Add sparse impulses
    for i in range(0, 10000, 500):
        sig[i] = 10.0
    sig += np.random.randn(10000) * 0.01
    return sig


class TestRMS:
    def test_sine_rms(self, sine_signal):
        rms = compute_rms(sine_signal)
        # RMS of sin(x) ≈ 1/sqrt(2) ≈ 0.7071
        assert abs(rms - 1.0 / np.sqrt(2)) < 0.01

    def test_zero_signal(self):
        assert compute_rms(np.zeros(100)) == 0.0

    def test_constant_signal(self):
        rms = compute_rms(np.ones(100) * 3.0)
        assert abs(rms - 3.0) < 1e-10

    def test_positive(self, sine_signal):
        assert compute_rms(sine_signal) > 0


class TestPeakAmplitude:
    def test_sine_peak(self, sine_signal):
        peak = compute_peak_amplitude(sine_signal)
        assert abs(peak - 1.0) < 0.01

    def test_negative_values(self):
        sig = np.array([-5.0, 1.0, 2.0, -3.0])
        assert compute_peak_amplitude(sig) == 5.0


class TestKurtosis:
    def test_normal_distribution(self):
        np.random.seed(42)
        normal = np.random.randn(100000)
        kurt = compute_kurtosis(normal)
        # Excess kurtosis of normal ≈ 0
        assert abs(kurt) < 0.1

    def test_impulse_high_kurtosis(self, impulse_signal):
        kurt = compute_kurtosis(impulse_signal)
        assert kurt > 5  # Impulsive signals have high kurtosis

    def test_uniform_negative_kurtosis(self):
        np.random.seed(42)
        uniform = np.random.uniform(-1, 1, 100000)
        kurt = compute_kurtosis(uniform)
        # Uniform distribution has negative excess kurtosis
        assert kurt < 0


class TestCrestFactor:
    def test_sine_crest_factor(self, sine_signal):
        cf = compute_crest_factor(sine_signal)
        # Crest factor of sine ≈ sqrt(2) ≈ 1.414
        assert abs(cf - np.sqrt(2)) < 0.02

    def test_impulse_high_crest(self, impulse_signal):
        cf = compute_crest_factor(impulse_signal)
        assert cf > 10  # Impulsive → high crest factor

    def test_zero_signal(self):
        assert compute_crest_factor(np.zeros(100)) == 0.0


class TestSkewness:
    def test_symmetric(self, sine_signal):
        skew = compute_skewness(sine_signal)
        assert abs(skew) < 0.1  # Symmetric signal


class TestAllFeatures:
    def test_all_keys(self, sine_signal):
        features = compute_all_features(sine_signal)
        assert "rms" in features
        assert "peak_amplitude" in features
        assert "kurtosis" in features
        assert "crest_factor" in features
        assert "skewness" in features
        assert "std" in features

    def test_with_temperature(self, sine_signal):
        features = compute_all_features(sine_signal, temperature=65.5)
        assert "temperature" in features
        assert features["temperature"] == 65.5

    def test_without_temperature(self, sine_signal):
        features = compute_all_features(sine_signal)
        assert "temperature" not in features

    def test_all_numeric(self, sine_signal):
        features = compute_all_features(sine_signal)
        for k, v in features.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"
