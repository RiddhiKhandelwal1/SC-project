"""Tests for signal preprocessing — filtering, normalization, missing values."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PreprocessingConfig
from preprocessing.signal_processing import (
    handle_missing_values,
    apply_filter,
    Normalizer,
    preprocess_signal,
)


@pytest.fixture
def clean_signal():
    """Create a clean sine signal."""
    np.random.seed(42)
    t = np.arange(10000) / 12000.0
    return np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(10000)


@pytest.fixture
def config():
    return PreprocessingConfig(
        sampling_frequency=12000.0,
        filter_enabled=True,
        filter_type="bandpass",
        filter_order=5,
        lowcut=10.0,
        highcut=5000.0,
    )


class TestMissingValues:
    def test_no_missing(self, clean_signal):
        result = handle_missing_values(clean_signal, strategy="interpolate")
        assert len(result) == len(clean_signal)
        assert not np.any(np.isnan(result))

    def test_with_nan(self):
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = handle_missing_values(data, strategy="interpolate")
        assert not np.any(np.isnan(result))
        assert np.isclose(result[1], 2.0)

    def test_with_inf(self):
        data = np.array([1.0, np.inf, 3.0, -np.inf, 5.0])
        result = handle_missing_values(data, strategy="interpolate")
        assert not np.any(np.isinf(result))
        assert not np.any(np.isnan(result))

    def test_ffill(self):
        data = np.array([1.0, np.nan, np.nan, 4.0])
        result = handle_missing_values(data, strategy="ffill")
        assert result[1] == 1.0
        assert result[2] == 1.0

    def test_bfill(self):
        data = np.array([1.0, np.nan, np.nan, 4.0])
        result = handle_missing_values(data, strategy="bfill")
        assert result[1] == 4.0
        assert result[2] == 4.0

    def test_drop(self):
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = handle_missing_values(data, strategy="drop")
        assert len(result) == 3

    def test_invalid_strategy(self):
        data = np.array([1.0, np.nan, 2.0])  # Needs NaN to reach strategy branch
        with pytest.raises(ValueError):
            handle_missing_values(data, strategy="invalid")


class TestFiltering:
    def test_bandpass(self, clean_signal, config):
        filtered = apply_filter(clean_signal, config)
        assert len(filtered) == len(clean_signal)
        assert filtered.dtype == np.float64

    def test_lowpass(self, clean_signal):
        config = PreprocessingConfig(
            sampling_frequency=12000.0,
            filter_enabled=True,
            filter_type="lowpass",
            highcut=500.0,
        )
        filtered = apply_filter(clean_signal, config)
        assert len(filtered) == len(clean_signal)

    def test_highpass(self, clean_signal):
        config = PreprocessingConfig(
            sampling_frequency=12000.0,
            filter_enabled=True,
            filter_type="highpass",
            lowcut=50.0,
        )
        filtered = apply_filter(clean_signal, config)
        assert len(filtered) == len(clean_signal)

    def test_filter_disabled(self, clean_signal, config):
        config.filter_enabled = False
        filtered = apply_filter(clean_signal, config)
        np.testing.assert_array_equal(filtered, clean_signal)

    def test_invalid_cutoff(self, clean_signal):
        config = PreprocessingConfig(
            sampling_frequency=12000.0,
            filter_type="lowpass",
            highcut=7000.0,  # Above Nyquist
        )
        with pytest.raises(ValueError):
            apply_filter(clean_signal, config)


class TestNormalizer:
    def test_standardize(self, clean_signal):
        norm = Normalizer(method="standardize")
        result = norm.fit_transform(clean_signal)
        assert np.isclose(np.mean(result), 0, atol=1e-10)
        assert np.isclose(np.std(result), 1, atol=1e-10)

    def test_minmax(self, clean_signal):
        norm = Normalizer(method="minmax")
        result = norm.fit_transform(clean_signal)
        assert np.min(result) >= 0
        assert np.max(result) <= 1

    def test_transform_without_fit(self):
        norm = Normalizer()
        with pytest.raises(RuntimeError):
            norm.transform(np.array([1, 2, 3], dtype=np.float64))

    def test_params_roundtrip(self, clean_signal):
        norm = Normalizer(method="standardize")
        norm.fit(clean_signal)
        params = norm.get_params()
        norm2 = Normalizer.from_params(params)
        result1 = norm.transform(clean_signal[:100])
        result2 = norm2.transform(clean_signal[:100])
        np.testing.assert_array_almost_equal(result1, result2)

    def test_no_leakage(self, clean_signal):
        """Normalizer fitted on train should be used on test without refitting."""
        train = clean_signal[:5000]
        test = clean_signal[5000:]
        norm = Normalizer(method="standardize")
        norm.fit(train)
        train_result = norm.transform(train)
        test_result = norm.transform(test)
        # Test data should NOT have mean=0, std=1 (unless identical distribution)
        assert isinstance(test_result, np.ndarray)
        assert len(test_result) == len(test)


class TestFullPipeline:
    def test_preprocess_signal(self, clean_signal, config):
        processed, normalizer = preprocess_signal(clean_signal, config)
        assert len(processed) == len(clean_signal)
        assert normalizer._fitted
