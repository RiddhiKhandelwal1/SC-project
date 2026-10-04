"""Tests for CSV loading and column extraction."""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ColumnMapping
from preprocessing.csv_loader import (
    load_csv_from_bytes,
    extract_vibration_signal,
    extract_labels,
    extract_metadata,
    get_column_summary,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame mimicking a vibration CSV."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "time": np.arange(n) / 12000.0,
        "vibration": np.random.randn(n),
        "fault": np.random.choice(["normal", "bearing", "unbalance"], n),
        "run_id": np.random.choice(["run_0", "run_1", "run_2"], n),
        "temperature": 40 + np.random.randn(n) * 5,
    })


@pytest.fixture
def mapping():
    return ColumnMapping(
        vibration="vibration",
        label="fault",
        time="time",
        run_id="run_id",
        temperature="temperature",
    )


def test_load_csv_from_bytes(sample_df):
    csv_bytes = sample_df.to_csv(index=False).encode()
    df = load_csv_from_bytes(csv_bytes)
    assert len(df) == 1000
    assert "vibration" in df.columns


def test_extract_vibration_signal(sample_df, mapping):
    signal = extract_vibration_signal(sample_df, mapping)
    assert isinstance(signal, np.ndarray)
    assert signal.shape == (1000,)
    assert signal.dtype == np.float64


def test_extract_vibration_missing_column(sample_df):
    bad_mapping = ColumnMapping(vibration="nonexistent")
    with pytest.raises(KeyError):
        extract_vibration_signal(sample_df, bad_mapping)


def test_extract_labels(sample_df, mapping):
    labels = extract_labels(sample_df, mapping)
    assert labels is not None
    assert len(labels) == 1000
    assert set(labels).issubset({"normal", "bearing", "unbalance"})


def test_extract_labels_none():
    df = pd.DataFrame({"vibration": [1, 2, 3]})
    mapping = ColumnMapping(vibration="vibration")
    labels = extract_labels(df, mapping)
    assert labels is None


def test_extract_metadata(sample_df, mapping):
    meta = extract_metadata(sample_df, mapping)
    assert "time" in meta
    assert meta["time"] is not None
    assert "temperature" in meta
    assert meta["temperature"] is not None
    assert "run_id" in meta
    assert meta["run_id"] is not None
    assert meta["rpm"] is None  # Not mapped


def test_get_column_summary(sample_df):
    summary = get_column_summary(sample_df)
    assert len(summary) == 5
    assert "Column" in summary.columns
    assert "Type" in summary.columns
    assert "Missing" in summary.columns


def test_non_numeric_vibration():
    df = pd.DataFrame({"vibration": ["a", "b", "1.0", "2.0", "c"]})
    mapping = ColumnMapping(vibration="vibration")
    signal = extract_vibration_signal(df, mapping)
    # Non-numeric values should become NaN
    assert np.isnan(signal[0])
    assert np.isnan(signal[1])
    assert signal[2] == 1.0
