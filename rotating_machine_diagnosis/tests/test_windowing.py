"""Tests for windowing — window generation and group-level splitting."""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import WindowingConfig
from preprocessing.windowing import (
    generate_windows,
    group_level_split,
    VibrationWindow,
)


@pytest.fixture
def long_signal():
    np.random.seed(42)
    return np.random.randn(20000)


@pytest.fixture
def config():
    return WindowingConfig(window_size=2048, overlap=0.5)


class TestGenerateWindows:
    def test_basic(self, long_signal, config):
        windows = generate_windows(long_signal, config)
        assert len(windows) > 0
        assert all(w.data.shape == (2048,) for w in windows)

    def test_window_count(self, long_signal, config):
        step = config.step_size  # 1024
        expected = (len(long_signal) - config.window_size) // step + 1
        windows = generate_windows(long_signal, config)
        assert len(windows) == expected

    def test_with_labels(self, long_signal, config):
        labels = np.array(["normal"] * 10000 + ["fault"] * 10000)
        windows = generate_windows(long_signal, config, labels=labels)
        assert all(w.label in ("normal", "fault") for w in windows)

    def test_with_metadata(self, long_signal, config):
        run_ids = np.array(["run_0"] * len(long_signal))
        machine_ids = np.array(["m1"] * len(long_signal))
        windows = generate_windows(
            long_signal, config,
            run_ids=run_ids,
            machine_ids=machine_ids,
            source_file="test.csv",
        )
        assert all(w.run_id == "run_0" for w in windows)
        assert all(w.machine_id == "m1" for w in windows)
        assert all(w.source_file == "test.csv" for w in windows)

    def test_short_signal(self, config):
        """Signal shorter than window → should pad and return 1 window."""
        short = np.random.randn(500)
        windows = generate_windows(short, config)
        assert len(windows) == 1
        assert windows[0].data.shape == (2048,)

    def test_exact_one_window(self, config):
        signal = np.random.randn(2048)
        windows = generate_windows(signal, config)
        assert len(windows) == 1

    def test_overlap_zero(self, long_signal):
        config = WindowingConfig(window_size=2048, overlap=0.0)
        windows = generate_windows(long_signal, config)
        expected = len(long_signal) // 2048
        assert len(windows) == expected

    def test_high_overlap(self, long_signal):
        config = WindowingConfig(window_size=2048, overlap=0.9)
        windows = generate_windows(long_signal, config)
        # More windows with higher overlap
        config_low = WindowingConfig(window_size=2048, overlap=0.1)
        windows_low = generate_windows(long_signal, config_low)
        assert len(windows) > len(windows_low)

    def test_window_ids_sequential(self, long_signal, config):
        windows = generate_windows(long_signal, config)
        ids = [w.window_id for w in windows]
        assert ids == list(range(len(windows)))


class TestGroupLevelSplit:
    def _make_windows(self, n=100, with_runs=True):
        windows = []
        for i in range(n):
            w = VibrationWindow(
                window_id=i,
                data=np.random.randn(2048),
                label="normal" if i < n // 2 else "fault",
                run_id=f"run_{i // 10}" if with_runs else None,
                machine_id=None,
            )
            windows.append(w)
        return windows

    def test_split_with_runs(self):
        windows = self._make_windows(100, with_runs=True)
        split = group_level_split(windows, val_ratio=0.2, test_ratio=0.2)
        assert "train" in split
        assert "val" in split
        assert "test" in split
        assert not split["leakage_warning"]
        total = len(split["train"]) + len(split["val"]) + len(split["test"])
        assert total == 100

    def test_split_without_runs(self):
        windows = self._make_windows(100, with_runs=False)
        split = group_level_split(windows, val_ratio=0.2, test_ratio=0.2)
        assert split["leakage_warning"] is True
        total = len(split["train"]) + len(split["val"]) + len(split["test"])
        assert total == 100

    def test_no_overlap_between_splits(self):
        windows = self._make_windows(200, with_runs=True)
        split = group_level_split(windows)
        train_ids = {w.window_id for w in split["train"]}
        val_ids = {w.window_id for w in split["val"]}
        test_ids = {w.window_id for w in split["test"]}
        assert train_ids & val_ids == set()
        assert train_ids & test_ids == set()
        assert val_ids & test_ids == set()

    def test_group_integrity(self):
        """All windows from same run should end up in the same split."""
        windows = self._make_windows(100, with_runs=True)
        split = group_level_split(windows)

        run_to_split = {}
        for split_name in ["train", "val", "test"]:
            for w in split[split_name]:
                if w.run_id in run_to_split:
                    assert run_to_split[w.run_id] == split_name, \
                        f"Run {w.run_id} appears in both {run_to_split[w.run_id]} and {split_name}"
                run_to_split[w.run_id] = split_name
