"""
Windowing — Segment continuous vibration signals into overlapping windows.

Preserves metadata (label, run_id, machine_id, etc.) for each window.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import WindowingConfig


@dataclass
class VibrationWindow:
    """One segmented vibration window with metadata."""
    window_id: int
    data: np.ndarray                    # shape (window_size,)
    label: Optional[str] = None
    source_file: Optional[str] = None
    run_id: Optional[str] = None
    machine_id: Optional[str] = None
    start_idx: int = 0
    end_idx: int = 0


def generate_windows(
    signal: np.ndarray,
    config: WindowingConfig,
    labels: Optional[np.ndarray] = None,
    run_ids: Optional[np.ndarray] = None,
    machine_ids: Optional[np.ndarray] = None,
    source_file: Optional[str] = None,
) -> List[VibrationWindow]:
    """
    Divide a continuous signal into fixed-size overlapping windows.

    Parameters
    ----------
    signal : np.ndarray
        1-D preprocessed vibration signal.
    config : WindowingConfig
        Window size, overlap, etc.
    labels : np.ndarray or None
        Per-sample labels (same length as signal).
    run_ids, machine_ids : np.ndarray or None
        Per-sample metadata.
    source_file : str or None
        Name of the source CSV file.

    Returns
    -------
    List[VibrationWindow]
    """
    n = len(signal)
    ws = config.window_size
    step = config.step_size

    if n < ws:
        # If the signal is shorter than one window, pad with zeros and warn
        padded = np.zeros(ws, dtype=signal.dtype)
        padded[:n] = signal
        label = _majority_label(labels, 0, n) if labels is not None else None
        rid = _first_value(run_ids, 0) if run_ids is not None else None
        mid = _first_value(machine_ids, 0) if machine_ids is not None else None
        return [VibrationWindow(
            window_id=0,
            data=padded,
            label=label,
            source_file=source_file,
            run_id=rid,
            machine_id=mid,
            start_idx=0,
            end_idx=n,
        )]

    windows = []
    wid = 0
    start = 0
    while start + ws <= n:
        end = start + ws
        chunk = signal[start:end].copy()

        label = _majority_label(labels, start, end) if labels is not None else None
        rid = _first_value(run_ids, start) if run_ids is not None else None
        mid = _first_value(machine_ids, start) if machine_ids is not None else None

        windows.append(VibrationWindow(
            window_id=wid,
            data=chunk,
            label=label,
            source_file=source_file,
            run_id=rid,
            machine_id=mid,
            start_idx=start,
            end_idx=end,
        ))
        wid += 1
        start += step

    return windows


def _majority_label(labels: np.ndarray, start: int, end: int) -> Optional[str]:
    """Return the most common label in [start, end)."""
    segment = labels[start:end]
    if len(segment) == 0:
        return None
    vals, counts = np.unique(segment, return_counts=True)
    return str(vals[np.argmax(counts)])


def _first_value(arr: np.ndarray, start: int) -> Optional[str]:
    """Return the first value in the segment."""
    if start < len(arr):
        return str(arr[start])
    return None


def group_level_split(
    windows: List[VibrationWindow],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[VibrationWindow]]:
    """
    Split windows at the run/machine level to prevent data leakage.

    If run_id is available, splits by run_id.
    Otherwise, falls back to machine_id.
    If neither exists, performs a random split and sets a leakage flag.

    Returns
    -------
    dict with keys 'train', 'val', 'test', and 'leakage_warning' (bool).
    """
    rng = np.random.RandomState(seed)

    # Determine grouping key
    has_run = any(w.run_id is not None for w in windows)
    has_machine = any(w.machine_id is not None for w in windows)

    if has_run:
        groups = _group_by(windows, lambda w: w.run_id or "unknown")
        group_key = "run_id"
    elif has_machine:
        groups = _group_by(windows, lambda w: w.machine_id or "unknown")
        group_key = "machine_id"
    else:
        # Random split with leakage warning
        indices = rng.permutation(len(windows))
        n = len(windows)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]
        return {
            "train": [windows[i] for i in train_idx],
            "val": [windows[i] for i in val_idx],
            "test": [windows[i] for i in test_idx],
            "leakage_warning": True,
            "group_key": None,
        }

    # Shuffle group keys and split
    keys = list(groups.keys())
    rng.shuffle(keys)
    n_keys = len(keys)
    n_test_keys = max(1, int(n_keys * test_ratio))
    n_val_keys = max(1, int(n_keys * val_ratio))

    test_keys = set(keys[:n_test_keys])
    val_keys = set(keys[n_test_keys:n_test_keys + n_val_keys])
    train_keys = set(keys[n_test_keys + n_val_keys:])

    train_w = [w for k in train_keys for w in groups[k]]
    val_w = [w for k in val_keys for w in groups[k]]
    test_w = [w for k in test_keys for w in groups[k]]

    return {
        "train": train_w,
        "val": val_w,
        "test": test_w,
        "leakage_warning": False,
        "group_key": group_key,
    }


def _group_by(windows, key_fn):
    """Group windows by a key function."""
    groups = {}
    for w in windows:
        k = key_fn(w)
        groups.setdefault(k, []).append(w)
    return groups
