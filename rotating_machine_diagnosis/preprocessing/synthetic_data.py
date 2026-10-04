"""
Synthetic Data Generator — Creates artificial vibration datasets
ONLY for testing/demonstrating the pipeline.

Each fault class has a distinct signal signature:
  - Normal:      Low-amplitude sinusoidal + noise
  - Unbalance:   Strong 1x rotational frequency component
  - Bearing:     Periodic impulses with random spacing
  - Misalignment: Strong 2x rotational frequency + harmonics
  - Gear:        Periodic meshing pattern

These are simplified synthetic models and do NOT represent
real-world fault signatures accurately. They are ONLY for
pipeline testing.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict


def generate_normal(
    n_samples: int = 10000,
    fs: float = 12000.0,
    amplitude: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate a 'normal' operating vibration signal."""
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / fs
    signal = amplitude * np.sin(2 * np.pi * 30 * t)  # 30 Hz base
    signal += 0.02 * np.sin(2 * np.pi * 60 * t)
    signal += rng.normal(0, amplitude * 0.3, n_samples)
    return signal


def generate_unbalance(
    n_samples: int = 10000,
    fs: float = 12000.0,
    amplitude: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Generate an 'unbalance' fault signal — strong 1x frequency."""
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / fs
    freq_1x = 30.0  # Rotational frequency
    signal = amplitude * np.sin(2 * np.pi * freq_1x * t)
    signal += 0.1 * np.sin(2 * np.pi * 2 * freq_1x * t)
    signal += rng.normal(0, amplitude * 0.15, n_samples)
    return signal


def generate_bearing_fault(
    n_samples: int = 10000,
    fs: float = 12000.0,
    amplitude: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Generate a 'bearing fault' signal — periodic impulses."""
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / fs
    signal = rng.normal(0, amplitude * 0.2, n_samples)

    # Periodic impulses at characteristic frequency
    char_freq = 120.0  # Hz
    period_samples = int(fs / char_freq)
    for i in range(0, n_samples, period_samples):
        imp_len = min(50, n_samples - i)
        decay = np.exp(-np.linspace(0, 5, imp_len))
        impulse = amplitude * rng.normal(0, 1, imp_len) * decay
        signal[i:i + imp_len] += impulse

    # Background
    signal += 0.05 * np.sin(2 * np.pi * 30 * t)
    return signal


def generate_misalignment(
    n_samples: int = 10000,
    fs: float = 12000.0,
    amplitude: float = 0.4,
    seed: int = 42,
) -> np.ndarray:
    """Generate a 'misalignment' fault signal — strong 2x component."""
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / fs
    freq_1x = 30.0
    signal = 0.15 * np.sin(2 * np.pi * freq_1x * t)  # 1x
    signal += amplitude * np.sin(2 * np.pi * 2 * freq_1x * t)  # 2x dominant
    signal += 0.2 * np.sin(2 * np.pi * 3 * freq_1x * t)  # 3x
    signal += rng.normal(0, amplitude * 0.1, n_samples)
    return signal


def generate_gear_fault(
    n_samples: int = 10000,
    fs: float = 12000.0,
    amplitude: float = 0.35,
    seed: int = 42,
) -> np.ndarray:
    """Generate a 'gear fault' signal — meshing frequency modulation."""
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / fs
    mesh_freq = 600.0  # Gear mesh frequency
    rot_freq = 30.0

    # Amplitude-modulated mesh signal
    carrier = np.sin(2 * np.pi * mesh_freq * t)
    modulator = 1 + amplitude * np.sin(2 * np.pi * rot_freq * t)
    signal = amplitude * carrier * modulator

    # Add sidebands
    signal += 0.15 * np.sin(2 * np.pi * (mesh_freq - rot_freq) * t)
    signal += 0.15 * np.sin(2 * np.pi * (mesh_freq + rot_freq) * t)
    signal += rng.normal(0, amplitude * 0.1, n_samples)
    return signal


def generate_synthetic_dataset(
    n_samples_per_class: int = 50000,
    fs: float = 12000.0,
    n_runs_per_class: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a complete synthetic CSV dataset with multiple fault classes.

    Parameters
    ----------
    n_samples_per_class : int
        Total vibration samples per class.
    fs : float
        Sampling frequency.
    n_runs_per_class : int
        Number of separate "runs" per class (for group-level splitting).
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame with columns:
        time, vibration, fault, run_id, temperature
    """
    rng = np.random.RandomState(seed)

    generators = {
        "normal": generate_normal,
        "unbalance": generate_unbalance,
        "bearing_fault": generate_bearing_fault,
        "misalignment": generate_misalignment,
        "gear_fault": generate_gear_fault,
    }

    all_data = []
    global_idx = 0

    for fault_name, gen_fn in generators.items():
        samples_per_run = n_samples_per_class // n_runs_per_class

        for run_i in range(n_runs_per_class):
            run_seed = seed + global_idx
            signal = gen_fn(
                n_samples=samples_per_run,
                fs=fs,
                seed=run_seed,
            )
            n = len(signal)
            time = np.arange(n) / fs + global_idx * (samples_per_run / fs)

            # Synthetic temperature: normal → low, faults → elevated
            base_temp = 40.0 if fault_name == "normal" else 55.0 + rng.uniform(-5, 15)
            temp = base_temp + rng.normal(0, 2, n)

            run_id = f"{fault_name}_run_{run_i}"

            for i in range(n):
                all_data.append({
                    "time": time[i],
                    "vibration": signal[i],
                    "fault": fault_name,
                    "run_id": run_id,
                    "temperature": temp[i],
                })
            global_idx += 1

    df = pd.DataFrame(all_data)
    # Shuffle rows (but keep run_id for group-level splitting)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df
