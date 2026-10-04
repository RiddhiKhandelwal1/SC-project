"""
Traditional Vibration Features — Computed per window for interpretability
and as inputs to the fuzzy risk assessment system.

These features are separate from CNN-learned features and serve as
domain-knowledge features for condition monitoring.
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Dict, Optional


def compute_rms(signal: np.ndarray) -> float:
    """
    Root Mean Square.
    RMS = sqrt(mean(x²))
    """
    return float(np.sqrt(np.mean(signal ** 2)))


def compute_peak_amplitude(signal: np.ndarray) -> float:
    """
    Peak amplitude.
    Peak = max(|x|)
    """
    return float(np.max(np.abs(signal)))


def compute_kurtosis(signal: np.ndarray) -> float:
    """
    Kurtosis (Fisher's definition, excess kurtosis).
    Normal distribution → 0.
    High values indicate impulsive/fault-like behavior.
    """
    return float(scipy_stats.kurtosis(signal, fisher=True))


def compute_crest_factor(signal: np.ndarray) -> float:
    """
    Crest Factor = Peak / RMS.
    High crest factor indicates sharp peaks (possible faults).
    """
    rms = compute_rms(signal)
    if rms < 1e-12:
        return 0.0
    peak = compute_peak_amplitude(signal)
    return float(peak / rms)


def compute_skewness(signal: np.ndarray) -> float:
    """Skewness of the signal distribution."""
    return float(scipy_stats.skew(signal))


def compute_std(signal: np.ndarray) -> float:
    """Standard deviation."""
    return float(np.std(signal))


def compute_all_features(
    signal: np.ndarray,
    temperature: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all traditional vibration features for a single window.

    Parameters
    ----------
    signal : np.ndarray
        1-D vibration window.
    temperature : float or None
        Temperature reading (if available).

    Returns
    -------
    dict
        Feature name → value.
    """
    features = {
        "rms": compute_rms(signal),
        "peak_amplitude": compute_peak_amplitude(signal),
        "kurtosis": compute_kurtosis(signal),
        "crest_factor": compute_crest_factor(signal),
        "skewness": compute_skewness(signal),
        "std": compute_std(signal),
    }
    if temperature is not None:
        features["temperature"] = temperature
    return features
