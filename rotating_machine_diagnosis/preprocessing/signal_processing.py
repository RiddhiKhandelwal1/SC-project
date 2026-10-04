"""
Signal Processing — Filtering, normalization, and missing-value handling.

Uses SciPy for digital filters and provides both standardization
and min-max normalization with the ability to fit on training data
and transform test/inference data without leakage.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, Dict, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PreprocessingConfig


# ─────────────────────────────────────────────────────────
# Missing Value Handling
# ─────────────────────────────────────────────────────────

def handle_missing_values(
    data: np.ndarray,
    strategy: str = "interpolate",
) -> np.ndarray:
    """
    Handle NaN and infinite values in a 1-D signal.

    Parameters
    ----------
    data : np.ndarray
        Raw vibration signal (1-D).
    strategy : str
        One of: 'drop', 'interpolate', 'ffill', 'bfill'.

    Returns
    -------
    np.ndarray
        Cleaned signal.
    """
    import pandas as pd

    # Replace inf with NaN first
    data = data.copy().astype(np.float64)
    data[~np.isfinite(data)] = np.nan

    if not np.any(np.isnan(data)):
        return data

    series = pd.Series(data)

    if strategy == "drop":
        series = series.dropna()
    elif strategy == "interpolate":
        series = series.interpolate(method="linear", limit_direction="both")
        # Fill any remaining edge NaNs
        series = series.ffill().bfill()
    elif strategy == "ffill":
        series = series.ffill().bfill()
    elif strategy == "bfill":
        series = series.bfill().ffill()
    else:
        raise ValueError(f"Unknown missing-value strategy: {strategy}")

    return series.values.astype(np.float64)


# ─────────────────────────────────────────────────────────
# Digital Filtering (SciPy)
# ─────────────────────────────────────────────────────────

def apply_filter(
    data: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """
    Apply a Butterworth filter to the vibration signal.

    Parameters
    ----------
    data : np.ndarray
        1-D vibration signal.
    config : PreprocessingConfig
        Contains filter_type, filter_order, lowcut, highcut, sampling_frequency.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    if not config.filter_enabled:
        return data

    fs = config.sampling_frequency
    nyq = fs / 2.0

    ftype = config.filter_type.lower()
    order = config.filter_order

    if ftype == "lowpass":
        if config.highcut >= nyq:
            raise ValueError(
                f"High-cut frequency ({config.highcut} Hz) must be below "
                f"the Nyquist frequency ({nyq} Hz)."
            )
        b, a = scipy_signal.butter(order, config.highcut / nyq, btype="low")
    elif ftype == "highpass":
        if config.lowcut >= nyq:
            raise ValueError(
                f"Low-cut frequency ({config.lowcut} Hz) must be below "
                f"the Nyquist frequency ({nyq} Hz)."
            )
        b, a = scipy_signal.butter(order, config.lowcut / nyq, btype="high")
    elif ftype == "bandpass":
        if config.lowcut >= config.highcut:
            raise ValueError(
                f"Low-cut ({config.lowcut} Hz) must be less than "
                f"high-cut ({config.highcut} Hz)."
            )
        if config.highcut >= nyq:
            raise ValueError(
                f"High-cut ({config.highcut} Hz) must be below Nyquist ({nyq} Hz)."
            )
        b, a = scipy_signal.butter(
            order,
            [config.lowcut / nyq, config.highcut / nyq],
            btype="band",
        )
    else:
        raise ValueError(f"Unsupported filter type: {ftype}")

    filtered = scipy_signal.filtfilt(b, a, data)
    return filtered


# ─────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────

class Normalizer:
    """
    Signal normalizer that fits on training data and transforms
    any data using the fitted parameters.
    """

    def __init__(self, method: str = "standardize"):
        """
        Parameters
        ----------
        method : str
            'standardize' (z-score) or 'minmax' (0-1 scaling).
        """
        self.method = method
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None
        self.min_: Optional[float] = None
        self.max_: Optional[float] = None
        self._fitted = False

    def fit(self, data: np.ndarray) -> "Normalizer":
        """Compute normalization parameters from training data."""
        clean = data[np.isfinite(data)]
        if len(clean) == 0:
            raise ValueError("Cannot fit normalizer on empty/all-NaN data.")

        if self.method == "standardize":
            self.mean_ = float(np.mean(clean))
            self.std_ = float(np.std(clean))
            if self.std_ == 0:
                self.std_ = 1.0
        elif self.method == "minmax":
            self.min_ = float(np.min(clean))
            self.max_ = float(np.max(clean))
            if self.max_ == self.min_:
                self.max_ = self.min_ + 1.0
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply fitted normalization."""
        if not self._fitted:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")

        result = data.copy().astype(np.float64)
        if self.method == "standardize":
            result = (result - self.mean_) / self.std_
        elif self.method == "minmax":
            result = (result - self.min_) / (self.max_ - self.min_)
        return result

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)

    def get_params(self) -> dict:
        """Return fitted parameters as a dict (for serialization)."""
        return {
            "method": self.method,
            "mean": self.mean_,
            "std": self.std_,
            "min": self.min_,
            "max": self.max_,
        }

    @classmethod
    def from_params(cls, params: dict) -> "Normalizer":
        """Reconstruct a fitted normalizer from saved parameters."""
        norm = cls(method=params["method"])
        norm.mean_ = params.get("mean")
        norm.std_ = params.get("std")
        norm.min_ = params.get("min")
        norm.max_ = params.get("max")
        norm._fitted = True
        return norm


def preprocess_signal(
    raw_signal: np.ndarray,
    config: PreprocessingConfig,
    normalizer: Optional[Normalizer] = None,
    fit_normalizer: bool = True,
) -> Tuple[np.ndarray, Normalizer]:
    """
    Full preprocessing pipeline for a single vibration signal.

    Steps: handle missing → filter → normalize.

    Parameters
    ----------
    raw_signal : np.ndarray
        Raw 1-D vibration signal.
    config : PreprocessingConfig
    normalizer : Normalizer or None
        If provided and fit_normalizer=False, uses this normalizer (inference).
    fit_normalizer : bool
        If True, fits a new normalizer on this signal (training).

    Returns
    -------
    (processed_signal, normalizer)
    """
    # Step 1: Missing values
    cleaned = handle_missing_values(raw_signal, strategy=config.missing_value_strategy)

    # Step 2: Filter
    filtered = apply_filter(cleaned, config)

    # Step 3: Normalize
    if normalizer is None or fit_normalizer:
        normalizer = Normalizer(method=config.normalization_method)
        normalized = normalizer.fit_transform(filtered)
    else:
        normalized = normalizer.transform(filtered)

    return normalized, normalizer
