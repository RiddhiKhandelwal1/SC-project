"""
Spectrogram — STFT-based time-frequency representation.

Uses scipy.signal.stft for the Short-Time Fourier Transform.
Produces magnitude spectrograms for the 2D CNN branch.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import STFTConfig


def compute_stft(
    window_data: np.ndarray,
    config: STFTConfig,
    fs: float = 12000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT for a single vibration window.

    Parameters
    ----------
    window_data : np.ndarray
        1-D array of shape (window_size,).
    config : STFTConfig
        STFT parameters.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    (frequencies, times, Zxx)
        - frequencies: 1-D array of frequency bins
        - times: 1-D array of time bins
        - Zxx: 2-D complex STFT matrix (freq × time)
    """
    f, t, Zxx = scipy_signal.stft(
        window_data,
        fs=fs,
        window=config.window,
        nperseg=config.win_length,
        noverlap=config.win_length - config.hop_length,
        nfft=config.n_fft,
    )
    return f, t, Zxx


def compute_magnitude_spectrogram(
    window_data: np.ndarray,
    config: STFTConfig,
    fs: float = 12000.0,
    log_scale: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute the magnitude spectrogram for a vibration window.

    Parameters
    ----------
    window_data : np.ndarray
        1-D array (window_size,).
    config : STFTConfig
    fs : float
        Sampling frequency.
    log_scale : bool
        If True, return 10*log10(|STFT|² + eps).
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    np.ndarray
        2-D spectrogram (frequency_bins × time_bins).
    """
    _, _, Zxx = compute_stft(window_data, config, fs)
    magnitude = np.abs(Zxx)

    if log_scale:
        magnitude = 10.0 * np.log10(magnitude ** 2 + eps)

    return magnitude


def compute_power_spectrogram(
    window_data: np.ndarray,
    config: STFTConfig,
    fs: float = 12000.0,
) -> np.ndarray:
    """Compute the power spectrogram (|STFT|²)."""
    _, _, Zxx = compute_stft(window_data, config, fs)
    return np.abs(Zxx) ** 2


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """
    Normalize a 2-D spectrogram to [0, 1] range.
    """
    s_min = spec.min()
    s_max = spec.max()
    if s_max - s_min < 1e-10:
        return np.zeros_like(spec)
    return (spec - s_min) / (s_max - s_min)


def batch_spectrograms(
    windows: list,
    config: STFTConfig,
    fs: float = 12000.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute normalized spectrograms for a batch of VibrationWindow objects.

    Returns
    -------
    np.ndarray
        Shape (num_windows, freq_bins, time_bins, 1) — ready for 2D CNN.
    """
    specs = []
    for w in windows:
        data = w.data if hasattr(w, "data") else w
        spec = compute_magnitude_spectrogram(data, config, fs)
        if normalize:
            spec = normalize_spectrogram(spec)
        specs.append(spec)

    # Stack and add channel dimension
    batch = np.array(specs)  # (N, F, T)
    batch = batch[..., np.newaxis]  # (N, F, T, 1)
    return batch
