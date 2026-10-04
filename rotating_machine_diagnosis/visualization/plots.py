"""
Visualization — Matplotlib-based plotting functions for the Streamlit dashboard.

All functions return Matplotlib Figure objects that Streamlit can display.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from scipy import signal as scipy_signal
from typing import Optional, Dict, List
from matplotlib.figure import Figure


# ─────────────────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6366F1",
    "secondary": "#8B5CF6",
    "accent": "#EC4899",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#3B82F6",
    "dark_bg": "#1E1E2E",
    "card_bg": "#2D2D44",
    "text": "#E4E4E7",
}

SEVERITY_COLORS = {
    "LOW": "#10B981",
    "MEDIUM": "#F59E0B",
    "HIGH": "#F97316",
    "CRITICAL": "#EF4444",
}


def set_dark_style():
    """Apply a consistent dark style to plots."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": COLORS["dark_bg"],
        "axes.facecolor": COLORS["card_bg"],
        "axes.edgecolor": "#4a4a6a",
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "text.color": COLORS["text"],
        "grid.color": "#3a3a5a",
        "grid.alpha": 0.3,
        "font.size": 10,
    })


def plot_raw_signal(
    signal: np.ndarray,
    fs: Optional[float] = None,
    title: str = "Raw Vibration Signal",
) -> Figure:
    """Plot a raw vibration waveform."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    if fs is not None:
        time = np.arange(len(signal)) / fs
        ax.plot(time, signal, color=COLORS["primary"], linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Time (s)")
    else:
        ax.plot(signal, color=COLORS["primary"], linewidth=0.5, alpha=0.8)
        ax.set_xlabel("Sample Index")

    ax.set_ylabel("Amplitude")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_filtered_signal(
    raw: np.ndarray,
    filtered: np.ndarray,
    fs: Optional[float] = None,
    title: str = "Raw vs Filtered Signal",
) -> Figure:
    """Overlay raw and filtered signals."""
    set_dark_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    if fs is not None:
        time = np.arange(len(raw)) / fs
        ax1.plot(time[:len(raw)], raw, color=COLORS["info"], linewidth=0.5, alpha=0.7, label="Raw")
        ax2.plot(time[:len(filtered)], filtered, color=COLORS["success"], linewidth=0.5, alpha=0.8, label="Filtered")
        ax2.set_xlabel("Time (s)")
    else:
        ax1.plot(raw, color=COLORS["info"], linewidth=0.5, alpha=0.7, label="Raw")
        ax2.plot(filtered, color=COLORS["success"], linewidth=0.5, alpha=0.8, label="Filtered")
        ax2.set_xlabel("Sample Index")

    ax1.set_ylabel("Amplitude")
    ax1.set_title("Raw Signal", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.2)

    ax2.set_ylabel("Amplitude")
    ax2.set_title("Filtered Signal", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_fft(
    signal: np.ndarray,
    fs: float,
    title: str = "Frequency Spectrum (FFT)",
) -> Figure:
    """Plot single-sided FFT magnitude spectrum."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals) / N
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    ax.plot(freqs, fft_mag, color=COLORS["accent"], linewidth=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, fs / 2)
    fig.tight_layout()
    return fig


def plot_spectrogram(
    spectrogram: np.ndarray,
    fs: float = 12000.0,
    hop_length: int = 64,
    title: str = "STFT Spectrogram",
) -> Figure:
    """Plot a 2D spectrogram heatmap."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    extent = [
        0, spectrogram.shape[1] * hop_length / fs,  # time
        0, fs / 2,  # frequency
    ]
    im = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=extent,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    return fig


def plot_training_history(
    history: Dict[str, list],
) -> Figure:
    """Plot training/validation loss and accuracy curves."""
    set_dark_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["loss"], color=COLORS["info"], linewidth=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], color=COLORS["accent"], linewidth=2, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Accuracy
    ax2.plot(epochs, history["accuracy"], color=COLORS["success"], linewidth=2, label="Train Accuracy")
    ax2.plot(epochs, history["val_accuracy"], color=COLORS["warning"], linewidth=2, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
) -> Figure:
    """Plot a color-mapped confusion matrix."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Annotate
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=10)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_class_probabilities(
    probabilities: np.ndarray,
    class_names: List[str],
    title: str = "Class Probabilities",
) -> Figure:
    """Bar chart of class probabilities for a single prediction."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    colors_list = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                   COLORS["success"], COLORS["warning"], COLORS["danger"],
                   COLORS["info"]]
    bar_colors = [colors_list[i % len(colors_list)] for i in range(len(class_names))]

    bars = ax.barh(class_names, probabilities, color=bar_colors, edgecolor="white", linewidth=0.5)

    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", fontsize=10, color=COLORS["text"])

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Probability")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    return fig


def plot_risk_gauge(
    risk_score: float,
    severity: str,
    title: str = "Machine Risk Assessment",
) -> Figure:
    """Create a semi-circular gauge for risk score visualization."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": "polar"})

    # Semi-circular gauge
    theta = np.linspace(np.pi, 0, 100)

    # Background bands
    bands = [
        (0, 25, "#10B981", "Low"),
        (25, 50, "#F59E0B", "Medium"),
        (50, 75, "#F97316", "High"),
        (75, 100, "#EF4444", "Critical"),
    ]

    for lo, hi, color, label in bands:
        theta_lo = np.pi * (1 - lo / 100)
        theta_hi = np.pi * (1 - hi / 100)
        theta_band = np.linspace(theta_lo, theta_hi, 50)
        ax.fill_between(theta_band, 0.6, 1.0, color=color, alpha=0.3)

    # Needle
    needle_angle = np.pi * (1 - risk_score / 100)
    ax.plot([needle_angle, needle_angle], [0, 0.95], color="white", linewidth=3)
    ax.plot(needle_angle, 0.95, "o", color=SEVERITY_COLORS.get(severity, "white"), markersize=10)

    ax.set_ylim(0, 1.2)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    # Center text
    ax.text(np.pi / 2, 0.3, f"{risk_score:.0f}", ha="center", va="center",
            fontsize=36, fontweight="bold", color=SEVERITY_COLORS.get(severity, "white"))
    ax.text(np.pi / 2, 0.05, severity, ha="center", va="center",
            fontsize=14, fontweight="bold", color=SEVERITY_COLORS.get(severity, "white"))

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    return fig


def plot_membership_functions(
    membership_values: Dict[str, Dict[str, float]],
    title: str = "Fuzzy Membership Values",
) -> Figure:
    """Bar chart showing active membership degrees for each input variable."""
    set_dark_style()
    n_vars = len(membership_values)
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))

    if n_vars == 1:
        axes = [axes]

    colors_map = {"low": COLORS["success"], "medium": COLORS["warning"], "high": COLORS["danger"]}

    for ax, (var_name, terms) in zip(axes, membership_values.items()):
        term_names = list(terms.keys())
        values = list(terms.values())
        bar_colors = [colors_map.get(t, COLORS["info"]) for t in term_names]
        bars = ax.bar(term_names, values, color=bar_colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", fontsize=9, color=COLORS["text"])
        ax.set_ylim(0, 1.15)
        ax.set_title(var_name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Membership Degree")
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_summary(
    features: Dict[str, float],
    title: str = "Vibration Feature Summary",
) -> Figure:
    """Bar chart of computed vibration features."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(features.keys())
    values = list(features.values())
    colors_list = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                   COLORS["success"], COLORS["warning"], COLORS["danger"]]
    bar_colors = [colors_list[i % len(colors_list)] for i in range(len(names))]

    bars = ax.bar(names, values, color=bar_colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, color=COLORS["text"])

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.2, axis="y")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig
