"""
Central Configuration for Rotating Machine Fault Diagnosis System.

All configurable parameters are defined here as dataclasses.
These are defaults — the Streamlit UI allows the user to override them.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# ─────────────────────────────────────────────────────────
# Column Mapping (user-configurable via UI)
# ─────────────────────────────────────────────────────────
@dataclass
class ColumnMapping:
    """Maps user's CSV column names to pipeline roles."""
    vibration: str = ""          # Required
    label: Optional[str] = None  # Required for training
    time: Optional[str] = None
    rpm: Optional[str] = None
    load: Optional[str] = None
    temperature: Optional[str] = None
    machine_id: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ColumnMapping":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

# ─────────────────────────────────────────────────────────
# Preprocessing Configuration
# ─────────────────────────────────────────────────────────
@dataclass
class PreprocessingConfig:
    """Signal preprocessing parameters."""
    # Sampling
    sampling_frequency: float = 12000.0  # Hz — user must confirm or set

    # Missing value handling
    missing_value_strategy: str = "interpolate"  # drop | interpolate | ffill | bfill

    # Filtering
    filter_enabled: bool = True
    filter_type: str = "bandpass"       # lowpass | highpass | bandpass
    filter_order: int = 5
    lowcut: float = 10.0                # Hz
    highcut: float = 5000.0             # Hz

    # Normalization
    normalization_method: str = "standardize"  # standardize | minmax

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

# ─────────────────────────────────────────────────────────
# Windowing Configuration
# ─────────────────────────────────────────────────────────
@dataclass
class WindowingConfig:
    """Sliding-window segmentation parameters."""
    window_size: int = 2048       # samples
    overlap: float = 0.5          # fraction 0–1
    min_signal_length: int = 4096 # minimum samples needed

    @property
    def step_size(self) -> int:
        return int(self.window_size * (1 - self.overlap))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["step_size"] = self.step_size
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WindowingConfig":
        allowed = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})

# ─────────────────────────────────────────────────────────
# STFT / Spectrogram Configuration
# ─────────────────────────────────────────────────────────
@dataclass
class STFTConfig:
    """Short-Time Fourier Transform parameters."""
    n_fft: int = 256
    hop_length: int = 64
    win_length: int = 256
    window: str = "hann"  # hann | hamming | blackman

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "STFTConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

# ─────────────────────────────────────────────────────────
# Model / Training Configuration
# ─────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Hybrid CNN-BiLSTM model and training parameters."""
    # 1D CNN
    cnn1d_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn1d_kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 3])

    # 2D CNN
    cnn2d_filters: List[int] = field(default_factory=lambda: [32, 64])
    cnn2d_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3])

    # BiLSTM
    lstm_units: int = 64
    sequence_length: int = 5  # number of consecutive windows

    # Dense / output
    dense_units: int = 64
    dropout_rate: float = 0.3
    num_classes: int = 0  # auto-determined from data

    # Training
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    early_stopping_patience: int = 7
    lr_reduce_patience: int = 3
    lr_reduce_factor: float = 0.5
    validation_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

# ─────────────────────────────────────────────────────────
# Fuzzy Logic Configuration
# ─────────────────────────────────────────────────────────
@dataclass
class FuzzyConfig:
    """Mamdani fuzzy inference system parameters."""
    # RMS membership ranges [low_end, low_peak, med_start, med_peak, med_end, high_start, high_peak]
    rms_universe: List[float] = field(default_factory=lambda: [0.0, 1.0])
    rms_low: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.3])
    rms_medium: List[float] = field(default_factory=lambda: [0.15, 0.5, 0.85])
    rms_high: List[float] = field(default_factory=lambda: [0.7, 1.0, 1.0])

    kurtosis_universe: List[float] = field(default_factory=lambda: [0.0, 20.0])
    kurtosis_low: List[float] = field(default_factory=lambda: [0.0, 0.0, 5.0])
    kurtosis_medium: List[float] = field(default_factory=lambda: [3.0, 7.0, 12.0])
    kurtosis_high: List[float] = field(default_factory=lambda: [8.0, 20.0, 20.0])

    fault_prob_universe: List[float] = field(default_factory=lambda: [0.0, 1.0])
    fault_prob_low: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.35])
    fault_prob_medium: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.8])
    fault_prob_high: List[float] = field(default_factory=lambda: [0.65, 1.0, 1.0])

    risk_universe: List[float] = field(default_factory=lambda: [0.0, 100.0])
    risk_low: List[float] = field(default_factory=lambda: [0.0, 0.0, 25.0, 35.0])
    risk_medium: List[float] = field(default_factory=lambda: [25.0, 40.0, 55.0, 65.0])
    risk_high: List[float] = field(default_factory=lambda: [55.0, 70.0, 80.0, 90.0])
    risk_critical: List[float] = field(default_factory=lambda: [80.0, 90.0, 100.0, 100.0])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FuzzyConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

# ─────────────────────────────────────────────────────────
# Recommendation thresholds
# ─────────────────────────────────────────────────────────
RECOMMENDATION_THRESHOLDS = {
    "routine": (0, 25),
    "monitor": (25, 50),
    "inspect_soon": (50, 75),
    "priority": (75, 90),
    "immediate": (90, 101),
}

RECOMMENDATION_MESSAGES = {
    "routine": "Routine monitoring recommended. No immediate action required.",
    "monitor": "Continue monitoring and schedule inspection at next convenient opportunity.",
    "inspect_soon": "Maintenance inspection recommended soon. Plan downtime for inspection.",
    "priority": "Priority inspection recommended. Schedule inspection as soon as possible.",
    "immediate": "Immediate inspection recommended. Stop machine if safe to do so.",
}

DISCLAIMER = (
    "⚠️ DISCLAIMER: This system provides an AI-based condition assessment and "
    "maintenance decision-support recommendation. It does not guarantee mechanical "
    "failure or replace professional inspection. Always consult qualified maintenance "
    "engineers before making critical decisions."
)

# ─────────────────────────────────────────────────────────
# Helpers to save/load config bundles
# ─────────────────────────────────────────────────────────
def save_config_bundle(path: str, **configs):
    """Save multiple config objects as a single JSON file."""
    bundle = {}
    for name, cfg in configs.items():
        if hasattr(cfg, "to_dict"):
            bundle[name] = cfg.to_dict()
        else:
            bundle[name] = cfg
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(bundle, f, indent=2)

def load_config_bundle(path: str) -> dict:
    """Load a config bundle JSON file."""
    with open(path, "r") as f:
        return json.load(f)
