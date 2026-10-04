"""
Training Pipeline — End-to-end training with early stopping,
LR scheduling, checkpointing, class weighting, and evaluation.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Optional, Tuple, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    ModelConfig, PreprocessingConfig, WindowingConfig, STFTConfig,
    SAVED_MODELS_DIR, save_config_bundle,
)
from utils.seed import set_global_seed
from preprocessing.signal_processing import Normalizer
from preprocessing.spectrogram import batch_spectrograms
from preprocessing.windowing import VibrationWindow
from models.hybrid_model import build_hybrid_model, prepare_sequences


def encode_labels(
    windows: List[VibrationWindow],
) -> Tuple[np.ndarray, Dict[int, str], Dict[str, int]]:
    """
    Encode string labels to integers.

    Returns
    -------
    (encoded, idx_to_class, class_to_idx)
    """
    labels = [w.label for w in windows]
    unique_labels = sorted(set(l for l in labels if l is not None))
    class_to_idx = {c: i for i, c in enumerate(unique_labels)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    encoded = np.array([class_to_idx.get(l, 0) for l in labels], dtype=np.int32)
    return encoded, idx_to_class, class_to_idx


def prepare_training_data(
    windows: List[VibrationWindow],
    stft_config: STFTConfig,
    fs: float,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple, tuple]:
    """
    Convert windows to model-ready arrays.

    Returns
    -------
    (wav_seq, spec_seq, labels, waveform_shape, spectrogram_shape)
    """
    # Waveforms: (N, window_size, 1)
    waveforms = np.array([w.data for w in windows], dtype=np.float32)
    waveforms = waveforms[..., np.newaxis]

    # Spectrograms: (N, F, T, 1)
    spectrograms = batch_spectrograms(windows, stft_config, fs, normalize=True)
    spectrograms = spectrograms.astype(np.float32)

    # Encode labels
    labels_encoded, idx_to_class, class_to_idx = encode_labels(windows)

    waveform_shape = waveforms.shape[1:]   # (window_size, 1)
    spectrogram_shape = spectrograms.shape[1:]  # (F, T, 1)

    # Create sequences
    wav_seq, spec_seq, label_seq = prepare_sequences(
        waveforms, spectrograms, labels_encoded, sequence_length
    )

    return wav_seq, spec_seq, label_seq, waveform_shape, spectrogram_shape, idx_to_class, class_to_idx


def compute_class_weights_from_labels(labels: np.ndarray) -> Dict[int, float]:
    """Compute class weights for handling imbalanced datasets."""
    unique = np.unique(labels)
    weights = compute_class_weight("balanced", classes=unique, y=labels)
    return {int(c): float(w) for c, w in zip(unique, weights)}


def train_model(
    train_windows: List[VibrationWindow],
    val_windows: List[VibrationWindow],
    model_config: ModelConfig,
    preprocess_config: PreprocessingConfig,
    windowing_config: WindowingConfig,
    stft_config: STFTConfig,
    normalizer: Normalizer,
    save_dir: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Full training pipeline.

    Parameters
    ----------
    train_windows : list of VibrationWindow
    val_windows : list of VibrationWindow
    model_config : ModelConfig
    preprocess_config : PreprocessingConfig
    windowing_config : WindowingConfig
    stft_config : STFTConfig
    normalizer : Normalizer
        Already fitted on training data.
    save_dir : str or None
        Directory to save the trained model.
    progress_callback : callable or None
        Called with (epoch, logs) after each epoch.

    Returns
    -------
    dict with keys: model, history, idx_to_class, class_to_idx, etc.
    """
    set_global_seed(model_config.random_seed)

    if save_dir is None:
        save_dir = os.path.join(SAVED_MODELS_DIR, "model")
    os.makedirs(save_dir, exist_ok=True)

    fs = preprocess_config.sampling_frequency
    seq_len = model_config.sequence_length

    # ── Prepare training data ──
    (train_wav, train_spec, train_labels,
     waveform_shape, spectrogram_shape,
     idx_to_class, class_to_idx) = prepare_training_data(
        train_windows, stft_config, fs, seq_len
    )

    # ── Prepare validation data ──
    val_waveforms = np.array([w.data for w in val_windows], dtype=np.float32)[..., np.newaxis]
    val_spectrograms = batch_spectrograms(val_windows, stft_config, fs, normalize=True).astype(np.float32)
    val_labels_encoded, _, _ = encode_labels(val_windows)
    val_wav, val_spec, val_labels = prepare_sequences(
        val_waveforms, val_spectrograms, val_labels_encoded, seq_len
    )

    num_classes = len(idx_to_class)
    model_config.num_classes = num_classes

    # ── Build model ──
    model = build_hybrid_model(
        waveform_shape=waveform_shape,
        spectrogram_shape=spectrogram_shape,
        num_classes=num_classes,
        config=model_config,
        sequence_length=seq_len if seq_len > 1 else 1,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=model_config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Class weights ──
    class_weights = compute_class_weights_from_labels(train_labels)

    # ── Callbacks ──
    checkpoint_path = os.path.join(save_dir, "best_model.keras")
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=model_config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=model_config.lr_reduce_factor,
            patience=model_config.lr_reduce_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Train ──
    history = model.fit(
        [train_wav, train_spec],
        train_labels,
        validation_data=([val_wav, val_spec], val_labels),
        epochs=model_config.epochs,
        batch_size=model_config.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save artifacts ──
    model.save(os.path.join(save_dir, "full_model.keras"))

    # Class mapping
    with open(os.path.join(save_dir, "class_mapping.json"), "w") as f:
        json.dump({"idx_to_class": {str(k): v for k, v in idx_to_class.items()},
                    "class_to_idx": class_to_idx}, f, indent=2)

    # Normalization params
    with open(os.path.join(save_dir, "normalization_params.json"), "w") as f:
        json.dump(normalizer.get_params(), f, indent=2)

    # Preprocessing config
    save_config_bundle(
        os.path.join(save_dir, "preprocessing_config.json"),
        preprocessing=preprocess_config,
        windowing=windowing_config,
        stft=stft_config,
        model=model_config,
    )

    # Training metadata
    metadata = {
        "num_classes": num_classes,
        "class_mapping": idx_to_class,
        "waveform_shape": list(waveform_shape),
        "spectrogram_shape": list(spectrogram_shape),
        "sequence_length": seq_len,
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }
    with open(os.path.join(save_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "model": model,
        "history": history.history,
        "idx_to_class": idx_to_class,
        "class_to_idx": class_to_idx,
        "waveform_shape": waveform_shape,
        "spectrogram_shape": spectrogram_shape,
        "metadata": metadata,
    }


def load_trained_model(model_dir: str) -> Dict[str, Any]:
    """
    Load a previously trained model and all its artifacts.

    Parameters
    ----------
    model_dir : str
        Path to the saved model directory.

    Returns
    -------
    dict with model, class_mapping, normalizer, configs, etc.
    """
    model_path = os.path.join(model_dir, "full_model.keras")
    if not os.path.exists(model_path):
        # Try .h5 fallback
        model_path = os.path.join(model_dir, "best_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found in {model_dir}")

    model = tf.keras.models.load_model(model_path)

    # Class mapping
    mapping_path = os.path.join(model_dir, "class_mapping.json")
    with open(mapping_path) as f:
        cm = json.load(f)
    idx_to_class = {int(k): v for k, v in cm["idx_to_class"].items()}
    class_to_idx = cm["class_to_idx"]

    # Normalizer
    norm_path = os.path.join(model_dir, "normalization_params.json")
    with open(norm_path) as f:
        norm_params = json.load(f)
    normalizer = Normalizer.from_params(norm_params)

    # Configs
    config_path = os.path.join(model_dir, "preprocessing_config.json")
    from config import load_config_bundle
    configs = load_config_bundle(config_path)

    preprocess_config = PreprocessingConfig.from_dict(configs.get("preprocessing", {}))
    windowing_config = WindowingConfig.from_dict(configs.get("windowing", {}))
    stft_config = STFTConfig.from_dict(configs.get("stft", {}))
    model_config = ModelConfig.from_dict(configs.get("model", {}))

    # Metadata
    meta_path = os.path.join(model_dir, "training_metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    return {
        "model": model,
        "idx_to_class": idx_to_class,
        "class_to_idx": class_to_idx,
        "normalizer": normalizer,
        "preprocess_config": preprocess_config,
        "windowing_config": windowing_config,
        "stft_config": stft_config,
        "model_config": model_config,
        "metadata": metadata,
    }
