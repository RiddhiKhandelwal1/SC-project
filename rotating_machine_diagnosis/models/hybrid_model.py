"""
Hybrid Model — Parallel 1D CNN + 2D CNN → Feature Fusion → BiLSTM → Softmax.

This is the main classification model.  It combines:
  • 1D CNN (raw waveform) → temporal features
  • 2D CNN (spectrogram)  → time-frequency features
  • Concatenation (feature fusion)
  • BiLSTM over sequences of consecutive windows
  • Dense + Dropout + Softmax

When only a single window is available (no sequential context),
the model uses a sequence length of 1 with a documented fallback.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ModelConfig
from models.cnn_1d import build_cnn1d_branch
from models.cnn_2d import build_cnn2d_branch


def build_hybrid_model(
    waveform_shape: tuple,
    spectrogram_shape: tuple,
    num_classes: int,
    config: ModelConfig,
    sequence_length: int = 1,
) -> Model:
    """
    Build the full hybrid CNN-BiLSTM model.

    Parameters
    ----------
    waveform_shape : tuple
        Shape of one waveform window, e.g. (2048, 1).
    spectrogram_shape : tuple
        Shape of one spectrogram, e.g. (freq_bins, time_bins, 1).
    num_classes : int
        Number of fault classes (determined from training data).
    config : ModelConfig
    sequence_length : int
        Number of consecutive windows in a sequence for the BiLSTM.
        Use 1 for single-window inference (fallback mode).

    Returns
    -------
    tf.keras.Model
        Inputs:  waveform_seq (batch, seq_len, *waveform_shape)
                 spectrogram_seq (batch, seq_len, *spectrogram_shape)
        Outputs: class_probabilities (batch, num_classes)
    """
    # ── Inputs ──
    waveform_input = layers.Input(
        shape=(sequence_length, *waveform_shape),
        name="waveform_input",
    )
    spectrogram_input = layers.Input(
        shape=(sequence_length, *spectrogram_shape),
        name="spectrogram_input",
    )

    # ── Build branches (shared weights across time steps) ──
    cnn1d = build_cnn1d_branch(waveform_shape, config)
    cnn2d = build_cnn2d_branch(spectrogram_shape, config)

    # ── Apply branches to each time step (TimeDistributed) ──
    wav_features = layers.TimeDistributed(cnn1d, name="td_cnn1d")(waveform_input)
    spec_features = layers.TimeDistributed(cnn2d, name="td_cnn2d")(spectrogram_input)

    # ── Feature Fusion (concatenation per time step) ──
    fused = layers.Concatenate(axis=-1, name="feature_fusion")([wav_features, spec_features])

    # ── BiLSTM ──
    x = layers.Bidirectional(
        layers.LSTM(config.lstm_units, return_sequences=False, name="lstm"),
        name="bilstm",
    )(fused)
    x = layers.Dropout(config.dropout_rate, name="dropout")(x)

    # ── Dense → Softmax ──
    x = layers.Dense(config.dense_units, activation="relu", name="dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax_output")(x)

    model = Model(
        inputs=[waveform_input, spectrogram_input],
        outputs=outputs,
        name="hybrid_cnn_bilstm",
    )
    return model


def build_feature_extractor(model: Model) -> Model:
    """
    Build a model that outputs fused features (before BiLSTM).
    Useful for explainability / visualization.
    """
    fusion_layer = model.get_layer("feature_fusion")
    return Model(
        inputs=model.inputs,
        outputs=fusion_layer.output,
        name="feature_extractor",
    )


def prepare_sequences(
    waveforms: np.ndarray,
    spectrograms: np.ndarray,
    labels: np.ndarray,
    sequence_length: int,
) -> tuple:
    """
    Organize individual windows into sequences for the BiLSTM.

    Parameters
    ----------
    waveforms : np.ndarray
        Shape (N, window_size, 1).
    spectrograms : np.ndarray
        Shape (N, freq_bins, time_bins, 1).
    labels : np.ndarray
        Shape (N,) integer labels.
    sequence_length : int
        Number of consecutive windows per sequence.

    Returns
    -------
    (wav_seq, spec_seq, label_seq)
        wav_seq:   (M, seq_len, window_size, 1)
        spec_seq:  (M, seq_len, freq, time, 1)
        label_seq: (M,) — label of the last window in each sequence.
    """
    N = len(waveforms)
    if sequence_length <= 1 or N < sequence_length:
        # Fallback: treat each window as a length-1 sequence
        wav_seq = waveforms[:, np.newaxis, ...]   # (N, 1, ws, 1)
        spec_seq = spectrograms[:, np.newaxis, ...]  # (N, 1, F, T, 1)
        return wav_seq, spec_seq, labels

    wav_seqs, spec_seqs, lab_seqs = [], [], []
    for i in range(N - sequence_length + 1):
        wav_seqs.append(waveforms[i:i + sequence_length])
        spec_seqs.append(spectrograms[i:i + sequence_length])
        lab_seqs.append(labels[i + sequence_length - 1])

    return (
        np.array(wav_seqs),
        np.array(spec_seqs),
        np.array(lab_seqs),
    )
