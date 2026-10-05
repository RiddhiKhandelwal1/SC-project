"""
1D CNN Branch — Learns temporal features directly from raw vibration waveforms.

Architecture:
    Conv1D → BatchNorm → ReLU → MaxPool
    Conv1D → BatchNorm → ReLU → MaxPool
    Conv1D → GlobalAveragePooling1D
    → Feature Vector
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ModelConfig


def build_cnn1d_branch(
    input_shape: tuple,
    config: ModelConfig,
    name: str = "cnn1d_branch",
) -> Model:
    """
    Build the 1D CNN branch for raw waveform feature extraction.

    Parameters
    ----------
    input_shape : tuple
        Shape of a single window, e.g., (2048, 1).
    config : ModelConfig
        Model hyperparameters.
    name : str
        Model name.

    Returns
    -------
    tf.keras.Model
        Input: (batch, window_size, 1)
        Output: (batch, feature_dim)
    """
    filters = config.cnn1d_filters
    kernels = config.cnn1d_kernel_sizes

    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    x = inp

    # Dynamic Conv blocks based on config
    for i, (f, k) in enumerate(zip(filters, kernels)):
        x = layers.Conv1D(f, k, padding="same", name=f"{name}_conv{i+1}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i+1}")(x)
        x = layers.ReLU(name=f"{name}_relu{i+1}")(x)
        # Add pooling on all blocks except the last
        if i < len(filters) - 1:
            x = layers.MaxPooling1D(pool_size=2, name=f"{name}_pool{i+1}")(x)

    # Global Average Pooling → Feature Vector
    x = layers.GlobalAveragePooling1D(name=f"{name}_gap")(x)

    model = Model(inputs=inp, outputs=x, name=name)
    return model
