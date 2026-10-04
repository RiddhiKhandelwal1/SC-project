"""
2D CNN Branch — Learns time-frequency features from STFT spectrograms.

Architecture:
    Conv2D → BatchNorm → ReLU → MaxPool
    Conv2D → BatchNorm → ReLU → MaxPool
    GlobalAveragePooling2D
    → Feature Vector

The 2D CNN captures spectral-temporal patterns that the 1D CNN cannot.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ModelConfig


def build_cnn2d_branch(
    input_shape: tuple,
    config: ModelConfig,
    name: str = "cnn2d_branch",
) -> Model:
    """
    Build the 2D CNN branch for spectrogram feature extraction.

    Parameters
    ----------
    input_shape : tuple
        Shape of a spectrogram, e.g., (freq_bins, time_bins, 1).
    config : ModelConfig
        Model hyperparameters.
    name : str
        Model name.

    Returns
    -------
    tf.keras.Model
        Input: (batch, freq_bins, time_bins, 1)
        Output: (batch, feature_dim)
    """
    filters = config.cnn2d_filters
    kernels = config.cnn2d_kernel_sizes

    inp = layers.Input(shape=input_shape, name=f"{name}_input")
    x = inp

    # Block 1
    x = layers.Conv2D(
        filters[0], (kernels[0], kernels[0]),
        padding="same", name=f"{name}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool1")(x)

    # Block 2
    x = layers.Conv2D(
        filters[1], (kernels[1], kernels[1]),
        padding="same", name=f"{name}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool2")(x)

    # Global Average Pooling → Feature Vector
    x = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)

    model = Model(inputs=inp, outputs=x, name=name)
    return model
