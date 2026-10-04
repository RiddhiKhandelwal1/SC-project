"""Utility: Reproducible random seeds."""

import os
import random
import numpy as np


def set_global_seed(seed: int = 42):
    """Set seeds for Python, NumPy, and TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
