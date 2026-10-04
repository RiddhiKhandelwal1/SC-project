"""
CSV Loader — Load and prepare CSV vibration datasets.

Handles file reading, basic type coercion, and column extraction
based on the user's ColumnMapping.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ColumnMapping


def load_csv(filepath: str, **read_kwargs) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    **read_kwargs
        Additional keyword arguments forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError, pd.errors.EmptyDataError, etc.
    """
    df = pd.read_csv(filepath, **read_kwargs)
    return df


def load_csv_from_bytes(file_bytes, filename: str = "upload.csv", **read_kwargs) -> pd.DataFrame:
    """Load a CSV from an in-memory file-like object (Streamlit upload)."""
    import io
    if isinstance(file_bytes, bytes):
        file_bytes = io.BytesIO(file_bytes)
    df = pd.read_csv(file_bytes, **read_kwargs)
    return df


def extract_vibration_signal(
    df: pd.DataFrame, mapping: ColumnMapping
) -> np.ndarray:
    """
    Extract the vibration column as a 1-D NumPy array.

    Non-numeric values are coerced to NaN.
    """
    col = mapping.vibration
    if col not in df.columns:
        raise KeyError(f"Vibration column '{col}' not found in DataFrame.")
    series = pd.to_numeric(df[col], errors="coerce")
    return series.values.astype(np.float64)


def extract_labels(
    df: pd.DataFrame, mapping: ColumnMapping
) -> Optional[np.ndarray]:
    """
    Extract labels as a 1-D array of strings (or None).
    """
    if not mapping.label or mapping.label not in df.columns:
        return None
    return df[mapping.label].astype(str).values


def extract_metadata(
    df: pd.DataFrame, mapping: ColumnMapping
) -> Dict[str, Optional[np.ndarray]]:
    """
    Extract optional metadata columns.

    Returns a dict with keys: time, rpm, load, temperature, machine_id, run_id.
    Values are NumPy arrays or None.
    """
    meta = {}
    for attr in ["time", "rpm", "load", "temperature", "machine_id", "run_id"]:
        col = getattr(mapping, attr, None)
        if col and col in df.columns:
            if attr in ("machine_id", "run_id"):
                meta[attr] = df[col].astype(str).values
            else:
                meta[attr] = pd.to_numeric(df[col], errors="coerce").values.astype(np.float64)
        else:
            meta[attr] = None
    return meta


def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame with dtype, missing count, unique count, etc."""
    summary = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Non-Null": [df[c].notna().sum() for c in df.columns],
        "Missing": [df[c].isna().sum() for c in df.columns],
        "Unique": [df[c].nunique() for c in df.columns],
    })
    return summary.reset_index(drop=True)
