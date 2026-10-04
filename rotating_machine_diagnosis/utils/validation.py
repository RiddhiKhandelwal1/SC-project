"""Utility: CSV and data validation helpers."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from config import ColumnMapping


class ValidationResult:
    """Container for validation messages."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str):
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_info(self, msg: str):
        self.info.append(msg)


def validate_csv(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    mode: str = "training",
) -> ValidationResult:
    """
    Validate a loaded CSV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame.
    mapping : ColumnMapping
        User-specified column mapping.
    mode : str
        'training' or 'inference'.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    # ── Empty check ──
    if df.empty:
        result.add_error("CSV file is empty (0 rows).")
        return result

    result.add_info(f"Rows: {len(df):,}")
    result.add_info(f"Columns: {len(df.columns):,}")

    # ── Vibration column ──
    if not mapping.vibration:
        result.add_error("No vibration column selected.")
        return result

    if mapping.vibration not in df.columns:
        result.add_error(
            f"Vibration column '{mapping.vibration}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
        return result

    vib = df[mapping.vibration]
    result.add_info(f"Vibration column: {mapping.vibration}")

    # Numeric check
    if not np.issubdtype(vib.dtype, np.number):
        # Try coercion
        coerced = pd.to_numeric(vib, errors="coerce")
        non_numeric = coerced.isna().sum() - vib.isna().sum()
        if non_numeric > 0:
            result.add_error(
                f"Vibration column contains {non_numeric:,} non-numeric values "
                f"that cannot be converted."
            )

    # NaN / Inf
    nan_count = vib.isna().sum()
    if nan_count > 0:
        pct = nan_count / len(vib) * 100
        if pct > 50:
            result.add_error(
                f"Vibration column has {nan_count:,} NaN values ({pct:.1f}%). "
                f"Too many missing values for reliable analysis."
            )
        else:
            result.add_warning(
                f"Vibration column has {nan_count:,} NaN values ({pct:.1f}%). "
                f"These will be handled during preprocessing."
            )

    if np.issubdtype(vib.dtype, np.number):
        inf_count = np.isinf(vib.values.astype(float)).sum()
        if inf_count > 0:
            result.add_warning(
                f"Vibration column has {inf_count:,} infinite values. "
                f"These will be replaced during preprocessing."
            )

    # Signal length
    valid_count = vib.dropna().shape[0]
    if valid_count < 256:
        result.add_error(
            f"Signal too short ({valid_count} valid samples). "
            f"At least 256 samples are required."
        )
    elif valid_count < 2048:
        result.add_warning(
            f"Signal is short ({valid_count} valid samples). "
            f"Results may be unreliable with very few windows."
        )

    # Duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        pct = dup_count / len(df) * 100
        result.add_warning(
            f"Found {dup_count:,} duplicate rows ({pct:.1f}%)."
        )

    # ── Label column (required for training) ──
    if mode == "training":
        if not mapping.label:
            result.add_error(
                "Label/fault column is required for training mode."
            )
        elif mapping.label not in df.columns:
            result.add_error(
                f"Label column '{mapping.label}' not found in CSV."
            )
        else:
            labels = df[mapping.label].dropna()
            n_classes = labels.nunique()
            result.add_info(f"Label column: {mapping.label}")
            result.add_info(f"Number of classes: {n_classes}")
            if n_classes < 2:
                result.add_error(
                    f"Only {n_classes} class found. Need at least 2 for classification."
                )
            class_counts = labels.value_counts()
            for cls_name, count in class_counts.items():
                result.add_info(f"  Class '{cls_name}': {count:,} samples")
            # Warn about imbalance
            if n_classes >= 2:
                min_c = class_counts.min()
                max_c = class_counts.max()
                if max_c > 10 * min_c:
                    result.add_warning(
                        "Severe class imbalance detected. "
                        "Class weighting will be applied during training."
                    )
    elif mode == "inference":
        if mapping.label and mapping.label in df.columns:
            result.add_info(
                f"Label column '{mapping.label}' detected (will be used for evaluation if desired)."
            )

    # ── Time column → estimate sampling frequency ──
    estimated_fs = None
    if mapping.time and mapping.time in df.columns:
        time_col = pd.to_numeric(df[mapping.time], errors="coerce")
        if time_col.notna().sum() > 1:
            diffs = time_col.dropna().diff().dropna()
            if len(diffs) > 0:
                median_dt = diffs.median()
                if median_dt > 0:
                    estimated_fs = 1.0 / median_dt
                    result.add_info(
                        f"Estimated sampling frequency: {estimated_fs:,.0f} Hz"
                    )
                else:
                    result.add_warning(
                        "Time column has zero or negative intervals. "
                        "Cannot estimate sampling frequency."
                    )
        else:
            result.add_warning("Time column has insufficient valid values.")

    if estimated_fs is None and mapping.time:
        result.add_warning(
            "Could not estimate sampling frequency from the time column. "
            "Please specify it manually."
        )

    # ── Optional columns ──
    for attr, col_name in [
        ("rpm", mapping.rpm),
        ("load", mapping.load),
        ("temperature", mapping.temperature),
        ("machine_id", mapping.machine_id),
        ("run_id", mapping.run_id),
    ]:
        if col_name:
            if col_name in df.columns:
                result.add_info(f"{attr} column: {col_name}")
            else:
                result.add_warning(
                    f"{attr} column '{col_name}' not found in CSV. Skipping."
                )

    return result


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Heuristic detection of likely column roles."""
    hints = {}
    for col in df.columns:
        lower = col.lower().strip()
        if any(k in lower for k in ["vibr", "accel", "vib", "signal", "amplitude"]):
            hints[col] = "vibration"
        elif any(k in lower for k in ["fault", "label", "class", "defect", "condition"]):
            hints[col] = "label"
        elif any(k in lower for k in ["time", "timestamp", "t_sec", "seconds"]):
            hints[col] = "time"
        elif any(k in lower for k in ["rpm", "speed", "rotational"]):
            hints[col] = "rpm"
        elif any(k in lower for k in ["load", "torque", "force"]):
            hints[col] = "load"
        elif any(k in lower for k in ["temp", "temperature", "thermo"]):
            hints[col] = "temperature"
        elif any(k in lower for k in ["machine", "asset", "equipment"]):
            hints[col] = "machine_id"
        elif any(k in lower for k in ["run", "record", "segment", "batch"]):
            hints[col] = "run_id"
        else:
            hints[col] = "unknown"
    return hints
