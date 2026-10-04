"""
Rotating Machine Fault Diagnosis — Streamlit Dashboard

Main application entry point.  Multi-page dashboard with:
  1. Home — Overview and architecture
  2. Dataset — CSV upload, validation, column mapping
  3. Signal Analysis — Raw / filtered / FFT / spectrogram
  4. Model Training — Configure & train the hybrid CNN-BiLSTM
  5. Fault Diagnosis — Upload new data for inference
  6. Fuzzy Risk Assessment — Risk score, severity, memberships
  7. Report — Downloadable machine condition report
"""

import os
import sys
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")

# ── Project root on path ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    ColumnMapping, PreprocessingConfig, WindowingConfig,
    STFTConfig, ModelConfig, FuzzyConfig,
    SAVED_MODELS_DIR, DISCLAIMER,
    RECOMMENDATION_THRESHOLDS, RECOMMENDATION_MESSAGES,
)
from utils.seed import set_global_seed
from utils.validation import validate_csv, detect_column_types, ValidationResult
from preprocessing.csv_loader import (
    load_csv_from_bytes, extract_vibration_signal,
    extract_labels, extract_metadata, get_column_summary,
)
from preprocessing.signal_processing import (
    handle_missing_values, apply_filter, Normalizer, preprocess_signal,
    PreprocessingConfig as PPConfig,
)
from preprocessing.windowing import (
    generate_windows, group_level_split, VibrationWindow, WindowingConfig as WConfig,
)
from preprocessing.spectrogram import (
    compute_magnitude_spectrogram, normalize_spectrogram, batch_spectrograms,
    STFTConfig as SConfig,
)
from preprocessing.synthetic_data import generate_synthetic_dataset
from fuzzy.features import compute_all_features
from fuzzy.fuzzy_system import FuzzyRiskAssessor
from fuzzy.rules import get_default_rules, rules_to_text
from visualization.plots import (
    plot_raw_signal, plot_filtered_signal, plot_fft,
    plot_spectrogram, plot_training_history, plot_confusion_matrix,
    plot_class_probabilities, plot_risk_gauge, plot_membership_functions,
    plot_feature_summary, SEVERITY_COLORS,
)

# ─────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rotating Machine Fault Diagnosis",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme enhancements */
    .main { background-color: #1E1E2E; }
    .stApp { background-color: #1E1E2E; }

    .metric-card {
        background: linear-gradient(135deg, #2D2D44 0%, #1E1E2E 100%);
        border: 1px solid #4a4a6a;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .severity-low { color: #10B981; font-weight: bold; }
    .severity-medium { color: #F59E0B; font-weight: bold; }
    .severity-high { color: #F97316; font-weight: bold; }
    .severity-critical { color: #EF4444; font-weight: bold; font-size: 1.5rem; }

    .report-box {
        background: linear-gradient(135deg, #2D2D44 0%, #1a1a2e 100%);
        border: 1px solid #6366F1;
        border-radius: 16px;
        padding: 30px;
        margin: 15px 0;
    }

    .architecture-box {
        background: #2D2D44;
        border: 1px solid #4a4a6a;
        border-radius: 12px;
        padding: 20px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.6;
        white-space: pre;
        overflow-x: auto;
    }

    .header-gradient {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }

    .disclaimer {
        background: #2D2D44;
        border-left: 4px solid #F59E0B;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
        font-size: 0.85rem;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #2D2D44 100%);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "df": None,
        "mapping": ColumnMapping(),
        "preprocess_config": PreprocessingConfig(),
        "windowing_config": WindowingConfig(),
        "stft_config": STFTConfig(),
        "model_config": ModelConfig(),
        "fuzzy_config": FuzzyConfig(),
        "validation_result": None,
        "raw_signal": None,
        "processed_signal": None,
        "normalizer": None,
        "windows": None,
        "split_data": None,
        "trained_model": None,
        "training_history": None,
        "idx_to_class": None,
        "class_to_idx": None,
        "metadata": None,
        "inference_df": None,
        "inference_mapping": ColumnMapping(),
        "diagnosis_results": None,
        "fuzzy_results": None,
        "features": None,
        "is_synthetic": False,
        "model_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ─────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────
st.sidebar.markdown('<p class="header-gradient" style="font-size:1.5rem;">⚙️ Fault Diagnosis</p>', unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 Signal Analysis",
        "🧠 Model Training",
        "🔍 Fault Diagnosis",
        "🎯 Fuzzy Risk Assessment",
        "📋 Report",
    ],
    label_visibility="collapsed",
)

# Model status indicator
model_dir = os.path.join(SAVED_MODELS_DIR, "model")
model_exists = os.path.exists(os.path.join(model_dir, "full_model.keras"))
st.sidebar.markdown("---")
if model_exists:
    st.sidebar.success("✅ Trained model available")
else:
    st.sidebar.warning("⚠️ No trained model found")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-size:0.75rem; color:#9CA3AF; text-align:center;'>"
    "Deep Learning + Fuzzy Logic<br>Machine Fault Diagnosis System</p>",
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<p class="header-gradient">Rotating Machine Fault Diagnosis</p>', unsafe_allow_html=True)
    st.markdown("### Deep Learning + Fuzzy Logic for Vibration-Based Condition Monitoring")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Deep Learning</p>
            <p class="metric-value">CNN + BiLSTM</p>
            <p style="color:#9CA3AF; font-size:0.85rem;">Parallel 1D/2D CNN branches with BiLSTM temporal modeling</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Fuzzy Logic</p>
            <p class="metric-value">Risk Score</p>
            <p style="color:#9CA3AF; font-size:0.85rem;">Mamdani inference for severity assessment (0–100)</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-label">Input Format</p>
            <p class="metric-value">CSV Upload</p>
            <p style="color:#9CA3AF; font-size:0.85rem;">Flexible column mapping — any CSV vibration dataset</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
    <div class="architecture-box">
CSV Vibration Data
       ↓
Data Validation + Column Mapping
       ↓
Signal Processing (Filter → Normalize)
       ↓
Overlapping Window Generation
       ↓
 ┌─────────────────────────────────────┐
 │                                     │
 ↓                                     ↓
Raw Waveform                    STFT Spectrogram
 ↓                                     ↓
1D CNN (Conv1D×3)               2D CNN (Conv2D×2)
 ↓                                     ↓
Feature Vector                  Feature Vector
 │                                     │
 └──────────────┬──────────────────────┘
                ↓
         Feature Fusion (Concatenation)
                ↓
         BiLSTM (Temporal Modeling)
                ↓
         Dense + Dropout
                ↓
         Softmax Classification
                ↓
    Fault Probability Distribution
                ↓
   ┌────────────────────────┐
   │  Fuzzy Logic System    │
   │  (Mamdani Inference)   │
   │                        │
   │  Inputs:               │
   │   • Fault Probability  │
   │   • RMS                │
   │   • Kurtosis           │
   │                        │
   │  Output:               │
   │   Risk Score (0–100)   │
   └────────────────────────┘
                ↓
    Severity: LOW / MEDIUM / HIGH / CRITICAL
                ↓
    Maintenance Recommendation
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚠️ Important Limitations")
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)

    st.markdown("""
    **Additional notes:**
    - The model learns from the training data you provide — results depend on data quality
    - Synthetic data mode is for pipeline testing only — not real-world performance
    - The 1D CNN learns from raw waveforms; the 2D CNN learns from STFT spectrograms
    - Both branches are genuine trainable neural networks — not hand-crafted features
    - The fuzzy system assesses **severity/risk** — the deep learning model classifies **fault type**
    """)


# ═════════════════════════════════════════════════════════
# PAGE: DATASET
# ═════════════════════════════════════════════════════════
elif page == "📊 Dataset":
    st.markdown('<p class="header-gradient">Dataset Configuration</p>', unsafe_allow_html=True)
    st.markdown("---")

    tab_upload, tab_synthetic = st.tabs(["📁 Upload CSV", "🧪 Synthetic Data (Demo)"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload your vibration CSV dataset",
            type=["csv"],
            help="CSV file with at least a vibration column",
        )

        if uploaded_file is not None:
            try:
                df = load_csv_from_bytes(uploaded_file.getvalue(), filename=uploaded_file.name)
                st.session_state.df = df
                st.session_state.is_synthetic = False
                st.success(f"✅ CSV loaded: **{uploaded_file.name}** — {len(df):,} rows × {len(df.columns)} columns")
            except Exception as e:
                st.error(f"❌ Failed to load CSV: {e}")

    with tab_synthetic:
        st.warning("⚠️ **DEMONSTRATION / SYNTHETIC DATA** — For pipeline testing only. Not real-world data.")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            synth_samples = st.number_input("Samples per class", 5000, 100000, 20000, 5000)
        with col_s2:
            synth_fs = st.number_input("Sampling freq (Hz)", 1000, 50000, 12000, 1000)
        with col_s3:
            synth_runs = st.number_input("Runs per class", 2, 20, 5)

        if st.button("🧪 Generate Synthetic Dataset", use_container_width=True):
            with st.spinner("Generating synthetic vibration data..."):
                df = generate_synthetic_dataset(
                    n_samples_per_class=synth_samples,
                    fs=synth_fs,
                    n_runs_per_class=synth_runs,
                )
                st.session_state.df = df
                st.session_state.is_synthetic = True
                st.success(f"✅ Synthetic dataset generated: {len(df):,} rows × {len(df.columns)} columns")

    # ── Column Mapping ──
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.markdown("### 🗂️ Column Mapping")
        st.markdown("Map your CSV columns to the pipeline roles.")

        if st.session_state.is_synthetic:
            st.info("🧪 **Synthetic data detected** — columns auto-mapped.")

        # Auto-detect hints
        hints = detect_column_types(df)
        columns = ["(none)"] + list(df.columns)

        def find_hint(role):
            for col, hint in hints.items():
                if hint == role:
                    return col
            return "(none)"

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            default_vib = find_hint("vibration")
            vib_col = st.selectbox("🔊 Vibration Column **(required)**", columns,
                                   index=columns.index(default_vib) if default_vib in columns else 0)
            default_label = find_hint("label")
            label_col = st.selectbox("🏷️ Label/Fault Column", columns,
                                     index=columns.index(default_label) if default_label in columns else 0)
            default_time = find_hint("time")
            time_col = st.selectbox("⏱️ Time Column", columns,
                                    index=columns.index(default_time) if default_time in columns else 0)
            default_rpm = find_hint("rpm")
            rpm_col = st.selectbox("🔄 RPM Column", columns,
                                   index=columns.index(default_rpm) if default_rpm in columns else 0)

        with col_m2:
            default_load = find_hint("load")
            load_col = st.selectbox("⚡ Load Column", columns,
                                    index=columns.index(default_load) if default_load in columns else 0)
            default_temp = find_hint("temperature")
            temp_col = st.selectbox("🌡️ Temperature Column", columns,
                                    index=columns.index(default_temp) if default_temp in columns else 0)
            default_machine = find_hint("machine_id")
            machine_col = st.selectbox("🏭 Machine ID Column", columns,
                                       index=columns.index(default_machine) if default_machine in columns else 0)
            default_run = find_hint("run_id")
            run_col = st.selectbox("📦 Run/Recording ID Column", columns,
                                   index=columns.index(default_run) if default_run in columns else 0)

        # Update mapping
        mapping = ColumnMapping(
            vibration=vib_col if vib_col != "(none)" else "",
            label=label_col if label_col != "(none)" else None,
            time=time_col if time_col != "(none)" else None,
            rpm=rpm_col if rpm_col != "(none)" else None,
            load=load_col if load_col != "(none)" else None,
            temperature=temp_col if temp_col != "(none)" else None,
            machine_id=machine_col if machine_col != "(none)" else None,
            run_id=run_col if run_col != "(none)" else None,
        )
        st.session_state.mapping = mapping

        # ── Validation ──
        st.markdown("---")
        st.markdown("### ✅ Dataset Validation")

        mode = st.radio("Mode", ["Training", "Inference"], horizontal=True)
        mode_key = "training" if mode == "Training" else "inference"

        result = validate_csv(df, mapping, mode=mode_key)
        st.session_state.validation_result = result

        if result.is_valid:
            for msg in result.info:
                st.markdown(f"✅ {msg}")
            for msg in result.warnings:
                st.warning(f"⚠️ {msg}")
        else:
            for msg in result.errors:
                st.error(f"❌ {msg}")
            for msg in result.warnings:
                st.warning(f"⚠️ {msg}")
            for msg in result.info:
                st.info(f"ℹ️ {msg}")

        # ── Data Preview ──
        st.markdown("---")
        st.markdown("### 📊 Data Preview")

        col_prev1, col_prev2 = st.columns(2)
        with col_prev1:
            st.markdown("**First 10 rows**")
            st.dataframe(df.head(10), use_container_width=True)
        with col_prev2:
            st.markdown("**Column Summary**")
            summary = get_column_summary(df)
            st.dataframe(summary, use_container_width=True)

        # ── Basic statistics ──
        if mapping.vibration and mapping.vibration in df.columns:
            st.markdown("---")
            st.markdown("### 📈 Vibration Signal Preview")
            raw_sig = extract_vibration_signal(df, mapping)
            raw_sig_clean = handle_missing_values(raw_sig, strategy="interpolate")
            st.session_state.raw_signal = raw_sig_clean

            # Show a slice
            max_display = min(50000, len(raw_sig_clean))
            fs_est = st.session_state.preprocess_config.sampling_frequency
            fig = plot_raw_signal(raw_sig_clean[:max_display], fs=fs_est,
                                  title=f"Vibration Signal (first {max_display:,} samples)")
            st.pyplot(fig)

            # Stats
            st.markdown("**Signal Statistics**")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Mean", f"{np.mean(raw_sig_clean):.6f}")
            col_s2.metric("Std Dev", f"{np.std(raw_sig_clean):.6f}")
            col_s3.metric("Min", f"{np.min(raw_sig_clean):.6f}")
            col_s4.metric("Max", f"{np.max(raw_sig_clean):.6f}")


# ═════════════════════════════════════════════════════════
# PAGE: SIGNAL ANALYSIS
# ═════════════════════════════════════════════════════════
elif page == "📈 Signal Analysis":
    st.markdown('<p class="header-gradient">Signal Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.df is None:
        st.warning("Please upload a dataset first in the **Dataset** page.")
        st.stop()

    df = st.session_state.df
    mapping = st.session_state.mapping

    if not mapping.vibration or mapping.vibration not in df.columns:
        st.error("Please select a valid vibration column in the Dataset page.")
        st.stop()

    # ── Preprocessing Configuration ──
    st.markdown("### ⚙️ Preprocessing Configuration")

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        fs = st.number_input("Sampling Frequency (Hz)", 100, 100000,
                             int(st.session_state.preprocess_config.sampling_frequency), 100)
        missing_strategy = st.selectbox("Missing Value Strategy",
                                        ["interpolate", "drop", "ffill", "bfill"],
                                        index=0)
    with col_c2:
        filter_enabled = st.checkbox("Enable Filtering", value=True)
        filter_type = st.selectbox("Filter Type", ["bandpass", "lowpass", "highpass"])
        filter_order = st.slider("Filter Order", 1, 10, 5)
    with col_c3:
        lowcut = st.number_input("Low Cutoff (Hz)", 1.0, float(fs//2 - 1), 10.0, 1.0)
        highcut = st.number_input("High Cutoff (Hz)", lowcut + 1, float(fs//2 - 1), min(5000.0, float(fs//2 - 1)), 1.0)
        norm_method = st.selectbox("Normalization", ["standardize", "minmax"])

    pp_config = PreprocessingConfig(
        sampling_frequency=float(fs),
        missing_value_strategy=missing_strategy,
        filter_enabled=filter_enabled,
        filter_type=filter_type,
        filter_order=filter_order,
        lowcut=lowcut,
        highcut=highcut,
        normalization_method=norm_method,
    )
    st.session_state.preprocess_config = pp_config

    # ── Process Signal ──
    if st.button("🔬 Process Signal", use_container_width=True):
        with st.spinner("Processing vibration signal..."):
            raw_signal = extract_vibration_signal(df, mapping)
            cleaned = handle_missing_values(raw_signal, strategy=pp_config.missing_value_strategy)

            try:
                filtered = apply_filter(cleaned, pp_config) if pp_config.filter_enabled else cleaned
            except ValueError as e:
                st.error(f"Filter error: {e}")
                filtered = cleaned

            normalizer = Normalizer(method=pp_config.normalization_method)
            normalized = normalizer.fit_transform(filtered)

            st.session_state.raw_signal = cleaned
            st.session_state.processed_signal = normalized
            st.session_state.normalizer = normalizer

            st.success("✅ Signal processed successfully!")

            # ── Visualizations ──
            st.markdown("---")
            st.markdown("### 📊 Signal Visualizations")

            # Raw vs Filtered
            max_disp = min(20000, len(cleaned))
            fig1 = plot_filtered_signal(cleaned[:max_disp], filtered[:max_disp], fs=fs)
            st.pyplot(fig1)

            # FFT
            fig2 = plot_fft(filtered[:max_disp], fs)
            st.pyplot(fig2)

            # STFT / Spectrogram
            st.markdown("### 🎵 STFT Spectrogram")
            col_stft1, col_stft2, col_stft3 = st.columns(3)
            with col_stft1:
                n_fft = st.number_input("n_fft", 64, 2048, 256, 64)
            with col_stft2:
                hop_length = st.number_input("hop_length", 16, 512, 64, 16)
            with col_stft3:
                win_length = st.number_input("win_length", 64, 2048, 256, 64)

            stft_config = STFTConfig(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            st.session_state.stft_config = stft_config

            # Compute spectrogram for a sample window
            sample_window = normalized[:min(2048, len(normalized))]
            spec = compute_magnitude_spectrogram(sample_window, stft_config, fs)
            fig3 = plot_spectrogram(spec, fs, hop_length, title="Sample Window Spectrogram")
            st.pyplot(fig3)


# ═════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING
# ═════════════════════════════════════════════════════════
elif page == "🧠 Model Training":
    st.markdown('<p class="header-gradient">Model Training</p>', unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.df is None:
        st.warning("Please upload a dataset first in the **Dataset** page.")
        st.stop()

    df = st.session_state.df
    mapping = st.session_state.mapping

    if not mapping.vibration or not mapping.label:
        st.error("Both vibration and label columns are required for training. "
                 "Please configure them in the Dataset page.")
        st.stop()

    if st.session_state.is_synthetic:
        st.warning("⚠️ **SYNTHETIC DATA MODE** — Training on synthetic data for demonstration. "
                   "Do not interpret these results as real-world performance.")

    # ── Training Configuration ──
    st.markdown("### ⚙️ Training Configuration")

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("**Signal Processing**")
        fs = st.number_input("Sampling Freq (Hz)", 100, 100000,
                             int(st.session_state.preprocess_config.sampling_frequency), key="train_fs")
        window_size = st.number_input("Window Size", 256, 8192, 2048, 256, key="train_ws")
        overlap = st.slider("Window Overlap", 0.0, 0.9, 0.5, 0.1, key="train_ov")

    with col_t2:
        st.markdown("**STFT Parameters**")
        n_fft = st.number_input("n_fft", 64, 2048, 256, key="train_nfft")
        hop = st.number_input("hop_length", 16, 512, 64, key="train_hop")
        seq_len = st.number_input("Sequence Length (BiLSTM)", 1, 20, 5, key="train_seq")

    with col_t3:
        st.markdown("**Training Hyper-parameters**")
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        epochs = st.number_input("Max Epochs", 5, 200, 50)
        lr = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
        val_split = st.slider("Validation Split", 0.05, 0.3, 0.15, 0.05)
        test_split = st.slider("Test Split", 0.05, 0.3, 0.15, 0.05)

    # Update configs
    pp_config = st.session_state.preprocess_config
    pp_config.sampling_frequency = float(fs)

    wc = WindowingConfig(window_size=window_size, overlap=overlap)
    st.session_state.windowing_config = wc

    sc = STFTConfig(n_fft=n_fft, hop_length=hop, win_length=n_fft)
    st.session_state.stft_config = sc

    mc = st.session_state.model_config
    mc.batch_size = batch_size
    mc.epochs = epochs
    mc.learning_rate = lr
    mc.validation_split = val_split
    mc.test_split = test_split
    mc.sequence_length = seq_len

    # ── Train Button ──
    st.markdown("---")
    if st.button("🚀 Start Training", use_container_width=True, type="primary"):
        progress_bar = st.progress(0, text="Preparing data...")

        try:
            # Step 1: Extract & preprocess
            progress_bar.progress(5, text="Extracting vibration signal...")
            raw_signal = extract_vibration_signal(df, mapping)
            labels = extract_labels(df, mapping)
            meta = extract_metadata(df, mapping)

            # Step 2: Preprocess
            progress_bar.progress(10, text="Preprocessing signal...")
            processed, normalizer = preprocess_signal(raw_signal, pp_config)
            st.session_state.normalizer = normalizer

            # Step 3: Windowing
            progress_bar.progress(20, text="Generating windows...")
            windows = generate_windows(
                processed, wc,
                labels=labels,
                run_ids=meta.get("run_id"),
                machine_ids=meta.get("machine_id"),
                source_file="training_data",
            )

            # Filter windows with labels
            windows = [w for w in windows if w.label is not None]

            if len(windows) < 10:
                st.error(f"Only {len(windows)} labeled windows generated. "
                         f"Need more data or smaller window size.")
                st.stop()

            st.info(f"Generated **{len(windows):,}** labeled windows")

            # Step 4: Split
            progress_bar.progress(30, text="Splitting dataset...")
            split = group_level_split(windows, val_ratio=val_split, test_ratio=test_split,
                                      seed=mc.random_seed)

            if split["leakage_warning"]:
                st.warning(
                    "⚠️ **Data Leakage Warning**: No run_id or machine_id found. "
                    "Windows are randomly split, which may produce optimistic results "
                    "if overlapping windows share the same recording."
                )

            st.info(f"Train: {len(split['train']):,} | Val: {len(split['val']):,} | Test: {len(split['test']):,}")

            # Step 5: Train
            progress_bar.progress(40, text="Building model and starting training...")

            from models.train import train_model as do_train

            results = do_train(
                train_windows=split["train"],
                val_windows=split["val"],
                model_config=mc,
                preprocess_config=pp_config,
                windowing_config=wc,
                stft_config=sc,
                normalizer=normalizer,
            )

            st.session_state.trained_model = results["model"]
            st.session_state.training_history = results["history"]
            st.session_state.idx_to_class = results["idx_to_class"]
            st.session_state.class_to_idx = results["class_to_idx"]
            st.session_state.metadata = results["metadata"]
            st.session_state.split_data = split
            st.session_state.model_loaded = True

            progress_bar.progress(90, text="Evaluating on test set...")

            # Step 6: Evaluate on test set
            from models.train import prepare_training_data
            from evaluation.metrics import compute_metrics, get_predictions

            test_windows = split["test"]
            if len(test_windows) > 0:
                (test_wav, test_spec, test_labels,
                 _, _, _, _) = prepare_training_data(test_windows, sc, float(fs), seq_len)

                test_preds = get_predictions(results["model"], test_wav, test_spec)

                metrics = compute_metrics(test_labels, test_preds, results["idx_to_class"])

                progress_bar.progress(100, text="✅ Training complete!")

                # Display results
                st.markdown("---")
                st.markdown("### 📊 Training Results")

                # Training curves
                fig_hist = plot_training_history(results["history"])
                st.pyplot(fig_hist)

                # Metrics
                st.markdown("### 📋 Test Set Evaluation")
                for w in metrics["warnings"]:
                    st.warning(w)

                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                col_r1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                col_r2.metric("Macro F1", f"{metrics['macro_f1']:.2%}")
                col_r3.metric("Weighted F1", f"{metrics['weighted_f1']:.2%}")
                col_r4.metric("Macro Precision", f"{metrics['macro_precision']:.2%}")

                # Confusion matrix
                class_names = [results["idx_to_class"][i] for i in sorted(results["idx_to_class"])]
                fig_cm = plot_confusion_matrix(metrics["confusion_matrix"], class_names)
                st.pyplot(fig_cm)

                # Classification report
                st.markdown("### 📝 Classification Report")
                st.code(metrics["classification_report"])
            else:
                progress_bar.progress(100, text="✅ Training complete (no test data)")
                st.warning("No test samples available for evaluation.")

            st.success("🎉 Model trained and saved successfully!")

        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            import traceback
            st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════
# PAGE: FAULT DIAGNOSIS
# ═════════════════════════════════════════════════════════
elif page == "🔍 Fault Diagnosis":
    st.markdown('<p class="header-gradient">Fault Diagnosis</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Check for model
    model_loaded = st.session_state.model_loaded and st.session_state.trained_model is not None

    if not model_loaded:
        # Try loading from disk
        model_dir = os.path.join(SAVED_MODELS_DIR, "model")
        if os.path.exists(os.path.join(model_dir, "full_model.keras")):
            if st.button("📥 Load Saved Model", use_container_width=True):
                try:
                    from models.train import load_trained_model
                    loaded = load_trained_model(model_dir)
                    st.session_state.trained_model = loaded["model"]
                    st.session_state.idx_to_class = loaded["idx_to_class"]
                    st.session_state.class_to_idx = loaded["class_to_idx"]
                    st.session_state.normalizer = loaded["normalizer"]
                    st.session_state.preprocess_config = loaded["preprocess_config"]
                    st.session_state.windowing_config = loaded["windowing_config"]
                    st.session_state.stft_config = loaded["stft_config"]
                    st.session_state.model_config = loaded["model_config"]
                    st.session_state.metadata = loaded["metadata"]
                    st.session_state.model_loaded = True
                    model_loaded = True
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
        else:
            st.warning("No trained model available. Please train a model first.")
            st.stop()

    if not model_loaded:
        st.stop()

    # ── Upload inference CSV ──
    st.markdown("### 📁 Upload Data for Diagnosis")
    inf_file = st.file_uploader("Upload vibration CSV for diagnosis", type=["csv"], key="inf_upload")

    if inf_file is not None:
        try:
            inf_df = load_csv_from_bytes(inf_file.getvalue(), filename=inf_file.name)
            st.session_state.inference_df = inf_df
            st.success(f"✅ Loaded: {len(inf_df):,} rows")
        except Exception as e:
            st.error(f"Failed to load: {e}")

    # Also allow using current dataset
    if st.session_state.inference_df is None and st.session_state.df is not None:
        if st.button("Use current training dataset for diagnosis"):
            st.session_state.inference_df = st.session_state.df
            st.session_state.inference_mapping = st.session_state.mapping
            st.rerun()

    if st.session_state.inference_df is not None:
        inf_df = st.session_state.inference_df
        columns = ["(none)"] + list(inf_df.columns)

        st.markdown("### 🗂️ Column Selection")
        vib_col = st.selectbox("Vibration Column", columns, key="inf_vib")

        inf_mapping = ColumnMapping(
            vibration=vib_col if vib_col != "(none)" else "",
        )
        st.session_state.inference_mapping = inf_mapping

        if inf_mapping.vibration and inf_mapping.vibration in inf_df.columns:
            if st.button("🔍 Run Diagnosis", use_container_width=True, type="primary"):
                with st.spinner("Running fault diagnosis pipeline..."):
                    try:
                        model = st.session_state.trained_model
                        normalizer = st.session_state.normalizer
                        pp_config = st.session_state.preprocess_config
                        wc = st.session_state.windowing_config
                        sc = st.session_state.stft_config
                        mc = st.session_state.model_config
                        idx_to_class = st.session_state.idx_to_class

                        # Extract & preprocess
                        raw_sig = extract_vibration_signal(inf_df, inf_mapping)
                        cleaned = handle_missing_values(raw_sig, strategy=pp_config.missing_value_strategy)

                        try:
                            filtered = apply_filter(cleaned, pp_config) if pp_config.filter_enabled else cleaned
                        except Exception:
                            filtered = cleaned

                        normalized = normalizer.transform(filtered)

                        # Window
                        windows = generate_windows(normalized, wc, source_file="inference")

                        if len(windows) == 0:
                            st.error("No windows could be generated. Signal may be too short.")
                            st.stop()

                        # Prepare data
                        waveforms = np.array([w.data for w in windows], dtype=np.float32)[..., np.newaxis]
                        spectrograms = batch_spectrograms(windows, sc, pp_config.sampling_frequency,
                                                          normalize=True).astype(np.float32)

                        seq_len = mc.sequence_length
                        from models.hybrid_model import prepare_sequences
                        dummy_labels = np.zeros(len(waveforms), dtype=np.int32)
                        wav_seq, spec_seq, _ = prepare_sequences(
                            waveforms, spectrograms, dummy_labels, seq_len
                        )

                        # Predict
                        probs = model.predict([wav_seq, spec_seq], verbose=0)

                        # Average predictions across all windows/sequences
                        avg_probs = np.mean(probs, axis=0)
                        pred_class_idx = int(np.argmax(avg_probs))
                        pred_class = idx_to_class.get(pred_class_idx, f"Class {pred_class_idx}")
                        confidence = float(avg_probs[pred_class_idx])

                        # Compute traditional features
                        features = compute_all_features(normalized)

                        # Store results
                        st.session_state.diagnosis_results = {
                            "predicted_fault": pred_class,
                            "confidence": confidence,
                            "probabilities": avg_probs,
                            "class_names": [idx_to_class.get(i, str(i)) for i in range(len(avg_probs))],
                            "per_window_probs": probs,
                        }
                        st.session_state.features = features

                        # ── Display Results ──
                        st.markdown("---")
                        st.markdown("### 🎯 Diagnosis Results")

                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <p class="metric-label">Predicted Fault</p>
                                <p class="metric-value">{pred_class}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_d2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <p class="metric-label">Confidence</p>
                                <p class="metric-value">{confidence:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Probability distribution
                        class_names = [idx_to_class.get(i, str(i)) for i in range(len(avg_probs))]
                        fig_prob = plot_class_probabilities(avg_probs, class_names)
                        st.pyplot(fig_prob)

                        # Feature summary
                        st.markdown("### 📊 Vibration Features")
                        fig_feat = plot_feature_summary(features)
                        st.pyplot(fig_feat)

                        st.success("✅ Diagnosis complete! Proceed to **Fuzzy Risk Assessment** for severity analysis.")

                    except Exception as e:
                        st.error(f"Diagnosis failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════
# PAGE: FUZZY RISK ASSESSMENT
# ═════════════════════════════════════════════════════════
elif page == "🎯 Fuzzy Risk Assessment":
    st.markdown('<p class="header-gradient">Fuzzy Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    The fuzzy inference system evaluates machine **risk/severity** based on:
    - **Fault probability** from the deep learning classifier
    - **RMS** vibration level
    - **Kurtosis** (impulsiveness indicator)

    This is separate from fault classification — it answers **"how serious is the condition?"**
    """)

    diagnosis = st.session_state.diagnosis_results
    features = st.session_state.features

    if diagnosis is None or features is None:
        st.warning("Please run **Fault Diagnosis** first.")

        # Manual input mode
        st.markdown("---")
        st.markdown("### 🔧 Manual Input Mode")
        st.info("You can also manually enter values to explore the fuzzy system.")

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            man_fp = st.slider("Fault Probability", 0.0, 1.0, 0.5, 0.01)
        with col_f2:
            man_rms = st.slider("RMS (normalized 0–1)", 0.0, 1.0, 0.3, 0.01)
        with col_f3:
            man_kurt = st.slider("Kurtosis", 0.0, 20.0, 3.0, 0.5)

        if st.button("🎯 Assess Risk (Manual)", use_container_width=True):
            fuzzy_system = FuzzyRiskAssessor(config=st.session_state.fuzzy_config)
            fuzzy_result = fuzzy_system.assess_risk(man_fp, man_rms, man_kurt)
            st.session_state.fuzzy_results = fuzzy_result

            st.markdown("---")
            _display_fuzzy_results(fuzzy_result) if False else None

            # Inline display
            severity = fuzzy_result["severity"]
            sev_class = f"severity-{severity.lower()}"
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <p class="metric-label">Machine Risk Score</p>
                <p class="metric-value">{fuzzy_result['risk_score']:.0f} / 100</p>
                <p class="{sev_class}" style="font-size:1.8rem;">⬤ {severity}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Recommendation:** {fuzzy_result['recommendation']}")

            fig_gauge = plot_risk_gauge(fuzzy_result["risk_score"], severity)
            st.pyplot(fig_gauge)

            fig_memb = plot_membership_functions(fuzzy_result["membership_values"])
            st.pyplot(fig_memb)

    else:
        # Automatic from diagnosis
        fault_prob = float(diagnosis["confidence"])
        rms_val = features.get("rms", 0.0)
        kurt_val = features.get("kurtosis", 3.0)

        # Normalize RMS to 0-1 range (heuristic)
        rms_normalized = min(rms_val, 1.0)

        st.markdown("### 📊 Input Values")
        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        col_i1.metric("Fault Probability", f"{fault_prob:.2%}")
        col_i2.metric("RMS", f"{rms_val:.4f}")
        col_i3.metric("Kurtosis", f"{kurt_val:.2f}")
        col_i4.metric("Crest Factor", f"{features.get('crest_factor', 0):.2f}")

        if features.get("temperature") is not None:
            st.metric("Temperature", f"{features['temperature']:.1f}°C")

        if st.button("🎯 Compute Risk Assessment", use_container_width=True, type="primary"):
            fuzzy_system = FuzzyRiskAssessor(config=st.session_state.fuzzy_config)
            fuzzy_result = fuzzy_system.assess_risk(fault_prob, rms_normalized, kurt_val)
            st.session_state.fuzzy_results = fuzzy_result

            severity = fuzzy_result["severity"]
            sev_class = f"severity-{severity.lower()}"

            st.markdown("---")
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <p class="metric-label">Machine Risk Score</p>
                <p class="metric-value">{fuzzy_result['risk_score']:.0f} / 100</p>
                <p class="{sev_class}" style="font-size:1.8rem;">⬤ {severity}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Recommendation:** {fuzzy_result['recommendation']}")

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_gauge = plot_risk_gauge(fuzzy_result["risk_score"], severity)
                st.pyplot(fig_gauge)
            with col_g2:
                fig_memb = plot_membership_functions(fuzzy_result["membership_values"])
                st.pyplot(fig_memb)

            # Show rules
            st.markdown("### 📜 Active Fuzzy Rules")
            rules_text = rules_to_text(get_default_rules())
            st.code(rules_text, language="text")

    st.markdown("---")
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
# PAGE: REPORT
# ═════════════════════════════════════════════════════════
elif page == "📋 Report":
    st.markdown('<p class="header-gradient">Machine Condition Report</p>', unsafe_allow_html=True)
    st.markdown("---")

    diagnosis = st.session_state.diagnosis_results
    features = st.session_state.features
    fuzzy_result = st.session_state.fuzzy_results

    if diagnosis is None:
        st.warning("Please run fault diagnosis first.")
        st.stop()

    pred_fault = diagnosis["predicted_fault"]
    confidence = diagnosis["confidence"]

    # ── Build Report ──
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("        MACHINE CONDITION REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append(f"  Predicted Fault:    {pred_fault}")
    report_lines.append(f"  Confidence:         {confidence:.1%}")
    report_lines.append("")

    if features:
        report_lines.append("  ── Vibration Features ──")
        report_lines.append(f"  RMS:                {features.get('rms', 'N/A'):.6f}" if isinstance(features.get('rms'), float) else f"  RMS:                N/A")
        report_lines.append(f"  Peak Amplitude:     {features.get('peak_amplitude', 'N/A'):.6f}" if isinstance(features.get('peak_amplitude'), float) else f"  Peak:               N/A")
        report_lines.append(f"  Kurtosis:           {features.get('kurtosis', 'N/A'):.4f}" if isinstance(features.get('kurtosis'), float) else f"  Kurtosis:           N/A")
        report_lines.append(f"  Crest Factor:       {features.get('crest_factor', 'N/A'):.4f}" if isinstance(features.get('crest_factor'), float) else f"  Crest Factor:       N/A")
        if features.get("temperature") is not None:
            report_lines.append(f"  Temperature:        {features['temperature']:.1f}°C")
        report_lines.append("")

    if fuzzy_result:
        report_lines.append("  ── Risk Assessment ──")
        report_lines.append(f"  Risk Score:         {fuzzy_result['risk_score']:.0f} / 100")
        report_lines.append(f"  Severity:           {fuzzy_result['severity']}")
        report_lines.append(f"  Recommendation:     {fuzzy_result['recommendation']}")
        report_lines.append("")

    report_lines.append("  ── Class Probabilities ──")
    for name, prob in zip(diagnosis["class_names"], diagnosis["probabilities"]):
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        report_lines.append(f"  {name:20s} {bar} {prob:.1%}")

    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append("  DISCLAIMER:")
    report_lines.append("  This system provides an AI-based condition assessment")
    report_lines.append("  and maintenance decision-support recommendation.")
    report_lines.append("  It does not guarantee mechanical failure or replace")
    report_lines.append("  professional inspection.")
    report_lines.append("-" * 60)

    report_text = "\n".join(report_lines)

    # ── Display ──
    st.markdown(f"""
    <div class="report-box">
        <h3 style="text-align:center; color:#6366F1;">⚙️ MACHINE CONDITION REPORT</h3>
    </div>
    """, unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Predicted Fault</p>
            <p class="metric-value">{pred_fault}</p>
            <p style="color:#9CA3AF;">Confidence: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_r2:
        if fuzzy_result:
            severity = fuzzy_result["severity"]
            sev_color = SEVERITY_COLORS.get(severity, "#fff")
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Machine Risk</p>
                <p class="metric-value" style="background:none; -webkit-text-fill-color:{sev_color}; color:{sev_color};">
                    {fuzzy_result['risk_score']:.0f} / 100
                </p>
                <p style="color:{sev_color}; font-weight:bold;">{severity}</p>
            </div>
            """, unsafe_allow_html=True)

    # Features
    if features:
        st.markdown("### 📊 Vibration Features")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        col_f1.metric("RMS", f"{features.get('rms', 0):.6f}")
        col_f2.metric("Peak", f"{features.get('peak_amplitude', 0):.6f}")
        col_f3.metric("Kurtosis", f"{features.get('kurtosis', 0):.4f}")
        col_f4.metric("Crest Factor", f"{features.get('crest_factor', 0):.4f}")

    # Class probabilities
    if diagnosis.get("probabilities") is not None:
        fig_prob = plot_class_probabilities(
            diagnosis["probabilities"], diagnosis["class_names"],
            title="Fault Class Probability Distribution"
        )
        st.pyplot(fig_prob)

    # Risk gauge
    if fuzzy_result:
        fig_gauge = plot_risk_gauge(fuzzy_result["risk_score"], fuzzy_result["severity"])
        st.pyplot(fig_gauge)

        st.markdown(f"**Recommended Action:** {fuzzy_result['recommendation']}")

    # Full text report
    st.markdown("### 📝 Full Report")
    st.code(report_text, language="text")

    # Download
    st.download_button(
        label="📥 Download Report",
        data=report_text,
        file_name="machine_condition_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
