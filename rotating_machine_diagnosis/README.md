# Rotating Machine Fault Diagnosis from Vibration Signals

## Deep Learning + Fuzzy Logic for Condition Monitoring

A complete, modular Python application for rotating machine fault diagnosis using a hybrid **CNN-BiLSTM** deep learning architecture combined with **Mamdani fuzzy logic** risk assessment, all accessible through a **Streamlit** dashboard.

---

## 🏗️ Architecture

```
CSV Vibration Data
       ↓
Data Validation + Column Mapping
       ↓
Signal Processing (Filter → Normalize)
       ↓
Overlapping Window Generation
       ↓
 ┌───────────────────────────────────┐
 │                                   │
 ↓                                   ↓
Raw Waveform                  STFT Spectrogram
 ↓                                   ↓
1D CNN (Conv1D×3)             2D CNN (Conv2D×2)
 ↓                                   ↓
Feature Vector                Feature Vector
 │                                   │
 └──────────────┬────────────────────┘
                ↓
         Feature Fusion
                ↓
              BiLSTM
                ↓
         Dense + Softmax
                ↓
    Fault Classification
                ↓
      Fuzzy Risk Assessment
                ↓
    Risk Score (0–100) + Severity
                ↓
    Maintenance Recommendation
```

---

## 📁 Project Structure

```
rotating_machine_diagnosis/
├── app.py                      # Streamlit dashboard (main entry point)
├── config.py                   # All configuration dataclasses
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── preprocessing/
│   ├── csv_loader.py           # CSV loading and column extraction
│   ├── signal_processing.py    # Filtering, normalization, missing values
│   ├── windowing.py            # Overlapping window generation + group splitting
│   ├── spectrogram.py          # STFT and magnitude spectrogram
│   └── synthetic_data.py       # Synthetic data generator (demo only)
│
├── models/
│   ├── cnn_1d.py               # 1D CNN branch (raw waveform)
│   ├── cnn_2d.py               # 2D CNN branch (spectrogram)
│   ├── hybrid_model.py         # Hybrid CNN-BiLSTM model + feature fusion
│   └── train.py                # Training pipeline, saving, loading
│
├── fuzzy/
│   ├── features.py             # Traditional vibration features (RMS, kurtosis, etc.)
│   ├── fuzzy_system.py         # Mamdani fuzzy inference system
│   └── rules.py                # Configurable fuzzy rule base
│
├── evaluation/
│   └── metrics.py              # Classification metrics and confusion matrix
│
├── visualization/
│   └── plots.py                # All Matplotlib plotting functions
│
├── utils/
│   ├── seed.py                 # Reproducible random seed setting
│   └── validation.py           # CSV validation and column type detection
│
├── saved_models/               # Trained model artifacts
│
├── tests/
│   ├── test_csv_loader.py
│   ├── test_preprocessing.py
│   ├── test_windowing.py
│   ├── test_spectrogram.py
│   ├── test_features.py
│   └── test_fuzzy.py
│
└── data/
    ├── raw/
    └── processed/
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
cd rotating_machine_diagnosis
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

### 3. Use the Application

1. **Dataset page**: Upload a CSV or generate synthetic demo data
2. **Signal Analysis**: Configure preprocessing and visualize signals
3. **Model Training**: Set hyperparameters and train the hybrid model
4. **Fault Diagnosis**: Upload new data for inference
5. **Fuzzy Risk Assessment**: View risk score and severity
6. **Report**: Download the machine condition report

---

## 📊 CSV Input Format

The application supports **any CSV** with a vibration column. Use the column mapping UI to select:

| Role | Required | Example Column Name |
|------|----------|-------------------|
| Vibration | ✅ Yes | `acceleration_x`, `vibration`, `signal` |
| Label/Fault | Training only | `fault`, `label`, `class` |
| Time | Optional | `time`, `timestamp` |
| RPM | Optional | `rpm`, `speed` |
| Temperature | Optional | `temperature`, `temp` |
| Machine ID | Optional | `machine_id`, `asset` |
| Run ID | Optional | `run_id`, `recording` |

---

## 🧠 Deep Learning Model

**Parallel CNN Branches:**
- **1D CNN**: 3 Conv1D blocks → learns temporal patterns from raw waveforms
- **2D CNN**: 2 Conv2D blocks → learns time-frequency patterns from STFT spectrograms

**Feature Fusion**: Concatenation of both branch outputs

**Temporal Modeling**: BiLSTM over sequences of consecutive windows

**Classification**: Dense → Dropout → Softmax (auto-determined number of classes)

---

## 🎯 Fuzzy Logic Risk Assessment

After deep learning classification, the Mamdani fuzzy inference system evaluates:

| Input | Range | Membership Functions |
|-------|-------|---------------------|
| Fault Probability | 0–1 | Low, Medium, High |
| RMS | 0–1 | Low, Medium, High |
| Kurtosis | 0–20 | Low, Medium, High |

| Output | Range | Membership Functions |
|--------|-------|---------------------|
| Risk | 0–100 | Low, Medium, High, Critical |

**Severity Levels:**
- `0–25`: LOW — Routine monitoring
- `25–50`: MEDIUM — Schedule inspection
- `50–75`: HIGH — Inspect soon
- `75–100`: CRITICAL — Immediate inspection

---

## ⚠️ Disclaimer

This system provides an AI-based condition assessment and maintenance decision-support recommendation. It **does not guarantee mechanical failure** or replace professional inspection. Always consult qualified maintenance engineers before making critical decisions.

---

## 🧪 Testing

```bash
cd rotating_machine_diagnosis
python -m pytest tests/ -v
```

---

## 📜 Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Deep Learning | TensorFlow / Keras |
| Signal Processing | SciPy |
| Fuzzy Logic | scikit-fuzzy |
| ML Utilities | scikit-learn |
| Dashboard | Streamlit |
| Visualization | Matplotlib |
| Data Processing | Pandas, NumPy |
