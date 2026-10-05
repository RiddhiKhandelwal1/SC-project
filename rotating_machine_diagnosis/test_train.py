import pandas as pd
from preprocessing.csv_loader import extract_vibration_signal, extract_labels
from preprocessing.signal_processing import preprocess_signal, Normalizer
from preprocessing.windowing import generate_windows, WindowingConfig
from config import PreprocessingConfig, STFTConfig, ModelConfig
from models.train import train_model

df = pd.read_csv('data/raw/hard_test_dataset.csv')
p_cfg = PreprocessingConfig(sampling_frequency=12000.0)
w_cfg = WindowingConfig(window_size=1024, overlap=0.5)
s_cfg = STFTConfig()
m_cfg = ModelConfig(epochs=2)

normalizer = Normalizer(method='standardize')
normalizer.fit(df['vibration'].values)
processed = normalizer.transform(df['vibration'].values)
labels = df['fault'].values
run_ids = df['run_id'].values

windows = generate_windows(processed, w_cfg, labels, run_ids)

# Train/val split
from sklearn.model_selection import train_test_split
import numpy as np

indices = np.arange(len(windows))
window_labels = np.array([w.label for w in windows])
train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=window_labels, random_state=42)

train_w = [windows[i] for i in train_idx]
val_w = [windows[i] for i in val_idx]

train_model(train_w, val_w, m_cfg, p_cfg, w_cfg, s_cfg, normalizer, 'test_save')
