import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_clean_dataset(output_path: str, fs=12000.0, seed=42):
    """
    Generate a clean dataset with very distinct, non-overlapping fault signatures
    to ensure the model can achieve high accuracy.
    """
    rng = np.random.RandomState(seed)

    n_runs = 10            # 10 runs per class for good splitting
    samples_per_run = 12000  # 1 second per run at 12 kHz

    all_dfs = []

    for run_i in range(n_runs):
        t = np.arange(samples_per_run) / fs
        n = samples_per_run
        run_seed = seed + run_i * 100
        run_rng = np.random.RandomState(run_seed)

        # Baseline noise (very low)
        bg_noise = run_rng.normal(0, 0.01, n)

        # === NORMAL: Just low amplitude 30 Hz + noise ===
        normal = 0.1 * np.sin(2 * np.pi * 30 * t) + bg_noise

        # === UNBALANCE: High amplitude 30 Hz ===
        # Very distinct from normal due to much higher amplitude
        unbalance = 0.8 * np.sin(2 * np.pi * 30 * t) + bg_noise

        # === BEARING FAULT: Strong distinct impulses ===
        bearing = 0.1 * np.sin(2 * np.pi * 30 * t) + bg_noise
        impulse_period = int(fs / 100) # 100 Hz impulses
        for j in range(0, n, impulse_period):
            imp_len = min(30, n - j)
            decay = np.exp(-np.linspace(0, 5, imp_len))
            bearing[j:j + imp_len] += 1.5 * decay * run_rng.choice([-1, 1])

        # === MISALIGNMENT: Strong 2x component (60 Hz) ===
        misalignment = 0.1 * np.sin(2 * np.pi * 30 * t) + 0.6 * np.sin(2 * np.pi * 60 * t) + bg_noise

        # === GEAR FAULT: Mesh frequency (600 Hz) ===
        gear = 0.5 * np.sin(2 * np.pi * 600 * t) + bg_noise

        # Build DataFrames for each class
        for fault_name, signal in [
            ("normal", normal),
            ("unbalance", unbalance),
            ("bearing_fault", bearing),
            ("misalignment", misalignment),
            ("gear_fault", gear),
        ]:
            base_temp = 38.0 if fault_name == "normal" else 55.0 + run_rng.uniform(0, 5)
            df_run = pd.DataFrame({
                "time": t,
                "vibration": signal,
                "fault": fault_name,
                "run_id": f"{fault_name}_run_{run_i}",
                "temperature": base_temp + run_rng.normal(0, 1.0, n),
            })
            all_dfs.append(df_run)

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Clean dataset saved: {output_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Classes: {sorted(df['fault'].unique())}")
    print(f"   Runs per class: {n_runs}")
    print(f"   Samples per run: {samples_per_run:,}")
    return df

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clean_test_dataset.csv")
    generate_clean_dataset(output_path)
