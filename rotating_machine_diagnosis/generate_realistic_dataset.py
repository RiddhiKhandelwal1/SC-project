import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_realistic_dataset(output_path: str, fs=12000.0, seed=42):
    """
    Generate a dataset with realistic noise and moderate overlap.
    This aims for a realistic accuracy (around 85-95%) by adding:
    - Moderate random background noise
    - Slight frequency variations
    - Amplitude variations per run
    """
    rng = np.random.RandomState(seed)

    n_runs = 10            # 10 runs per class
    samples_per_run = 12000  # 1 second per run at 12 kHz

    all_dfs = []

    for run_i in range(n_runs):
        t = np.arange(samples_per_run) / fs
        n = samples_per_run
        run_seed = seed + run_i * 123
        run_rng = np.random.RandomState(run_seed)

        # Vary the baseline RPM slightly per run (e.g. 29.5 to 30.5 Hz)
        rot_freq = 30.0 + run_rng.uniform(-0.5, 0.5)
        
        # Moderate baseline noise (simulating sensor noise + background machine vibration)
        bg_noise = run_rng.normal(0, 0.08, n)

        # Baseline rotational vibration present in all classes
        base_rotation = 0.15 * np.sin(2 * np.pi * rot_freq * t)

        # === NORMAL: Base rotation + noise ===
        normal = base_rotation + bg_noise + run_rng.normal(0, 0.02, n)

        # === UNBALANCE: Higher 1x amplitude ===
        # Amplitude varies per run to create some harder cases
        unbalance_amp = run_rng.uniform(0.3, 0.6)
        unbalance = unbalance_amp * np.sin(2 * np.pi * rot_freq * t) + bg_noise

        # === BEARING FAULT: Weak impulses + base rotation ===
        bearing = base_rotation + bg_noise
        # BPFO around 110-120 Hz
        bpfo = 115.0 + run_rng.uniform(-3, 3)
        impulse_period = int(fs / bpfo) 
        impulse_amp = run_rng.uniform(0.4, 0.8)
        for j in range(0, n, impulse_period):
            imp_len = min(40, n - j)
            decay = np.exp(-np.linspace(0, 4, imp_len))
            bearing[j:j + imp_len] += impulse_amp * decay * run_rng.choice([-1, 1])

        # === MISALIGNMENT: Strong 2x component + base rotation ===
        misalign_2x_amp = run_rng.uniform(0.3, 0.5)
        misalignment = base_rotation + misalign_2x_amp * np.sin(2 * np.pi * (2 * rot_freq) * t) + bg_noise

        # === GEAR FAULT: Mesh frequency (600 Hz) amplitude modulated by rotation ===
        mesh_freq = 600.0 + run_rng.uniform(-5, 5)
        carrier = np.sin(2 * np.pi * mesh_freq * t)
        modulator = 1 + run_rng.uniform(0.3, 0.6) * np.sin(2 * np.pi * rot_freq * t)
        gear_amp = run_rng.uniform(0.2, 0.4)
        gear = gear_amp * carrier * modulator + base_rotation + bg_noise

        # Build DataFrames for each class
        for fault_name, signal in [
            ("normal", normal),
            ("unbalance", unbalance),
            ("bearing_fault", bearing),
            ("misalignment", misalignment),
            ("gear_fault", gear),
        ]:
            base_temp = 40.0 if fault_name == "normal" else 50.0 + run_rng.uniform(0, 10)
            df_run = pd.DataFrame({
                "time": t,
                "vibration": signal,
                "fault": fault_name,
                "run_id": f"{fault_name}_run_{run_i}",
                "temperature": base_temp + run_rng.normal(0, 1.5, n),
            })
            all_dfs.append(df_run)

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Realistic dataset saved: {output_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Classes: {sorted(df['fault'].unique())}")

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "realistic_test_dataset.csv")
    generate_realistic_dataset(output_path)
