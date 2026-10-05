import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_hard_dataset(output_path: str, fs=12000.0, seed=42):
    """
    Generate a very challenging dataset aiming for 70-80% accuracy.
    This simulates early-stage faults in a very noisy industrial environment:
    - High background noise (low Signal-to-Noise Ratio)
    - Very weak fault signatures (early stage faults)
    - Significant frequency and amplitude drift
    - Normal class sometimes has transient spikes imitating faults
    """
    rng = np.random.RandomState(seed)

    n_runs = 10            # 10 runs per class
    samples_per_run = 12000  # 1 second per run at 12 kHz

    all_dfs = []

    for run_i in range(n_runs):
        t = np.arange(samples_per_run) / fs
        n = samples_per_run
        run_seed = seed + run_i * 999
        run_rng = np.random.RandomState(run_seed)

        # High RPM drift (28 Hz to 32 Hz)
        rot_freq = 30.0 + run_rng.uniform(-2.0, 2.0)
        
        # Heavy background noise masking the signals
        bg_noise = run_rng.normal(0, 0.25, n)  # Increased noise floor

        # Base rotation is dominant and highly variable
        base_rotation = run_rng.uniform(0.1, 0.3) * np.sin(2 * np.pi * rot_freq * t)

        # === NORMAL: Base rotation + high noise + random transient spikes ===
        normal = base_rotation + bg_noise
        # Sometimes normal has random mechanical bumps
        if run_rng.rand() > 0.5:
            spike_idx = run_rng.randint(0, n)
            normal[spike_idx:spike_idx+20] += run_rng.uniform(0.5, 1.0)

        # === UNBALANCE: Only slightly elevated 1x ===
        # Very hard to distinguish from normal runs that naturally have higher 1x
        unbalance_amp = run_rng.uniform(0.2, 0.4) 
        unbalance = unbalance_amp * np.sin(2 * np.pi * rot_freq * t) + bg_noise

        # === BEARING FAULT: Very weak impulses heavily masked by noise ===
        bearing = base_rotation + bg_noise
        bpfo = 115.0 + run_rng.uniform(-5, 5)
        impulse_period = int(fs / bpfo) 
        impulse_amp = run_rng.uniform(0.2, 0.3) # Weak early-stage spalling
        for j in range(0, n, impulse_period):
            imp_len = min(20, n - j)
            decay = np.exp(-np.linspace(0, 6, imp_len))
            bearing[j:j + imp_len] += impulse_amp * decay * run_rng.choice([-1, 1])

        # === MISALIGNMENT: Weak 2x component ===
        misalign_2x_amp = run_rng.uniform(0.15, 0.25) # Blends into noise
        misalignment = base_rotation + misalign_2x_amp * np.sin(2 * np.pi * (2 * rot_freq) * t) + bg_noise

        # === GEAR FAULT: Mesh frequency but highly attenuated ===
        mesh_freq = 600.0 + run_rng.uniform(-15, 15)
        carrier = np.sin(2 * np.pi * mesh_freq * t)
        modulator = 1 + run_rng.uniform(0.1, 0.3) * np.sin(2 * np.pi * rot_freq * t)
        gear_amp = run_rng.uniform(0.1, 0.2)
        gear = gear_amp * carrier * modulator + base_rotation + bg_noise

        # Build DataFrames for each class
        for fault_name, signal in [
            ("normal", normal),
            ("unbalance", unbalance),
            ("bearing_fault", bearing),
            ("misalignment", misalignment),
            ("gear_fault", gear),
        ]:
            # Temperatures are also overlapping heavily now
            base_temp = run_rng.uniform(40.0, 65.0) 
            df_run = pd.DataFrame({
                "time": t,
                "vibration": signal,
                "fault": fault_name,
                "run_id": f"{fault_name}_run_{run_i}",
                "temperature": base_temp + run_rng.normal(0, 3.0, n),
            })
            all_dfs.append(df_run)

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Hard dataset saved: {output_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Classes: {sorted(df['fault'].unique())}")

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hard_test_dataset.csv")
    generate_hard_dataset(output_path)
