"""
Generate Demo Dataset — Creates a compact CSV with realistic fault signatures
that share enough overlap to prevent trivial 100% classification.

Run:  python3.12 generate_demo_data.py

Design choices for realistic accuracy:
  - All faults share a common 30 Hz rotational base frequency
  - Fault severity varies across runs (mild → severe)
  - Inter-class noise contamination adds realistic overlap
  - Run-to-run amplitude/phase variation prevents memorization
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_demo_dataset(output_path: str, fs=12000.0, seed=42):
    """
    Generate a demo dataset with overlapping fault signatures.

    Key differences from the previous generator:
      - Normal and unbalance both have 30 Hz; unbalance just has ~2-3x higher
        amplitude but with overlap in the mild-severity runs
      - Misalignment has both 1x and 2x present (not just 2x)
      - Bearing impulses are weaker and share the 30 Hz background
      - Gear fault has lower mesh amplitude with noise
      - Severity varies per run (0.3 to 1.0 scale)
    """
    rng = np.random.RandomState(seed)

    n_runs = 10            # 10 runs per class → enough for group-level split
    samples_per_run = 12000  # 1 second per run at 12 kHz

    all_dfs = []

    for run_i in range(n_runs):
        t = np.arange(samples_per_run) / fs
        n = samples_per_run
        run_seed = seed + run_i * 17 + 3
        run_rng = np.random.RandomState(run_seed)

        # Severity factor varies per run (0.3 = mild, 1.0 = severe)
        severity = 0.3 + 0.7 * (run_i / max(n_runs - 1, 1))
        # Add some random jitter to severity
        severity = np.clip(severity + run_rng.uniform(-0.15, 0.15), 0.2, 1.0)

        # Random phase offsets per run
        phase_1x = run_rng.uniform(0, 2 * np.pi)
        phase_2x = run_rng.uniform(0, 2 * np.pi)

        # Shared background noise level (all classes get this)
        bg_noise = run_rng.normal(0, 0.08, n)

        # Common 30 Hz rotational component (present in ALL classes)
        common_1x = 0.15 * np.sin(2 * np.pi * 30 * t + phase_1x)

        # === NORMAL: common rotation + moderate noise ===
        normal = common_1x.copy()
        normal += 0.03 * np.sin(2 * np.pi * 60 * t)  # weak 2x
        normal += bg_noise
        normal += run_rng.normal(0, 0.04, n)

        # === UNBALANCE: elevated 1x (but overlaps with normal at low severity) ===
        unbal_amp = 0.15 + 0.45 * severity  # 0.15 to 0.60 (normal is 0.15)
        unbalance = unbal_amp * np.sin(2 * np.pi * 30 * t + phase_1x)
        unbalance += 0.08 * np.sin(2 * np.pi * 60 * t)  # some 2x
        unbalance += bg_noise
        unbalance += run_rng.normal(0, 0.06, n)

        # === BEARING FAULT: impulses on top of normal background ===
        bearing = common_1x.copy()
        bearing += bg_noise
        impulse_period = int(fs / (90 + run_rng.uniform(-10, 10)))  # ~90 Hz BPFO
        impulse_amp = 0.3 + 1.0 * severity  # weaker impulses
        for j in range(0, n, impulse_period):
            imp_len = min(25, n - j)
            decay = np.exp(-np.linspace(0, 6, imp_len))
            bearing[j:j + imp_len] += impulse_amp * decay * run_rng.choice([-1, 1])
        bearing += run_rng.normal(0, 0.05, n)

        # === MISALIGNMENT: elevated 2x, but 1x still present ===
        misalign_2x_amp = 0.15 + 0.40 * severity
        misalignment = common_1x.copy()  # 1x still there
        misalignment += misalign_2x_amp * np.sin(2 * np.pi * 60 * t + phase_2x)
        misalignment += (0.08 * severity) * np.sin(2 * np.pi * 90 * t)  # 3x
        misalignment += bg_noise
        misalignment += run_rng.normal(0, 0.05, n)

        # === GEAR FAULT: AM at mesh frequency, but with strong low-freq content ===
        carrier = np.sin(2 * np.pi * 600 * t)
        modulator = 1 + 0.5 * severity * np.sin(2 * np.pi * 30 * t)
        gear_amp = 0.15 + 0.30 * severity
        gear = gear_amp * carrier * modulator
        gear += common_1x  # rotational component still present
        gear += 0.08 * severity * np.sin(2 * np.pi * 570 * t)  # sidebands
        gear += 0.08 * severity * np.sin(2 * np.pi * 630 * t)
        gear += bg_noise
        gear += run_rng.normal(0, 0.06, n)

        # Build DataFrames for each class
        for fault_name, signal in [
            ("normal", normal),
            ("unbalance", unbalance),
            ("bearing_fault", bearing),
            ("misalignment", misalignment),
            ("gear_fault", gear),
        ]:
            base_temp = 38.0 if fault_name == "normal" else 48.0 + run_rng.uniform(0, 12)
            df_run = pd.DataFrame({
                "time": t,
                "vibration": signal,
                "fault": fault_name,
                "run_id": f"{fault_name}_run_{run_i}",
                "temperature": base_temp + run_rng.normal(0, 2.0, n),
            })
            all_dfs.append(df_run)

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset saved: {output_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Classes: {sorted(df['fault'].unique())}")
    print(f"   Runs per class: {n_runs}")
    print(f"   Samples per run: {samples_per_run:,}")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    print()
    print("   Class distribution:")
    for cls, count in df["fault"].value_counts().sort_index().items():
        print(f"     {cls:20s}  {count:,} samples")
    print()
    print("   Signal amplitude ranges:")
    for cls in sorted(df["fault"].unique()):
        subset = df[df["fault"] == cls]["vibration"]
        print(f"     {cls:20s}  std={subset.std():.4f}  range=[{subset.min():.3f}, {subset.max():.3f}]")

    return df


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "demo_vibration_dataset.csv")

    print("=" * 60)
    print("  Generating Demo Vibration Dataset")
    print("  ⚠️  SYNTHETIC DATA — FOR DEMONSTRATION ONLY")
    print("=" * 60)
    print()

    generate_demo_dataset(output_path)

    print()
    print("=" * 60)
    print("  Upload this file in the Streamlit dashboard:")
    print("  1. Go to Dataset page")
    print("  2. Upload demo_vibration_dataset.csv")
    print("  3. Go to Model Training → Start Training")
    print("=" * 60)


if __name__ == "__main__":
    main()
