"""
Generate Demo Synthetic CSV — Creates a ready-to-use CSV file
in data/raw/ for demonstrating the project.

Run this script once:
    python3.12 generate_demo_data.py

It produces:  data/raw/demo_vibration_dataset.csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.synthetic_data import generate_synthetic_dataset


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "demo_vibration_dataset.csv")

    print("=" * 60)
    print("  Generating Demo Synthetic Vibration Dataset")
    print("  ⚠️  DEMONSTRATION DATA ONLY — NOT REAL-WORLD")
    print("=" * 60)
    print()

    df = generate_synthetic_dataset(
        n_samples_per_class=20000,   # 20k samples per fault class
        fs=12000.0,                  # 12 kHz sampling frequency
        n_runs_per_class=5,          # 5 separate runs per class
        seed=42,
    )

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset saved to: {output_path}")
    print(f"   Rows:    {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    print()
    print("   Column mapping for the dashboard:")
    print("     Vibration Column:  vibration")
    print("     Label Column:      fault")
    print("     Time Column:       time")
    print("     Run ID Column:     run_id")
    print("     Temperature:       temperature")
    print()

    # Print class distribution
    print("   Class distribution:")
    for cls, count in df["fault"].value_counts().sort_index().items():
        print(f"     {cls:20s}  {count:,} samples")
    print()

    # Quick stats
    print("   Vibration signal stats:")
    print(f"     Mean:  {df['vibration'].mean():.6f}")
    print(f"     Std:   {df['vibration'].std():.6f}")
    print(f"     Min:   {df['vibration'].min():.6f}")
    print(f"     Max:   {df['vibration'].max():.6f}")
    print()
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    print()
    print("=" * 60)
    print("  To use this file in the dashboard:")
    print("  1. Open http://localhost:8501")
    print("  2. Go to 📊 Dataset page")
    print("  3. Upload  data/raw/demo_vibration_dataset.csv")
    print("  4. Columns will auto-map")
    print("=" * 60)


if __name__ == "__main__":
    main()
