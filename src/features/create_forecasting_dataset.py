"""Forecasting dataset construction utilities for the research project."""
"""
create_forecasting_dataset.py

Creates separate datasets for different forecast horizons.

Input:
    data/features/features.csv

Outputs:
    data/features/features_1day.csv
    data/features/features_5day.csv
    data/features/features_22day.csv
"""

import os
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = "data/features/features.csv"
OUTPUT_DIR = "data/features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Forecast horizons
HORIZONS = [1, 5, 22]

# --------------------------------------------------
# Load feature dataset
# --------------------------------------------------

print("Loading feature dataset...")

df = pd.read_csv(
    INPUT_FILE,
    index_col=0,
    parse_dates=True
)

print(f"Loaded dataset with shape: {df.shape}")

# --------------------------------------------------
# Create datasets
# --------------------------------------------------

for horizon in HORIZONS:

    print(f"\nCreating {horizon}-day forecast dataset...")

    dataset = df.copy()

    rv_columns = [col for col in dataset.columns if col.endswith("_rv")]

    # Create future target columns
    for col in rv_columns:

        index_name = col.replace("_rv", "")

        dataset[f"{index_name}_target"] = dataset[col].shift(-horizon)

    # Remove rows without future targets
    dataset.dropna(inplace=True)

    output_file = os.path.join(
        OUTPUT_DIR,
        f"features_{horizon}day.csv"
    )

    dataset.to_csv(output_file)

    print(f"Saved: {output_file}")
    print(f"Shape: {dataset.shape}")

print("\nDone!")