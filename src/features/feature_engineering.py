"""Feature engineering utilities for the research project."""
"""
feature_engineering.py

Creates the features used by all forecasting models.

Features:
- Daily log returns
- 22-day realized volatility (target)
- Lagged realized volatility (1, 2, 3 days)

Output:
data/features/features.csv
"""

import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = "data/processed/cleaned_indices.csv"

OUTPUT_DIR = "data/features"
OUTPUT_FILE = "features.csv"

RV_WINDOW = 22          # 22 trading days (~1 month)
ANNUALIZATION = 252     # Trading days in a year

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Load cleaned prices
# --------------------------------------------------

print("Loading cleaned price data...")

prices = pd.read_csv(
    INPUT_FILE,
    index_col=0,
    parse_dates=True
)

print("Done.\n")

# --------------------------------------------------
# Calculate daily log returns
# --------------------------------------------------

print("Calculating log returns...")

log_returns = np.log(prices / prices.shift(1))

# --------------------------------------------------
# Calculate realized volatility
# --------------------------------------------------

print("Calculating realized volatility...")

realized_vol = (
    log_returns
    .rolling(RV_WINDOW)
    .std()
    * np.sqrt(ANNUALIZATION)
)

# --------------------------------------------------
# Build feature dataset
# --------------------------------------------------

print("Creating feature dataset...")

feature_df = pd.DataFrame(index=prices.index)

# --------------------------------------------------
# Create features for each index
# --------------------------------------------------

for index in prices.columns:

    # Daily return
    feature_df[f"{index}_return"] = log_returns[index]

    # Realized volatility (target)
    feature_df[f"{index}_rv"] = realized_vol[index]

    # Lagged RV features
    feature_df[f"{index}_rv_lag1"] = realized_vol[index].shift(1)
    feature_df[f"{index}_rv_lag2"] = realized_vol[index].shift(2)
    feature_df[f"{index}_rv_lag3"] = realized_vol[index].shift(3)

# --------------------------------------------------
# Remove rows with missing values
# --------------------------------------------------

feature_df.dropna(inplace=True)

# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

feature_df.to_csv(output_path)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nFeature engineering complete!")

print(f"\nSaved to:\n{output_path}")

print("\nDataset Shape:")
print(feature_df.shape)

print("\nFirst 5 rows:")
print(feature_df.head())

print("\nMissing Values:")
print(feature_df.isna().sum().sum())