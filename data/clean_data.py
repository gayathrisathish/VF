"""
clean_data.py

Reads the raw Yahoo Finance data,
cleans and aligns all indices,
and saves a clean dataset for feature engineering.
"""

import os
import pandas as pd

# -----------------------------
# File paths
# -----------------------------
RAW_DATA_PATH = "data/raw/global_indices.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = "cleaned_indices.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load raw data
# -----------------------------
print("Loading raw data...")

df = pd.read_csv(RAW_DATA_PATH, header=[0, 1], index_col=0)

# Convert index to datetime
df.index = pd.to_datetime(df.index)

# Sort by date
df = df.sort_index()

# -----------------------------
# Keep only Adjusted Close prices
# (auto_adjust=True in yfinance means Close contains adjusted prices)
# -----------------------------
close_prices = pd.DataFrame()

for ticker in df.columns.levels[0]:
    if ("Close" in df[ticker].columns):
        close_prices[ticker] = df[ticker]["Close"]

# -----------------------------
# Remove rows where every market is missing
# -----------------------------
close_prices = close_prices.dropna(how="all")

# -----------------------------
# Forward-fill missing values
# (handles holidays between markets)
# -----------------------------
close_prices = close_prices.ffill()

# -----------------------------
# Remove any remaining missing values
# -----------------------------
close_prices = close_prices.dropna()

# -----------------------------
# Save cleaned dataset
# -----------------------------
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

close_prices.to_csv(output_path)

print("Cleaning complete!")
print(f"Saved cleaned data to: {output_path}")

print("\nDataset Shape:")
print(close_prices.shape)

print("\nFirst 5 rows:")
print(close_prices.head())

print("\nMissing Values:")
print(close_prices.isna().sum())