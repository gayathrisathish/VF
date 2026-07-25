"""
split_data.py

Chronologically splits forecasting datasets into:
- Training (70%)
- Validation (15%)
- Testing (15%)

Input:
    data/features/features_1day.csv
    data/features/features_5day.csv
    data/features/features_22day.csv

Output:
    data/splits/
        ├── 1day/
        ├── 5day/
        └── 22day/
"""

import os
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

FEATURE_DIR = "data/features"
OUTPUT_DIR = "data/splits"

HORIZONS = [1, 5, 22]

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# --------------------------------------------------
# Create output folder
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Process each forecasting horizon
# --------------------------------------------------

for horizon in HORIZONS:

    print("=" * 60)
    print(f"{horizon}-Day Forecast")
    print("=" * 60)

    file = os.path.join(
        FEATURE_DIR,
        f"features_{horizon}day.csv"
    )

    df = pd.read_csv(
        file,
        index_col=0,
        parse_dates=True
    )

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------
    # Separate Features and Targets
    # --------------------------------------------------

    target_columns = [
        col for col in df.columns
        if col.endswith("_target")
    ]

    feature_columns = [
        col for col in df.columns
        if col not in target_columns
    ]

    X = df[feature_columns]
    y = df[target_columns]

    # --------------------------------------------------
    # Chronological Split
    # --------------------------------------------------

    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)

    X_train = X.iloc[:train_end]
    X_validation = X.iloc[train_end:valid_end]
    X_test = X.iloc[valid_end:]

    y_train = y.iloc[:train_end]
    y_validation = y.iloc[train_end:valid_end]
    y_test = y.iloc[valid_end:]

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    save_folder = os.path.join(
        OUTPUT_DIR,
        f"{horizon}day"
    )

    os.makedirs(save_folder, exist_ok=True)

    X_train.to_csv(os.path.join(save_folder, "X_train.csv"))
    X_validation.to_csv(os.path.join(save_folder, "X_validation.csv"))
    X_test.to_csv(os.path.join(save_folder, "X_test.csv"))

    y_train.to_csv(os.path.join(save_folder, "y_train.csv"))
    y_validation.to_csv(os.path.join(save_folder, "y_validation.csv"))
    y_test.to_csv(os.path.join(save_folder, "y_test.csv"))

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\nSaved to:")
    print(save_folder)

    print("\nTrain:")
    print(X_train.shape)

    print("Validation:")
    print(X_validation.shape)

    print("Test:")
    print(X_test.shape)

print("\nAll forecasting datasets successfully split.")