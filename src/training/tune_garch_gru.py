"""
Optuna hyperparameter tuning script — GARCH-GRU (hybrid)
Volatility Forecasting Project (VF)

Tuning index: S&P 500 (^GSPC)
Target: annualized 22-day realized volatility (^GSPC_target)
Optimization metric: QLIKE (standard loss function for volatility forecast evaluation)
Framework: TensorFlow / Keras

ONLY DIFFERENCE FROM tune_gru.py: the GARCH conditional volatility series
(from fit_garch_family.py, saved at data/features/garch_volatility_feature.csv)
is merged in as one extra input feature column. Everything else — architecture,
search space, sequence length, training loop, QLIKE scoring — is IDENTICAL to
the standalone GRU script, per your finalized spec (item 4): hybrids reuse the
same neural search space, no additional tunable hyperparameters.

HOW TO USE:
1. Run this once per horizon. Change HORIZON below to "1day", "5day", or "22day"
   and re-run — 3 separate runs total.
2. Run: python src/tuning/tune_garch_gru.py   (from project root)
3. Best hyperparameters get saved to results/best_params/best_params_garch_gru_<horizon>.json

PREREQUISITE: fit_garch_family.py must have already been run, since this script
reads data/features/garch_volatility_feature.csv, which that script produces.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import optuna
import json
import os

# ============================================================
# STEP 1: SETTINGS
# ============================================================

HORIZON = "1day"   # <-- change to "1day", "5day", or "22day" and re-run for each

DATA_PATH = f"data/features/features_{HORIZON}.csv"
GARCH_FEATURE_PATH = "data/features/garch_volatility_feature.csv"   # produced by fit_garch_family.py
TARGET_COLUMN = "^GSPC_target"
DATE_COLUMN = "Date"          # <-- confirm this matches your actual date column name in the CSV

ALL_TARGET_COLUMNS = [
    "000001.SS_target", "^FTSE_target", "^GDAXI_target",
    "^GSPC_target", "^HSI_target", "^IXIC_target", "^N225_target",
]

SEQUENCE_LENGTH = 30   # fixed — same rationale as standalone GRU script

N_TRIALS = 50
RANDOM_SEED = 42

TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"
# everything after VAL_END is test data — never touched during tuning

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12

STUDY_NAME = f"garch_gru_{HORIZON}"
STORAGE_PATH = f"sqlite:///results/studies/{STUDY_NAME}.db"
RESULTS_DIR = "results/best_params"

os.makedirs("results/studies", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# STEP 2: QLIKE METRIC
# ============================================================

def qlike(y_true, y_pred, eps=1e-8):
    """
    QLIKE loss — standard for volatility forecast evaluation.
    Lower is better. Requires strictly positive values (volatility is always >= 0,
    but predictions are clipped to avoid log(0) or division by zero).
    """
    y_pred = np.clip(y_pred, eps, None)
    y_true = np.clip(y_true, eps, None)
    return np.mean(np.log(y_pred) + y_true / y_pred)


# ============================================================
# STEP 3: LOAD, MERGE GARCH FEATURE, SPLIT, SEQUENCE (chronological)
# ============================================================

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    df = df.dropna(subset=[TARGET_COLUMN])  # target is pre-computed in your feature files already

    garch_df = pd.read_csv(GARCH_FEATURE_PATH, parse_dates=[DATE_COLUMN])
    df = df.merge(garch_df, on=DATE_COLUMN, how="left")

    # merge should not introduce NaNs if garch_volatility_feature.csv covers the full date
    # range — if this fires, fit_garch_family.py may not have covered all your dates
    if df["garch_vol_feature"].isna().any():
        missing_count = df["garch_vol_feature"].isna().sum()
        raise ValueError(
            f"{missing_count} rows have no matching GARCH feature after merge — "
            f"check that garch_volatility_feature.csv covers the full date range."
        )

    feature_cols = [c for c in df.columns if c not in ALL_TARGET_COLUMNS + [DATE_COLUMN]]

    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    dates = df[DATE_COLUMN]

    return X, y, dates


def chronological_split(X, y, dates):
    train_mask = dates <= TRAIN_END
    val_mask = (dates > TRAIN_END) & (dates <= VAL_END)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    return X_train, y_train, X_val, y_val


def make_sequences(X_scaled, y_values, seq_len):
    """
    Turns flat (rows, features) data into (samples, seq_len, features) windows
    for the GRU, with y aligned to the day immediately after each window.
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_scaled)):
        X_seq.append(X_scaled[i - seq_len:i])
        y_seq.append(y_values[i])
    return np.array(X_seq), np.array(y_seq)


# ============================================================
# STEP 4: MODEL BUILDER (identical to standalone GRU)
# ============================================================

def build_gru(input_shape, hidden_units, num_layers, dropout, learning_rate):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for i in range(num_layers):
        return_sequences = i < num_layers - 1  # only last GRU layer returns a single vector
        model.add(layers.GRU(hidden_units, return_sequences=return_sequences))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation="relu"))  # relu keeps volatility predictions non-negative

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


# ============================================================
# STEP 5: OPTUNA OBJECTIVE (identical search space to standalone GRU)
# ============================================================

def objective(trial, X_train_seq, y_train_seq, X_val_seq, y_val_seq):
    hidden_units = trial.suggest_categorical("hidden_units", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    tf.keras.backend.clear_session()

    model = build_gru(
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        hidden_units=hidden_units,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=MAX_EPOCHS,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    preds = model.predict(X_val_seq, verbose=0).flatten()
    preds = np.clip(preds, 1e-8, None)

    score = qlike(y_val_seq, preds)
    return score


# ============================================================
# STEP 6: RUN TUNING
# ============================================================

def main():
    print(f"Loading data: {DATA_PATH}")
    print(f"Merging GARCH feature: {GARCH_FEATURE_PATH}")
    X, y, dates = load_data()

    X_train, y_train, X_val, y_val = chronological_split(X, y, dates)
    print(f"Train size (pre-sequencing): {len(X_train)} | Val size (pre-sequencing): {len(X_val)}")
    print(f"Feature count (including GARCH): {X_train.shape[1]}")

    # Scale features — fit ONLY on train to avoid leakage, apply same scaler to val
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train.values, SEQUENCE_LENGTH)
    X_val_seq, y_val_seq = make_sequences(X_val_scaled, y_val.values, SEQUENCE_LENGTH)

    print(f"Train sequences: {X_train_seq.shape} | Val sequences: {X_val_seq.shape}")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(),
    )

    print(f"Starting tuning: {N_TRIALS} trials — {STUDY_NAME} (metric: QLIKE)")
    study.optimize(
        lambda trial: objective(trial, X_train_seq, y_train_seq, X_val_seq, y_val_seq),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("\n=== TUNING COMPLETE ===")
    print(f"Best QLIKE: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    result_path = os.path.join(RESULTS_DIR, f"best_params_{STUDY_NAME}.json")
    with open(result_path, "w") as f:
        json.dump({
            "model": "garch_gru",
            "index": "^GSPC",
            "horizon": HORIZON,
            "metric": "QLIKE",
            "sequence_length": SEQUENCE_LENGTH,
            "best_score": study.best_value,
            "best_params": study.best_params,
            "n_trials": N_TRIALS,
            "random_seed": RANDOM_SEED,
        }, f, indent=2)

    print(f"Saved best params to {result_path}")


if __name__ == "__main__":
    main()