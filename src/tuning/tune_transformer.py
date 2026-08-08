"""
Optuna hyperparameter tuning script — Transformer
Volatility Forecasting Project (VF)

Tuning index: S&P 500 (^GSPC)
Target: annualized 22-day realized volatility (^GSPC_target)
Optimization metric: QLIKE (standard loss function for volatility forecast evaluation)
Framework: TensorFlow / Keras

HOW TO USE:
1. Run this once per horizon. Change HORIZON below to "1day", "5day", or "22day"
   and re-run — that's 3 separate runs total for Transformer.
2. Run: python src/tuning/tune_transformer.py   (from project root)
3. Best hyperparameters get saved to results/best_params/best_params_transformer_<horizon>.json
4. Progress prints trial by trial. Safe to stop and resume later (see STORAGE_PATH).

Sequence length note: fixed at 30 trading days (NOT tuned via Optuna), same
rationale as the LSTM/GRU scripts — one full realized-vol cycle of context,
kept fixed to control per-trial cost.

Search space note: matches the finalized spec exactly (d_model, num_heads,
num_layers, dropout, learning_rate). Unlike LSTM/GRU, batch_size is NOT part
of the Transformer search space per the finalized spec, so it's fixed below
(BATCH_SIZE) rather than tuned.

num_heads note: d_model choices (32/64/128) and num_heads choices (2/4/8) are
selected so that num_heads always evenly divides d_model, which is a hard
requirement of multi-head attention — no additional conditional logic needed.

Training loss note: trained with MSE, early stopping monitors validation MSE.
QLIKE is computed separately on validation predictions and is what Optuna
optimizes against — same split used in the other model scripts.
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
TARGET_COLUMN = "^GSPC_target"
DATE_COLUMN = "Date"          # <-- confirm this matches your actual date column name in the CSV

ALL_TARGET_COLUMNS = [
    "000001.SS_target", "^FTSE_target", "^GDAXI_target",
    "^GSPC_target", "^HSI_target", "^IXIC_target", "^N225_target",
]

SEQUENCE_LENGTH = 30   # fixed — see rationale in module docstring above
BATCH_SIZE = 32        # fixed — not part of the Transformer search space per spec

N_TRIALS = 50
RANDOM_SEED = 42

TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"
# everything after VAL_END is test data — never touched during tuning

MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12

STUDY_NAME = f"transformer_{HORIZON}"
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
# STEP 3: LOAD, SPLIT, AND SEQUENCE DATA (chronological)
# ============================================================

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    df = df.dropna(subset=[TARGET_COLUMN])  # target is pre-computed in your feature files already

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
    Turns flat (rows, features) data into (samples, seq_len, features) windows,
    with y aligned to the day immediately after each window.
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_scaled)):
        X_seq.append(X_scaled[i - seq_len:i])
        y_seq.append(y_values[i])
    return np.array(X_seq), np.array(y_seq)


# ============================================================
# STEP 4: POSITIONAL ENCODING + TRANSFORMER ENCODER BLOCK
# ============================================================

def positional_encoding(seq_len, d_model):
    """Standard sinusoidal positional encoding, added to the input embeddings
    so the model has a sense of sequence order (Transformers have no
    recurrence, unlike LSTM/GRU, so this is required)."""
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
    angle_rads = positions * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)


def transformer_encoder_block(x, d_model, num_heads, dropout):
    """One encoder block: multi-head self-attention + residual + norm,
    followed by a feed-forward network + residual + norm."""
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads
    )(x, x)
    attn_output = layers.Dropout(dropout)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ff_output = layers.Dense(d_model * 4, activation="relu")(x)
    ff_output = layers.Dense(d_model)(ff_output)
    ff_output = layers.Dropout(dropout)(ff_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

    return x


def build_transformer(input_shape, d_model, num_heads, num_layers, dropout, learning_rate):
    seq_len, num_features = input_shape

    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(d_model)(inputs)  # project raw features into d_model space
    x = x + positional_encoding(seq_len, d_model)

    for _ in range(num_layers):
        x = transformer_encoder_block(x, d_model, num_heads, dropout)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation="relu")(x)  # relu keeps volatility predictions non-negative

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


# ============================================================
# STEP 5: OPTUNA OBJECTIVE
# ============================================================

def objective(trial, X_train_seq, y_train_seq, X_val_seq, y_val_seq):
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])  # always divides d_model choices above
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    tf.keras.backend.clear_session()

    model = build_transformer(
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        d_model=d_model,
        num_heads=num_heads,
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
        batch_size=BATCH_SIZE,
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
    X, y, dates = load_data()

    X_train, y_train, X_val, y_val = chronological_split(X, y, dates)
    print(f"Train size (pre-sequencing): {len(X_train)} | Val size (pre-sequencing): {len(X_val)}")

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
            "model": "transformer",
            "index": "^GSPC",
            "horizon": HORIZON,
            "metric": "QLIKE",
            "sequence_length": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "best_score": study.best_value,
            "best_params": study.best_params,
            "n_trials": N_TRIALS,
            "random_seed": RANDOM_SEED,
        }, f, indent=2)

    print(f"Saved best params to {result_path}")


if __name__ == "__main__":
    main()