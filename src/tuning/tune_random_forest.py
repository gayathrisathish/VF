"""
Optuna hyperparameter tuning script — Random Forest
Volatility Forecasting Project (VF)

Tuning index: S&P 500 (^GSPC)
Target: annualized 22-day realized volatility (^GSPC_target)
Optimization metric: QLIKE (standard loss function for volatility forecast evaluation)

HOW TO USE:
1. Run this once per horizon. Change HORIZON below to "1day", "5day", or "22day"
   and re-run — that's 3 separate runs total for Random Forest.
2. Run: python src/tuning/tune_random_forest.py   (from project root)
3. Best hyperparameters get saved to results/best_params/best_params_random_forest_<horizon>.json
4. Progress prints trial by trial. Safe to stop and resume later (see STORAGE_PATH).

Note: Random Forest has no early-stopping / epoch concept the way XGBoost or neural
models do — each tree in the forest is built independently, not iteratively refined
against a validation loss. So n_estimators is tuned directly as a hyperparameter
rather than treated as a ceiling with early stopping.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import optuna
import json
import os

# ============================================================
# STEP 1: SETTINGS
# ============================================================

HORIZON = "22day"   # <-- change to "1day", "5day", or "22day" and re-run for each

DATA_PATH = f"data/features/features_{HORIZON}.csv"
TARGET_COLUMN = "^GSPC_target"
DATE_COLUMN = "Date"          # <-- confirm this matches your actual date column name in the CSV

# All 7 indices' target columns — these must be dropped from the feature set
# (you can't use future-looking target columns from OTHER indices as features either,
# since they're computed the same forward-looking way and would leak information)
ALL_TARGET_COLUMNS = [
    "000001.SS_target", "^FTSE_target", "^GDAXI_target",
    "^GSPC_target", "^HSI_target", "^IXIC_target", "^N225_target",
]

N_TRIALS = 50
RANDOM_SEED = 42

TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"
# everything after VAL_END is test data — never touched during tuning

STUDY_NAME = f"random_forest_{HORIZON}"
STORAGE_PATH = f"sqlite:///results/studies/{STUDY_NAME}.db"
RESULTS_DIR = "results/best_params"

os.makedirs("results/studies", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


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
# STEP 3: LOAD AND SPLIT DATA (chronological)
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


# ============================================================
# STEP 4: OPTUNA OBJECTIVE
# ============================================================

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
    "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
    "max_depth": trial.suggest_int("max_depth", 3, 30),
    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),

    "max_features": trial.suggest_categorical(
        "max_features",
        ["sqrt", "log2", 0.5, 0.7, 1.0]
    ),

    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    # predictions can occasionally be at/near zero for a volatility target — clip before scoring
    preds = np.clip(preds, 1e-8, None)

    score = qlike(y_val.values, preds)
    return score


# ============================================================
# STEP 5: RUN TUNING
# ============================================================

def main():
    print(f"Loading data: {DATA_PATH}")
    X, y, dates = load_data()

    X_train, y_train, X_val, y_val = chronological_split(X, y, dates)
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")
    print(f"Feature count: {X_train.shape[1]}")

    np.random.seed(RANDOM_SEED)

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
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("\n=== TUNING COMPLETE ===")
    print(f"Best QLIKE: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    result_path = os.path.join(RESULTS_DIR, f"best_params_{STUDY_NAME}.json")
    with open(result_path, "w") as f:
        json.dump({
            "model": "random_forest",
            "index": "^GSPC",
            "horizon": HORIZON,
            "metric": "QLIKE",
            "best_score": study.best_value,
            "best_params": study.best_params,
            "n_trials": N_TRIALS,
            "random_seed": RANDOM_SEED,
        }, f, indent=2)

    print(f"Saved best params to {result_path}")


if __name__ == "__main__":
    main()