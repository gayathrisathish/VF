"""
Model fitting script — Linear Regression
Volatility Forecasting Project (VF)

Tuning index: S&P 500 (^GSPC)
Target: annualized 22-day realized volatility (^GSPC_target)

NO OPTUNA HERE — Linear Regression has no hyperparameters to tune (per your
finalized spec, item 13). This script just fits the model directly and reports
metrics, no search loop. It DOES loop over all 3 horizons in one run though,
since there's no tuning cost to worry about — unlike your other model scripts,
you don't need to run this 3 separate times.

HOW TO USE:
Run once: python src/training/fit_linear_regression.py   (from project root)
Fits and evaluates all 3 horizons (1day, 5day, 22day) in a single run.
Metrics saved to results/metrics/, predictions saved to results/predictions/.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import json
import os

# ============================================================
# STEP 1: SETTINGS
# ============================================================

HORIZONS = ["1day", "5day", "22day"]

TARGET_COLUMN = "^GSPC_target"
DATE_COLUMN = "Date"          # <-- confirm this matches your actual date column name in the CSV

ALL_TARGET_COLUMNS = [
    "000001.SS_target", "^FTSE_target", "^GDAXI_target",
    "^GSPC_target", "^HSI_target", "^IXIC_target", "^N225_target",
]

TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"
# everything after VAL_END is test data — held out here too, same as your other model scripts

RANDOM_SEED = 42   # kept for consistency across scripts, though LinearRegression itself
                    # is deterministic and doesn't actually use a seed

METRICS_DIR = "results/metrics"
PREDICTIONS_DIR = "results/predictions"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# ============================================================
# STEP 2: METRICS
# ============================================================

def qlike(y_true, y_pred, eps=1e-8):
    """QLIKE loss — standard for volatility forecast evaluation. Lower is better."""
    y_pred = np.clip(y_pred, eps, None)
    y_true = np.clip(y_true, eps, None)
    return np.mean(np.log(y_pred) + y_true / y_pred)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# ============================================================
# STEP 3: LOAD AND SPLIT DATA (chronological)
# ============================================================

def load_data(horizon):
    data_path = f"data/features/features_{horizon}.csv"
    df = pd.read_csv(data_path, parse_dates=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    df = df.dropna(subset=[TARGET_COLUMN])

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
    dates_val = dates[val_mask]

    return X_train, y_train, X_val, y_val, dates_val


# ============================================================
# STEP 4: FIT AND EVALUATE, PER HORIZON
# ============================================================

def run_horizon(horizon):
    print(f"\n=== Linear Regression — horizon: {horizon} ===")
    X, y, dates = load_data(horizon)
    X_train, y_train, X_val, y_val, dates_val = chronological_split(X, y, dates)

    print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    preds = np.clip(preds, 1e-8, None)   # volatility predictions can't be negative

    val_qlike = qlike(y_val.values, preds)
    val_rmse = rmse(y_val.values, preds)
    val_mae = mae(y_val.values, preds)

    print(f"Val QLIKE: {val_qlike:.6f} | Val RMSE: {val_rmse:.6f} | Val MAE: {val_mae:.6f}")

    # save predictions
    pred_df = pd.DataFrame({
        DATE_COLUMN: dates_val.values,
        "linear_regression_prediction": preds,
        "actual": y_val.values,
    })
    pred_path = os.path.join(PREDICTIONS_DIR, f"linear_regression_predictions_{horizon}.csv")
    pred_df.to_csv(pred_path, index=False)

    # save metrics
    metrics = {
        "model": "linear_regression",
        "index": "^GSPC",
        "horizon": horizon,
        "val_qlike": val_qlike,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "n_features": X_train.shape[1],
        "coefficients": dict(zip(X_train.columns, model.coef_.tolist())),
        "intercept": float(model.intercept_),
    }
    metrics_path = os.path.join(METRICS_DIR, f"linear_regression_metrics_{horizon}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved predictions to {pred_path}")
    print(f"Saved metrics to {metrics_path}")

    return metrics


# ============================================================
# STEP 5: RUN ALL HORIZONS
# ============================================================

def main():
    all_results = {}
    for horizon in HORIZONS:
        all_results[horizon] = run_horizon(horizon)

    print("\n=== SUMMARY ===")
    for horizon, m in all_results.items():
        print(f"{horizon}: QLIKE={m['val_qlike']:.6f}  RMSE={m['val_rmse']:.6f}  MAE={m['val_mae']:.6f}")


if __name__ == "__main__":
    main()