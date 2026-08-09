"""
GARCH / EGARCH / GJR-GARCH — order selection + conditional volatility generation
Volatility Forecasting Project (VF)

Index: S&P 500 (^GSPC)
Order selection metric: AIC/BIC grid search over p, q in [1,2,3] (NOT Optuna — these
are econometric models fit via maximum likelihood, order selection is a standard
grid + information-criterion procedure, not a hyperparameter search).

WHAT THIS SCRIPT DOES:
1. Loads ^GSPC daily log returns from your training period only.
2. For each of GARCH, EGARCH, GJR-GARCH: grid-searches p,q in {1,2,3} x {1,2,3},
   fits each combination via MLE on TRAINING DATA ONLY, and picks the order with
   the lowest BIC (BIC penalizes extra parameters more than AIC, standard choice
   for order selection to avoid overfitting on a fairly small parameter grid).
3. Re-fits each selected model, but with parameters FIXED from the training-only
   fit, and filters that fixed-parameter model through the FULL return series
   (train + val + test). This is the leakage-free part: GARCH's variance
   recursion at time t depends only on returns and variances up to t-1, so
   running the recursion forward with train-only-estimated parameters never uses
   information from the future relative to any given day — even though the
   recursion mechanically passes through validation/test dates.
4. Saves each model's full conditional volatility series (aligned to date) to
   results/predictions/ — these are your standalone GARCH/EGARCH/GJR-GARCH
   forecasts for the results table.
5. Additionally saves the plain GARCH conditional volatility series separately
   to data/features/garch_volatility_feature.csv — this is the file the
   GARCH-LSTM/GARCH-GRU/GARCH-Transformer scripts will read in and append as an
   extra input feature (per your spec: hybrids use plain GARCH output, not
   EGARCH/GJR-GARCH).

HOW TO USE:
Run once: python src/tuning/fit_garch_family.py   (from project root)
No horizon looping needed — GARCH is fit on the return series itself, which is
horizon-independent, per your confirmed design.
"""

import pandas as pd
import numpy as np
from arch import arch_model
import json
import os

# ============================================================
# STEP 1: SETTINGS
# ============================================================

# Any one of the horizon feature files works here since ^GSPC_return is identical
# across all three — the return series itself doesn't change by horizon, only the
# forward-looking target does. Using the 1day file as the source.
DATA_PATH = "data/features/features_1day.csv"
RETURN_COLUMN = "^GSPC_return"
DATE_COLUMN = "Date"

TRAIN_END = "2018-12-31"
VAL_END = "2021-12-31"
# everything after VAL_END is test — the fitted params never see val or test returns,
# only the filtering recursion passes through those dates (see docstring above)

P_RANGE = [1, 2, 3]
Q_RANGE = [1, 2, 3]

PREDICTIONS_DIR = "results/predictions"
FEATURES_DIR = "data/features"
ORDER_SELECTION_LOG = "results/metrics/garch_family_order_selection.json"

os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

# Returns are typically small decimals (e.g. 0.01). The arch package recommends
# scaling returns up (commonly x100, i.e. percentage returns) for numerical
# stability during MLE optimization. This is standard practice, not a modeling choice.
RETURN_SCALE = 100


# ============================================================
# STEP 2: LOAD DATA
# ============================================================

def load_returns():
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    df = df.dropna(subset=[RETURN_COLUMN])

    returns = df[RETURN_COLUMN] * RETURN_SCALE
    dates = df[DATE_COLUMN]

    return returns, dates


# ============================================================
# STEP 3: ORDER SELECTION (train data only, BIC-based)
# ============================================================

def select_order(train_returns, vol_model, dist="normal"):
    """
    Grid-searches p, q for a given volatility model family, fit on TRAIN DATA ONLY.
    Returns the (p, q) with the lowest BIC, plus a log of every combination tried.
    """
    best_bic = np.inf
    best_order = None
    results_log = []

    for p in P_RANGE:
        for q in Q_RANGE:
            try:
                model = arch_model(
                    train_returns,
                    vol=vol_model,
                    p=p,
                    q=q,
                    dist=dist,
                    mean="constant",
                )
                fit_result = model.fit(disp="off")

                results_log.append({
                    "p": p, "q": q,
                    "aic": float(fit_result.aic),
                    "bic": float(fit_result.bic),
                })

                if fit_result.bic < best_bic:
                    best_bic = fit_result.bic
                    best_order = (p, q)

            except Exception as e:
                # some p,q combinations can fail to converge — skip and log, don't crash the whole run
                results_log.append({"p": p, "q": q, "error": str(e)})

    return best_order, results_log


# ============================================================
# STEP 4: LEAKAGE-FREE CONDITIONAL VOLATILITY GENERATION
# ============================================================

def generate_full_series(train_returns, full_returns, vol_model, order, dist="normal"):
    """
    Re-fits the model on TRAIN DATA ONLY to obtain final parameter estimates,
    then FIXES those parameters and filters them through the FULL return series
    (train + val + test). Because the GARCH-family variance recursion at time t
    only uses returns/variance up through t-1, this produces a conditional
    volatility value for every date using only information available up to that
    date under the training-estimated parameters — no look-ahead into val/test.
    """
    p, q = order

    # Step A: fit on train only to get final parameter estimates
    train_model = arch_model(train_returns, vol=vol_model, p=p, q=q, dist=dist, mean="constant")
    train_fit = train_model.fit(disp="off")

    # Step B: build a model over the FULL return series, but fix params from train_fit
    # (fix(), not fit() — this does NOT re-estimate parameters using val/test data)
    full_model = arch_model(full_returns, vol=vol_model, p=p, q=q, dist=dist, mean="constant")
    fixed_result = full_model.fix(train_fit.params)

    conditional_vol = fixed_result.conditional_volatility / RETURN_SCALE  # undo the x100 scaling
    return conditional_vol, train_fit.params.to_dict()


# ============================================================
# STEP 5: RUN FOR ALL THREE MODEL TYPES
# ============================================================

MODEL_SPECS = {
    "garch": {"vol_model": "GARCH", "dist": "normal"},
    "egarch": {"vol_model": "EGARCH", "dist": "normal"},
    "gjr_garch": {"vol_model": "GARCH", "dist": "normal", "power": 2.0, "o": 1},
    # Note: GJR-GARCH is implemented in the `arch` package as GARCH with the
    # asymmetry term `o=1` (adds the leverage/asymmetric shock term). This is
    # the standard way to specify GJR-GARCH in this library.
}


def fit_gjr_garch_order(train_returns):
    best_bic = np.inf
    best_order = None
    results_log = []

    for p in P_RANGE:
        for q in Q_RANGE:
            try:
                model = arch_model(
                    train_returns, vol="GARCH", p=p, o=1, q=q, dist="normal", mean="constant"
                )
                fit_result = model.fit(disp="off")

                results_log.append({"p": p, "q": q, "aic": float(fit_result.aic), "bic": float(fit_result.bic)})

                if fit_result.bic < best_bic:
                    best_bic = fit_result.bic
                    best_order = (p, q)
            except Exception as e:
                results_log.append({"p": p, "q": q, "error": str(e)})

    return best_order, results_log


def generate_gjr_full_series(train_returns, full_returns, order):
    p, q = order
    train_model = arch_model(train_returns, vol="GARCH", p=p, o=1, q=q, dist="normal", mean="constant")
    train_fit = train_model.fit(disp="off")

    full_model = arch_model(full_returns, vol="GARCH", p=p, o=1, q=q, dist="normal", mean="constant")
    fixed_result = full_model.fix(train_fit.params)

    conditional_vol = fixed_result.conditional_volatility / RETURN_SCALE
    return conditional_vol, train_fit.params.to_dict()


def main():
    print(f"Loading returns: {DATA_PATH}")
    returns, dates = load_returns()

    train_mask = dates <= TRAIN_END
    train_returns = returns[train_mask].reset_index(drop=True)
    full_returns = returns.reset_index(drop=True)

    print(f"Train returns: {len(train_returns)} | Full returns: {len(full_returns)}")

    all_order_logs = {}
    all_params = {}

    # ---------------- GARCH ----------------
    print("\n--- GARCH: selecting order (p,q) via BIC ---")
    garch_order, garch_log = select_order(train_returns, vol_model="GARCH", dist="normal")
    print(f"GARCH best order: {garch_order}")
    garch_vol, garch_params = generate_full_series(train_returns, full_returns, "GARCH", garch_order)
    all_order_logs["garch"] = {"selected_order": garch_order, "grid": garch_log}
    all_params["garch"] = garch_params

    garch_df = pd.DataFrame({DATE_COLUMN: dates.values, "garch_conditional_volatility": garch_vol.values})
    garch_df.to_csv(os.path.join(PREDICTIONS_DIR, "garch_conditional_volatility.csv"), index=False)

    # this is the file the GARCH-hybrid scripts will read in as an extra feature
    garch_df.rename(columns={"garch_conditional_volatility": "garch_vol_feature"}).to_csv(
        os.path.join(FEATURES_DIR, "garch_volatility_feature.csv"), index=False
    )

    # ---------------- EGARCH ----------------
    print("\n--- EGARCH: selecting order (p,q) via BIC ---")
    egarch_order, egarch_log = select_order(train_returns, vol_model="EGARCH", dist="normal")
    print(f"EGARCH best order: {egarch_order}")
    egarch_vol, egarch_params = generate_full_series(train_returns, full_returns, "EGARCH", egarch_order)
    all_order_logs["egarch"] = {"selected_order": egarch_order, "grid": egarch_log}
    all_params["egarch"] = egarch_params

    egarch_df = pd.DataFrame({DATE_COLUMN: dates.values, "egarch_conditional_volatility": egarch_vol.values})
    egarch_df.to_csv(os.path.join(PREDICTIONS_DIR, "egarch_conditional_volatility.csv"), index=False)

    # ---------------- GJR-GARCH ----------------
    print("\n--- GJR-GARCH: selecting order (p,q) via BIC ---")
    gjr_order, gjr_log = fit_gjr_garch_order(train_returns)
    print(f"GJR-GARCH best order: {gjr_order}")
    gjr_vol, gjr_params = generate_gjr_full_series(train_returns, full_returns, gjr_order)
    all_order_logs["gjr_garch"] = {"selected_order": gjr_order, "grid": gjr_log}
    all_params["gjr_garch"] = gjr_params

    gjr_df = pd.DataFrame({DATE_COLUMN: dates.values, "gjr_garch_conditional_volatility": gjr_vol.values})
    gjr_df.to_csv(os.path.join(PREDICTIONS_DIR, "gjr_garch_conditional_volatility.csv"), index=False)

    # ---------------- Save order selection log ----------------
    with open(ORDER_SELECTION_LOG, "w") as f:
        json.dump({
            "order_selection": all_order_logs,
            "fitted_params": all_params,
            "note": "Orders selected via BIC on training data only (through {}). "
                    "Conditional volatility series generated by fixing train-estimated "
                    "parameters and filtering through the full return series — leakage-free "
                    "since the recursion at time t only uses information through t-1.".format(TRAIN_END),
        }, f, indent=2, default=str)

    print(f"\n=== DONE ===")
    print(f"Saved standalone forecasts to {PREDICTIONS_DIR}/")
    print(f"Saved GARCH hybrid feature to {FEATURES_DIR}/garch_volatility_feature.csv")
    print(f"Saved order selection log to {ORDER_SELECTION_LOG}")


if __name__ == "__main__":
    main()