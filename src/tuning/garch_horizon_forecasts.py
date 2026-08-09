"""
GARCH / EGARCH / GJR-GARCH — horizon-aligned forecast generation (CORRECTED)
Volatility Forecasting Project (VF)

WHAT CHANGED FROM THE FIRST VERSION:
The first version assumed each horizon's target was a fully forward-looking
22-day window starting h days from the origin. Checking against your actual
create_forecasting_dataset.py showed that's wrong for h=1 and h=5:

    dataset[f"{index}_target"] = dataset[f"{index}_rv"].shift(-horizon)

Your `_rv` column is a BACKWARD-looking 22-day rolling window ending at each
row. So target at origin date t for horizon h = the RV value at date t+h,
which itself covers the window [t+h-21, t+h] — of those 22 days, (22-h) are
ALREADY KNOWN at origin t (they're in the past), and only h days are genuinely
in the future. Only h=22 happens to be a fully-forward window (22-h=0 known
days), which is why the first version was closest to correct there and most
wrong at h=1.

CORRECTED APPROACH, per horizon h:
  - known part: actual observed squared returns for the (22-h) already-past
    days in the window — no forecasting needed, they already happened
  - future part: GARCH's h-step-ahead variance forecast for the h genuinely
    future days
  - combine, average over 22, sqrt, annualize (x sqrt(252))

This also means max forecast horizon needed is just 22 steps (not 43 like the
first version), since the future portion is always <= h <= 22 — this version
is actually faster to run, not slower.

WHAT THIS SCRIPT DOES, IN ORDER:
1. Loads ^GSPC daily returns, re-fits GARCH/EGARCH/GJR-GARCH on train-only data
   (using orders already selected by fit_garch_family.py), fixes those
   parameters on the full return series (leakage-free — see fit_garch_family.py
   for the full explanation of why fix() vs fit() matters here).
2. Generates up-to-22-step-ahead variance forecasts from every origin date.
3. For each horizon, blends known realized squared returns with the
   appropriate number of GARCH forecast steps, per the corrected logic above.
4. Saves one CSV per model per horizon to results/predictions/.

VERIFY BEFORE TRUSTING THE RESULTS: run the alignment check (correlate this
output against your existing target column, testing a small range of date
shifts) before scoring. Correlation should now peak at shift=0, not several
days negative like the first version did.

RUNTIME NOTE: simulation-based forecasting across ~5,400 origins x 3 models
still takes real time (though less than the first version, since horizon is
capped at 22 steps instead of 43) — expect minutes, not seconds.

HOW TO USE:
Run once: python src/tuning/garch_horizon_forecasts.py   (from project root)
"""

import pandas as pd
import numpy as np
from arch import arch_model
import json
import os

# ============================================================
# STEP 1: SETTINGS
# ============================================================

DATA_PATH = "data/features/features_1day.csv"   # ^GSPC_return is identical across horizon files
RETURN_COLUMN = "^GSPC_return"
DATE_COLUMN = "Date"

TRAIN_END = "2018-12-31"
RETURN_SCALE = 100   # must match fit_garch_family.py

HORIZONS = [1, 5, 22]
RV_WINDOW = 22          # realized volatility window length (trading days) — matches
                         # the .rolling(22) used in your feature_engineering.py
ANNUALIZE_FACTOR = np.sqrt(252)

N_SIMULATIONS = 200     # simulation paths per forecast origin — reduce if this runs too slowly,
                         # increase for a smoother/more stable forecast if you have time to spare

ORDER_SELECTION_LOG = "results/metrics/garch_family_order_selection.json"
PREDICTIONS_DIR = "results/predictions"

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

MODEL_SPECS = {
    "garch": {"vol": "GARCH", "o": 0},
    "egarch": {"vol": "EGARCH", "o": 0},
    "gjr_garch": {"vol": "GARCH", "o": 1},   # GJR-GARCH = GARCH + asymmetry term o=1
}


# ============================================================
# STEP 2: LOAD DATA + PREVIOUSLY SELECTED ORDERS
# ============================================================

def load_returns():
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    df = df.dropna(subset=[RETURN_COLUMN])

    returns = df[RETURN_COLUMN] * RETURN_SCALE
    dates = df[DATE_COLUMN]
    return returns.reset_index(drop=True), dates.reset_index(drop=True)


def load_selected_orders():
    with open(ORDER_SELECTION_LOG, "r") as f:
        log = json.load(f)
    orders = {}
    for model_name in MODEL_SPECS:
        p, q = log["order_selection"][model_name]["selected_order"]
        orders[model_name] = (p, q)
    return orders


# ============================================================
# STEP 3: FIT (TRAIN-ONLY) + FIX ON FULL SERIES
# ============================================================

def fit_and_fix(train_returns, full_returns, vol, o, p, q):
    train_model = arch_model(train_returns, vol=vol, p=p, o=o, q=q, dist="normal", mean="constant")
    train_fit = train_model.fit(disp="off")

    full_model = arch_model(full_returns, vol=vol, p=p, o=o, q=q, dist="normal", mean="constant")
    fixed_result = full_model.fix(train_fit.params)

    return fixed_result


# ============================================================
# STEP 4: MULTI-STEP FORECAST -> HORIZON-ALIGNED RV FORECASTS
# ============================================================

def generate_multistep_forecasts(fixed_result, max_steps_needed):
    """
    Produces multi-step-ahead variance forecasts from every origin in the series.
    Returns a DataFrame indexed by origin position, columns 'h.01' .. 'h.<max_steps_needed>'
    (arch package zero-pads column names to the width of the largest step number).
    """
    forecasts = fixed_result.forecast(
        horizon=max_steps_needed,
        start=0,
        method="simulation",
        simulations=N_SIMULATIONS,
        reindex=False,
    )
    return forecasts.variance


def build_rv_forecast_for_horizon(variance_df, known_sq_returns, horizon):
    """
    CORRECTED LOGIC — matches how your targets are actually built.

    Target at origin date t for horizon h = the RV value at date t+h, which
    covers the backward-looking window [t+h-21, t+h]. Of those 22 days,
    (22-h) are already known at origin t, and h are genuinely future.

      - known part: actual observed squared returns for the (22-h) most
        recent already-past days (rolling sum, right-aligned at t)
      - future part: GARCH's h-step-ahead variance forecasts, summed
      - combine, average over 22, sqrt, annualize

    For h=22, known_window=0, so this reduces to a pure forward-looking
    GARCH forecast (matches intuition: at h=22 the whole window is future).
    """
    known_window = RV_WINDOW - horizon

    if known_window > 0:
        known_sum = known_sq_returns.rolling(window=known_window).sum()
    else:
        known_sum = pd.Series(0.0, index=known_sq_returns.index)
    known_sum = known_sum.reset_index(drop=True)

    if horizon > 0:
        pad_width = len(variance_df.columns[-1].split(".")[1])
        future_step_cols = [f"h.{str(step).zfill(pad_width)}" for step in range(1, horizon + 1)]
        missing = [c for c in future_step_cols if c not in variance_df.columns]
        if missing:
            raise ValueError(f"Missing forecast steps {missing}")
        future_sum = variance_df[future_step_cols].sum(axis=1)
    else:
        future_sum = pd.Series(0.0, index=variance_df.index)
    future_sum = future_sum.reset_index(drop=True)

    total_scaled_var = known_sum + future_sum
    avg_var = (total_scaled_var / RV_WINDOW) / (RETURN_SCALE ** 2)   # undo x100 return scaling
    forecast_vol = np.sqrt(avg_var) * ANNUALIZE_FACTOR
    return forecast_vol


# ============================================================
# STEP 5: RUN FOR ALL MODELS x ALL HORIZONS
# ============================================================

def main():
    print(f"Loading returns: {DATA_PATH}")
    returns, dates = load_returns()

    train_mask = dates <= TRAIN_END
    train_returns = returns[train_mask].reset_index(drop=True)
    full_returns = returns

    print(f"Train returns: {len(train_returns)} | Full returns: {len(full_returns)}")

    orders = load_selected_orders()
    print(f"Loaded previously selected orders: {orders}")

    # known squared returns, in the SAME scale as the GARCH variance forecasts (x100 returns)
    known_sq_returns = (full_returns ** 2)

    max_steps_needed = max(HORIZONS)   # now just 22, not 43 — future portion is capped at h

    for model_name, spec in MODEL_SPECS.items():
        p, q = orders[model_name]
        vol, o = spec["vol"], spec["o"]

        print(f"\n--- {model_name.upper()}: fitting (order p={p}, q={q}, o={o}) ---")
        fixed_result = fit_and_fix(train_returns, full_returns, vol, o, p, q)

        print(f"--- {model_name.upper()}: generating multi-step forecasts "
              f"({max_steps_needed} steps, {N_SIMULATIONS} simulations per origin) ---")
        print("This is the slow step — expect a few minutes, not seconds.")
        variance_df = generate_multistep_forecasts(fixed_result, max_steps_needed)

        for h in HORIZONS:
            forecast_vol = build_rv_forecast_for_horizon(variance_df, known_sq_returns, h)

            # variance_df's index corresponds to origin positions in the full return series
            forecast_dates = dates.iloc[variance_df.index].reset_index(drop=True)

            out_df = pd.DataFrame({
                DATE_COLUMN: forecast_dates.values,
                f"{model_name}_forecast_h{h}": forecast_vol.values,
            })
            out_df = out_df.dropna()   # early rows won't have a full known_window — drop, matches
                                        # your own pipeline's dropna() behavior on incomplete windows

            out_path = os.path.join(PREDICTIONS_DIR, f"{model_name}_forecast_h{h}.csv")
            out_df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}  ({len(out_df)} rows)")

    print("\n=== DONE ===")
    print("Re-run your alignment check now — correlation should peak at shift=0.")


if __name__ == "__main__":
    main()