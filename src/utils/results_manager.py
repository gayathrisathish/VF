"""
Utilities for saving experiment results.

This module:
- Appends evaluation metrics to one metrics.csv file.
- Appends predictions to one CSV per forecast horizon.
"""

from pathlib import Path
import pandas as pd

# =============================================================================
# Directories
# =============================================================================

RESULTS_DIR = Path("results")

METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Save Metrics
# =============================================================================

def save_metrics(
    horizon,
    index,
    model_name,
    validation_metrics,
    test_metrics
):
    """
    Appends one experiment to metrics.csv
    """

    metrics_file = METRICS_DIR / "metrics.csv"

    row = pd.DataFrame([{
        "Horizon": horizon,
        "Index": index,
        "Model": model_name,

        "Validation_RMSE": validation_metrics["RMSE"],
        "Validation_MAE": validation_metrics["MAE"],
        "Validation_QLIKE": validation_metrics["QLIKE"],

        "Test_RMSE": test_metrics["RMSE"],
        "Test_MAE": test_metrics["MAE"],
        "Test_QLIKE": test_metrics["QLIKE"]
    }])

    if metrics_file.exists():
        row.to_csv(
            metrics_file,
            mode="a",
            header=False,
            index=False
        )
    else:
        row.to_csv(
            metrics_file,
            index=False
        )


# =============================================================================
# Save Predictions
# =============================================================================

def save_predictions(
    horizon,
    index,
    model_name,
    dates,
    actual,
    predicted
):
    """
    Appends predictions to

    predictions_1day.csv
    predictions_5day.csv
    predictions_22day.csv
    """

    prediction_file = (
        PREDICTIONS_DIR /
        f"predictions_{horizon}.csv"
    )

    prediction_df = pd.DataFrame({

        "Date": dates,

        "Index": index,

        "Model": model_name,

        "Actual": actual,

        "Predicted": predicted

    })

    if prediction_file.exists():

        prediction_df.to_csv(
            prediction_file,
            mode="a",
            header=False,
            index=False
        )

    else:

        prediction_df.to_csv(
            prediction_file,
            index=False
        )