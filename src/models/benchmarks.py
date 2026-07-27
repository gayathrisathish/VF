"""
Benchmark Models

Contains:

1. Historical Volatility
2. Persistence

Each function follows the same interface so that it can be called
directly by run_experiments.py.
"""


from src.evaluation.metrics import calculate_metrics


# ============================================================================
# Historical Volatility
# ============================================================================

def train_historical_volatility(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    index,
):
    """
    Historical Volatility Benchmark.

    Uses the most recent realized volatility observation as the prediction.
    """

    # Last realized volatility feature
    lag_column = f"{index}_rv_lag1"

    validation_predictions = X_validation[lag_column].values
    test_predictions = X_test[lag_column].values
    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    return {

        "model": "HistoricalVolatility",

        "validation_predictions": validation_predictions,

        "test_predictions": test_predictions,

        "validation_metrics": validation_metrics,

        "test_metrics": test_metrics

    }


# ============================================================================
# Persistence Model
# ============================================================================

def train_persistence(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    index,
):
    """
    Persistence Benchmark.

    Forecast equals the previous realized volatility.
    """

    lag_column = f"{index}_rv_lag1"

    validation_predictions = X_validation[lag_column].values
    test_predictions = X_test[lag_column].values

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    return {

        "model": "Persistence",

        "validation_predictions": validation_predictions,

        "test_predictions": test_predictions,

        "validation_metrics": validation_metrics,

        "test_metrics": test_metrics

    }