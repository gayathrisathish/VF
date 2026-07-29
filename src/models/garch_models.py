"""ARCH-family volatility models used by the experiment registry.

The models are estimated from the daily return column for an index and their
horizon-matched conditional standard-deviation forecasts are annualized to be
comparable with the realized-volatility targets produced by this project.
"""

import warnings

import numpy as np
from arch import arch_model

from src.evaluation.metrics import calculate_metrics


ANNUALIZATION_FACTOR = 252


def _validate_inputs(X_train, X_validation, X_test, y_train, index, horizon):
    """Validate data shared by all ARCH-family model functions.

    Returns
    -------
    tuple
        The return-column name and a finite, one-dimensional training target.
    """
    return_column = f"{index}_return"

    for split_name, features in (
        ("X_train", X_train),
        ("X_validation", X_validation),
        ("X_test", X_test),
    ):
        if return_column not in features.columns:
            raise ValueError(
                f"{split_name} must contain the required column "
                f"'{return_column}'."
            )

    training_returns = np.asarray(X_train[return_column], dtype=float)
    if training_returns.size == 0:
        raise ValueError("X_train must contain at least one observation.")
    if not np.isfinite(training_returns).all():
        raise ValueError("X_train return values must be finite.")

    training_target = np.asarray(y_train, dtype=float).reshape(-1)
    if training_target.size == 0 or not np.isfinite(training_target).all():
        raise ValueError("y_train must contain finite observations.")
    if training_target.size != training_returns.size:
        raise ValueError("X_train and y_train must have matching lengths.")
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool):
        raise ValueError("horizon must be a positive integer.")
    if horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    return return_column, training_target


def _fallback_predictions(X, index, training_target):
    """Return a stable fallback forecast if ARCH estimation cannot converge."""
    lag_column = f"{index}_rv_lag1"
    if lag_column in X.columns:
        predictions = np.asarray(X[lag_column], dtype=float)
        if np.isfinite(predictions).all():
            return np.maximum(predictions, 0.0)

    return np.full(len(X), np.mean(training_target), dtype=float)


def _forecast_out_of_sample(
    fitted_model,
    model_specification,
    all_returns,
    training_size,
    validation_size,
    test_size,
    horizon,
):
    """Produce horizon-matched forecasts without estimating new parameters.

    The fitted parameters are reused with the observed return history; no
    re-estimation occurs outside the training split.  Forecast origins begin
    at the first validation observation so each output aligns with its target.
    """
    forecast_model = arch_model(
        all_returns,
        mean="Zero",
        vol=model_specification["vol"],
        p=model_specification["p"],
        o=model_specification["o"],
        q=model_specification["q"],
        dist="normal",
        rescale=False,
    )
    fixed_model = forecast_model.fix(fitted_model.params)
    forecast_arguments = {
        "start": training_size,
        "horizon": horizon,
        "reindex": False,
    }
    if model_specification["vol"] == "EGARCH" and horizon > 1:
        forecast_arguments.update(
            {
                "method": "simulation",
                "simulations": 1000,
                "rng": np.random.default_rng(0).standard_normal,
            }
        )

    variance = fixed_model.forecast(**forecast_arguments).variance.iloc[
        :, horizon - 1
    ].to_numpy()

    predictions = np.sqrt(np.maximum(variance, 0.0)) * np.sqrt(
        ANNUALIZATION_FACTOR
    )
    validation_predictions = predictions[:validation_size]
    test_predictions = predictions[
        validation_size:validation_size + test_size
    ]

    return validation_predictions, test_predictions


def _train_arch_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    index,
    horizon,
    model_name,
    model_specification,
):
    """Fit an ARCH-family specification and evaluate its forecasts."""
    return_column, training_target = _validate_inputs(
        X_train,
        X_validation,
        X_test,
        y_train,
        index,
        horizon,
    )

    validation_size = len(X_validation)
    test_size = len(X_test)
    if len(y_validation) != validation_size or len(y_test) != test_size:
        raise ValueError(
            "Feature and target splits must have matching lengths."
        )

    training_returns = np.asarray(X_train[return_column], dtype=float)
    all_returns = np.concatenate(
        (
            training_returns,
            np.asarray(X_validation[return_column], dtype=float),
            np.asarray(X_test[return_column], dtype=float),
        )
    )
    if not np.isfinite(all_returns).all():
        raise ValueError("Return values must be finite in every data split.")

    try:
        model = arch_model(
            training_returns,
            mean="Zero",
            vol=model_specification["vol"],
            p=model_specification["p"],
            o=model_specification["o"],
            q=model_specification["q"],
            dist="normal",
            rescale=False,
        )
        fitted_model = model.fit(disp="off", show_warning=False)
        if getattr(fitted_model, "convergence_flag", 0) != 0:
            raise RuntimeError("ARCH optimizer did not converge.")

        validation_predictions, test_predictions = _forecast_out_of_sample(
            fitted_model,
            model_specification,
            all_returns,
            len(training_returns),
            validation_size,
            test_size,
            horizon,
        )
        if (
            len(validation_predictions) != validation_size
            or len(test_predictions) != test_size
        ):
            raise RuntimeError("ARCH forecast did not cover all observations.")
    except (
        ArithmeticError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        warnings.warn(
            f"{model_name} fitting failed ({error}); using a stable fallback "
            "forecast.",
            RuntimeWarning,
            stacklevel=2,
        )
        validation_predictions = _fallback_predictions(
            X_validation,
            index,
            training_target,
        )
        test_predictions = _fallback_predictions(
            X_test,
            index,
            training_target,
        )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )
    test_metrics = calculate_metrics(y_test, test_predictions)

    return {
        "model": model_name,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }


def train_garch(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    index,
    horizon,
):
    """Train a GARCH(1,1) model and forecast horizon-matched volatility."""
    return _train_arch_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        index,
        horizon,
        "GARCH",
        {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
    )


def train_egarch(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    index,
    horizon,
):
    """Train an EGARCH(1,1) model and forecast horizon-matched volatility."""
    return _train_arch_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        index,
        horizon,
        "EGARCH",
        {"vol": "EGARCH", "p": 1, "o": 0, "q": 1},
    )


def train_gjr_garch(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    index,
    horizon,
):
    """Train a GJR-GARCH(1,1) model and forecast horizon-matched volatility."""
    return _train_arch_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        index,
        horizon,
        "GJRGARCH",
        {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
    )
