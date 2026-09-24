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

# Scale applied to the returns when the optimizer fails on the original scale.
# Daily log returns have a standard deviation near 0.015, which puts the
# variance intercept around 1e-6 and leaves the likelihood poorly conditioned;
# on some SciPy builds the optimizer then reports a convergence failure. The
# same specification estimated on percentage returns is well conditioned, and
# the estimates convert back exactly, so nothing downstream has to know that a
# different scale was used during estimation.
RETRY_RETURN_SCALE = 100.0


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


def _build_arch_model(returns, model_specification):
    """Construct one ARCH-family specification on the supplied return series."""
    return arch_model(
        returns,
        mean="Zero",
        vol=model_specification["vol"],
        p=model_specification["p"],
        o=model_specification["o"],
        q=model_specification["q"],
        dist="normal",
        rescale=False,
    )


def _parameters_in_original_units(params, model_specification, scale):
    """Convert estimates made on ``scale * returns`` back to the original units.

    Scaling the returns by ``c`` scales the conditional variance by ``c ** 2``.
    For the GARCH variance recursion only the intercept absorbs that factor; the
    ARCH, asymmetry and GARCH coefficients are scale free. EGARCH models the log
    variance, so its intercept shifts by ``2 * log(c) * (1 - persistence)``
    instead. Both conversions are exact, not approximations.
    """
    converted = params.copy()

    if model_specification["vol"] == "EGARCH":
        persistence = sum(
            value
            for name, value in params.items()
            if name.startswith("beta")
        )
        converted["omega"] = params["omega"] - 2.0 * np.log(scale) * (
            1.0 - persistence
        )
    else:
        converted["omega"] = params["omega"] / (scale ** 2)

    return converted


def _fit_on_rescaled_returns(
    training_returns,
    model_specification,
    scale=RETRY_RETURN_SCALE,
):
    """Estimate one specification on rescaled returns, reported in original units.

    Used only after the optimizer has failed on the original scale. The returned
    object holds the maximum-likelihood estimates converted back to the original
    units, so it forecasts and serializes exactly like a direct fit; evaluating
    the model at those parameters estimates nothing further.
    """
    rescaled_fit = _build_arch_model(
        scale * training_returns,
        model_specification,
    ).fit(disp="off", show_warning=False)

    if getattr(rescaled_fit, "convergence_flag", 0) != 0:
        raise RuntimeError(
            "ARCH optimizer did not converge on the original or the rescaled "
            "return series."
        )

    original_parameters = _parameters_in_original_units(
        rescaled_fit.params,
        model_specification,
        scale,
    )

    return _build_arch_model(training_returns, model_specification).fix(
        original_parameters
    )


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

    # Stays None unless estimation converges and its forecasts are usable, so a
    # non-converged run is never persisted as if it were a fitted model. The
    # candidate is only promoted after every check below has passed.
    fitted_model = None

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
        candidate_model = model.fit(disp="off", show_warning=False)
        if getattr(candidate_model, "convergence_flag", 0) != 0:
            # Retry the same specification on a better conditioned scale rather
            # than accepting a failed optimization. Raises if that also fails,
            # which leaves the fallback forecast and saves no checkpoint.
            candidate_model = _fit_on_rescaled_returns(
                training_returns,
                model_specification,
            )

        validation_predictions, test_predictions = _forecast_out_of_sample(
            candidate_model,
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

        fitted_model = candidate_model
    except (
        ArithmeticError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        fitted_model = None
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
        "fitted_model": fitted_model,
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
