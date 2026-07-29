"""Baseline machine-learning models for volatility forecasting.

Each training function follows the common experiment interface: it fits on
the training split only, forecasts the validation and test splits, and
returns predictions together with their evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.evaluation.metrics import calculate_metrics


RANDOM_STATE = 42


def _as_feature_matrix(features, split_name):
    """Return a numeric two-dimensional feature matrix for one data split.

    Missing feature values are retained as ``numpy.nan`` so the training-only
    imputer can replace them. Infinite or non-numeric values are not valid
    model inputs and raise a clear error.
    """
    try:
        matrix = np.asarray(features, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{split_name} must contain only numeric feature values."
        ) from error

    if matrix.ndim != 2:
        raise ValueError(f"{split_name} must be a two-dimensional array.")
    if matrix.shape[0] == 0:
        raise ValueError(
            f"{split_name} must contain at least one observation."
        )
    if matrix.shape[1] == 0:
        raise ValueError(f"{split_name} must contain at least one feature.")
    if np.isinf(matrix).any():
        raise ValueError(f"{split_name} must not contain infinite values.")

    return matrix


def _as_target_vector(target, split_name):
    """Return a finite one-dimensional target vector for one data split."""
    try:
        target_array = np.asarray(target, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{split_name} must contain only numeric target values."
        ) from error

    if target_array.ndim == 1:
        vector = target_array
    elif target_array.ndim == 2 and target_array.shape[1] == 1:
        vector = target_array[:, 0]
    else:
        raise ValueError(
            f"{split_name} must have shape (n,) or (n, 1)."
        )

    if vector.size == 0:
        raise ValueError(
            f"{split_name} must contain at least one observation."
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"{split_name} must contain finite target values.")

    return vector


def _validate_feature_schema(X_train, X_validation, X_test):
    """Ensure DataFrame feature splits have identical ordered columns."""
    if not all(
        isinstance(features, pd.DataFrame)
        for features in (X_train, X_validation, X_test)
    ):
        return

    if not X_train.columns.equals(X_validation.columns):
        raise ValueError(
            "X_train and X_validation must have identical ordered columns."
        )
    if not X_train.columns.equals(X_test.columns):
        raise ValueError(
            "X_train and X_test must have identical ordered columns."
        )


def _validate_inputs(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    horizon,
):
    """Validate split alignment and return model-ready arrays.

    Feature missing values are allowed and will be median-imputed by a
    pipeline fitted solely on the training split. Targets must be finite
    because they are used for fitting or evaluation.
    """
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool):
        raise ValueError("horizon must be a positive integer.")
    if horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    _validate_feature_schema(X_train, X_validation, X_test)

    feature_splits = {
        "X_train": _as_feature_matrix(X_train, "X_train"),
        "X_validation": _as_feature_matrix(X_validation, "X_validation"),
        "X_test": _as_feature_matrix(X_test, "X_test"),
    }
    target_splits = {
        "y_train": _as_target_vector(y_train, "y_train"),
        "y_validation": _as_target_vector(y_validation, "y_validation"),
        "y_test": _as_target_vector(y_test, "y_test"),
    }

    for split in ("train", "validation", "test"):
        features = feature_splits[f"X_{split}"]
        target = target_splits[f"y_{split}"]
        if len(features) != len(target):
            raise ValueError(
                f"X_{split} and y_{split} must have matching lengths."
            )

    training_feature_count = feature_splits["X_train"].shape[1]
    for split in ("X_validation", "X_test"):
        if feature_splits[split].shape[1] != training_feature_count:
            raise ValueError(
                "All feature splits must have the same number of columns."
            )

    if np.isnan(feature_splits["X_train"]).all(axis=0).any():
        raise ValueError(
            "X_train must not contain entirely missing feature columns."
        )

    return (
        feature_splits["X_train"],
        target_splits["y_train"],
        feature_splits["X_validation"],
        target_splits["y_validation"],
        feature_splits["X_test"],
        target_splits["y_test"],
    )


def _validate_predictions(predictions):
    """Return finite, non-negative one-dimensional volatility forecasts."""
    try:
        validated_predictions = np.asarray(predictions, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError("Model produced non-numeric predictions.") from error

    if not np.isfinite(validated_predictions).all():
        raise ValueError("Model produced non-finite predictions.")

    return np.maximum(validated_predictions, 0.0)


def _train_and_evaluate(
    estimator,
    model_name,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    horizon,
    scale_features=False,
):
    """Fit one baseline estimator and evaluate its volatility forecasts."""
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = _validate_inputs(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        horizon,
    )

    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
    ]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("regressor", estimator))

    model = Pipeline(steps=steps)
    model.fit(X_train, y_train)

    validation_predictions = _validate_predictions(
        model.predict(X_validation)
    )
    test_predictions = _validate_predictions(model.predict(X_test))

    return {
        "model": model_name,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "validation_metrics": calculate_metrics(
            y_validation,
            validation_predictions,
        ),
        "test_metrics": calculate_metrics(y_test, test_predictions),
    }


def train_linear_regression(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    index,
    horizon,
):
    """Train a baseline linear-regression volatility forecaster.

    ``index`` is accepted to maintain the shared model-training interface;
    this feature-based model does not require it directly.
    """
    del index
    return _train_and_evaluate(
        LinearRegression(),
        "LinearRegression",
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        horizon,
        scale_features=True,
    )


def train_random_forest(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    index,
    horizon,
):
    """Train a reproducible baseline random-forest volatility forecaster.

    ``index`` is accepted to maintain the shared model-training interface;
    this feature-based model does not require it directly.
    """
    del index
    return _train_and_evaluate(
        RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "RandomForest",
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        horizon,
    )


def train_xgboost(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    index,
    horizon,
):
    """Train a reproducible baseline XGBoost volatility forecaster.

    ``index`` is accepted to maintain the shared model-training interface;
    this feature-based model does not require it directly.
    """
    del index
    return _train_and_evaluate(
        XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "XGBoost",
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        horizon,
    )
