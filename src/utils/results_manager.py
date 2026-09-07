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

# =============================================================================
# Save Trained Models
# =============================================================================

MODELS_DIR = RESULTS_DIR / "models"


def _slugify_index(index):
    """Return a filesystem-safe form of a market index symbol."""

    slug = "".join(
        character
        for character in str(index)
        if character.isalnum()
    )

    if not slug:
        raise ValueError(f"Index {index!r} produced an empty filename slug.")

    return slug


def _is_keras_model(fitted_model):
    """Return True when the object is a Keras model that owns a .save method."""

    try:
        import keras
    except ImportError:
        keras = None

    if keras is not None and isinstance(fitted_model, keras.Model):
        return True

    module_name = type(fitted_model).__module__ or ""

    return (
        module_name.startswith(("keras.", "tensorflow.python.keras"))
        and hasattr(fitted_model, "save")
    )


def save_model(
    horizon,
    index,
    model_name,
    fitted_model
):
    """
    Persist one fitted model to

    results/models/<horizon>/<index>/<model_name>.<extension>

    Keras models are written with the native Keras v3 ".keras" format.
    Every other estimator (scikit-learn pipelines, fitted arch results) is
    written with joblib as ".joblib".

    Raises
    ------
    ValueError
        If no fitted estimator was supplied.
    TypeError
        If a model-name string is supplied instead of a fitted estimator.
    RuntimeError
        If serialization fails.
    """

    if fitted_model is None:
        raise ValueError(
            f"No fitted estimator supplied for {model_name} "
            f"({horizon} | {index}); refusing to save."
        )

    if isinstance(fitted_model, str):
        raise TypeError(
            f"Refusing to save the model-name string {fitted_model!r} for "
            f"{model_name} ({horizon} | {index}); a fitted estimator is "
            "required."
        )

    output_directory = (
        MODELS_DIR /
        horizon /
        _slugify_index(index)
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    if _is_keras_model(fitted_model):
        model_path = output_directory / f"{model_name}.keras"
        try:
            fitted_model.save(model_path)
        except Exception as error:
            raise RuntimeError(
                f"Failed to save Keras model {model_name} "
                f"({horizon} | {index}) to {model_path}: {error}"
            ) from error
    else:
        import joblib

        model_path = output_directory / f"{model_name}.joblib"
        try:
            joblib.dump(fitted_model, model_path)
        except Exception as error:
            raise RuntimeError(
                f"Failed to save model {model_name} "
                f"({horizon} | {index}) to {model_path}: {error}"
            ) from error

    if not model_path.exists():
        raise RuntimeError(
            f"Model file was not created for {model_name} "
            f"({horizon} | {index}): {model_path}"
        )

    return model_path
