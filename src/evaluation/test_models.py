"""Held-out test inference for every saved model.

This stage is deliberately separate from training and from evaluation:

    saved checkpoint  ->  held-out test split  ->  test predictions

Nothing here fits, refits or updates a model. Every estimator is read back
from ``results/models/<horizon>/<index>/<Model>.<ext>`` and applied to the test
split only, and the predictions are written to ``results/test_predictions``
so the training-time files under ``results/predictions`` are left untouched.

Preprocessing follows whatever each family did during training:

* The scikit-learn and XGBoost checkpoints are whole ``Pipeline`` objects, so
  the fitted imputer and scaler travel with the estimator. Raw test features go
  straight to ``predict``.
* The deep-learning and hybrid networks were trained on tensors normalised
  *outside* the graph, so the saved ``.keras`` file expects normalised input.
  This module rebuilds that normaliser by adapting it on the same training
  tensor the model was trained on, exactly as ``_normalize_features`` did.
  Test data is never adapted on.
* The ARCH-family checkpoints hold fitted parameters. Forecasts reuse those
  parameters over the observed return history through the project's own
  ``_forecast_out_of_sample``; no parameter is re-estimated.
* The two benchmarks are rules, not estimators, and have no checkpoint. They
  are reproduced from the same lagged-volatility column the trainer used.

Run it with::

    python -m src.evaluation.test_models
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.experiments.run_experiments import (
    DEEP_LEARNING_MODELS,
    HORIZONS,
    HYBRID_MODELS,
    INDICES,
    MODELS,
    TABULAR_MODELS,
    _extract_index_target,
    _slugify_index,
    load_garch_augmented_sequence_split,
    load_sequence_split,
)
from src.models.deep_learning import _validate_predictions as _validate_network_predictions
from src.models.garch_models import (
    ANNUALIZATION_FACTOR,
    _fallback_predictions,
    _forecast_out_of_sample,
)
from src.models.ml_models import _validate_predictions as _validate_estimator_predictions
from src.utils.data_loader import load_split


MODELS_DIR = Path("results/models")
DEFAULT_OUTPUT_DIR = Path("results/test_predictions")
SEQUENCE_LENGTH = 30

BENCHMARK_MODELS = {"HistoricalVolatility", "Persistence"}
ARCH_MODELS = {"GARCH", "EGARCH", "GJRGARCH"}
ESTIMATOR_MODELS = {"LinearRegression", "RandomForest", "XGBoost"}
NETWORK_MODELS = DEEP_LEARNING_MODELS | HYBRID_MODELS

# Specifications copied from the train_* wrappers in src.models.garch_models so
# a fixed-parameter forecast rebuilds the same process it was fitted on.
ARCH_SPECIFICATIONS = {
    "GARCH": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
    "EGARCH": {"vol": "EGARCH", "p": 1, "o": 0, "q": 1},
    "GJRGARCH": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
}

OUTPUT_COLUMNS = ["Horizon", "Index", "Model", "Date", "Actual", "Predicted"]


# =============================================================================
# Checkpoints
# =============================================================================

def checkpoint_path(model_name: str, index: str, horizon: str) -> Optional[Path]:
    """Return the saved checkpoint for one experiment, or None for a rule."""

    if model_name in BENCHMARK_MODELS:
        return None

    directory = MODELS_DIR / horizon / _slugify_index(index)
    extension = "keras" if model_name in NETWORK_MODELS else "joblib"

    return directory / f"{model_name}.{extension}"


def _file_fingerprint(path: Path) -> Dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "modified": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256_prefix": digest,
    }


def _load_checkpoint(model_name: str, path: Path):
    if model_name in NETWORK_MODELS:
        import keras

        return keras.saving.load_model(path)

    import joblib

    return joblib.load(path)


# =============================================================================
# Datasets
# =============================================================================

def _load_dataset(model_name: str, horizon: str, index: str) -> Dict[str, object]:
    """Return the split set the trainer used for this model family."""

    if model_name in TABULAR_MODELS:
        return load_split(horizon)

    if model_name in DEEP_LEARNING_MODELS:
        return load_sequence_split(horizon)

    if model_name in HYBRID_MODELS:
        return load_garch_augmented_sequence_split(horizon, index)

    raise ValueError(f"Unknown model family for model '{model_name}'.")


def _shared_test_dates(horizon: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(load_split(horizon)["y_test"].index))


def _expected_test_dates(model_name: str, horizon: str) -> pd.DatetimeIndex:
    """Return the test dates this model family should predict on."""

    shared = _shared_test_dates(horizon)
    if model_name in TABULAR_MODELS:
        return shared

    return shared[SEQUENCE_LENGTH - 1 :]


# =============================================================================
# Inference, one family at a time
# =============================================================================

def _predict_benchmark(index: str, dataset: Dict[str, object]) -> np.ndarray:
    """Reproduce the rule both benchmarks use: realized volatility at row t."""

    rv_column = f"{index}_rv"
    features = dataset["X_test"]
    if rv_column not in features.columns:
        raise KeyError(f"Benchmark column '{rv_column}' missing from X_test.")

    return np.asarray(features[rv_column], dtype=float)


def _predict_arch(
    model_name: str,
    index: str,
    horizon: str,
    dataset: Dict[str, object],
    fitted_model,
) -> np.ndarray:
    """Forecast with saved ARCH parameters; nothing is re-estimated."""

    return_column = f"{index}_return"
    splits = [dataset["X_train"], dataset["X_validation"], dataset["X_test"]]
    for split in splits:
        if return_column not in split.columns:
            raise KeyError(f"Return column '{return_column}' missing from a split.")

    all_returns = np.concatenate(
        [np.asarray(split[return_column], dtype=float) for split in splits]
    )

    if fitted_model is None:
        # The trainer stores None when the optimiser failed and it fell back to
        # the lagged-volatility forecast; testing reproduces that same fallback.
        return _fallback_predictions(
            dataset["X_test"],
            index,
            np.asarray(
                _extract_index_target(dataset["y_train"], index),
                dtype=float,
            ),
        )

    _, test_predictions = _forecast_out_of_sample(
        fitted_model,
        ARCH_SPECIFICATIONS[model_name],
        all_returns,
        len(dataset["X_train"]),
        len(dataset["X_validation"]),
        len(dataset["X_test"]),
        int(horizon.removesuffix("day")),
    )

    return np.asarray(test_predictions, dtype=float)


def _predict_estimator(dataset: Dict[str, object], pipeline) -> np.ndarray:
    """Predict with a saved scikit-learn pipeline, preprocessing included."""

    features = np.asarray(dataset["X_test"], dtype=float)

    return _validate_estimator_predictions(pipeline.predict(features))


def _predict_network(dataset: Dict[str, object], network) -> np.ndarray:
    """Predict with a saved network, rebuilding the training-time normaliser.

    ``adapt`` runs on the training tensor only, which is what
    ``deep_learning._normalize_features`` did while the model was fitted. Keras
    ``predict`` runs the graph in inference mode, so dropout is bypassed and no
    weights are updated; no gradient tape is opened anywhere in this module.
    """

    import tensorflow as tf

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(dataset["X_train"])
    normalized_test = np.asarray(normalizer(dataset["X_test"]))

    weights_before = [np.array(weight) for weight in network.get_weights()]
    predictions = network.predict(normalized_test, verbose=0)
    weights_after = network.get_weights()

    for before, after in zip(weights_before, weights_after):
        if not np.array_equal(before, after):
            raise RuntimeError("Model weights changed during inference.")

    return _validate_network_predictions(predictions)


# =============================================================================
# One experiment
# =============================================================================

def run_one(model_name: str, index: str, horizon: str) -> Dict[str, object]:
    """Load one saved model, predict the test split, and return tidy rows."""

    dataset = _load_dataset(model_name, horizon, index)
    actual = _extract_index_target(dataset["y_test"], index)
    dates = pd.DatetimeIndex(pd.to_datetime(actual.index))

    path = checkpoint_path(model_name, index, horizon)
    checkpoint: Dict[str, object]

    if path is None:
        checkpoint = {"path": None, "note": "rule-based model, no checkpoint"}
        predictions = _predict_benchmark(index, dataset)
    else:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for {model_name} | {index} | {horizon}: {path}"
            )
        checkpoint = _file_fingerprint(path)
        loaded = _load_checkpoint(model_name, path)

        if model_name in ARCH_MODELS:
            predictions = _predict_arch(model_name, index, horizon, dataset, loaded)
        elif model_name in ESTIMATOR_MODELS:
            predictions = _predict_estimator(dataset, loaded)
        elif model_name in NETWORK_MODELS:
            predictions = _predict_network(dataset, loaded)
        else:
            raise ValueError(f"No inference path defined for '{model_name}'.")

    predictions = np.asarray(predictions, dtype=float).reshape(-1)

    problems = _check_predictions(model_name, horizon, dates, predictions)

    frame = pd.DataFrame(
        {
            "Horizon": horizon,
            "Index": index,
            "Model": model_name,
            "Date": dates,
            "Actual": np.asarray(actual, dtype=float),
            "Predicted": predictions,
        }
    )

    return {
        "frame": frame[OUTPUT_COLUMNS],
        "checkpoint": checkpoint,
        "rows": len(frame),
        "test_start": str(dates[0].date()),
        "test_end": str(dates[-1].date()),
        "problems": problems,
    }


def _check_predictions(
    model_name: str,
    horizon: str,
    dates: pd.DatetimeIndex,
    predictions: np.ndarray,
) -> List[str]:
    """Verify shape, alignment and finiteness for one experiment."""

    problems: List[str] = []

    if predictions.shape[0] != len(dates):
        problems.append(
            f"predicted {predictions.shape[0]} values for {len(dates)} test dates"
        )
    if not np.isfinite(predictions).all():
        problems.append("predictions contain NaN or infinite values")
    if dates.has_duplicates:
        problems.append("test dates contain duplicates")
    if not dates.is_monotonic_increasing:
        problems.append("test dates are not in chronological order")

    expected = _expected_test_dates(model_name, horizon)
    if not dates.equals(expected):
        problems.append(
            f"test window {len(dates)} rows {dates[0].date()}..{dates[-1].date()} "
            f"does not match the shared split's {len(expected)} rows "
            f"{expected[0].date()}..{expected[-1].date()}"
        )

    return problems


# =============================================================================
# Runner
# =============================================================================

def run_testing(
    models: Sequence[str] = tuple(MODELS),
    indices: Sequence[str] = tuple(INDICES),
    horizons: Sequence[str] = tuple(HORIZONS),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write: bool = True,
) -> Dict[str, object]:
    """Run held-out inference for every requested experiment."""

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "sequence_length": SEQUENCE_LENGTH,
        "experiments": [],
    }
    failures: List[str] = []
    flagged: List[str] = []

    for horizon in horizons:
        frames: List[pd.DataFrame] = []

        for index in indices:
            for model_name in models:
                label = f"{horizon} | {index} | {model_name}"
                try:
                    result = run_one(model_name, index, horizon)
                except Exception as error:  # noqa: BLE001 - reported, not swallowed
                    failures.append(f"{label}: {type(error).__name__}: {error}")
                    print(f"[FAIL] {label}: {type(error).__name__}: {error}")
                    continue

                frames.append(result["frame"])
                manifest["experiments"].append(
                    {
                        "horizon": horizon,
                        "index": index,
                        "model": model_name,
                        "rows": result["rows"],
                        "test_start": result["test_start"],
                        "test_end": result["test_end"],
                        "checkpoint": result["checkpoint"],
                        "problems": result["problems"],
                    }
                )

                if result["problems"]:
                    flagged.append(f"{label}: {'; '.join(result['problems'])}")
                    print(f"[WARN] {label}: {'; '.join(result['problems'])}")
                else:
                    print(
                        f"[OK]   {label}: {result['rows']} rows "
                        f"{result['test_start']}..{result['test_end']}"
                    )

        if frames and write:
            combined = pd.concat(frames, ignore_index=True)
            destination = output_dir / f"test_predictions_{horizon}.csv"
            combined.to_csv(destination, index=False)
            print(f"       wrote {len(combined)} rows -> {destination}")

    if write:
        manifest_path = output_dir / "test_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
        print(f"       wrote manifest -> {manifest_path}")

    manifest["failures"] = failures
    manifest["flagged"] = flagged

    return manifest


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load every saved model and predict the held-out test split. "
            "No model is trained, tuned or modified."
        )
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--indices", nargs="+", default=list(INDICES), choices=list(INDICES))
    parser.add_argument(
        "--horizons", nargs="+", default=list(HORIZONS), choices=list(HORIZONS)
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference and report, but write no files.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _build_argument_parser().parse_args(argv)

    manifest = run_testing(
        models=arguments.models,
        indices=arguments.indices,
        horizons=arguments.horizons,
        output_dir=arguments.output_dir,
        write=not arguments.dry_run,
    )

    total = len(manifest["experiments"])
    failures = manifest["failures"]
    flagged = manifest["flagged"]

    print()
    print(f"Experiments completed : {total}")
    print(f"Failed                : {len(failures)}")
    print(f"Flagged               : {len(flagged)}")

    for entry in failures:
        print(f"  [FAIL] {entry}")
    for entry in flagged:
        print(f"  [WARN] {entry}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
