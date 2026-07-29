"""GARCH-derived feature generation for hybrid volatility models.

This module creates a split-preserving, leakage-safe feature set by fitting a
GARCH(1,1) model on the training split only, then reusing the fitted
parameters for validation and test without re-estimation.

The output is a single additional feature column, ``GARCH_sigma``, appended to
the existing forecasting features while leaving the targets unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd
from arch import arch_model

from src.utils.data_loader import load_split


ANNUALIZATION_FACTOR = 252
FEATURE_NAME = "GARCH_sigma"
DEFAULT_OUTPUT_ROOT = Path("data/garch_splits")
DEFAULT_HORIZON = "1day"
GARCH_SPECIFICATION = {
    "mean": "Zero",
    "vol": "GARCH",
    "p": 1,
    "o": 0,
    "q": 1,
    "dist": "normal",
    "rescale": False,
}


@dataclass(frozen=True)
class GarchFeatureDatasetResult:
    """Container for augmented splits and validation metadata."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_validation: pd.DataFrame
    y_test: pd.DataFrame
    metadata: Dict[str, object]


def _slugify(value: str) -> str:
    """Create a filesystem-safe version of an index name."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value.strip("^"))
    cleaned = cleaned.strip("_")
    return cleaned or "index"


def _expected_return_column(index: str) -> str:
    return f"{index}_return"


def _ensure_datetime_index(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Return a sorted copy of ``frame`` with a valid datetime index."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{split_name} must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError(f"{split_name} must contain at least one row.")

    sorted_frame = frame.copy()
    sorted_frame.index = pd.to_datetime(sorted_frame.index)
    if sorted_frame.index.has_duplicates:
        raise ValueError(f"{split_name} contains duplicate dates.")
    sorted_frame = sorted_frame.sort_index()
    return sorted_frame


def _validate_split_boundaries(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_validation: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
    """Validate split-local chronology and row alignment."""
    for split_name, split in (
        ("X_train", X_train),
        ("X_validation", X_validation),
        ("X_test", X_test),
        ("y_train", y_train),
        ("y_validation", y_validation),
        ("y_test", y_test),
    ):
        if not split.index.is_monotonic_increasing:
            raise ValueError(f"{split_name} must be sorted by date.")

    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train must have identical dates.")
    if not X_validation.index.equals(y_validation.index):
        raise ValueError("X_validation and y_validation must have identical dates.")
    if not X_test.index.equals(y_test.index):
        raise ValueError("X_test and y_test must have identical dates.")

    if X_train.index[-1] >= X_validation.index[0]:
        raise ValueError("Training and validation splits overlap.")
    if X_validation.index[-1] >= X_test.index[0]:
        raise ValueError("Validation and test splits overlap.")


def _validate_return_column(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    index: str,
) -> str:
    """Ensure the required return column exists in each feature split."""
    return_column = _expected_return_column(index)

    for split_name, split in (
        ("X_train", X_train),
        ("X_validation", X_validation),
        ("X_test", X_test),
    ):
        if return_column not in split.columns:
            raise ValueError(
                f"{split_name} must contain the required column '{return_column}'."
            )

    return return_column


def _fit_garch_feature_series(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    return_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the training GARCH model and generate aligned sigma values."""
    training_returns = np.asarray(X_train[return_column], dtype=float)
    validation_returns = np.asarray(X_validation[return_column], dtype=float)
    test_returns = np.asarray(X_test[return_column], dtype=float)

    if not np.isfinite(training_returns).all():
        raise ValueError("Training returns must be finite.")
    if not np.isfinite(validation_returns).all():
        raise ValueError("Validation returns must be finite.")
    if not np.isfinite(test_returns).all():
        raise ValueError("Test returns must be finite.")

    training_model = arch_model(
        training_returns,
        mean=GARCH_SPECIFICATION["mean"],
        vol=GARCH_SPECIFICATION["vol"],
        p=GARCH_SPECIFICATION["p"],
        o=GARCH_SPECIFICATION["o"],
        q=GARCH_SPECIFICATION["q"],
        dist=GARCH_SPECIFICATION["dist"],
        rescale=GARCH_SPECIFICATION["rescale"],
    )
    fitted_model = training_model.fit(disp="off", show_warning=False)
    if getattr(fitted_model, "convergence_flag", 0) != 0:
        raise RuntimeError("GARCH optimizer did not converge on the training split.")

    training_sigma = np.asarray(fitted_model.conditional_volatility, dtype=float)
    training_sigma *= np.sqrt(ANNUALIZATION_FACTOR)

    if training_sigma.shape[0] != training_returns.shape[0]:
        raise RuntimeError("Training sigma length does not match the training split.")

    all_returns = np.concatenate((training_returns, validation_returns, test_returns))
    fixed_model = arch_model(
        all_returns,
        mean=GARCH_SPECIFICATION["mean"],
        vol=GARCH_SPECIFICATION["vol"],
        p=GARCH_SPECIFICATION["p"],
        o=GARCH_SPECIFICATION["o"],
        q=GARCH_SPECIFICATION["q"],
        dist=GARCH_SPECIFICATION["dist"],
        rescale=GARCH_SPECIFICATION["rescale"],
    ).fix(fitted_model.params)

    forecast = fixed_model.forecast(
        start=len(training_returns),
        horizon=1,
        reindex=False,
    )
    out_of_sample_variance = forecast.variance.iloc[:, 0].to_numpy(dtype=float)
    expected_out_of_sample = validation_returns.shape[0] + test_returns.shape[0]
    if out_of_sample_variance.shape[0] != expected_out_of_sample:
        raise RuntimeError(
            "Out-of-sample GARCH forecast length does not match the validation and test splits."
        )

    validation_sigma = np.sqrt(
        np.maximum(out_of_sample_variance[: validation_returns.shape[0]], 0.0)
    )
    validation_sigma *= np.sqrt(ANNUALIZATION_FACTOR)

    test_sigma = np.sqrt(
        np.maximum(out_of_sample_variance[validation_returns.shape[0] :], 0.0)
    )
    test_sigma *= np.sqrt(ANNUALIZATION_FACTOR)

    return training_sigma, validation_sigma, test_sigma


def _append_feature_column(split: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    """Append the GARCH feature column without altering the split boundary."""
    augmented = split.copy()
    augmented[FEATURE_NAME] = values
    return augmented


def _validate_augmented_dataset(
    original: Mapping[str, pd.DataFrame],
    augmented: Mapping[str, pd.DataFrame],
) -> Dict[str, object]:
    """Check the generated feature set for leakage, alignment, and shape changes."""
    checks: Dict[str, object] = {
        "boundary_crossing": False,
        "date_alignment": {},
        "row_counts": {},
        "feature_column_added": {},
        "missing_values": {},
        "index_monotonic": {},
    }

    for split_name in ("train", "validation", "test"):
        original_frame = original[split_name]
        augmented_frame = augmented[split_name]

        checks["row_counts"][split_name] = {
            "original": len(original_frame),
            "augmented": len(augmented_frame),
        }
        checks["date_alignment"][split_name] = original_frame.index.equals(
            augmented_frame.index
        )
        checks["feature_column_added"][split_name] = FEATURE_NAME in augmented_frame.columns
        checks["missing_values"][split_name] = bool(augmented_frame[FEATURE_NAME].isna().any())
        checks["index_monotonic"][split_name] = augmented_frame.index.is_monotonic_increasing

        if not checks["date_alignment"][split_name]:
            raise RuntimeError(f"{split_name} dates changed during feature generation.")
        if checks["row_counts"][split_name]["original"] != checks["row_counts"][split_name]["augmented"]:
            raise RuntimeError(f"{split_name} row count changed during feature generation.")
        if checks["missing_values"][split_name]:
            raise RuntimeError(f"{split_name} contains missing GARCH_sigma values.")

    if not augmented["train"].index[-1] < augmented["validation"].index[0]:
        raise RuntimeError("Training and validation dates overlap after augmentation.")
    if not augmented["validation"].index[-1] < augmented["test"].index[0]:
        raise RuntimeError("Validation and test dates overlap after augmentation.")

    return checks


def _build_result(
    horizon: str,
    index: str,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_validation: pd.DataFrame,
    y_test: pd.DataFrame,
) -> GarchFeatureDatasetResult:
    """Create the augmented dataset and metadata for a single index."""
    return_column = _validate_return_column(X_train, X_validation, X_test, index)
    _validate_split_boundaries(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )

    training_sigma, validation_sigma, test_sigma = _fit_garch_feature_series(
        X_train,
        X_validation,
        X_test,
        return_column,
    )

    X_train_augmented = _append_feature_column(X_train, training_sigma)
    X_validation_augmented = _append_feature_column(X_validation, validation_sigma)
    X_test_augmented = _append_feature_column(X_test, test_sigma)

    original_frames = {
        "train": X_train,
        "validation": X_validation,
        "test": X_test,
    }
    augmented_frames = {
        "train": X_train_augmented,
        "validation": X_validation_augmented,
        "test": X_test_augmented,
    }
    checks = _validate_augmented_dataset(original_frames, augmented_frames)

    metadata: Dict[str, object] = {
        "feature_name": FEATURE_NAME,
        "index": index,
        "horizon": horizon,
        "return_column": return_column,
        "annualization_factor": ANNUALIZATION_FACTOR,
        "garch_specification": GARCH_SPECIFICATION,
        "source_split_dir": str(Path("data") / "splits" / horizon),
        "feature_column_count": {
            "original": len(X_train.columns),
            "augmented": len(X_train_augmented.columns),
        },
        "split_shapes": {
            "train": {
                "original": list(X_train.shape),
                "augmented": list(X_train_augmented.shape),
            },
            "validation": {
                "original": list(X_validation.shape),
                "augmented": list(X_validation_augmented.shape),
            },
            "test": {
                "original": list(X_test.shape),
                "augmented": list(X_test_augmented.shape),
            },
        },
        "checks": checks,
    }

    return GarchFeatureDatasetResult(
        X_train=X_train_augmented,
        X_validation=X_validation_augmented,
        X_test=X_test_augmented,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        metadata=metadata,
    )


def save_garch_feature_dataset(
    result: GarchFeatureDatasetResult,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    overwrite: bool = True,
) -> Path:
    """Persist one augmented split set to disk and return the output directory."""
    index_slug = _slugify(str(result.metadata["index"]))
    horizon = str(result.metadata["horizon"])
    output_dir = output_root / horizon / index_slug

    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    result.X_train.to_csv(output_dir / "X_train.csv")
    result.X_validation.to_csv(output_dir / "X_validation.csv")
    result.X_test.to_csv(output_dir / "X_test.csv")

    result.y_train.to_csv(output_dir / "y_train.csv")
    result.y_validation.to_csv(output_dir / "y_validation.csv")
    result.y_test.to_csv(output_dir / "y_test.csv")

    metadata_path = output_dir / "garch_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(result.metadata, metadata_file, indent=2, default=str)

    return output_dir


def build_garch_feature_dataset(
    horizon: str,
    index: str,
    overwrite: bool = True,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> GarchFeatureDatasetResult:
    """Load one split set, generate the GARCH feature, and save the result."""
    split = load_split(horizon)

    result = _build_result(
        horizon=horizon,
        index=index,
        X_train=_ensure_datetime_index(split["X_train"], "X_train"),
        X_validation=_ensure_datetime_index(split["X_validation"], "X_validation"),
        X_test=_ensure_datetime_index(split["X_test"], "X_test"),
        y_train=_ensure_datetime_index(split["y_train"], "y_train"),
        y_validation=_ensure_datetime_index(split["y_validation"], "y_validation"),
        y_test=_ensure_datetime_index(split["y_test"], "y_test"),
    )

    save_garch_feature_dataset(result, output_root=output_root, overwrite=overwrite)
    return result


def build_all_garch_feature_datasets(
    horizon: str = DEFAULT_HORIZON,
    overwrite: bool = True,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Dict[str, GarchFeatureDatasetResult]:
    """Generate GARCH-augmented split sets for every index in one horizon."""
    split = load_split(horizon)
    X_train = _ensure_datetime_index(split["X_train"], "X_train")
    X_validation = _ensure_datetime_index(split["X_validation"], "X_validation")
    X_test = _ensure_datetime_index(split["X_test"], "X_test")
    y_train = _ensure_datetime_index(split["y_train"], "y_train")
    y_validation = _ensure_datetime_index(split["y_validation"], "y_validation")
    y_test = _ensure_datetime_index(split["y_test"], "y_test")

    return_columns = [
        column[:-len("_return")]
        for column in X_train.columns
        if column.endswith("_return")
    ]

    results: Dict[str, GarchFeatureDatasetResult] = {}
    for index in return_columns:
        result = _build_result(
            horizon=horizon,
            index=index,
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
        )
        save_garch_feature_dataset(result, output_root=output_root, overwrite=overwrite)
        results[index] = result

    return results


if __name__ == "__main__":
    generated = build_garch_feature_dataset(DEFAULT_HORIZON, "^GSPC")
    print("GARCH feature generation complete.")
    print(f"Train shape: {generated.X_train.shape}")
    print(f"Validation shape: {generated.X_validation.shape}")
    print(f"Test shape: {generated.X_test.shape}")