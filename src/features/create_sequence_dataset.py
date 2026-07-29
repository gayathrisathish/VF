"""Sequence dataset construction utilities for deep-learning models.

This module converts the existing split-specific tabular forecasting datasets
into true sliding-window sequence datasets without crossing train/validation/
test boundaries.

Policy
------
- Windows are created independently within each split.
- The first ``sequence_length - 1`` rows of each split are discarded.
- Targets remain aligned with the final timestep of each window.
- Existing tabular datasets are never overwritten.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_OUTPUT_ROOT = os.path.join("data", "sequence_datasets")
DATE_COLUMN = "Date"


@dataclass(frozen=True)
class SequenceDatasetResult:
    """Container for sequence dataset outputs and validation metadata."""

    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: pd.DataFrame
    y_validation: pd.DataFrame
    y_test: pd.DataFrame
    metadata: Dict[str, object]


def _validate_split_inputs(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_validation: pd.DataFrame,
    y_test: pd.DataFrame,
    sequence_length: int,
) -> None:
    """Validate the split inputs before sequence construction."""
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool):
        raise ValueError("sequence_length must be a positive integer.")
    if sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer.")

    for split_name, split in (
        ("X_train", X_train),
        ("X_validation", X_validation),
        ("X_test", X_test),
        ("y_train", y_train),
        ("y_validation", y_validation),
        ("y_test", y_test),
    ):
        if not isinstance(split, pd.DataFrame):
            raise TypeError(f"{split_name} must be a pandas DataFrame.")
        if split.empty:
            raise ValueError(f"{split_name} must contain at least one row.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have matching lengths.")
    if len(X_validation) != len(y_validation):
        raise ValueError("X_validation and y_validation must have matching lengths.")
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have matching lengths.")

    for split_name, feature_split in (
        ("X_train", X_train),
        ("X_validation", X_validation),
        ("X_test", X_test),
    ):
        if DATE_COLUMN not in feature_split.columns:
            raise ValueError(f"{split_name} must contain a '{DATE_COLUMN}' column.")

    if not X_train.columns.equals(X_validation.columns):
        raise ValueError("X_train and X_validation must have identical ordered columns.")
    if not X_train.columns.equals(X_test.columns):
        raise ValueError("X_train and X_test must have identical ordered columns.")

    if not y_train.columns.equals(y_validation.columns):
        raise ValueError("y_train and y_validation must have identical ordered columns.")
    if not y_train.columns.equals(y_test.columns):
        raise ValueError("y_train and y_test must have identical ordered columns.")


def _sort_by_date(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Return a date-sorted copy of one split."""
    sorted_frame = frame.copy()
    try:
        sorted_frame[DATE_COLUMN] = pd.to_datetime(sorted_frame[DATE_COLUMN])
    except Exception as error:  # pragma: no cover - pandas raises varied subclasses
        raise ValueError(f"{split_name} has an invalid '{DATE_COLUMN}' column.") from error

    sorted_frame = sorted_frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    return sorted_frame


def _numeric_feature_columns(frame: pd.DataFrame) -> Iterable[str]:
    """Return the feature columns used for window construction."""
    return [column for column in frame.columns if column != DATE_COLUMN]


def _build_sequence_windows(
    X_split: pd.DataFrame,
    y_split: pd.DataFrame,
    sequence_length: int,
    split_name: str,
) -> Tuple[np.ndarray, pd.DataFrame, int]:
    """Build independent sliding windows for one split.

    The output features are stored as a true 3D NumPy tensor with shape
    ``(samples, sequence_length, n_features)``.
    """
    X_split = _sort_by_date(X_split, split_name)
    y_split = _sort_by_date(y_split, split_name)

    if len(X_split) != len(y_split):
        raise ValueError(f"{split_name} features and targets must have matching lengths.")

    feature_columns = list(_numeric_feature_columns(X_split))
    if not feature_columns:
        raise ValueError(f"{split_name} must contain at least one feature column.")

    feature_values = X_split[feature_columns].to_numpy(dtype=float)
    target_values = y_split.drop(columns=[DATE_COLUMN]).to_numpy(dtype=float)
    target_columns = [column for column in y_split.columns if column != DATE_COLUMN]

    if target_values.ndim != 2 or target_values.shape[1] != len(target_columns):
        raise ValueError(f"{split_name} targets must contain at least one target column.")

    if len(X_split) < sequence_length:
        raise ValueError(
            f"{split_name} has only {len(X_split)} rows, which is fewer than the required "
            f"sequence_length={sequence_length}."
        )

    sequence_rows = []
    sequence_targets = []
    sequence_dates = []

    for end_index in range(sequence_length - 1, len(X_split)):
        start_index = end_index - sequence_length + 1
        window = feature_values[start_index : end_index + 1]
        target_row = target_values[end_index]
        target_date = X_split.loc[end_index, DATE_COLUMN]

        if np.isnan(window).any():
            raise ValueError(
                f"{split_name} contains missing values inside a sequence window. "
                "Clean the source split before building sequences."
            )
        if not np.isfinite(window).all():
            raise ValueError(f"{split_name} contains non-finite feature values.")
        if not np.isfinite(target_row).all():
            raise ValueError(f"{split_name} contains non-finite target values.")

        sequence_rows.append(window)
        sequence_targets.append(target_row)
        sequence_dates.append(target_date)

    X_sequence = np.stack(sequence_rows, axis=0)

    y_sequence = pd.DataFrame(sequence_targets, columns=target_columns)
    y_sequence.insert(0, DATE_COLUMN, sequence_dates)

    discarded_rows = sequence_length - 1
    return X_sequence, y_sequence, discarded_rows


def create_sequence_dataset(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_validation: pd.DataFrame,
    y_test: pd.DataFrame,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> SequenceDatasetResult:
    """Create split-local sequence datasets from existing tabular splits.

    Parameters
    ----------
    X_train, X_validation, X_test, y_train, y_validation, y_test:
        Existing split datasets. Each split must already be chronologically
        ordered or contain a sortable ``Date`` column.
    sequence_length:
        Fixed lookback window length. Default is 30 trading days.

    Returns
    -------
    SequenceDatasetResult
        Flattened sequence datasets and metadata describing the conversion.
    """
    _validate_split_inputs(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        sequence_length,
    )

    X_train_seq, y_train_seq, discarded_train = _build_sequence_windows(
        X_train,
        y_train,
        sequence_length,
        "X_train",
    )
    X_validation_seq, y_validation_seq, discarded_validation = _build_sequence_windows(
        X_validation,
        y_validation,
        sequence_length,
        "X_validation",
    )
    X_test_seq, y_test_seq, discarded_test = _build_sequence_windows(
        X_test,
        y_test,
        sequence_length,
        "X_test",
    )

    metadata = {
        "sequence_length": sequence_length,
        "original_shapes": {
            "train": tuple(X_train.shape),
            "validation": tuple(X_validation.shape),
            "test": tuple(X_test.shape),
        },
        "sequence_shapes": {
            "train": tuple(X_train_seq.shape),
            "validation": tuple(X_validation_seq.shape),
            "test": tuple(X_test_seq.shape),
        },
        "discarded_rows": {
            "train": discarded_train,
            "validation": discarded_validation,
            "test": discarded_test,
        },
        "boundary_crossing": False,
    }

    return SequenceDatasetResult(
        X_train=X_train_seq,
        X_validation=X_validation_seq,
        X_test=X_test_seq,
        y_train=y_train_seq,
        y_validation=y_validation_seq,
        y_test=y_test_seq,
        metadata=metadata,
    )


def save_sequence_dataset(
    result: SequenceDatasetResult,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    split_name: str = "30day",
) -> Dict[str, str]:
    """Save a sequence dataset result to a clean, separate project location."""
    output_dir = os.path.join(output_root, split_name)
    os.makedirs(output_dir, exist_ok=True)

    outputs = {
        "X_train": os.path.join(output_dir, "X_train_sequence.npy"),
        "X_validation": os.path.join(output_dir, "X_validation_sequence.npy"),
        "X_test": os.path.join(output_dir, "X_test_sequence.npy"),
        "y_train": os.path.join(output_dir, "y_train_sequence.csv"),
        "y_validation": os.path.join(output_dir, "y_validation_sequence.csv"),
        "y_test": os.path.join(output_dir, "y_test_sequence.csv"),
    }

    np.save(outputs["X_train"], result.X_train)
    np.save(outputs["X_validation"], result.X_validation)
    np.save(outputs["X_test"], result.X_test)
    result.y_train.to_csv(outputs["y_train"], index=False)
    result.y_validation.to_csv(outputs["y_validation"], index=False)
    result.y_test.to_csv(outputs["y_test"], index=False)

    metadata_path = os.path.join(output_dir, "sequence_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(result.metadata, metadata_file, indent=2, default=str)
    outputs["metadata"] = metadata_path

    return outputs


def load_split_csvs(split_dir: str) -> Dict[str, pd.DataFrame]:
    """Load an existing split directory containing tabular forecasting data."""
    required_files = {
        "X_train": "X_train.csv",
        "X_validation": "X_validation.csv",
        "X_test": "X_test.csv",
        "y_train": "y_train.csv",
        "y_validation": "y_validation.csv",
        "y_test": "y_test.csv",
    }

    datasets: Dict[str, pd.DataFrame] = {}
    for key, file_name in required_files.items():
        file_path = os.path.join(split_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required split file: {file_path}")
        datasets[key] = pd.read_csv(file_path)

    return datasets


def build_and_save_from_split_dir(
    split_dir: str,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    split_name: str = "30day",
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> Tuple[SequenceDatasetResult, Dict[str, str]]:
    """Convenience wrapper for loading, building, validating, and saving."""
    datasets = load_split_csvs(split_dir)
    result = create_sequence_dataset(
        datasets["X_train"],
        datasets["X_validation"],
        datasets["X_test"],
        datasets["y_train"],
        datasets["y_validation"],
        datasets["y_test"],
        sequence_length=sequence_length,
    )
    outputs = save_sequence_dataset(result, output_root=output_root, split_name=split_name)
    return result, outputs


if __name__ == "__main__":
    default_split_dir = os.path.join("data", "splits", "1day")
    result, outputs = build_and_save_from_split_dir(default_split_dir)
    print("Sequence dataset creation complete.")
    print(f"Original training shape: {result.metadata['original_shapes']['train']}")
    print(f"Sequence training shape: {result.metadata['sequence_shapes']['train']}")
    print(f"Original validation shape: {result.metadata['original_shapes']['validation']}")
    print(f"Sequence validation shape: {result.metadata['sequence_shapes']['validation']}")
    print(f"Original test shape: {result.metadata['original_shapes']['test']}")
    print(f"Sequence test shape: {result.metadata['sequence_shapes']['test']}")
    print(f"Number of discarded rows per split: {result.metadata['discarded_rows']}")
    print(f"Boundary crossing: {result.metadata['boundary_crossing']}")
    print("Saved files:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")