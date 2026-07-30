""" the structure of this file:
1. Imports
2. Constants
   - Horizons
   - Indices
   - Models
3. main()
4. Run all experiments
5. Save results
6. Entry point """
"""
Main Experiment Runner

Runs all volatility forecasting experiments.

Workflow:
1. Load data
2. Loop through forecast horizons
3. Loop through indices
4. Loop through models
5. Train
6. Validate
7. Test
8. Save metrics
9. Save predictions
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.data_loader import load_split
from src.utils.model_registry import get_model
from src.utils.results_manager import (
    save_metrics,
    save_predictions,
)

# =============================================================================
# Forecast Horizons
# =============================================================================

HORIZONS = [
    "1day",
    "5day",
    "22day",
]

# =============================================================================
# Market Indices
# =============================================================================

INDICES = [
    "^GSPC",
    "^IXIC",
    "^FTSE",
    "^GDAXI",
    "^N225",
    "^HSI",
    "000001.SS",
]

# =============================================================================
# Models
# =============================================================================

MODELS = [
    "HistoricalVolatility",
    "Persistence",

    "GARCH",
    "EGARCH",
    "GJRGARCH",

    "LinearRegression",
    "RandomForest",
    "XGBoost",

    "LSTM",
    "GRU",
    "Transformer",

    "GARCHLSTM",
    "GARCHGRU",
    "GARCHTransformer",
]

TABULAR_MODELS = {
    "HistoricalVolatility",
    "Persistence",
    "GARCH",
    "EGARCH",
    "GJRGARCH",
    "LinearRegression",
    "RandomForest",
    "XGBoost",
}

DEEP_LEARNING_MODELS = {
    "LSTM",
    "GRU",
    "Transformer",
}

HYBRID_MODELS = {
    "GARCHLSTM",
    "GARCHGRU",
    "GARCHTransformer",
}

SEQUENCE_ROOT_CANDIDATES = (
    Path("data/sequence_datasets"),
    Path("data/sequence_datasets_test"),
)

GARCH_SEQUENCE_ROOT_CANDIDATES = (
    Path("data/garch_augmented_sequence_datasets"),
    Path("data/garch_sequence_datasets"),
    Path("data/garch_sequence_datasets_test"),
    Path("data/sequence_garch_datasets"),
    Path("data/sequence_datasets_garch"),
)


def _slugify_index(index):
    return "".join(character for character in index if character.isalnum())


def _find_existing_directory(candidates):
    for directory in candidates:
        if directory.exists():
            return directory

    candidate_text = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No validated dataset directory found. Checked: "
        f"{candidate_text}"
    )


def _has_required_sequence_files(directory):
    required_files = (
        "X_train_sequence.npy",
        "X_validation_sequence.npy",
        "X_test_sequence.npy",
        "y_train_sequence.csv",
        "y_validation_sequence.csv",
        "y_test_sequence.csv",
    )
    return all((directory / file_name).exists() for file_name in required_files)


def _load_sequence_split_from_directory(split_directory):
    required_files = {
        "X_train": split_directory / "X_train_sequence.npy",
        "X_validation": split_directory / "X_validation_sequence.npy",
        "X_test": split_directory / "X_test_sequence.npy",
        "y_train": split_directory / "y_train_sequence.csv",
        "y_validation": split_directory / "y_validation_sequence.csv",
        "y_test": split_directory / "y_test_sequence.csv",
    }

    missing_files = [
        str(file_path)
        for file_path in required_files.values()
        if not file_path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing required sequence dataset files: "
            + ", ".join(missing_files)
        )

    return {
        "X_train": np.load(required_files["X_train"]),
        "X_validation": np.load(required_files["X_validation"]),
        "X_test": np.load(required_files["X_test"]),
        "y_train": pd.read_csv(
            required_files["y_train"],
            index_col="Date",
            parse_dates=["Date"],
        ),
        "y_validation": pd.read_csv(
            required_files["y_validation"],
            index_col="Date",
            parse_dates=["Date"],
        ),
        "y_test": pd.read_csv(
            required_files["y_test"],
            index_col="Date",
            parse_dates=["Date"],
        ),
    }


def load_sequence_split(horizon):
    split_candidates = tuple(
        root / horizon
        for root in SEQUENCE_ROOT_CANDIDATES
    )
    for split_directory in split_candidates:
        if split_directory.exists() and _has_required_sequence_files(split_directory):
            return _load_sequence_split_from_directory(split_directory)

    for train_file in Path("data").glob("**/X_train_sequence.npy"):
        split_directory = train_file.parent
        directory_path = split_directory.as_posix().lower()
        if "garch" in directory_path:
            continue
        if horizon not in split_directory.parts:
            continue
        if _has_required_sequence_files(split_directory):
            return _load_sequence_split_from_directory(split_directory)

    candidate_text = ", ".join(str(path) for path in split_candidates)
    raise FileNotFoundError(
        f"Missing sequence dataset folder for horizon '{horizon}'. "
        f"Checked: {candidate_text}"
    )


def load_garch_augmented_sequence_split(horizon, index):
    index_slug = _slugify_index(index)
    split_candidates = []
    for root in GARCH_SEQUENCE_ROOT_CANDIDATES:
        split_candidates.extend(
            (
                root / horizon / index_slug,
                root / horizon / index,
                root / horizon,
            )
        )

    for split_directory in split_candidates:
        if split_directory.exists() and _has_required_sequence_files(split_directory):
            return _load_sequence_split_from_directory(split_directory)

    for train_file in Path("data").glob("**/X_train_sequence.npy"):
        split_directory = train_file.parent
        directory_path = split_directory.as_posix().lower()
        if "garch" not in directory_path:
            continue
        if horizon not in split_directory.parts:
            continue
        if (
            index_slug not in split_directory.parts
            and index not in split_directory.parts
            and split_directory.name != horizon
        ):
            continue
        if _has_required_sequence_files(split_directory):
            return _load_sequence_split_from_directory(split_directory)

    candidate_text = ", ".join(str(path) for path in split_candidates)
    raise FileNotFoundError(
        "Missing GARCH-augmented sequence dataset for "
        f"horizon '{horizon}' and index '{index}'. "
        f"Checked: {candidate_text}"
    )


def _extract_index_target(split, index):
    target_column = f"{index}_target"
    if target_column not in split.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in dataset."
        )
    return split[target_column]


def _select_dataset(
    model_name,
    horizon,
    index,
    tabular_split,
    sequence_splits,
    hybrid_sequence_splits,
):
    if model_name in TABULAR_MODELS:
        return tabular_split

    if model_name in DEEP_LEARNING_MODELS:
        if horizon not in sequence_splits:
            sequence_splits[horizon] = load_sequence_split(horizon)
        return sequence_splits[horizon]

    if model_name in HYBRID_MODELS:
        cache_key = (horizon, index)
        if cache_key not in hybrid_sequence_splits:
            hybrid_sequence_splits[cache_key] = (
                load_garch_augmented_sequence_split(horizon, index)
            )
        return hybrid_sequence_splits[cache_key]

    raise ValueError(f"Unknown model family for model '{model_name}'.")


def _validate_runner_configuration():
    if len(MODELS) != len(set(MODELS)):
        raise ValueError("MODELS contains duplicate entries.")

    for model_name in MODELS:
        get_model(model_name)


def main():

    print("=" * 60)
    print("VOLATILITY FORECASTING EXPERIMENTS")
    print("=" * 60)
    _validate_runner_configuration()
    print(f"Registered models: {len(MODELS)}")
    print(
        "Expected experiments: "
        f"{len(HORIZONS)} horizons x {len(INDICES)} indices x {len(MODELS)} models "
        f"= {len(HORIZONS) * len(INDICES) * len(MODELS)}"
    )

    summary = run_all_experiments()
    print("\nBenchmark run complete.")
    print(
        f"Successful experiments: {summary['successful_experiments']} | "
        f"Failed experiments: {summary['failed_experiments']} | "
        f"Total attempted: {summary['total_experiments']}"
    )

def run_all_experiments():
    total_experiments = len(HORIZONS) * len(INDICES) * len(MODELS)
    successful_experiments = 0
    failed_experiments = 0
    sequence_splits = {}
    hybrid_sequence_splits = {}

    for horizon in HORIZONS:

        print(f"\nLoading {horizon} tabular dataset...")

        tabular_split = load_split(horizon)
        forecast_horizon = int(horizon.removesuffix("day"))

        for index in INDICES:

            print(f"\nIndex: {index}")

            for model_name in MODELS:

                print(f"Running {horizon} | {index} | {model_name}...")

                try:
                    dataset_split = _select_dataset(
                        model_name,
                        horizon,
                        index,
                        tabular_split,
                        sequence_splits,
                        hybrid_sequence_splits,
                    )
                    y_train_index = _extract_index_target(
                        dataset_split["y_train"],
                        index,
                    )
                    y_validation_index = _extract_index_target(
                        dataset_split["y_validation"],
                        index,
                    )
                    y_test_index = _extract_index_target(
                        dataset_split["y_test"],
                        index,
                    )

                    model = get_model(model_name)
                    results = model(
                        dataset_split["X_train"],
                        y_train_index,
                        dataset_split["X_validation"],
                        y_validation_index,
                        dataset_split["X_test"],
                        y_test_index,
                        index,
                        forecast_horizon,
                    )

                    save_metrics(
                        horizon,
                        index,
                        model_name,
                        results["validation_metrics"],
                        results["test_metrics"],
                    )

                    save_predictions(
                        horizon,
                        index,
                        model_name,
                        y_test_index.index,
                        y_test_index.values,
                        results["test_predictions"],
                    )

                    successful_experiments += 1
                    print(
                        f"[SUCCESS] {horizon} | {index} | {model_name}"
                    )
                except Exception as error:
                    failed_experiments += 1
                    print(
                        f"[FAILED] {horizon} | {index} | {model_name} - "
                        f"{error}"
                    )
                    continue

    return {
        "total_experiments": total_experiments,
        "successful_experiments": successful_experiments,
        "failed_experiments": failed_experiments,
    }
# entry point
if __name__ == "__main__":
    main()
