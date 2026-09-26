"""Score the held-out test predictions on the window every model shares.

This stage follows :mod:`src.evaluation.test_models` and reads nothing else:

    results/test_predictions/  ->  common-window metrics and per-date losses

The training-time files under ``results/metrics`` and ``results/predictions``
are never opened. Every number here comes from reloaded-checkpoint inference.

Tabular models forecast from the first test date, while sequence models need a
30-step history and start 29 dates later. Scoring each family on its own dates
would compare different market periods, so every model is scored only on the
dates all 14 models cover. That window is derived from the predictions and must
equal ``EXPECTED_COMMON_WINDOWS``, so a change in the test calendar fails here
instead of silently shifting the results.

RMSE, MAE and QLIKE come from :mod:`src.evaluation.metrics` unchanged. The
per-observation losses are the same functions applied to one observation at a
time, and their means are checked against the cell metrics, so the loss series
later used for Diebold-Mariano and Model Confidence Set tests are the metrics
reported here.

Run it with::

    python -m src.evaluation.evaluate_predictions
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_metrics, mae, qlike, rmse
from src.experiments.run_experiments import (
    DEEP_LEARNING_MODELS,
    HORIZONS,
    HYBRID_MODELS,
    INDICES,
    MODELS,
    TABULAR_MODELS,
)
from src.utils.model_registry import MODEL_REGISTRY


SOURCE_ROOT = Path("results/test_predictions")
SOURCE_MANIFEST = SOURCE_ROOT / "test_manifest.json"
OUTPUT_ROOT = Path("results/evaluation")

KEY_COLUMNS = ["Horizon", "Index", "Model"]
PREDICTION_COLUMNS = KEY_COLUMNS + ["Date", "Actual", "Predicted"]
METRIC_NAMES = ("RMSE", "MAE", "QLIKE")
WINDOW = "common"

METRIC_COLUMNS = [
    "Horizon", "Index", "Model", "Part", "Family", "Window", "N", "Start", "End",
    "RMSE", "MAE", "QLIKE", "N_clipped",
]
LOSS_COLUMNS = PREDICTION_COLUMNS + ["SquaredError", "AbsoluteError", "QLIKE_Loss"]

# The dates every model covers at each horizon, fixed by the test run that
# Step 3 audited. A mismatch means the test calendar changed and the results
# must not be produced until that is understood.
EXPECTED_COMMON_WINDOWS = {
    "1day": {"N": 787, "Start": "2022-12-21", "End": "2025-12-29"},
    "5day": {"N": 787, "Start": "2022-12-15", "End": "2025-12-23"},
    "22day": {"N": 784, "Start": "2022-11-25", "End": "2025-11-28"},
}

# Parts follow the training wrappers in scripts/run_part{1,2,3}_*.sh, which
# split the models exactly as run_experiments groups them.
PARTS = (
    (1, TABULAR_MODELS),
    (2, DEEP_LEARNING_MODELS),
    (3, HYBRID_MODELS),
)

# Families are the section headings of src/utils/model_registry.py.
FAMILIES = {
    "HistoricalVolatility": "Benchmark",
    "Persistence": "Benchmark",
    "GARCH": "Econometric",
    "EGARCH": "Econometric",
    "GJRGARCH": "Econometric",
    "LinearRegression": "Machine Learning",
    "RandomForest": "Machine Learning",
    "XGBoost": "Machine Learning",
    "LSTM": "Deep Learning",
    "GRU": "Deep Learning",
    "Transformer": "Deep Learning",
    "GARCHLSTM": "Hybrid",
    "GARCHGRU": "Hybrid",
    "GARCHTransformer": "Hybrid",
}

# The floor qlike applies to forecasts; read from the function so N_clipped
# always counts exactly the observations the canonical metric clips.
QLIKE_EPSILON = inspect.signature(qlike).parameters["epsilon"].default


class EvaluationError(RuntimeError):
    """Raised when the test predictions fail a precondition for scoring."""


def _part(model: str) -> int:
    parts = [part for part, members in PARTS if model in members]
    if len(parts) != 1:
        raise EvaluationError(f"{model} belongs to {len(parts)} parts, expected 1.")
    return parts[0]


def _check_taxonomy() -> None:
    """The labels must cover exactly the models the project trains."""

    if set(FAMILIES) != set(MODELS) or set(MODEL_REGISTRY) != set(MODELS):
        raise EvaluationError(
            "Model taxonomy is out of step with run_experiments.MODELS and the "
            "model registry."
        )
    for model in MODELS:
        _part(model)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_path(source_root: Path, horizon: str) -> Path:
    return source_root / f"test_predictions_{horizon}.csv"


def _fail(problems: List[str]) -> None:
    if problems:
        listing = "\n".join(f"  - {problem}" for problem in problems)
        raise EvaluationError(
            f"{len(problems)} validation problem(s); nothing was written:\n{listing}"
        )


def _expected_cells() -> set:
    return set(product(HORIZONS, INDICES, MODELS))


def load_manifest(path: Path) -> dict:
    """Load the test manifest and require one clean entry per cell."""

    if not path.exists():
        raise EvaluationError(f"Missing test manifest: {path}")

    manifest = json.loads(path.read_text())
    experiments = manifest.get("experiments", [])
    keys = [(entry["horizon"], entry["index"], entry["model"]) for entry in experiments]
    expected = _expected_cells()
    problems: List[str] = []

    if len(experiments) != len(expected):
        problems.append(
            f"manifest lists {len(experiments)} cells, expected {len(expected)}"
        )

    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        problems.append(f"manifest repeats {len(duplicates)} cell(s): {duplicates}")

    missing = sorted(expected - set(keys))
    if missing:
        problems.append(f"manifest is missing {len(missing)} cell(s): {missing}")

    unexpected = sorted(set(keys) - expected)
    if unexpected:
        problems.append(f"manifest has {len(unexpected)} unknown cell(s): {unexpected}")

    flagged = [entry for entry in experiments if entry.get("problems")]
    for entry in flagged:
        problems.append(
            f"manifest flags {entry['horizon']} {entry['index']} {entry['model']}: "
            f"{entry['problems']}"
        )

    _fail(problems)
    return manifest


def load_predictions(source_root: Path, horizon: str) -> pd.DataFrame:
    """Read one horizon's test predictions and check their basic integrity."""

    path = _prediction_path(source_root, horizon)
    if not path.exists():
        raise EvaluationError(f"Missing test predictions: {path}")

    frame = pd.read_csv(path, parse_dates=["Date"])
    problems: List[str] = []

    if list(frame.columns) != PREDICTION_COLUMNS:
        raise EvaluationError(
            f"{path}: columns {list(frame.columns)}, expected {PREDICTION_COLUMNS}"
        )

    other_horizons = sorted(set(frame["Horizon"]) - {horizon})
    if other_horizons:
        problems.append(f"{path}: contains rows for {other_horizons}")

    for column in ("Actual", "Predicted"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        bad = int((~np.isfinite(values)).sum())
        if bad:
            problems.append(f"{path}: {bad} missing or non-finite {column} value(s)")

    duplicate_keys = frame.duplicated(KEY_COLUMNS + ["Date"]).sum()
    if duplicate_keys:
        problems.append(f"{path}: {duplicate_keys} duplicate Horizon/Index/Model/Date key(s)")

    cells = set(frame[KEY_COLUMNS].drop_duplicates().itertuples(index=False, name=None))
    expected = {cell for cell in _expected_cells() if cell[0] == horizon}
    if cells != expected:
        problems.append(
            f"{path}: missing cells {sorted(expected - cells)}, "
            f"unknown cells {sorted(cells - expected)}"
        )

    _fail(problems)
    return frame


def check_against_manifest(frame: pd.DataFrame, manifest: dict, horizon: str) -> None:
    """Row counts and date ranges must be the ones the test run recorded."""

    problems: List[str] = []
    grouped = frame.groupby(KEY_COLUMNS)["Date"].agg(["size", "min", "max"])

    for entry in manifest["experiments"]:
        if entry["horizon"] != horizon:
            continue
        key = (entry["horizon"], entry["index"], entry["model"])
        rows, start, end = grouped.loc[key]
        recorded = (entry["rows"], entry["test_start"], entry["test_end"])
        observed = (int(rows), start.date().isoformat(), end.date().isoformat())
        if recorded != observed:
            problems.append(f"{key}: manifest records {recorded}, file has {observed}")

    _fail(problems)


def check_actuals_agree(frame: pd.DataFrame, horizon: str) -> None:
    """Every model must be scored against the same target on a given date."""

    spread = frame.groupby(["Index", "Date"])["Actual"].nunique()
    disagreeing = spread[spread > 1]
    if len(disagreeing):
        examples = [f"{index} {date.date()}" for index, date in disagreeing.index[:5]]
        _fail([
            f"{horizon}: Actual differs between models on {len(disagreeing)} "
            f"Index/Date pair(s), e.g. {examples}"
        ])


def common_window(frame: pd.DataFrame, horizon: str) -> pd.DatetimeIndex:
    """Return the dates every model covers, identical across indices."""

    problems: List[str] = []
    windows = {}

    for index, rows in frame.groupby("Index"):
        dates_by_model = [
            set(model_rows["Date"]) for _, model_rows in rows.groupby("Model")
        ]
        windows[index] = pd.DatetimeIndex(sorted(set.intersection(*dates_by_model)))

    reference = windows[INDICES[0]]
    for index, dates in windows.items():
        if not dates.equals(reference):
            problems.append(
                f"{horizon} {index}: common window differs from {INDICES[0]} "
                f"({len(dates)} vs {len(reference)} dates)"
            )

    expected = EXPECTED_COMMON_WINDOWS[horizon]
    observed = {
        "N": len(reference),
        "Start": reference.min().date().isoformat(),
        "End": reference.max().date().isoformat(),
    }
    if observed != expected:
        problems.append(
            f"{horizon}: common window is {observed}, expected {expected}"
        )

    # Every model's own dates must contain the window as one unbroken tail, so
    # restricting to it drops only a leading stretch and never skips a date.
    for (index, model), rows in frame.groupby(["Index", "Model"]):
        own = pd.DatetimeIndex(sorted(rows["Date"]))
        if not own[len(own) - len(reference):].equals(reference):
            problems.append(
                f"{horizon} {index} {model}: common window is not the tail of "
                f"its {len(own)} test dates"
            )

    _fail(problems)
    return reference


def _per_observation(metric, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Apply a canonical metric to each observation on its own.

    Each metric is a mean over observations, so on a single observation it
    returns that observation's loss.
    """

    return np.array(
        [metric(actual[i : i + 1], predicted[i : i + 1]) for i in range(len(actual))],
        dtype=float,
    )


def evaluate_horizon(frame: pd.DataFrame, horizon: str, window: pd.DatetimeIndex):
    """Return the metric rows and per-observation losses for one horizon."""

    scored = frame[frame["Date"].isin(window)].copy()
    order = {
        "Index": {name: position for position, name in enumerate(INDICES)},
        "Model": {name: position for position, name in enumerate(MODELS)},
    }
    scored = scored.sort_values(
        ["Index", "Model", "Date"],
        key=lambda column: column.map(order[column.name]) if column.name in order else column,
    ).reset_index(drop=True)

    metric_rows = []
    loss_frames = []
    problems: List[str] = []

    for (index, model), rows in scored.groupby(["Index", "Model"], sort=False):
        actual = rows["Actual"].to_numpy(dtype=float)
        predicted = rows["Predicted"].to_numpy(dtype=float)
        dates = pd.DatetimeIndex(rows["Date"])

        if not dates.equals(window):
            problems.append(f"{horizon} {index} {model}: scored dates differ from the window")
            continue

        metrics = calculate_metrics(actual, predicted)

        losses = rows[PREDICTION_COLUMNS].copy()
        losses["SquaredError"] = _per_observation(rmse, actual, predicted) ** 2
        losses["AbsoluteError"] = _per_observation(mae, actual, predicted)
        losses["QLIKE_Loss"] = _per_observation(qlike, actual, predicted)

        # The loss series must reproduce the reported metrics.
        reproduced = {
            "RMSE": float(np.sqrt(losses["SquaredError"].mean())),
            "MAE": float(losses["AbsoluteError"].mean()),
            "QLIKE": float(losses["QLIKE_Loss"].mean()),
        }
        for name in METRIC_NAMES:
            if not np.isclose(reproduced[name], metrics[name], rtol=1e-10, atol=0.0):
                problems.append(
                    f"{horizon} {index} {model}: per-observation {name} "
                    f"{reproduced[name]!r} != metric {metrics[name]!r}"
                )

        values = [metrics[name] for name in METRIC_NAMES]
        if not np.all(np.isfinite(values)):
            problems.append(f"{horizon} {index} {model}: non-finite metric {metrics}")

        metric_rows.append({
            "Horizon": horizon,
            "Index": index,
            "Model": model,
            "Part": _part(model),
            "Family": FAMILIES[model],
            "Window": WINDOW,
            "N": len(rows),
            "Start": window.min().date().isoformat(),
            "End": window.max().date().isoformat(),
            "RMSE": float(metrics["RMSE"]),
            "MAE": float(metrics["MAE"]),
            "QLIKE": float(metrics["QLIKE"]),
            "N_clipped": int((predicted < QLIKE_EPSILON).sum()),
        })
        loss_frames.append(losses)

    _fail(problems)
    return metric_rows, pd.concat(loss_frames, ignore_index=True)[LOSS_COLUMNS]


def run(source_root: Path = SOURCE_ROOT, output_root: Path = OUTPUT_ROOT) -> dict:
    """Validate every source file, then score all cells and write the outputs.

    Every check runs before anything is written, so a failure leaves the output
    directory as it was.
    """

    _check_taxonomy()
    manifest_path = source_root / SOURCE_MANIFEST.name
    manifest = load_manifest(manifest_path)

    frames: Dict[str, pd.DataFrame] = {}
    windows: Dict[str, pd.DatetimeIndex] = {}
    for horizon in HORIZONS:
        frame = load_predictions(source_root, horizon)
        check_against_manifest(frame, manifest, horizon)
        check_actuals_agree(frame, horizon)
        windows[horizon] = common_window(frame, horizon)
        frames[horizon] = frame

    metric_rows = []
    losses: Dict[str, pd.DataFrame] = {}
    for horizon in HORIZONS:
        rows, horizon_losses = evaluate_horizon(frames[horizon], horizon, windows[horizon])
        metric_rows.extend(rows)
        losses[horizon] = horizon_losses

    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    expected_cells = len(_expected_cells())
    if len(metrics) != expected_cells or metrics.duplicated(KEY_COLUMNS).any():
        raise EvaluationError(
            f"Produced {len(metrics)} metric rows, expected {expected_cells} unique."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_root / "final_metrics.csv", index=False)
    for horizon, frame in losses.items():
        frame.to_csv(
            output_root / f"losses_{horizon}.csv", index=False, date_format="%Y-%m-%d"
        )

    source_files = [manifest_path] + [
        _prediction_path(source_root, horizon) for horizon in HORIZONS
    ]
    evaluation_manifest = {
        "evaluated": datetime.now(timezone.utc).isoformat(),
        "source_test_manifest_generated": manifest.get("generated"),
        "source_files": {str(path): _sha256(path) for path in source_files},
        "cells": len(metrics),
        "cells_per_horizon": {
            horizon: int((metrics["Horizon"] == horizon).sum()) for horizon in HORIZONS
        },
        "horizons": list(HORIZONS),
        "indices": list(INDICES),
        "models": list(MODELS),
        "window": WINDOW,
        "common_windows": {
            horizon: {
                "N": len(window),
                "Start": window.min().date().isoformat(),
                "End": window.max().date().isoformat(),
            }
            for horizon, window in windows.items()
        },
        "metrics": {
            "implementation": "src/evaluation/metrics.py::calculate_metrics",
            "names": list(METRIC_NAMES),
            "qlike_epsilon": QLIKE_EPSILON,
        },
        "loss_rows": {horizon: len(frame) for horizon, frame in losses.items()},
        "total_loss_rows": int(sum(len(frame) for frame in losses.values())),
        "cells_with_clipped_predictions": metrics.loc[
            metrics["N_clipped"] > 0, KEY_COLUMNS + ["N_clipped"]
        ].to_dict(orient="records"),
    }
    (output_root / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2) + "\n"
    )

    return evaluation_manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score the held-out test predictions on the common window."
    )
    parser.add_argument("--source-dir", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args(argv)

    try:
        summary = run(arguments.source_dir, arguments.output_dir)
    except EvaluationError as error:
        print(f"[FAIL] {error}")
        return 1

    print(f"Source test run : {summary['source_test_manifest_generated']}")
    print(f"Cells scored    : {summary['cells']} {summary['cells_per_horizon']}")
    for horizon, window in summary["common_windows"].items():
        print(
            f"  {horizon:>5}: {window['N']} dates {window['Start']}..{window['End']}, "
            f"{summary['loss_rows'][horizon]} loss rows"
        )
    print(f"Loss rows       : {summary['total_loss_rows']}")
    for cell in summary["cells_with_clipped_predictions"]:
        print(
            f"  clipped: {cell['Horizon']} {cell['Index']} {cell['Model']} "
            f"({cell['N_clipped']} forecast(s) floored by QLIKE)"
        )
    print(f"Written to      : {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
