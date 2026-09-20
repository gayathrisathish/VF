"""Rebuild the plain sequence datasets consumed by the Part 2 models.

Counterpart to :mod:`src.features.build_garch_sequence_datasets`, covering the
stage that had no committed driver:

    data/splits/<horizon>                          shared calendar
      -> data/sequence_datasets/<horizon>/         30-step windows
        -> Part 2 training

``data/splits/<horizon>`` is the only source of split dates. This module adds
no calendar of its own: it windows exactly the rows the shared split contains,
one split at a time, so a window can never span a split boundary. The output
directory layout is the one ``load_sequence_split`` already expects, so nothing
downstream changes.

``create_sequence_dataset`` remains the library that does the windowing; this
module only drives it per horizon and verifies the result against the shared
split, so drift between the two cannot go unnoticed again.

Run it with::

    python -m src.features.build_plain_sequence_datasets --horizons 5day 22day
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.features.create_sequence_dataset import (
    DEFAULT_SEQUENCE_LENGTH,
    create_sequence_dataset,
    load_split_csvs,
    save_sequence_dataset,
)
from src.utils.data_loader import load_split


HORIZONS = ("1day", "5day", "22day")
SHARED_SPLIT_ROOT = Path("data/splits")
SEQUENCE_ROOT = Path("data/sequence_datasets")
DATE_COLUMN = "Date"
SPLIT_NAMES = ("train", "validation", "test")


def _shared_split_dates(horizon: str) -> Dict[str, pd.DatetimeIndex]:
    """Return the authoritative calendar for one horizon, split by name."""

    split = load_split(horizon)

    return {
        name: pd.DatetimeIndex(pd.to_datetime(split[f"y_{name}"].index))
        for name in SPLIT_NAMES
    }


def build_horizon(
    horizon: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    sequence_root: Path = SEQUENCE_ROOT,
    shared_split_root: Path = SHARED_SPLIT_ROOT,
) -> Path:
    """Rebuild one horizon's sequence dataset from its shared split."""

    split_directory = shared_split_root / horizon
    datasets = load_split_csvs(str(split_directory))

    result = create_sequence_dataset(
        datasets["X_train"],
        datasets["X_validation"],
        datasets["X_test"],
        datasets["y_train"],
        datasets["y_validation"],
        datasets["y_test"],
        sequence_length=sequence_length,
    )

    save_sequence_dataset(
        result,
        output_root=str(sequence_root),
        split_name=horizon,
    )

    return sequence_root / horizon


def verify_horizon(
    horizon: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    sequence_root: Path = SEQUENCE_ROOT,
) -> List[str]:
    """Check one horizon's sequence dataset against the shared split.

    Returns a list of problems; an empty list means the dates, counts, tensor
    shapes and split boundaries all agree with ``data/splits/<horizon>``.
    """

    problems: List[str] = []
    shared = _shared_split_dates(horizon)
    feature_count = len(load_split(horizon)["X_train"].columns)
    directory = sequence_root / horizon

    if not directory.exists():
        return [f"{horizon}: missing sequence directory {directory}"]

    boundaries = []

    for name in SPLIT_NAMES:
        expected_dates = shared[name][sequence_length - 1 :]
        targets_path = directory / f"y_{name}_sequence.csv"
        features_path = directory / f"X_{name}_sequence.npy"

        if not targets_path.exists() or not features_path.exists():
            problems.append(f"{horizon}: missing {name} sequence files")
            continue

        targets = pd.read_csv(targets_path, parse_dates=[DATE_COLUMN])
        dates = pd.DatetimeIndex(targets[DATE_COLUMN])
        boundaries.append((name, dates))

        if not dates.equals(expected_dates):
            problems.append(
                f"{horizon}: {name} dates differ from data/splits/{horizon} "
                f"({len(dates)} rows {dates[0].date()}..{dates[-1].date()} vs "
                f"{len(expected_dates)} rows {expected_dates[0].date()}.."
                f"{expected_dates[-1].date()})"
            )

        features = np.load(features_path, mmap_mode="r")
        expected_shape = (len(expected_dates), sequence_length, feature_count)
        if tuple(features.shape) != expected_shape:
            problems.append(
                f"{horizon}: {name} tensor is {tuple(features.shape)}, "
                f"expected {expected_shape}"
            )

    for (earlier, earlier_dates), (later, later_dates) in zip(
        boundaries, boundaries[1:]
    ):
        if earlier_dates.max() >= later_dates.min():
            problems.append(
                f"{horizon}: {earlier} sequences overlap {later} "
                f"({earlier_dates.max().date()} >= {later_dates.min().date()})"
            )

    for name, dates in boundaries:
        outside = dates.difference(shared[name])
        if len(outside):
            problems.append(
                f"{horizon}: {len(outside)} {name} sequence dates fall outside "
                f"that split in data/splits/{horizon}"
            )

    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild or verify the plain sequence datasets from the shared "
            "split calendar."
        )
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=list(HORIZONS),
        choices=list(HORIZONS),
        help="Horizons to process (default: all).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check the files already on disk without regenerating them.",
    )
    arguments = parser.parse_args(argv)

    all_problems: List[str] = []

    for horizon in arguments.horizons:
        if not arguments.verify_only:
            print(f"Rebuilding {horizon}...")
            print(f"  -> {build_horizon(horizon)}")

        problems = verify_horizon(horizon)
        all_problems.extend(problems)

        if problems:
            print(f"[FAIL] {horizon}: {len(problems)} problem(s)")
            for problem in problems:
                print(f"   {problem}")
        else:
            counts = {
                name: len(
                    pd.read_csv(
                        SEQUENCE_ROOT / horizon / f"y_{name}_sequence.csv",
                        usecols=[DATE_COLUMN],
                    )
                )
                for name in SPLIT_NAMES
            }
            print(
                f"[OK]   {horizon}: matches data/splits/{horizon} "
                f"(train {counts['train']}, validation {counts['validation']}, "
                f"test {counts['test']})"
            )

    if all_problems:
        print(f"\n{len(all_problems)} problem(s) found.")
        return 1

    print("\nAll requested horizons match the shared split calendar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
