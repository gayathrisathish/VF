"""Rebuild the GARCH-augmented datasets consumed by the Part 3 hybrid models.

This module owns the pipeline stage that previously existed only as ad-hoc
commands, which is how the Part 3 data drifted away from the calendar used by
Parts 1 and 2:

    data/splits/<horizon>                                    shared calendar
      -> data/garch_splits/<horizon>/<index>/                GARCH_sigma appended
        -> data/garch_augmented_sequence_datasets/<horizon>/<slug>/   30-step windows
          -> Part 3 training

``data/splits/<horizon>`` is the only source of split dates. Nothing here
defines a calendar of its own: the GARCH feature builder reads the shared split
through :func:`src.utils.data_loader.load_split`, and the sequence stage windows
whatever that produced. A horizon's boundaries therefore always match the ones
Parts 1 and 2 train on, and :func:`verify_horizon` fails loudly when the files on
disk no longer agree with the shared split.

The GARCH specification, the target definition and the 30-step window policy are
unchanged; this module only decides which rows those calculations run on.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.features.create_garch_features import (
    DEFAULT_OUTPUT_ROOT as GARCH_SPLIT_ROOT,
    FEATURE_NAME,
    build_all_garch_feature_datasets,
)
from src.features.create_sequence_dataset import (
    DEFAULT_SEQUENCE_LENGTH,
    create_sequence_dataset,
    save_sequence_dataset,
)
from src.utils.data_loader import load_split


HORIZONS = ("1day", "5day", "22day")
SEQUENCE_ROOT = Path("data/garch_augmented_sequence_datasets")
SHARED_SPLIT_ROOT = Path("data/splits")
DATE_COLUMN = "Date"
SPLIT_NAMES = ("train", "validation", "test")


def training_slug(index: str) -> str:
    """Return the directory name Part 3 training looks for.

    Must stay identical to ``_slugify_index`` in
    ``src.experiments.run_experiments``: alphanumeric characters only, so
    ``^GSPC`` becomes ``GSPC`` and ``000001.SS`` becomes ``000001SS``. The
    intermediate ``data/garch_splits`` directories keep their own slug, which is
    only read by this module.
    """

    return "".join(character for character in index if character.isalnum())


def _with_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose dates live in a ``Date`` column, not the index."""

    if DATE_COLUMN in frame.columns:
        return frame.copy()

    reset = frame.reset_index()
    reset = reset.rename(columns={reset.columns[0]: DATE_COLUMN})
    reset[DATE_COLUMN] = pd.to_datetime(reset[DATE_COLUMN])
    return reset


def _shared_split_dates(horizon: str) -> Dict[str, pd.DatetimeIndex]:
    """Return the shared calendar for one horizon, split by name."""

    split = load_split(horizon)
    return {
        name: pd.DatetimeIndex(pd.to_datetime(split[f"y_{name}"].index))
        for name in SPLIT_NAMES
    }


def build_horizon(
    horizon: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    sequence_root: Path = SEQUENCE_ROOT,
    garch_split_root: Path = GARCH_SPLIT_ROOT,
) -> Dict[str, Path]:
    """Rebuild every GARCH split and sequence dataset for one horizon.

    Returns a mapping of index name to the sequence directory that was written.
    """

    results = build_all_garch_feature_datasets(
        horizon=horizon,
        overwrite=True,
        output_root=garch_split_root,
    )

    written: Dict[str, Path] = {}

    for index, result in results.items():
        sequence_result = create_sequence_dataset(
            _with_date_column(result.X_train),
            _with_date_column(result.X_validation),
            _with_date_column(result.X_test),
            _with_date_column(result.y_train),
            _with_date_column(result.y_validation),
            _with_date_column(result.y_test),
            sequence_length=sequence_length,
        )

        slug = training_slug(index)
        save_sequence_dataset(
            sequence_result,
            output_root=str(sequence_root / horizon),
            split_name=slug,
        )
        written[index] = sequence_root / horizon / slug

    return written


def verify_horizon(
    horizon: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    sequence_root: Path = SEQUENCE_ROOT,
    garch_split_root: Path = GARCH_SPLIT_ROOT,
) -> List[str]:
    """Check the generated data for one horizon against the shared split.

    Returns a list of human-readable problems; an empty list means the GARCH
    splits and sequence datasets on disk match the shared calendar exactly.
    """

    problems: List[str] = []
    shared = _shared_split_dates(horizon)
    shared_features = load_split(horizon)["X_train"]

    for garch_directory in sorted((garch_split_root / horizon).glob("*")):
        if not garch_directory.is_dir():
            continue

        for name in SPLIT_NAMES:
            for prefix in ("X", "y"):
                path = garch_directory / f"{prefix}_{name}.csv"
                if not path.exists():
                    problems.append(f"{horizon} {garch_directory.name}: missing {path}")
                    continue

                frame = pd.read_csv(path, index_col=0, parse_dates=[0])
                dates = pd.DatetimeIndex(frame.index)
                if not dates.equals(shared[name]):
                    problems.append(
                        f"{horizon} {garch_directory.name}: {prefix}_{name} dates differ "
                        f"from data/splits/{horizon} "
                        f"({len(dates)} rows {dates[0].date()}..{dates[-1].date()} vs "
                        f"{len(shared[name])} rows {shared[name][0].date()}.."
                        f"{shared[name][-1].date()})"
                    )

                if prefix == "X":
                    extra = [
                        column
                        for column in frame.columns
                        if column not in shared_features.columns
                    ]
                    if extra != [FEATURE_NAME]:
                        problems.append(
                            f"{horizon} {garch_directory.name}: X_{name} should add only "
                            f"{FEATURE_NAME}, found {extra}"
                        )

    for index_column in shared_features.columns:
        if not index_column.endswith("_return"):
            continue
        index = index_column[: -len("_return")]
        directory = sequence_root / horizon / training_slug(index)

        if not directory.exists():
            problems.append(f"{horizon} {index}: missing sequence directory {directory}")
            continue

        for name in SPLIT_NAMES:
            expected_dates = shared[name][sequence_length - 1 :]
            targets_path = directory / f"y_{name}_sequence.csv"
            features_path = directory / f"X_{name}_sequence.npy"

            if not targets_path.exists() or not features_path.exists():
                problems.append(f"{horizon} {index}: missing {name} sequence files")
                continue

            targets = pd.read_csv(targets_path, parse_dates=[DATE_COLUMN])
            dates = pd.DatetimeIndex(targets[DATE_COLUMN])
            if not dates.equals(expected_dates):
                problems.append(
                    f"{horizon} {index}: {name} sequence dates differ from the shared "
                    f"split ({len(dates)} rows {dates[0].date()}..{dates[-1].date()} vs "
                    f"{len(expected_dates)} rows {expected_dates[0].date()}.."
                    f"{expected_dates[-1].date()})"
                )

            features = np.load(features_path, mmap_mode="r")
            expected_features = len(shared_features.columns) + 1
            expected_shape = (len(expected_dates), sequence_length, expected_features)
            if tuple(features.shape) != expected_shape:
                problems.append(
                    f"{horizon} {index}: {name} sequence tensor is {tuple(features.shape)}, "
                    f"expected {expected_shape}"
                )

        boundaries = []
        for name in SPLIT_NAMES:
            targets_path = directory / f"y_{name}_sequence.csv"
            if targets_path.exists():
                dates = pd.to_datetime(
                    pd.read_csv(targets_path, usecols=[DATE_COLUMN])[DATE_COLUMN]
                )
                boundaries.append((name, dates.min(), dates.max()))

        for (earlier, _, earlier_last), (later, later_first, _) in zip(
            boundaries, boundaries[1:]
        ):
            if earlier_last >= later_first:
                problems.append(
                    f"{horizon} {index}: {earlier} sequences overlap {later} "
                    f"({earlier_last.date()} >= {later_first.date()})"
                )

    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild or verify the GARCH-augmented Part 3 datasets from the "
            "shared split calendar."
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
            written = build_horizon(horizon)
            for index, directory in sorted(written.items()):
                print(f"  {index:>10} -> {directory}")

        problems = verify_horizon(horizon)
        all_problems.extend(problems)

        if problems:
            print(f"[FAIL] {horizon}: {len(problems)} problem(s)")
            for problem in problems:
                print(f"   {problem}")
        else:
            print(f"[OK]   {horizon}: matches data/splits/{horizon}")

    if all_problems:
        print(f"\n{len(all_problems)} problem(s) found.")
        return 1

    print("\nAll horizons match the shared split calendar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
