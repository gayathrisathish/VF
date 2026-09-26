"""Statistical comparison of forecast losses on the common evaluation window.

This stage follows :mod:`src.evaluation.evaluate_predictions` and reads only
its per-observation losses and manifest:

    results/evaluation/losses_<horizon>.csv  ->  results/statistics/

Every inferential test compares models within one horizon, one index and one
loss, on the dates every model shares. Nothing is pooled across indices or
horizons.

Diebold-Mariano tests
    d_t = L_A,t - L_B,t; a positive mean means Model B has the lower loss.
    H0: E[d_t] = 0, two-sided. The long-run variance of d_t is a Bartlett
    (Newey-West) HAC estimate. The primary lag is arch's automatic Newey-West
    (1994) bandwidth, rounded up and floored at h - 1; a fixed lag of 21, the
    realised-volatility window minus one, is reported as a sensitivity. The
    reference distribution is Student-t with T - 1 degrees of freedom and no
    Harvey-Leybourne-Newbold factor, because the loss differentials remain
    autocorrelated well beyond h - 1. Models were estimated once on a fixed
    training window, so each test compares the forecasts of the fitted models
    (the Giacomini-White reading), which stays valid for nested pairs such as
    a hybrid and its plain counterpart.

    QLIKE is the primary loss and squared error the secondary one; both are
    minimised by the conditional mean of the target. Absolute error targets the
    conditional median and is reported descriptively only.

Multiple testing
    Holm, alpha = 0.05, applied separately within each family:
    benchmark (each model vs Persistence, 21 cells), ablation (hybrid vs plain,
    21 cells), hybrid vs GARCH (21 cells) and the exploratory all-pairs matrix
    (per cell). Each loss and each lag rule is its own family.

Model Confidence Set
    arch.bootstrap.MCS, method "R", stationary bootstrap, 10,000 repetitions,
    alpha = 0.10, a fixed seed per run and the largest Politis-White stationary
    block length across the run's pairwise loss differentials. Persistence and
    HistoricalVolatility forecast identically, so the set is computed on 13
    models and HistoricalVolatility is reported with Persistence's result.

QLIKE clipped forecasts
    The primary analysis uses the canonical losses unchanged. A separate
    sensitivity repeats the QLIKE all-pairs tests for each affected cell with
    the clipped dates removed for every model in that cell.

Run it with::

    python -m src.evaluation.statistical_tests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.evaluate_predictions import (
    FAMILIES,
    LOSS_COLUMNS,
    QLIKE_EPSILON,
    _part,
)
from src.experiments.run_experiments import HORIZONS, INDICES, MODELS


EVALUATION_ROOT = Path("results/evaluation")
EVALUATION_MANIFEST = "evaluation_manifest.json"
OUTPUT_ROOT = Path("results/statistics")

INFERENTIAL_LOSSES = ("QLIKE_Loss", "SquaredError")
DESCRIPTIVE_LOSSES = ("QLIKE_Loss", "SquaredError", "AbsoluteError")
LOSS_LABELS = {
    "QLIKE_Loss": "QLIKE",
    "SquaredError": "SquaredError",
    "AbsoluteError": "AbsoluteError",
}

ALPHA_DM = 0.05
ALPHA_MCS = 0.10
MCS_REPS = 10_000
MCS_METHOD = "R"
MCS_BOOTSTRAP = "stationary"
MCS_SEED = 20260925

HAC_KERNEL = "Bartlett"
REF_DIST = "t(T-1)"
CORRECTION = "holm"
# The realised-volatility window, RV_WINDOW in src/features/feature_engineering.py.
# That module runs the feature pipeline on import, so the value is not imported.
RV_WINDOW = 22
FIXED_LAG = RV_WINDOW - 1
LAG_RULES = ("auto_floor_h-1", f"fixed_{FIXED_LAG}")

BENCHMARK = "Persistence"
EQUIVALENT_BENCHMARK = "HistoricalVolatility"
CLASSICAL_GARCH = "GARCH"
ABLATION_PAIRS = (
    ("GARCHLSTM", "LSTM"),
    ("GARCHGRU", "GRU"),
    ("GARCHTransformer", "Transformer"),
)
MCS_MODELS = [model for model in MODELS if model != EQUIVALENT_BENCHMARK]

DM_COLUMNS = [
    "Analysis", "Horizon", "Index", "Loss", "Model_A", "Model_B",
    "Part_A", "Family_A", "Part_B", "Family_B",
    "N", "Start", "End", "Mean_Loss_A", "Mean_Loss_B", "Mean_Diff", "Direction",
    "DM_Stat", "Ref_Dist", "HAC_Kernel", "HAC_Lag", "HAC_Lag_Rule", "LongRun_Var",
    "P_Raw", "Correction", "Test_Family", "Family_Size", "P_Adj", "Alpha",
    "Significant", "Status",
]
SENSITIVITY_COLUMNS = DM_COLUMNS + ["Excluded_Dates"]
MCS_COLUMNS = [
    "Horizon", "Index", "Loss", "Model", "N", "Mean_Loss", "In_MCS", "MCS_PValue",
    "Elimination_Order", "Method", "Alpha", "Reps", "Block_Size", "Seed", "Note",
]
EQUIVALENCE_COLUMNS = [
    "Horizon", "Index", "Model_A", "Model_B", "N", "Max_Abs_Pred_Diff", "Identical",
]
SUMMARY_COLUMNS = [
    "Horizon", "Loss", "Model", "Cells", "Mean_Rank",
    "Median_Loss_Ratio_vs_Persistence", "MCS_Inclusions",
]

DISCLOSURES = [
    "Targets overlap: Actual on date t is the 22-day rolling realised volatility "
    "ending h days later, so consecutive targets share 21 of 22 returns at every "
    "horizon and loss differentials are strongly autocorrelated.",
    "Models were estimated once on a fixed training window; tests compare the "
    "forecasts of the fitted models, not the population models.",
    "Hybrid and plain networks were tuned separately, so the ablation compares a "
    "tuned hybrid with a tuned plain model rather than a single added feature.",
    "Each neural network was trained once with a fixed seed; training-seed "
    "variability is not reflected in any test.",
    "At the 22-day horizon the 784 overlapping observations carry roughly 36 "
    "non-overlapping months of information, so power is low and the asymptotic "
    "approximations are rough.",
    "Four LinearRegression 22-day forecasts (^FTSE 1, ^N225 2, ^HSI 1) are zero and "
    "are floored at 1e-8 by the canonical QLIKE. Each produces a per-observation "
    "QLIKE of millions that dominates the mean and the HAC variance of any "
    "differential involving that model in that cell, and destabilises the MCS "
    "bootstrap. Primary results keep these observations; "
    "dm_qlike_clipped_sensitivity.csv repeats the QLIKE tests without those dates.",
    "Median_Loss_Ratio_vs_Persistence is left empty for QLIKE: the canonical QLIKE "
    "is negative and defined only up to an additive constant, so a ratio of means "
    "has no stable interpretation. It is reported for squared and absolute error.",
    "Absolute error is minimised by the conditional median rather than the mean, "
    "so it is reported descriptively only.",
]


class StatisticsError(RuntimeError):
    """Raised when the evaluation losses fail a precondition for testing."""


def _fail(problems: List[str]) -> None:
    if problems:
        listing = "\n".join(f"  - {problem}" for problem in problems)
        raise StatisticsError(
            f"{len(problems)} validation problem(s); nothing was written:\n{listing}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _horizon_days(horizon: str) -> int:
    return int(horizon.replace("day", ""))


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def load_evaluation_manifest(root: Path) -> dict:
    path = root / EVALUATION_MANIFEST
    if not path.exists():
        raise StatisticsError(f"Missing evaluation manifest: {path}")

    manifest = json.loads(path.read_text())
    problems = []
    if manifest.get("cells") != len(HORIZONS) * len(INDICES) * len(MODELS):
        problems.append(f"evaluation manifest reports {manifest.get('cells')} cells")
    if list(manifest.get("horizons", [])) != list(HORIZONS):
        problems.append(f"evaluation manifest horizons {manifest.get('horizons')}")
    if manifest.get("window") != "common":
        problems.append(f"evaluation manifest window is {manifest.get('window')!r}")
    for horizon in HORIZONS:
        if horizon not in manifest.get("common_windows", {}):
            problems.append(f"evaluation manifest has no common window for {horizon}")
    _fail(problems)
    return manifest


def load_losses(root: Path, horizon: str, manifest: dict) -> pd.DataFrame:
    """Read one horizon's losses and require a complete, clean panel."""

    path = root / f"losses_{horizon}.csv"
    if not path.exists():
        raise StatisticsError(f"Missing loss file: {path}")

    frame = pd.read_csv(path, parse_dates=["Date"])
    if list(frame.columns) != LOSS_COLUMNS:
        raise StatisticsError(f"{path}: columns {list(frame.columns)}, expected {LOSS_COLUMNS}")

    problems: List[str] = []
    window = manifest["common_windows"][horizon]
    key = ["Horizon", "Index", "Model", "Date"]

    if set(frame["Horizon"]) != {horizon}:
        problems.append(f"{path}: Horizon values {sorted(set(frame['Horizon']))}")

    numeric = ["Actual", "Predicted"] + list(DESCRIPTIVE_LOSSES)
    values = frame[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    bad = int((~np.isfinite(values)).sum())
    if bad:
        problems.append(f"{path}: {bad} missing or non-finite numeric value(s)")

    duplicates = int(frame.duplicated(key).sum())
    if duplicates:
        problems.append(f"{path}: {duplicates} duplicate loss key(s)")

    cells = set(frame[["Index", "Model"]].drop_duplicates().itertuples(index=False, name=None))
    expected_cells = {(index, model) for index in INDICES for model in MODELS}
    if cells != expected_cells:
        problems.append(
            f"{path}: missing cells {sorted(expected_cells - cells)}, "
            f"unknown cells {sorted(cells - expected_cells)}"
        )

    dates = pd.DatetimeIndex(sorted(frame["Date"].unique()))
    observed = {
        "N": len(dates),
        "Start": dates.min().date().isoformat(),
        "End": dates.max().date().isoformat(),
    }
    if observed != window:
        problems.append(f"{path}: window {observed}, manifest records {window}")

    sizes = frame.groupby(["Index", "Model"]).size()
    short = sizes[sizes != window["N"]]
    if len(short):
        problems.append(
            f"{path}: {len(short)} cell(s) without exactly {window['N']} observations"
        )

    if len(frame) != manifest["loss_rows"][horizon]:
        problems.append(
            f"{path}: {len(frame)} rows, manifest records {manifest['loss_rows'][horizon]}"
        )

    spread = frame.groupby(["Index", "Date"])["Actual"].nunique()
    if (spread > 1).any():
        problems.append(
            f"{path}: Actual differs between models on {int((spread > 1).sum())} Index/Date pair(s)"
        )

    _fail(problems)
    return frame


def wide(frame: pd.DataFrame, index: str, column: str) -> pd.DataFrame:
    """Date x Model matrix of one column for one index, in MODELS order."""

    table = frame[frame["Index"] == index].pivot(index="Date", columns="Model", values=column)
    return table.sort_index()[list(MODELS)]


def check_equivalence(frame: pd.DataFrame, horizon: str) -> List[dict]:
    """Record every pair of models with identical forecasts in each cell.

    Persistence and HistoricalVolatility must be identical everywhere; any
    other identical pair is recorded but does not stop the run.
    """

    rows = []
    problems = []

    for index in INDICES:
        predictions = wide(frame, index, "Predicted")
        for model_a, model_b in combinations(MODELS, 2):
            difference = float(np.max(np.abs(predictions[model_a] - predictions[model_b])))
            required = {model_a, model_b} == {BENCHMARK, EQUIVALENT_BENCHMARK}
            if difference == 0.0 or required:
                rows.append({
                    "Horizon": horizon,
                    "Index": index,
                    "Model_A": model_a,
                    "Model_B": model_b,
                    "N": len(predictions),
                    "Max_Abs_Pred_Diff": difference,
                    "Identical": difference == 0.0,
                })
            if required and difference != 0.0:
                problems.append(
                    f"{horizon} {index}: {BENCHMARK} and {EQUIVALENT_BENCHMARK} differ "
                    f"(max {difference})"
                )

    _fail(problems)
    return rows


def clipped_dates(frame: pd.DataFrame) -> Dict[str, List[pd.Timestamp]]:
    """Dates in each index whose forecast the canonical QLIKE floors."""

    clipped = frame[frame["Predicted"] < QLIKE_EPSILON]
    return {
        index: sorted(pd.DatetimeIndex(rows["Date"].unique()))
        for index, rows in clipped.groupby("Index")
    }


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------

def diebold_mariano(differential: np.ndarray, horizon: str, rule: str) -> dict:
    """Two-sided DM test of E[d] = 0 with a Bartlett HAC long-run variance."""

    from arch.covariance.kernel import Bartlett

    T = len(differential)
    mean = float(differential.mean())

    if np.all(differential == 0.0):
        return {"Status": "identical_forecasts", "Mean_Diff": mean}

    if rule == LAG_RULES[0]:
        automatic = Bartlett(differential, force_int=True).bandwidth
        lag = int(max(automatic, _horizon_days(horizon) - 1))
    else:
        lag = FIXED_LAG

    long_run = float(np.asarray(Bartlett(differential, bandwidth=lag).cov.long_run).squeeze())
    if not np.isfinite(long_run) or long_run <= 0.0:
        return {"Status": "zero_variance", "Mean_Diff": mean, "HAC_Lag": lag,
                "LongRun_Var": long_run}

    statistic = mean / math.sqrt(long_run / T)
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=T - 1))

    return {
        "Status": "tested",
        "Mean_Diff": mean,
        "DM_Stat": statistic,
        "HAC_Lag": lag,
        "LongRun_Var": long_run,
        "P_Raw": p_value,
    }


def _direction(mean_difference: float) -> str:
    if mean_difference < 0:
        return "A_lower"
    if mean_difference > 0:
        return "B_lower"
    return "equal"


def dm_row(analysis, horizon, index, loss, model_a, model_b, losses, rule) -> dict:
    loss_a = losses[model_a].to_numpy(dtype=float)
    loss_b = losses[model_b].to_numpy(dtype=float)
    result = diebold_mariano(loss_a - loss_b, horizon, rule)

    return {
        "Analysis": analysis,
        "Horizon": horizon,
        "Index": index,
        "Loss": LOSS_LABELS[loss],
        "Model_A": model_a,
        "Model_B": model_b,
        "Part_A": _part(model_a),
        "Family_A": FAMILIES[model_a],
        "Part_B": _part(model_b),
        "Family_B": FAMILIES[model_b],
        "N": len(loss_a),
        "Start": losses.index.min().date().isoformat(),
        "End": losses.index.max().date().isoformat(),
        "Mean_Loss_A": float(loss_a.mean()),
        "Mean_Loss_B": float(loss_b.mean()),
        "Mean_Diff": result["Mean_Diff"],
        "Direction": _direction(result["Mean_Diff"]),
        "DM_Stat": result.get("DM_Stat", np.nan),
        "Ref_Dist": REF_DIST,
        "HAC_Kernel": HAC_KERNEL,
        "HAC_Lag": result.get("HAC_Lag", np.nan),
        "HAC_Lag_Rule": rule,
        "LongRun_Var": result.get("LongRun_Var", np.nan),
        "P_Raw": result.get("P_Raw", np.nan),
        "Correction": CORRECTION,
        "Test_Family": "",
        "Family_Size": np.nan,
        "P_Adj": np.nan,
        "Alpha": ALPHA_DM,
        "Significant": np.nan,
        "Status": result["Status"],
    }


def apply_holm(rows: pd.DataFrame, family_columns: Sequence[str], family_name) -> pd.DataFrame:
    """Holm-adjust the tested rows within each family; others stay empty."""

    from statsmodels.stats.multitest import multipletests

    rows = rows.copy()
    rows["Significant"] = rows["Significant"].astype(object)

    for key, members in rows.groupby(list(family_columns), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        name = family_name(dict(zip(family_columns, key)))
        rows.loc[members.index, "Test_Family"] = name

        tested = members[members["Status"] == "tested"]
        rows.loc[members.index, "Family_Size"] = len(tested)
        if tested.empty:
            continue

        reject, adjusted, _, _ = multipletests(
            tested["P_Raw"].to_numpy(dtype=float), alpha=ALPHA_DM, method=CORRECTION
        )
        rows.loc[tested.index, "P_Adj"] = adjusted
        rows.loc[tested.index, "Significant"] = reject.astype(bool)

    rows["Family_Size"] = rows["Family_Size"].astype(int)
    return rows


def comparison_pairs(analysis: str):
    if analysis == "benchmark":
        return [
            (model, BENCHMARK)
            for model in MODELS
            if model not in (BENCHMARK, EQUIVALENT_BENCHMARK)
        ]
    if analysis == "ablation":
        return list(ABLATION_PAIRS)
    if analysis == "hybrid_vs_garch":
        return [(hybrid, CLASSICAL_GARCH) for hybrid, _ in ABLATION_PAIRS]
    if analysis == "pairwise":
        return list(combinations(MODELS, 2))
    raise ValueError(analysis)


EXPECTED_PAIRS_PER_CELL = {
    "benchmark": len(MODELS) - 2,
    "ablation": 3,
    "hybrid_vs_garch": 3,
    "pairwise": len(MODELS) * (len(MODELS) - 1) // 2,
}

# Confirmatory families span the whole grid for one loss and lag rule; the
# exploratory matrix is corrected within each cell.
FAMILY_COLUMNS = {
    "benchmark": ["Loss", "HAC_Lag_Rule"],
    "ablation": ["Loss", "HAC_Lag_Rule"],
    "hybrid_vs_garch": ["Loss", "HAC_Lag_Rule"],
    "pairwise": ["Horizon", "Index", "Loss", "HAC_Lag_Rule"],
}


def _family_name(analysis: str, columns: Sequence[str]):
    def name(key: dict) -> str:
        return "|".join([analysis] + [str(key[column]) for column in columns])
    return name


def run_dm(frames: Dict[str, pd.DataFrame], analysis: str) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        for index in INDICES:
            for loss in INFERENTIAL_LOSSES:
                losses = wide(frames[horizon], index, loss)
                for rule in LAG_RULES:
                    for model_a, model_b in comparison_pairs(analysis):
                        rows.append(dm_row(
                            analysis, horizon, index, loss, model_a, model_b, losses, rule
                        ))

    table = pd.DataFrame(rows, columns=DM_COLUMNS)
    return apply_holm(
        table, FAMILY_COLUMNS[analysis], _family_name(analysis, FAMILY_COLUMNS[analysis])
    )


def run_clipped_sensitivity(frames, clipped) -> pd.DataFrame:
    """QLIKE all-pairs tests without the clipped dates, per affected cell."""

    rows = []
    for horizon in HORIZONS:
        for index in INDICES:
            dates = clipped[horizon].get(index)
            if not dates:
                continue
            losses = wide(frames[horizon], index, "QLIKE_Loss")
            losses = losses.drop(index=pd.DatetimeIndex(dates))
            excluded = ";".join(date.date().isoformat() for date in dates)
            for rule in LAG_RULES:
                for model_a, model_b in comparison_pairs("pairwise"):
                    row = dm_row(
                        "pairwise_qlike_clipped_sensitivity", horizon, index,
                        "QLIKE_Loss", model_a, model_b, losses, rule,
                    )
                    row["Excluded_Dates"] = excluded
                    rows.append(row)

    table = pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)
    return apply_holm(
        table,
        FAMILY_COLUMNS["pairwise"],
        _family_name("pairwise_qlike_clipped_sensitivity", FAMILY_COLUMNS["pairwise"]),
    )


# ---------------------------------------------------------------------------
# Model Confidence Set
# ---------------------------------------------------------------------------

def block_length(losses: pd.DataFrame) -> int:
    """Largest Politis-White stationary block length over pairwise differentials."""

    from arch.bootstrap import optimal_block_length

    differentials = pd.DataFrame({
        f"{a}-{b}": losses[a] - losses[b] for a, b in combinations(losses.columns, 2)
    })
    lengths = optimal_block_length(differentials)["stationary"]
    return int(math.ceil(float(lengths.max())))


def run_mcs(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    from arch.bootstrap import MCS

    rows = []
    for horizon in HORIZONS:
        for index in INDICES:
            for loss in INFERENTIAL_LOSSES:
                losses = wide(frames[horizon], index, loss)[MCS_MODELS]
                block = block_length(losses)
                procedure = MCS(
                    losses,
                    size=ALPHA_MCS,
                    reps=MCS_REPS,
                    block_size=block,
                    method=MCS_METHOD,
                    bootstrap=MCS_BOOTSTRAP,
                    seed=np.random.default_rng(MCS_SEED),
                )
                procedure.compute()

                pvalues = procedure.pvalues["Pvalue"]
                included = set(procedure.included)
                order = {model: position + 1 for position, model in enumerate(pvalues.index)}

                for model in MODELS:
                    source = BENCHMARK if model == EQUIVALENT_BENCHMARK else model
                    rows.append({
                        "Horizon": horizon,
                        "Index": index,
                        "Loss": LOSS_LABELS[loss],
                        "Model": model,
                        "N": len(losses),
                        "Mean_Loss": float(losses[source].mean()),
                        "In_MCS": source in included,
                        "MCS_PValue": float(pvalues[source]),
                        "Elimination_Order": order[source],
                        "Method": MCS_METHOD,
                        "Alpha": ALPHA_MCS,
                        "Reps": MCS_REPS,
                        "Block_Size": block,
                        "Seed": MCS_SEED,
                        "Note": (
                            f"identical forecasts to {BENCHMARK}; result is {BENCHMARK}'s"
                            if model == EQUIVALENT_BENCHMARK else ""
                        ),
                    })

    return pd.DataFrame(rows, columns=MCS_COLUMNS)


# ---------------------------------------------------------------------------
# Descriptive cross-index summary
# ---------------------------------------------------------------------------

def descriptive_summary(frames, mcs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        for loss in DESCRIPTIVE_LOSSES:
            means = pd.DataFrame({
                index: wide(frames[horizon], index, loss).mean() for index in INDICES
            }).T[list(MODELS)]
            ranks = means.rank(axis=1, method="average")
            ratios = means.div(means[BENCHMARK], axis=0)

            for model in MODELS:
                if loss == "QLIKE_Loss":
                    ratio = np.nan
                else:
                    ratio = float(ratios[model].median())

                if loss in INFERENTIAL_LOSSES:
                    members = mcs[
                        (mcs["Horizon"] == horizon)
                        & (mcs["Loss"] == LOSS_LABELS[loss])
                        & (mcs["Model"] == model)
                    ]
                    inclusions = int(members["In_MCS"].sum())
                else:
                    inclusions = np.nan

                rows.append({
                    "Horizon": horizon,
                    "Loss": LOSS_LABELS[loss],
                    "Model": model,
                    "Cells": len(means),
                    "Mean_Rank": float(ranks[model].mean()),
                    "Median_Loss_Ratio_vs_Persistence": ratio,
                    "MCS_Inclusions": inclusions,
                })

    table = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    table["MCS_Inclusions"] = table["MCS_Inclusions"].astype("Int64")
    return table


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _check_counts(tables: Dict[str, pd.DataFrame]) -> None:
    cells = len(HORIZONS) * len(INDICES)
    per_cell_rows = len(INFERENTIAL_LOSSES) * len(LAG_RULES)
    problems = []

    for analysis, table in tables.items():
        expected = EXPECTED_PAIRS_PER_CELL[analysis] * cells * per_cell_rows
        if len(table) != expected:
            problems.append(f"{analysis}: {len(table)} rows, expected {expected}")
        key = ["Horizon", "Index", "Loss", "Model_A", "Model_B", "HAC_Lag_Rule"]
        if table.duplicated(key).any():
            problems.append(f"{analysis}: duplicate comparison keys")

        not_tested = table[table["Status"] != "tested"]
        allowed = not_tested.apply(
            lambda row: {row["Model_A"], row["Model_B"]} == {BENCHMARK, EQUIVALENT_BENCHMARK}
            and row["Status"] == "identical_forecasts",
            axis=1,
        )
        if len(not_tested) and not allowed.all():
            problems.append(
                f"{analysis}: {int((~allowed).sum())} comparison(s) could not be tested"
            )

    if tables["ablation"].groupby(["Loss", "HAC_Lag_Rule"]).size().ne(63).any():
        problems.append("ablation: families are not 63 comparisons each")

    _fail(problems)


def run(evaluation_root: Path = EVALUATION_ROOT, output_root: Path = OUTPUT_ROOT) -> dict:
    manifest = load_evaluation_manifest(evaluation_root)
    frames = {horizon: load_losses(evaluation_root, horizon, manifest) for horizon in HORIZONS}

    equivalences = pd.DataFrame(
        [row for horizon in HORIZONS for row in check_equivalence(frames[horizon], horizon)],
        columns=EQUIVALENCE_COLUMNS,
    )

    clipped = {horizon: clipped_dates(frames[horizon]) for horizon in HORIZONS}
    recorded = {
        (cell["Horizon"], cell["Index"]): cell["N_clipped"]
        for cell in manifest.get("cells_with_clipped_predictions", [])
    }
    found = {
        (horizon, index): len(dates)
        for horizon, by_index in clipped.items()
        for index, dates in by_index.items()
    }
    if found != recorded:
        _fail([f"clipped forecasts {found} disagree with the evaluation manifest {recorded}"])

    dm_tables = {
        analysis: run_dm(frames, analysis)
        for analysis in ("benchmark", "ablation", "hybrid_vs_garch", "pairwise")
    }
    _check_counts(dm_tables)

    sensitivity = run_clipped_sensitivity(frames, clipped)
    mcs = run_mcs(frames)
    summary = descriptive_summary(frames, mcs)

    output_root.mkdir(parents=True, exist_ok=True)
    for analysis, table in dm_tables.items():
        table.to_csv(output_root / f"dm_{analysis}.csv", index=False)
    sensitivity.to_csv(output_root / "dm_qlike_clipped_sensitivity.csv", index=False)
    mcs.to_csv(output_root / "mcs_results.csv", index=False)
    equivalences.to_csv(output_root / "equivalences.csv", index=False)
    summary.to_csv(output_root / "descriptive_summary.csv", index=False)

    import arch
    import scipy
    import statsmodels

    source_files = [evaluation_root / EVALUATION_MANIFEST] + [
        evaluation_root / f"losses_{horizon}.csv" for horizon in HORIZONS
    ]
    all_dm = pd.concat(dm_tables.values(), ignore_index=True)
    statistics_manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_evaluation_manifest_evaluated": manifest.get("evaluated"),
        "source_test_manifest_generated": manifest.get("source_test_manifest_generated"),
        "source_files": {str(path): _sha256(path) for path in source_files},
        "horizons": list(HORIZONS),
        "indices": list(INDICES),
        "models": list(MODELS),
        "common_windows": manifest["common_windows"],
        "diebold_mariano": {
            "differential": "L_A - L_B (positive mean: Model B has lower loss)",
            "null": "E[d] = 0",
            "alternative": "two-sided",
            "losses": [LOSS_LABELS[loss] for loss in INFERENTIAL_LOSSES],
            "hac_kernel": HAC_KERNEL,
            "lag_rules": {
                LAG_RULES[0]: "arch automatic Newey-West (1994) bandwidth, rounded up, "
                              "floored at h - 1",
                LAG_RULES[1]: f"fixed lag {FIXED_LAG} (RV_WINDOW - 1)",
            },
            "reference_distribution": REF_DIST,
            "hln_correction": False,
            "alpha": ALPHA_DM,
        },
        "multiple_testing": {
            "correction": CORRECTION,
            "family_definitions": {
                analysis: FAMILY_COLUMNS[analysis] for analysis in FAMILY_COLUMNS
            },
            "family_sizes": {
                analysis: sorted({int(size) for size in table["Family_Size"]})
                for analysis, table in dm_tables.items()
            },
        },
        "model_confidence_set": {
            "implementation": "arch.bootstrap.MCS",
            "models": MCS_MODELS,
            "represented": {EQUIVALENT_BENCHMARK: BENCHMARK},
            "losses": [LOSS_LABELS[loss] for loss in INFERENTIAL_LOSSES],
            "method": MCS_METHOD,
            "bootstrap": MCS_BOOTSTRAP,
            "reps": MCS_REPS,
            "alpha": ALPHA_MCS,
            "seed": MCS_SEED,
            "block_size_rule": "ceil(max Politis-White stationary block length over "
                               "pairwise loss differentials)",
            "runs": int(len(mcs) / len(MODELS)),
        },
        "qlike_clipped_sensitivity": {
            "cells": [
                {"Horizon": horizon, "Index": index,
                 "Excluded_Dates": [date.date().isoformat() for date in dates]}
                for horizon, by_index in clipped.items()
                for index, dates in sorted(by_index.items())
            ],
            "rows": len(sensitivity),
            "tests": int((sensitivity["Status"] == "tested").sum()),
        },
        "counts": {
            "dm_rows": {analysis: len(table) for analysis, table in dm_tables.items()},
            "dm_tests": int((all_dm["Status"] == "tested").sum()),
            "dm_not_tested": all_dm.loc[all_dm["Status"] != "tested", "Status"]
                                   .value_counts().to_dict(),
            "mcs_rows": len(mcs),
            "equivalence_rows": len(equivalences),
            "summary_rows": len(summary),
        },
        "library_versions": {
            "arch": arch.__version__,
            "statsmodels": statsmodels.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "disclosures": DISCLOSURES,
    }
    (output_root / "statistics_manifest.json").write_text(
        json.dumps(statistics_manifest, indent=2, default=str) + "\n"
    )

    return statistics_manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diebold-Mariano and Model Confidence Set analysis of the evaluation losses."
    )
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args(argv)

    try:
        summary = run(arguments.evaluation_dir, arguments.output_dir)
    except StatisticsError as error:
        print(f"[FAIL] {error}")
        return 1

    counts = summary["counts"]
    print(f"Source evaluation : {summary['source_evaluation_manifest_evaluated']}")
    print(f"DM rows           : {counts['dm_rows']}")
    print(f"DM tests run      : {counts['dm_tests']}")
    print(f"Not tested        : {counts['dm_not_tested']}")
    print(f"MCS runs          : {summary['model_confidence_set']['runs']}")
    print(f"Sensitivity tests : {summary['qlike_clipped_sensitivity']['tests']}")
    print(f"Written to        : {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
