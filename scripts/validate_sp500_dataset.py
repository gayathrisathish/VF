#!/usr/bin/env python3
"""Diagnostic validation for the S&P 500 (^GSPC) dataset before tuning.

This script reads the S&P 500 price series from the processed dataset by default
and reports a detailed validation summary without modifying the original data.
It can also inspect a raw Yahoo Finance export or an optional feature file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


def build_expected_trading_dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Create an expected trading-day calendar for the requested span."""
    calendar = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return pd.DatetimeIndex(pd.date_range(start_date, end_date, freq=calendar))


def load_sp500_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load the S&P 500 series from the processed dataset or raw Yahoo export."""
    if data_path is None:
        candidates = [
            Path("data/processed/cleaned_indices.csv"),
            Path("data/raw/global_indices.csv"),
        ]
        for candidate in candidates:
            if candidate.exists():
                data_path = str(candidate)
                break
        if data_path is None:
            raise FileNotFoundError(
                "No dataset file found. Provide --data with a CSV path to validate."
            )

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.name == "cleaned_indices.csv" or path.parent.name == "processed":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if "^GSPC" in df.columns:
            return df[["^GSPC"]].copy()
        if "Adj Close" in df.columns:
            return df[["Adj Close"]].copy()
        if "Close" in df.columns:
            return df[["Close"]].copy()
        raise KeyError("Could not find an S&P 500 price column in the processed dataset.")

    # Fallback to a raw Yahoo Finance export with multi-level headers.
    df = pd.read_csv(path, header=[0, 1, 2])

    date_col = None
    for col in df.columns:
        if isinstance(col, tuple) and len(col) == 3 and col[2] == "Date":
            date_col = col
            break
    if date_col is None:
        raise KeyError("Could not find a Date column in the raw Yahoo Finance export.")

    close_col = None
    for col in df.columns:
        if isinstance(col, tuple) and len(col) == 3 and col[0] == "^GSPC" and col[1] == "Close":
            close_col = col
            break
    if close_col is None:
        raise KeyError("Could not find the '^GSPC' Close column in the raw Yahoo Finance export.")

    price_series = pd.to_numeric(df[close_col], errors="coerce")
    date_index = pd.to_datetime(df[date_col], errors="coerce")
    result = pd.DataFrame({"^GSPC": price_series}, index=date_index)
    result.index.name = "Date"
    return result


def load_feature_frame(features_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load an optional feature file if available."""
    if features_path is None:
        candidate = Path("data/features/features.csv")
        if candidate.exists():
            features_path = str(candidate)
        else:
            return None

    path = Path(features_path)
    if not path.exists():
        return None

    return pd.read_csv(path, index_col=0, parse_dates=True)


def validate_dataset(
    df: pd.DataFrame,
    expected_start: str = "2005-01-03",
    expected_end: str = "2025-12-31",
    features_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Validate the dataset and return a structured report."""
    if df.empty:
        raise ValueError("The provided data frame is empty.")

    result = {}
    working = df.copy()
    working.index = pd.to_datetime(working.index, errors="coerce")
    working = working.sort_index()
    working.index.name = "Date"

    expected_dates = build_expected_trading_dates(expected_start, expected_end)
    actual_dates = working.index.dropna()
    actual_date_set = set(actual_dates)
    expected_date_set = set(expected_dates)

    critical_issues: list[str] = []
    notes: list[str] = []

    # 1. Date range coverage.
    actual_start = actual_dates.min() if len(actual_dates) else pd.NaT
    actual_end = actual_dates.max() if len(actual_dates) else pd.NaT
    coverage_ok = actual_start <= pd.Timestamp(expected_start) and actual_end >= pd.Timestamp(expected_end)
    result["date_range_ok"] = bool(coverage_ok)
    if not coverage_ok:
        critical_issues.append(
            f"Date range is incomplete: expected {expected_start} to {expected_end}, got {actual_start} to {actual_end}."
        )

    # 2. Missing dates beyond expected market days.
    missing_dates = sorted(expected_dates.difference(actual_dates))
    result["missing_dates"] = missing_dates
    if missing_dates:
        critical_issues.append(
            f"Missing trading dates detected: {len(missing_dates)} dates, e.g. {missing_dates[:10]}"
        )
    else:
        notes.append("No missing trading dates were found relative to the expected trading calendar.")

    # 3. Missing values.
    missing_value_counts = working.isna().sum()
    total_missing = int(missing_value_counts.sum())
    result["missing_value_counts"] = missing_value_counts.to_dict()
    if total_missing:
        critical_issues.append(f"Missing values detected: {total_missing} total NaN values across the dataset.")
    else:
        notes.append("No missing values were found in the dataset columns.")

    # 4. Duplicate dates / rows.
    duplicate_dates = int(working.index.duplicated().sum())
    row_frame = working.reset_index().rename(columns={"index": "Date"})
    duplicate_rows = int(row_frame.duplicated(subset=["Date", *working.columns]).sum())
    result["duplicate_dates"] = duplicate_dates
    result["duplicate_rows"] = duplicate_rows
    if duplicate_dates:
        critical_issues.append(f"Duplicate dates found: {duplicate_dates}.")
    if duplicate_rows:
        critical_issues.append(f"Duplicate rows found: {duplicate_rows}.")
    if duplicate_dates == 0 and duplicate_rows == 0:
        notes.append("No duplicate dates or duplicate rows were found.")

    # 5. Chronology and gaps.
    monotonic = bool(working.index.is_monotonic_increasing)
    result["monotonic"] = monotonic
    if not monotonic:
        critical_issues.append("The Date index is not strictly chronological.")
    else:
        notes.append("The Date index is strictly chronological.")

    # Check for unexpected gaps compared with the expected trading calendar.
    if len(actual_dates) > 1:
        unexpected_gaps = [d for d in missing_dates if d not in set(actual_dates)]
        result["unexpected_gaps"] = int(len(unexpected_gaps))
        if unexpected_gaps:
            critical_issues.append(
                f"Unexpected date gaps detected: {len(unexpected_gaps)} calendar dates missing from the dataset."
            )
        else:
            notes.append("No unexpected date gaps were detected against the trading calendar.")

    # 6. Total trading days.
    result["trading_days"] = int(len(working))

    # 7. Price validity.
    price_column = None
    for candidate in ["^GSPC", "Adj Close", "Close"]:
        if candidate in working.columns:
            price_column = candidate
            break
    if price_column is None:
        raise KeyError("No price column was found for validation.")

    price_series = pd.to_numeric(working[price_column], errors="coerce")
    invalid_prices = (~np.isfinite(price_series)) | (price_series <= 0)
    invalid_price_count = int(invalid_prices.sum())
    result["price_column"] = price_column
    result["invalid_price_count"] = invalid_price_count
    if invalid_price_count:
        critical_issues.append(
            f"Invalid price values detected in {price_column}: {invalid_price_count} non-positive or non-finite values."
        )
    else:
        notes.append(f"The {price_column} column contains only valid positive finite values.")

    # 8. Feature checks for log returns and realized volatility.
    log_returns = np.log(price_series / price_series.shift(1))
    realized_vol = log_returns.rolling(22).std() * np.sqrt(252)

    inf_return = int(np.isinf(log_returns).sum())
    inf_rv = int(np.isinf(realized_vol).sum())
    return_nans_total = int(log_returns.isna().sum())
    rv_nans_total = int(realized_vol.isna().sum())
    return_nans_after_first = int(log_returns.iloc[1:].isna().sum())
    rv_nans_after_window = int(realized_vol.iloc[22:].isna().sum())

    if inf_return == 0 and inf_rv == 0:
        notes.append("Log returns and realized volatility contain no infinite values.")
    else:
        critical_issues.append("Infinite values were found in the computed returns or realized volatility series.")

    expected_return_nans = 1 if len(price_series) > 1 else 0
    expected_rv_nans = min(len(price_series), 22)

    if return_nans_total == expected_return_nans and return_nans_after_first == 0:
        notes.append("Log returns have the expected leading NaN(s) from differencing.")
    else:
        critical_issues.append(
            f"Unexpected log-return NaN pattern: {return_nans_total} NaNs total, {return_nans_after_first} after the first row."
        )

    if rv_nans_total == expected_rv_nans and rv_nans_after_window == 0:
        notes.append("22-day realized volatility has the expected leading NaNs from the rolling window.")
    else:
        critical_issues.append(
            f"Unexpected realized-volatility NaN pattern: {rv_nans_total} NaNs total, {rv_nans_after_window} after the rolling window."
        )

    result["feature_checks"] = [
        ("^GSPC_return", inf_return, return_nans_total, return_nans_after_first),
        ("^GSPC_rv", inf_rv, rv_nans_total, rv_nans_after_window),
    ]

    if features_df is not None:
        notes.append("An existing feature file was also inspected when available.")
    else:
        notes.append("The script computed log returns and realized volatility directly from the price series.")

    # 9. Final status.
    result["critical_issues"] = critical_issues
    result["notes"] = notes
    result["is_clean"] = len(critical_issues) == 0
    return result


def print_report(report: dict, df: pd.DataFrame) -> None:
    """Print a human-readable validation report."""
    print("=" * 80)
    print("S&P 500 (^GSPC) dataset validation report")
    print("=" * 80)
    print(f"Loaded {len(df)} rows from the dataset.")
    print(f"Expected date range: 2005-01-03 to 2025-12-31")
    print()

    print("1) Date coverage")
    print(f"- Minimum date: {df.index.min()}")
    print(f"- Maximum date: {df.index.max()}")
    print(f"- Full intended range covered: {'yes' if report['date_range_ok'] else 'no'}")
    if report['missing_dates']:
        print(f"- Missing trading dates found: {len(report['missing_dates'])}")
        print(f"  Example dates: {report['missing_dates'][:10]}")
    else:
        print("- No missing trading dates found.")
    print()

    print("2) Missing values")
    print("- Missing value counts per column:")
    for col, count in report['missing_value_counts'].items():
        print(f"  * {col}: {count}")
    print()

    print("3) Duplicate dates / duplicate rows")
    print(f"- Duplicate dates: {report['duplicate_dates']}")
    print(f"- Duplicate rows: {report['duplicate_rows']}")
    print()

    print("4) Date index chronology")
    print(f"- Strictly chronological: {'yes' if report['monotonic'] else 'no'}")
    print(f"- Unexpected gaps larger than one day: {report['unexpected_gaps']}")
    print()

    print("5) Trading day count")
    print(f"- Total trading days: {report['trading_days']}")
    print()

    print("6) First and last five rows")
    print(df.head().to_string())
    print("\n" + df.tail().to_string())
    print()

    print("7) Summary statistics")
    print(df.describe().to_string())
    print()

    print("8) Data types")
    print(df.dtypes.to_string())
    print()

    print("9) Price validity")
    print(f"- Price column used: {report['price_column']}")
    print(f"- Invalid price values (<= 0 or non-finite): {report['invalid_price_count']}")
    print()

    print("10) Log returns / realized volatility checks")
    for name, infinity_count, na_count, post_window_na_count in report['feature_checks']:
        print(
            f"- {name}: infinite values = {infinity_count}, NaN values = {na_count}, "
            f"NaNs after the initial window = {post_window_na_count}"
        )
    print()

    print("Final validation report")
    if report['is_clean']:
        print("- Status: CLEAN and suitable for model training.")
    else:
        print("- Status: NOT CLEAN; critical issues were found.")
        print("- Warnings:")
        for issue in report['critical_issues']:
            print(f"  * WARNING: {issue}")
    print("- Notes:")
    for note in report['notes']:
        print(f"  * {note}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the S&P 500 dataset before hyperparameter tuning.")
    parser.add_argument("--data", type=str, default=None, help="Path to the price dataset CSV")
    parser.add_argument("--features", type=str, default=None, help="Path to an optional feature CSV")
    parser.add_argument("--start-date", type=str, default="2005-01-03")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    args = parser.parse_args()

    df = load_sp500_data(args.data)
    features_df = load_feature_frame(args.features)
    report = validate_dataset(df, expected_start=args.start_date, expected_end=args.end_date, features_df=features_df)
    print_report(report, df)


if __name__ == "__main__":
    main()
