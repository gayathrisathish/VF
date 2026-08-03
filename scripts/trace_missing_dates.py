#!/usr/bin/env python3
"""Trace why specific dates are absent from the processed datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/global_indices.csv")
PROCESSED_DATA_PATH = Path("data/processed/cleaned_indices.csv")
FEATURES_DATA_PATH = Path("data/features/features.csv")


def main() -> None:
    dates = ["2005-01-03", "2025-12-31"]

    print("Tracing missing dates through the preprocessing pipeline")
    print("=" * 80)

    raw_df = pd.read_csv(RAW_DATA_PATH, header=[0, 1], index_col=0)
    raw_df.index = pd.to_datetime(raw_df.index)
    raw_df = raw_df.sort_index()

    print("Raw Yahoo Finance export")
    print(f"- min date: {raw_df.index.min()}")
    print(f"- max date: {raw_df.index.max()}")
    print()

    for date_str in dates:
        target = pd.Timestamp(date_str)
        print(f"Date: {date_str}")
        print(f"- Present in raw export: {'yes' if target in raw_df.index else 'no'}")

        if target not in raw_df.index:
            print("- Root cause: this date was not present in the raw Yahoo Finance download.")
            print("- Responsible step: the download stage in src/data/download_data.py")
            print("  yfinance.download(..., start='2005-01-01', end='2025-12-31')")
            print("- Result: it never reaches preprocessing and therefore is not removed later.")
            print()
            continue

        # Reproduce the cleaning pipeline from src/data/clean_data.py.
        close_prices = pd.DataFrame()
        for ticker in raw_df.columns.levels[0]:
            if "Close" in raw_df[ticker].columns:
                close_prices[ticker] = raw_df[ticker]["Close"]

        print("- Present after selecting Close columns: yes")
        before_drop = close_prices.copy()
        after_all_na = before_drop.dropna(how="all")
        print("- Present after dropna(how='all'): yes")
        after_ffill = after_all_na.ffill()
        print("- Present after ffill(): yes")
        after_dropna = after_ffill.dropna()
        print("- Present after final dropna():", target in after_dropna.index)

        if target not in after_dropna.index:
            missing_cols = after_ffill.loc[target].isna()
            missing_cols = missing_cols[missing_cols].index.tolist()
            print("- Responsible step: close_prices = close_prices.dropna()")
            print("  in src/data/clean_data.py")
            print(f"- Reason: the row still had NaNs for {missing_cols} after ffill(), so it was removed.")
        else:
            print("- Responsible step: none; it survived the cleaning step.")

        print("- Feature-engineering impact:")
        if FEATURES_DATA_PATH.exists():
            feature_df = pd.read_csv(FEATURES_DATA_PATH, index_col=0, parse_dates=True)
            print(f"  * Present in feature file: {'yes' if target in feature_df.index else 'no'}")
        else:
            print("  * Feature file not found")
        print()


if __name__ == "__main__":
    main()
