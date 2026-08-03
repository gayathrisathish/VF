from pathlib import Path

import pandas as pd

from scripts.validate_sp500_dataset import validate_dataset


def test_validate_dataset_reports_clean_status_for_clean_input(tmp_path):
    dates = pd.date_range("2005-01-03", "2005-01-07", freq="B")
    df = pd.DataFrame({"^GSPC": [100.0, 101.0, 101.5, 102.0, 102.2]}, index=dates)

    report = validate_dataset(df, expected_start="2005-01-03", expected_end="2005-01-07")

    assert report["date_range_ok"] is True
    assert report["missing_dates"] == []
    assert report["duplicate_dates"] == 0
    assert report["duplicate_rows"] == 0
    assert report["monotonic"] is True
    assert report["invalid_price_count"] == 0
    assert report["is_clean"] is True
