"""
Data loading utilities.

Loads the train, validation, and test datasets for a given forecast horizon.
"""

from pathlib import Path

import pandas as pd


BASE_PATH = Path("data/splits")


def load_split(horizon: str):
    """
    Load train, validation, and test datasets.

    Parameters
    ----------
    horizon : str
        One of:
        - "1day"
        - "5day"
        - "22day"

    Returns
    -------
    dict
        Dictionary containing:

        X_train
        X_validation
        X_test
        y_train
        y_validation
        y_test
    """

    split_path = BASE_PATH / horizon

    data = {
        "X_train": pd.read_csv(split_path / "X_train.csv"),
        "X_validation": pd.read_csv(split_path / "X_validation.csv"),
        "X_test": pd.read_csv(split_path / "X_test.csv"),

        "y_train": pd.read_csv(split_path / "y_train.csv"),
        "y_validation": pd.read_csv(split_path / "y_validation.csv"),
        "y_test": pd.read_csv(split_path / "y_test.csv")
    }

    return data