"""
Evaluation metrics for volatility forecasting.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """
    Mean Absolute Error.
    """
    return mean_absolute_error(y_true, y_pred)


def qlike(y_true, y_pred, epsilon=1e-8):
    """
    QLIKE loss for volatility forecasting.

    Lower values are better.

    Parameters
    ----------
    y_true : array-like
        Actual realized volatility.
    y_pred : array-like
        Predicted volatility.
    epsilon : float
        Prevents division by zero.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_pred = np.maximum(y_pred, epsilon)

    return np.mean(
        np.log(y_pred) + (y_true / y_pred)
    )


def calculate_metrics(y_true, y_pred):
    """
    Returns all evaluation metrics in a dictionary.
    """

    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "QLIKE": qlike(y_true, y_pred)
    }