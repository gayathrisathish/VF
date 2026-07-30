"""
Model Registry

Maps model names to their corresponding training functions.

This allows the experiment runner to dynamically select and execute
any model without using long if/elif statements.
"""

# ============================================================================
# Benchmark Models
# ============================================================================

from src.models.benchmarks import (
    train_historical_volatility,
    train_persistence,
)

# ============================================================================
# Econometric Models
# ============================================================================

from src.models.garch_models import (
    train_garch,
    train_egarch,
    train_gjr_garch,
)

# ============================================================================
# Machine Learning Models
# ============================================================================

from src.models.ml_models import (
    train_linear_regression,
    train_random_forest,
    train_xgboost,
)

# ============================================================================
# Deep Learning Models
# ============================================================================

from src.models.deep_learning import (
    train_lstm,
    train_gru,
    train_transformer,
)

# ============================================================================
# Hybrid Models
# ============================================================================

from src.models.hybrid_models import (
    train_garch_lstm,
    train_garch_gru,
    train_garch_transformer,
)

# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY = {

    # ------------------------------------------------------------------------
    # Benchmark Models
    # ------------------------------------------------------------------------
    "HistoricalVolatility": train_historical_volatility,
    "Persistence": train_persistence,

    # ------------------------------------------------------------------------
    # Econometric Models
    # ------------------------------------------------------------------------
    "GARCH": train_garch,
    "EGARCH": train_egarch,
    "GJRGARCH": train_gjr_garch,

    # ------------------------------------------------------------------------
    # Machine Learning Models
    # ------------------------------------------------------------------------
    "LinearRegression": train_linear_regression,
    "RandomForest": train_random_forest,
    "XGBoost": train_xgboost,

    # ------------------------------------------------------------------------
    # Deep Learning Models
    # ------------------------------------------------------------------------
    "LSTM": train_lstm,
    "GRU": train_gru,
    "Transformer": train_transformer,

    # ------------------------------------------------------------------------
    # Hybrid Models
    # ------------------------------------------------------------------------
    "GARCHLSTM": train_garch_lstm,
    "GARCHGRU": train_garch_gru,
    "GARCHTransformer": train_garch_transformer,
}


def get_model(model_name):
    """
    Return the training function for the requested model.

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    callable
        Training function corresponding to the model.
    """

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    return MODEL_REGISTRY[model_name]


def list_models():
    """
    Return a list of all available models.
    """

    return list(MODEL_REGISTRY.keys())