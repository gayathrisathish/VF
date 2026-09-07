"""
Loader for the best hyperparameters produced by the Optuna tuning scripts.

The tuning scripts write one JSON file per (model, horizon) study into
``results/best_params`` using the study name as the filename stem:

    src/tuning/tune_lstm.py            -> best_params_lstm_<horizon>.json
    src/tuning/tune_gru.py             -> best_params_gru_<horizon>.json
    src/tuning/tune_transformer.py     -> best_params_transformer_<horizon>.json
    src/training/tune_garch_lstm.py    -> best_params_garch_lstm_<horizon>.json
    src/training/tune_garch_gru.py     -> best_params_garch_gru_<horizon>.json
    src/training/tune_garch_transformer.py
                                       -> best_params_garch_transformer_<horizon>.json

Every study was run on a single index (``^GSPC``), so the saved parameters are
keyed by (model, horizon) only and are reused across all indices. This module
never runs tuning; it only reads what tuning already wrote.
"""

import json
from pathlib import Path

# =============================================================================
# Directories
# =============================================================================

BEST_PARAMS_DIR = Path("results/best_params")

# =============================================================================
# Model name -> tuning study stem
# =============================================================================

STUDY_STEMS = {
    "LSTM": "lstm",
    "GRU": "gru",
    "Transformer": "transformer",
    "GARCHLSTM": "garch_lstm",
    "GARCHGRU": "garch_gru",
    "GARCHTransformer": "garch_transformer",
}

# Parameters each architecture expects from its study.
REQUIRED_PARAMETERS = {
    "LSTM": ("hidden_units", "num_layers", "dropout", "learning_rate"),
    "GRU": ("hidden_units", "num_layers", "dropout", "learning_rate"),
    "Transformer": (
        "d_model",
        "num_heads",
        "num_layers",
        "dropout",
        "learning_rate",
    ),
    "GARCHLSTM": ("hidden_units", "num_layers", "dropout", "learning_rate"),
    "GARCHGRU": ("hidden_units", "num_layers", "dropout", "learning_rate"),
    "GARCHTransformer": (
        "d_model",
        "num_heads",
        "num_layers",
        "dropout",
        "learning_rate",
    ),
}


def is_tuned_model(model_name):
    """Return True when the model has an associated tuning study."""

    return model_name in STUDY_STEMS


def best_params_path(model_name, horizon):
    """Return the expected JSON path for one (model, horizon) study."""

    if model_name not in STUDY_STEMS:
        raise ValueError(
            f"No tuning study is defined for model '{model_name}'. "
            f"Tuned models: {sorted(STUDY_STEMS)}"
        )

    stem = STUDY_STEMS[model_name]

    return BEST_PARAMS_DIR / f"best_params_{stem}_{horizon}.json"


def load_best_params(model_name, horizon):
    """
    Load the tuned hyperparameters for one model and horizon.

    Returns
    -------
    dict
        The tuned hyperparameters, including ``batch_size`` when the study
        recorded one.

    Raises
    ------
    FileNotFoundError
        If tuning has not been completed for this model and horizon.
    ValueError
        If the file is unreadable or is missing expected parameters.
    """

    params_path = best_params_path(model_name, horizon)

    if not params_path.exists():
        raise FileNotFoundError(
            f"Missing tuned hyperparameters for {model_name} | {horizon}: "
            f"{params_path} does not exist. Run the corresponding tuning "
            "script before training this model."
        )

    try:
        with open(params_path, "r", encoding="utf-8") as params_file:
            payload = json.load(params_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read tuned hyperparameters for {model_name} | "
            f"{horizon} from {params_path}: {error}"
        ) from error

    if not isinstance(payload, dict) or "best_params" not in payload:
        raise ValueError(
            f"{params_path} does not contain a 'best_params' object."
        )

    best_params = dict(payload["best_params"])

    missing_parameters = [
        parameter
        for parameter in REQUIRED_PARAMETERS[model_name]
        if parameter not in best_params
    ]
    if missing_parameters:
        raise ValueError(
            f"{params_path} is missing required parameters for "
            f"{model_name}: {', '.join(missing_parameters)}"
        )

    # The transformer studies record batch_size at the top level of the file
    # rather than inside best_params; the recurrent studies record it inside.
    if "batch_size" not in best_params and "batch_size" in payload:
        best_params["batch_size"] = payload["batch_size"]

    return best_params


def tuning_coverage(model_names, horizons):
    """
    Report which (model, horizon) studies have a usable best-parameter file.

    Returns
    -------
    dict
        Maps (model_name, horizon) to a status dict with 'path', 'present',
        'readable', and 'error' keys.
    """

    coverage = {}

    for model_name in model_names:
        for horizon in horizons:
            params_path = best_params_path(model_name, horizon)
            status = {
                "path": str(params_path),
                "present": params_path.exists(),
                "readable": False,
                "error": None,
            }

            if status["present"]:
                try:
                    load_best_params(model_name, horizon)
                    status["readable"] = True
                except (ValueError, FileNotFoundError) as error:
                    status["error"] = str(error)

            coverage[(model_name, horizon)] = status

    return coverage
