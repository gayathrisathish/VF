"""Deep learning volatility models built with TensorFlow/Keras.

Each training function follows the shared experiment interface: fit only on
the training split, forecast the validation and test splits, validate the
predictions, and return the standardized results dictionary used throughout
the project.
"""

import os
import random

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np

from src.evaluation.metrics import calculate_metrics


RANDOM_SEED = 42
SEQUENCE_LENGTH = 30
MAX_EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10
TRANSFORMER_MODEL_DIM = 64
TRANSFORMER_NUM_HEADS = 4

# Defaults preserving the original untuned architecture. Tuned runs override
# these with the values loaded from results/best_params.
DEFAULT_HIDDEN_UNITS = 64
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.0
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TRANSFORMER_DROPOUT = 0.1


def _get_tensorflow():
    """Import TensorFlow lazily so the module stays importable without it."""
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required to train deep learning models."
        ) from error

    return tf


def _set_random_seeds(seed=RANDOM_SEED):
    """Set Python, NumPy, and TensorFlow seeds for reproducibility."""
    tf = _get_tensorflow()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _as_feature_array(features, split_name):
    """Return a validated 3D sequence tensor for one data split."""
    try:
        array = np.asarray(features, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{split_name} must contain only numeric feature values."
        ) from error

    if array.ndim != 3:
        raise ValueError(f"{split_name} must be a three-dimensional array.")
    if array.shape[0] == 0:
        raise ValueError(f"{split_name} must contain at least one observation.")
    if array.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            f"{split_name} must have sequence length {SEQUENCE_LENGTH}; "
            f"received {array.shape[1]}."
        )
    if any(dimension == 0 for dimension in array.shape[1:]):
        raise ValueError(f"{split_name} must contain at least one feature.")
    if np.isinf(array).any():
        raise ValueError(f"{split_name} must not contain infinite values.")

    return array


def _as_target_vector(target, split_name):
    """Return a finite one-dimensional target vector for one data split."""
    try:
        target_array = np.asarray(target, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{split_name} must contain only numeric target values."
        ) from error

    if target_array.ndim == 1:
        vector = target_array
    elif target_array.ndim == 2 and target_array.shape[1] == 1:
        vector = target_array[:, 0]
    else:
        raise ValueError(f"{split_name} must have shape (n,) or (n, 1).")

    if vector.size == 0:
        raise ValueError(f"{split_name} must contain at least one observation.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{split_name} must contain finite target values.")

    return vector


def _validate_inputs(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    horizon,
):
    """Validate split alignment and return model-ready arrays."""
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool):
        raise ValueError("horizon must be a positive integer.")
    if horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    feature_splits = {
        "X_train": _as_feature_array(X_train, "X_train"),
        "X_validation": _as_feature_array(X_validation, "X_validation"),
        "X_test": _as_feature_array(X_test, "X_test"),
    }
    target_splits = {
        "y_train": _as_target_vector(y_train, "y_train"),
        "y_validation": _as_target_vector(y_validation, "y_validation"),
        "y_test": _as_target_vector(y_test, "y_test"),
    }

    for split in ("train", "validation", "test"):
        features = feature_splits[f"X_{split}"]
        target = target_splits[f"y_{split}"]
        if len(features) != len(target):
            raise ValueError(f"X_{split} and y_{split} must have matching lengths.")

    feature_shape = feature_splits["X_train"].shape[1:]
    for split in ("X_validation", "X_test"):
        if feature_splits[split].shape[1:] != feature_shape:
            raise ValueError(
                "All feature splits must have the same sequence shape."
            )

    if np.isnan(feature_splits["X_train"]).all(axis=(0, 1)).any():
        raise ValueError(
            "X_train must not contain entirely missing features across the training split."
        )

    return (
        feature_splits["X_train"],
        target_splits["y_train"],
        feature_splits["X_validation"],
        target_splits["y_validation"],
        feature_splits["X_test"],
        target_splits["y_test"],
    )


def _validate_predictions(predictions):
    """Return finite, non-negative one-dimensional volatility forecasts."""
    try:
        validated_predictions = np.asarray(predictions, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError("Model produced non-numeric predictions.") from error

    if not np.isfinite(validated_predictions).all():
        raise ValueError("Model produced non-finite predictions.")

    return np.maximum(validated_predictions, 0.0)


def _normalize_features(X_train, X_validation, X_test):
    """Normalize sequence tensors using statistics learned only from training data."""
    tf = _get_tensorflow()
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train)

    return (
        np.asarray(normalizer(X_train)),
        np.asarray(normalizer(X_validation)),
        np.asarray(normalizer(X_test)),
    )


def _build_recurrent_model(
    recurrent_layer,
    model_name,
    input_shape,
    hyperparameters=None,
):
    """Build a stacked recurrent forecaster from tuned or default settings."""
    tf = _get_tensorflow()
    hyperparameters = hyperparameters or {}

    hidden_units = int(hyperparameters.get("hidden_units", DEFAULT_HIDDEN_UNITS))
    num_layers = int(hyperparameters.get("num_layers", DEFAULT_NUM_LAYERS))
    dropout = float(hyperparameters.get("dropout", DEFAULT_DROPOUT))
    learning_rate = float(
        hyperparameters.get("learning_rate", DEFAULT_LEARNING_RATE)
    )

    if hidden_units < 1:
        raise ValueError("hidden_units must be a positive integer.")
    if num_layers < 1:
        raise ValueError("num_layers must be a positive integer.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for layer_index in range(num_layers):
        return_sequences = layer_index < num_layers - 1
        x = recurrent_layer(hidden_units, return_sequences=return_sequences)(x)
        if dropout > 0.0:
            x = tf.keras.layers.Dropout(dropout)(x)

    # The tuning studies feed the recurrent stack straight into the output
    # layer. Without tuned settings the original 32-unit dense head is kept.
    if not hyperparameters:
        x = tf.keras.layers.Dense(32, activation="relu")(x)

    outputs = tf.keras.layers.Dense(1, activation="softplus")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def _build_lstm_model(input_shape, hyperparameters=None):
    tf = _get_tensorflow()
    return _build_recurrent_model(
        tf.keras.layers.LSTM,
        "LSTM",
        input_shape,
        hyperparameters,
    )


def _build_gru_model(input_shape, hyperparameters=None):
    tf = _get_tensorflow()
    return _build_recurrent_model(
        tf.keras.layers.GRU,
        "GRU",
        input_shape,
        hyperparameters,
    )


def _build_transformer_model(input_shape, hyperparameters=None):
    tf = _get_tensorflow()
    hyperparameters = hyperparameters or {}

    model_dim = int(hyperparameters.get("d_model", TRANSFORMER_MODEL_DIM))
    num_heads = int(hyperparameters.get("num_heads", TRANSFORMER_NUM_HEADS))
    num_layers = int(hyperparameters.get("num_layers", DEFAULT_NUM_LAYERS))
    dropout = float(
        hyperparameters.get("dropout", DEFAULT_TRANSFORMER_DROPOUT)
    )
    learning_rate = float(
        hyperparameters.get("learning_rate", DEFAULT_LEARNING_RATE)
    )

    if model_dim < 1:
        raise ValueError("d_model must be a positive integer.")
    if num_heads < 1:
        raise ValueError("num_heads must be a positive integer.")
    if model_dim % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    if num_layers < 1:
        raise ValueError("num_layers must be a positive integer.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(model_dim)(inputs)

    position_indices = tf.range(start=0, limit=input_shape[0], delta=1)
    positional_embeddings = tf.keras.layers.Embedding(
        input_dim=SEQUENCE_LENGTH,
        output_dim=model_dim,
        name="positional_embedding",
    )(position_indices)
    positional_embeddings = tf.keras.layers.Lambda(
        lambda tensor: tf.expand_dims(tensor, axis=0),
        name="expand_positional_embedding",
    )(positional_embeddings)

    x = tf.keras.layers.Add()([x, positional_embeddings])
    x = tf.keras.layers.LayerNormalization()(x)

    for _ in range(num_layers):
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=model_dim // num_heads,
            dropout=dropout,
        )(x, x)
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(model_dim * 2, activation="relu")(x)
        x = tf.keras.layers.Dense(model_dim, activation="relu")(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation="softplus")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Transformer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def _train_and_evaluate(
    model_builder,
    model_name,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    horizon,
    hyperparameters=None,
):
    """Fit one deep-learning estimator and evaluate its volatility forecasts.

    ``hyperparameters`` carries the tuned settings loaded from
    results/best_params. When it is None the original defaults are used.
    """
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = _validate_inputs(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        horizon,
    )

    tf = _get_tensorflow()
    _set_random_seeds()
    tf.keras.backend.clear_session()

    X_train, X_validation, X_test = _normalize_features(
        X_train,
        X_validation,
        X_test,
    )

    model = model_builder(X_train.shape[1:], hyperparameters)
    batch_size = int((hyperparameters or {}).get("batch_size", BATCH_SIZE))
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=MAX_EPOCHS,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0,
        shuffle=False,
    )

    validation_predictions = _validate_predictions(
        model.predict(X_validation, verbose=0)
    )
    test_predictions = _validate_predictions(model.predict(X_test, verbose=0))

    return {
        "model": model_name,
        "fitted_model": model,
        "hyperparameters": dict(hyperparameters or {}),
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "validation_metrics": calculate_metrics(
            y_validation,
            validation_predictions,
        ),
        "test_metrics": calculate_metrics(y_test, test_predictions),
    }


def train_lstm(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    index,
    horizon,
    hyperparameters=None,
):
    """Train a reproducible LSTM volatility forecaster."""
    del index
    return _train_and_evaluate(
        _build_lstm_model,
        "LSTM",
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        horizon,
        hyperparameters,
    )


def train_gru(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    index,
    horizon,
    hyperparameters=None,
):
    """Train a reproducible GRU volatility forecaster."""
    del index
    return _train_and_evaluate(
        _build_gru_model,
        "GRU",
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        horizon,
        hyperparameters,
    )


def train_transformer(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    index,
    horizon,
    hyperparameters=None,
):
    """Train a reproducible Transformer volatility forecaster."""
    del index
    return _train_and_evaluate(
        _build_transformer_model,
        "Transformer",
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        horizon,
        hyperparameters,
    )