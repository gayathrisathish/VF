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


def _build_lstm_model(input_shape):
    tf = _get_tensorflow()
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(64)(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="softplus")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )
    return model


def _build_gru_model(input_shape):
    tf = _get_tensorflow()
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.GRU(64)(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="softplus")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="GRU")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )
    return model


def _build_transformer_model(input_shape):
    tf = _get_tensorflow()
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(TRANSFORMER_MODEL_DIM)(inputs)

    position_indices = tf.range(start=0, limit=input_shape[0], delta=1)
    positional_embeddings = tf.keras.layers.Embedding(
        input_dim=SEQUENCE_LENGTH,
        output_dim=TRANSFORMER_MODEL_DIM,
        name="positional_embedding",
    )(position_indices)
    positional_embeddings = tf.keras.layers.Lambda(
        lambda tensor: tf.expand_dims(tensor, axis=0),
        name="expand_positional_embedding",
    )(positional_embeddings)

    x = tf.keras.layers.Add()([x, positional_embeddings])
    x = tf.keras.layers.LayerNormalization()(x)
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=TRANSFORMER_NUM_HEADS,
        key_dim=TRANSFORMER_MODEL_DIM // TRANSFORMER_NUM_HEADS,
        dropout=0.1,
    )(x, x)
    x = tf.keras.layers.Add()([x, attention_output])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(TRANSFORMER_MODEL_DIM * 2, activation="relu")(x)
    x = tf.keras.layers.Dense(TRANSFORMER_MODEL_DIM, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation="softplus")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Transformer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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
):
    """Fit one deep-learning estimator and evaluate its volatility forecasts."""
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

    model = model_builder(X_train.shape[1:])
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
        batch_size=BATCH_SIZE,
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
    )