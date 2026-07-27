"""
Generic model training pipeline.
"""

from src.evaluation.metrics import calculate_metrics


def train_and_evaluate(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):
    """
    Generic pipeline used by every model.
    """

    # Train
    model.fit(X_train, y_train)

    # Validation predictions
    validation_predictions = model.predict(X_validation)

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions
    )

    # Test predictions
    test_predictions = model.predict(X_test)

    test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    return {
        "model": model,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics
    }