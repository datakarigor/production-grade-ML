"""Model evaluation helpers."""

import logging

from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


def evaluate(model, features_test, target_test):
    """
    Evaluate a single trained model on a test set.

    Args:
        model: Fitted scikit-learn model.
        features_test: Test feature matrix.
        target_test: Ground-truth labels for the test set.

    Returns:
        dict: Dictionary containing model predictions and accuracy score.
    """
    predictions = model.predict(features_test)
    accuracy = accuracy_score(target_test, predictions)
    return {"predictions": predictions, "accuracy": accuracy}


def evaluate_models(models, features_test, target_test):
    """
    Evaluate multiple trained models and return the best-performing one.

    Args:
        models: Dictionary of trained model objects keyed by name.
        features_test: Test feature matrix.
        target_test: Ground-truth labels for the test set.

    Returns:
        dict: Best-performing model result including its name, predictions, and accuracy.
    """
    results = []
    for name, model in models.items():
        metric = evaluate(model, features_test, target_test)
        logger.info("%s accuracy: %.3f", name, metric["accuracy"])
        results.append({"name": name, **metric})

    best_result = max(results, key=lambda item: item["accuracy"])
    logger.info("Selected best model: %s", best_result["name"])
    return best_result
