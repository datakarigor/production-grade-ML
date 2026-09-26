"""Model training helpers."""

import logging

logger = logging.getLogger(__name__)


def train_model(model, features_train, target_train):
    """
    Fit one model on the training data.

    Args:
        model: Scikit-learn estimator instance.
        features_train: Training feature matrix.
        target_train: Target labels for the training set.

    Returns:
        Any: Fitted estimator instance.
    """
    return model.fit(features_train, target_train)


def train_all_models(model_config, features_train, target_train):
    """
    Instantiate and train every configured model.

    Args:
        model_config: Model names, classes, and constructor arguments.
        features_train: Training feature matrix.
        target_train: Target labels for the training set.

    Returns:
        dict: Dictionary of fitted model objects keyed by model name.
    """
    trained_models = {}
    for model_name, model_class, params in model_config:
        model = model_class(**params)
        trained_models[model_name] = train_model(model, features_train, target_train)
        logger.info("Trained model: %s", model_name)
    return trained_models
