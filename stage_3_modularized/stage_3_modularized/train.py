import json
import logging
import os

import joblib

from configs.config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    ENCODERS_PATH,
    FEATURE_COLS,
    FEATURE_COLS_PATH,
    METRICS_PATH,
    MODEL_CONFIG,
    MODEL_PATH,
    RANDOM_SEED,
    SCALER_PATH,
    TARGET_COL,
    TEST_SIZE,
)
from src.data import clean_data, load_data, split_data
from src.evaluate import evaluate, evaluate_models
from src.features import build_features, scale_features
from src.model import train_all_models


def train_pipeline():
    """Train all candidate models and return the selected model and metadata."""
    df = clean_data(load_data(DATA_PATH))
    features, encoders = build_features(df)
    target = df[TARGET_COL]

    train_features, test_features, train_target, test_target = split_data(
        features, target, TEST_SIZE, RANDOM_SEED
    )
    scaled_train, scaled_test, scaler = scale_features(train_features, test_features)
    models = train_all_models(MODEL_CONFIG, scaled_train, train_target)
    best_model = evaluate_models(models, scaled_test, test_target)

    all_model_accuracies = {
        name: evaluate(model, scaled_test, test_target)["accuracy"]
        for name, model in models.items()
    }

    return models[best_model["name"]], scaler, encoders, best_model, all_model_accuracies


def save_artifacts(model, scaler, encoders, feature_cols, metrics):
    """Persist the trained model and metadata to disk."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)

    with open(FEATURE_COLS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    """Run the training pipeline and save the artifacts."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model, scaler, encoders, best_model, all_model_accuracies = train_pipeline()
    metrics = {
        "best_model": best_model["name"],
        "best_accuracy": best_model["accuracy"],
        "all_model_accuracies": all_model_accuracies,
    }

    save_artifacts(model, scaler, encoders, FEATURE_COLS, metrics)
    logger.info("Best model: %s (accuracy=%.3f)", best_model["name"], best_model["accuracy"])
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
