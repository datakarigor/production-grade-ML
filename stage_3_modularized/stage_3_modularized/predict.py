import json

import joblib
from configs.config import (
    ENCODERS_PATH,
    FEATURE_COLS_PATH,
    MODEL_PATH,
    SAMPLE_PATH,
    SCALER_PATH,
)
from src.predict import predict_species


def load_artifacts():
    """Load the persisted model, scaler, encoders, and feature columns."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)

    with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    return model, scaler, encoders, feature_cols


def main():
    """Load saved artifacts and predict the species for the sample record."""


    model, scaler, encoders, feature_cols = load_artifacts()

    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        sample = json.load(f)

    prediction = predict_species(sample, model, scaler, encoders, feature_cols)
    print(prediction)
    return prediction


if __name__ == "__main__":
    main()
