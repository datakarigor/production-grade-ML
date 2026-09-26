import logging

import pandas as pd

from src.features import add_engineered_features

logger = logging.getLogger(__name__)


def predict_species(sample, model, scaler, encoders, feature_cols):
    """
    Predict the species for one raw penguin observation.

    Args:
        sample: Raw penguin record as a dictionary with the original columns from the
            source dataset, including bill length, bill depth, flipper length,
            body mass, island, and sex.
        model: Fitted scikit-learn estimator used to predict the species.
        scaler: Fitted StandardScaler used to scale the feature matrix.
        encoders: Dictionary of fitted LabelEncoder objects for the island and sex
            categorical columns.
        feature_cols: Ordered list of feature column names expected by the trained model.

    Returns:
        str: Predicted penguin species label.
    """
    sample_df = pd.DataFrame([sample])
    sample_df = add_engineered_features(sample_df)

    for column in ["island", "sex"]:
        if column in sample_df.columns and column in encoders:
            sample_df[column] = encoders[column].transform(sample_df[column])

    prepared = sample_df[feature_cols]
    scaled = scaler.transform(prepared)
    prediction = model.predict(scaled)
    logger.info("Predicted species: %s", prediction[0])
    return prediction[0]
