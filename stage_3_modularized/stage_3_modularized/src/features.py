"""Feature preparation helpers."""

import logging
from typing import cast

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def encode_categoricals(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns and return the fitted encoders.

    Args:
        dataframe: Input dataframe containing categorical columns.

    Returns:
        tuple: Encoded dataframe and fitted encoders keyed by column name.
    """
    encoded = dataframe.copy()
    encoders: dict[str, LabelEncoder] = {}
    for column in ["island", "sex"]:
        if column in encoded.columns:
            encoder = LabelEncoder()
            encoded[column] = encoder.fit_transform(encoded[column].astype(str))
            encoders[column] = encoder
    return encoded, encoders


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived numerical features used by the penguin model.

    Args:
        dataframe: Input dataframe containing penguin measurements.

    Returns:
        pd.DataFrame: Dataframe with bill_ratio and body_mass_scaled added.
    """
    return dataframe.assign(
        bill_ratio=lambda df: df["bill_length_mm"] / df["bill_depth_mm"],
        body_mass_scaled=lambda df: df["body_mass_g"] / 1000.0,
    )


def build_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Add engineered features and encode categorical columns for modeling.

    Args:
        dataframe: Cleaned penguin dataframe.

    Returns:
        tuple: Final feature matrix and fitted encoders for the categorical columns.
    """
    features = add_engineered_features(dataframe)
    features, encoders = encode_categoricals(features)
    feature_columns = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "bill_ratio",
        "body_mass_scaled",
        "island",
        "sex",
    ]
    logger.info("Built feature matrix with shape %s", features[feature_columns].shape)
    return cast(pd.DataFrame, features[feature_columns]), encoders


def scale_features(
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
) -> tuple[object, object, StandardScaler]:
    """
    Fit a scaler on the training features and apply it to both splits.

    Args:
        features_train: Training feature matrix.
        features_test: Test feature matrix.

    Returns:
        tuple: Scaled training features, scaled test features, and the fitted scaler.
    """
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(features_train)
    scaled_test = scaler.transform(features_test)
    return scaled_train, scaled_test, scaler
