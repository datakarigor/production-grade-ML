import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import clean_data, load_data
from src.features import (
    add_engineered_features,
    build_features,
    encode_categoricals,
    scale_features,
)


def test_encode_categoricals_label_encodes_categories():
    df = load_data("data/penguins.csv")
    cleaned = clean_data(df)

    encoded, encoders = encode_categoricals(cleaned[["island", "sex"]])  # type: ignore

    assert isinstance(encoded, pd.DataFrame)
    assert list(encoded.columns) == ["island", "sex"]
    assert set(encoders) == {"island", "sex"}
    assert pd.api.types.is_integer_dtype(encoded["island"])
    assert pd.api.types.is_integer_dtype(encoded["sex"])


def test_add_engineered_features_creates_bill_ratio_and_scaled_mass():
    df = load_data("data/penguins.csv")
    cleaned = clean_data(df)

    engineered = add_engineered_features(cleaned)

    assert "bill_ratio" in engineered.columns
    assert "body_mass_scaled" in engineered.columns
    assert (
        engineered["bill_ratio"]
        == engineered["bill_length_mm"] / engineered["bill_depth_mm"]
    ).all()
    assert (engineered["body_mass_scaled"] == engineered["body_mass_g"] / 1000).all()


def test_build_features_uses_expected_columns():
    df = load_data("data/penguins.csv")
    cleaned = clean_data(df)

    features, encoders = build_features(cleaned)

    assert list(features.columns) == [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "bill_ratio",
        "body_mass_scaled",
        "island",
        "sex",
    ]
    assert set(encoders) == {"island", "sex"}
    assert not features.isnull().any().any()


def test_scale_features_fits_on_train_only():
    df = load_data("data/penguins.csv")
    cleaned = clean_data(df)
    features, _ = build_features(cleaned)

    train = features.iloc[: int(len(features) * 0.8)]
    test = features.iloc[int(len(features) * 0.8) :]

    scaled_train, scaled_test, scaler = scale_features(train, test)  # type: ignore

    assert isinstance(scaler, StandardScaler)
    assert scaled_train.shape == train.shape
    assert scaled_test.shape == test.shape
