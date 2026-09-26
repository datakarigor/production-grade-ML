import pandas as pd

from src.data import clean_data, load_data, split_data


def test_load_data_returns_dataframe():
    df = load_data("data/penguins.csv")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"species", "island", "bill_length_mm", "body_mass_g"}.issubset(
        set(df.columns)
    )


def test_clean_data_removes_missing_rows_and_resets_index():
    raw = load_data("data/penguins.csv")
    cleaned = clean_data(raw)

    assert cleaned.isna().sum().sum() == 0
    assert len(cleaned) <= len(raw)


def test_split_data_returns_80_20_split():
    df = clean_data(load_data("data/penguins.csv"))

    features_train, features_test, target_train, target_test = split_data(
        df.drop(columns="species"), df["species"], test_size=0.2, random_state=42
    )

    assert len(features_train) == len(target_train)
    assert len(features_test) == len(target_test)
    assert round(len(features_test) / len(df), 1) == 0.2
