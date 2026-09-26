"""Data loading and cleaning helpers."""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """
    Load the penguins dataset from a CSV file.

    Args:
        path (str): Path to the CSV file containing the penguins dataset.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove incomplete rows from the dataset.

    Args:
        dataframe (pd.DataFrame): Input dataset as a pandas DataFrame.

    Returns:
        pd.DataFrame: Cleaned dataset with incomplete rows removed.
    """

    cleaned = dataframe.dropna().reset_index(drop=True)
    logger.info("Cleaned data: %d rows remain", len(cleaned))
    return cleaned


def split_data(
    features: pd.DataFrame, target: pd.Series, test_size: float, random_state: int
):
    """
    Split features and target into train and test sets.

    Args:
        features (pd.DataFrame): Feature matrix.
        target (pd.Series): Target labels.
        test_size (float): Fraction of rows to hold out for testing.
        random_state (int): Seed for reproducible splits.

    Returns:
        tuple: features_train, features_test, target_train, target_test.
    """

    return train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
