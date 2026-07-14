"""
preprocessing.py
Shared data cleaning and preprocessing utilities for all four modules
(Boiler, Throughput, Energy, Yield).
"""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load raw refinery dataset from CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and basic sanity checks."""
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    # TODO: add column-specific missing value handling
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived/engineered features used across modules."""
    # TODO: add feature engineering logic per module
    return df


if __name__ == "__main__":
    df = load_data("data/raw_dataset.csv")
    df = clean_data(df)
    df = engineer_features(df)
    print(df.head())
