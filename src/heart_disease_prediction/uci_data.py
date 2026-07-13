"""UCI Heart Disease dataset loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


UCI_HEART_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


@dataclass(frozen=True)
class UciDatasetSplits:
    """Train and test splits for compact UCI experiments.

    Attributes:
        x_train: Training features.
        x_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
    """

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_uci_heart_file(
    data_path: Path,
    source_name: str,
) -> pd.DataFrame:
    """Load one processed UCI Heart Disease data file.

    Args:
        data_path: Path to a processed UCI file.
        source_name: Source cohort name, such as ``cleveland``.

    Returns:
        Loaded dataframe with a source column.

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    if not data_path.exists():
        raise FileNotFoundError(f"UCI file not found: {data_path}")

    data = pd.read_csv(
        data_path,
        names=UCI_HEART_COLUMNS,
        na_values="?",
    )
    data["source"] = source_name
    return data


def load_uci_heart_directory(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all downloaded processed UCI Heart Disease cohorts.

    Args:
        data_dir: Directory containing the processed UCI files.

    Returns:
        Mapping of cohort names to dataframes.
    """

    files = {
        "cleveland": "processed.cleveland.data",
        "hungarian": "processed.hungarian.data",
        "switzerland": "processed.switzerland.data",
        "va": "processed.va.data",
    }
    return {
        name: load_uci_heart_file(data_dir / filename, name)
        for name, filename in files.items()
    }


def clean_uci_heart_data(
    data: pd.DataFrame,
    missing_threshold: float = 0.35,
) -> pd.DataFrame:
    """Clean UCI Heart Disease data for binary prediction.

    The original UCI target is multi-class: 0 means no disease and 1-4 indicate
    disease severity. This function converts it to a binary target.

    Args:
        data: Raw UCI dataframe.
        missing_threshold: Maximum feature missingness allowed before dropping
            a column.

    Returns:
        Cleaned dataframe.
    """

    clean_data = data.copy()
    clean_data["target"] = (clean_data["target"] > 0).astype(int)

    feature_missing_rate = clean_data.drop(columns=["target"]).isna().mean()
    columns_to_drop = feature_missing_rate[
        feature_missing_rate > missing_threshold
    ].index.tolist()
    clean_data = clean_data.drop(columns=columns_to_drop)

    numeric_columns = clean_data.select_dtypes(include=["number"]).columns
    categorical_columns = [
        column
        for column in clean_data.columns
        if column not in numeric_columns and column != "target"
    ]

    for column in numeric_columns:
        if column != "target":
            clean_data[column] = clean_data[column].fillna(
                clean_data[column].median()
            )
    for column in categorical_columns:
        clean_data[column] = clean_data[column].fillna("unknown")

    return clean_data.drop_duplicates().reset_index(drop=True)


def split_uci_features_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split UCI dataframe into features and binary target."""

    return data.drop(columns=["target"]), data["target"].astype(int)


def make_uci_train_test_split(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    random_state: int,
    test_size: float = 0.20,
) -> UciDatasetSplits:
    """Create a stratified train/test split for UCI experiments."""

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        stratify=y_data,
    )
    return UciDatasetSplits(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )
