"""Data loading, validation, and preparation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


KAGGLE_CVD_COLUMNS = {
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio",
}


@dataclass(frozen=True)
class DatasetSplits:
    """Train, validation, and test splits.

    Attributes:
        x_train: Training features.
        x_valid: Validation features.
        x_test: Final test features.
        y_train: Training labels.
        y_valid: Validation labels.
        y_test: Final test labels.
    """

    x_train: pd.DataFrame
    x_valid: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def load_csv(data_path: Path) -> pd.DataFrame:
    """Load a CSV file with automatic delimiter detection.

    Args:
        data_path: Path to the dataset CSV file.

    Returns:
        Loaded dataframe.

    Raises:
        FileNotFoundError: If the dataset path does not exist.
    """

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    return pd.read_csv(data_path, sep=None, engine="python")


def validate_kaggle_cvd_schema(
    data: pd.DataFrame,
    target_column: str,
) -> None:
    """Validate the expected Kaggle cardiovascular disease schema.

    Args:
        data: Raw dataframe.
        target_column: Expected target column.

    Raises:
        ValueError: If required columns are missing or target is not binary.
    """

    required_columns = KAGGLE_CVD_COLUMNS - {"cardio"} | {target_column}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    target_values = set(data[target_column].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Target column must be binary 0/1. Found: {target_values}"
        )


def clean_kaggle_cvd_data(
    data: pd.DataFrame,
    target_column: str = "cardio",
) -> pd.DataFrame:
    """Clean the Kaggle cardiovascular disease dataset.

    The cleaning keeps clinically plausible ranges, converts age from days to
    years, adds BMI, and drops the row identifier if present.

    Args:
        data: Raw Kaggle cardiovascular disease dataframe.
        target_column: Binary target column name.

    Returns:
        Cleaned dataframe.
    """

    clean_data = data.copy()
    if "id" in clean_data.columns:
        clean_data = clean_data.drop(columns=["id"])

    clean_data["age_years"] = clean_data["age"] / 365.25
    clean_data["bmi"] = clean_data["weight"] / ((clean_data["height"] / 100) ** 2)
    clean_data = clean_data.drop(columns=["age"])

    plausible_mask = (
        clean_data["age_years"].between(18, 100)
        & clean_data["height"].between(120, 220)
        & clean_data["weight"].between(30, 250)
        & clean_data["ap_hi"].between(80, 250)
        & clean_data["ap_lo"].between(40, 160)
        & (clean_data["ap_hi"] > clean_data["ap_lo"])
        & clean_data["bmi"].between(12, 80)
    )

    clean_data = clean_data.loc[plausible_mask].drop_duplicates()
    clean_data = add_clinical_features(clean_data)
    clean_data[target_column] = clean_data[target_column].astype(int)
    return clean_data.reset_index(drop=True)


def add_clinical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add clinically motivated derived features.

    Args:
        data: Cleaned cardiovascular disease dataframe.

    Returns:
        Dataframe with additional derived features.
    """

    featured_data = data.copy()
    featured_data["pulse_pressure"] = (
        featured_data["ap_hi"] - featured_data["ap_lo"]
    )
    featured_data["mean_arterial_pressure"] = (
        featured_data["ap_hi"] + (2 * featured_data["ap_lo"])
    ) / 3
    featured_data["hypertension_stage"] = pd.cut(
        featured_data["ap_hi"],
        bins=[0, 120, 130, 140, float("inf")],
        labels=["normal", "elevated", "stage_1", "stage_2"],
        right=False,
    ).astype(str)
    featured_data["bmi_category"] = pd.cut(
        featured_data["bmi"],
        bins=[0, 18.5, 25, 30, float("inf")],
        labels=["underweight", "normal", "overweight", "obese"],
        right=False,
    ).astype(str)
    featured_data["age_group"] = pd.cut(
        featured_data["age_years"],
        bins=[0, 40, 50, 60, float("inf")],
        labels=["under_40", "40_49", "50_59", "60_plus"],
        right=False,
    ).astype(str)
    featured_data["is_hypertensive"] = (
        (featured_data["ap_hi"] >= 140) | (featured_data["ap_lo"] >= 90)
    ).astype(int)
    featured_data["lifestyle_risk_count"] = (
        featured_data["smoke"]
        + featured_data["alco"]
        + (1 - featured_data["active"])
    )
    featured_data["age_systolic_interaction"] = (
        featured_data["age_years"] * featured_data["ap_hi"]
    )
    featured_data["bmi_systolic_interaction"] = (
        featured_data["bmi"] * featured_data["ap_hi"]
    )
    return featured_data


def split_features_target(
    data: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and labels.

    Args:
        data: Cleaned dataframe.
        target_column: Binary target column name.

    Returns:
        Feature dataframe and target series.
    """

    x_data = data.drop(columns=[target_column])
    y_data = data[target_column]
    return x_data, y_data


def make_dataset_splits(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    random_state: int,
    test_size: float,
    validation_size: float,
) -> DatasetSplits:
    """Create stratified train, validation, and test splits.

    Args:
        x_data: Feature dataframe.
        y_data: Binary labels.
        random_state: Reproducibility seed.
        test_size: Fraction reserved for final test.
        validation_size: Fraction of remaining data reserved for validation.

    Returns:
        DatasetSplits containing train, validation, and test partitions.
    """

    x_train_valid, x_test, y_train_valid, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        stratify=y_data,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid,
        y_train_valid,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_valid,
    )
    return DatasetSplits(
        x_train=x_train,
        x_valid=x_valid,
        x_test=x_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
    )
