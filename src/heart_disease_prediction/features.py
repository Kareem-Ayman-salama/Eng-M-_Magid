"""Feature preprocessing helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_groups(
    x_data: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical columns.

    Args:
        x_data: Feature dataframe.

    Returns:
        Numeric column names and categorical column names.
    """

    categorical_candidates = {
        "gender",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "hypertension_stage",
        "bmi_category",
        "age_group",
        "is_hypertensive",
    }
    categorical_columns = [
        column for column in x_data.columns if column in categorical_candidates
    ]
    numeric_columns = [
        column for column in x_data.columns if column not in categorical_columns
    ]
    return numeric_columns, categorical_columns


def build_preprocessor(x_data: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing transformer for tabular clinical data.

    Args:
        x_data: Feature dataframe used to infer column groups.

    Returns:
        ColumnTransformer with numeric scaling and categorical encoding.
    """

    numeric_columns, categorical_columns = get_feature_groups(x_data)
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
