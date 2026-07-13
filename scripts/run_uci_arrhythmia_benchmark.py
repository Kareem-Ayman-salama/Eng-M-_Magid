"""Benchmark UCI Arrhythmia prediction and classification tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ArrhythmiaConfig:
    """Configuration for UCI Arrhythmia experiments."""

    project_root: Path
    output_dir: Path
    random_state: int = 42


def load_arrhythmia_data(path: Path) -> pd.DataFrame:
    """Load UCI Arrhythmia data.

    Args:
        path: arrhythmia.data path.

    Returns:
        Loaded dataframe with named feature columns and target.
    """

    data = pd.read_csv(path, header=None, na_values="?")
    feature_columns = [f"feature_{index:03d}" for index in range(data.shape[1] - 1)]
    data.columns = [*feature_columns, "arrhythmia_class"]
    data["arrhythmia_class"] = data["arrhythmia_class"].astype(int)
    data["binary_target"] = (data["arrhythmia_class"] != 1).astype(int)
    return data


def build_preprocessor(x_data: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for mixed tabular Arrhythmia features."""

    categorical_columns = [
        column
        for column in x_data.columns
        if x_data[column].nunique(dropna=True) <= 8
        and pd.api.types.is_integer_dtype(x_data[column].dropna())
    ]
    numeric_columns = [column for column in x_data.columns if column not in categorical_columns]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ],
        verbose_feature_names_out=False,
    )


def optional_xgb(random_state: int, multiclass: bool) -> BaseEstimator | None:
    """Create XGBoost classifier if available."""

    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    params: dict[str, Any] = {
        "n_estimators": 250,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 3.0,
        "random_state": random_state,
        "n_jobs": -1,
        "eval_metric": "mlogloss" if multiclass else "logloss",
    }
    if multiclass:
        params["objective"] = "multi:softprob"
    else:
        params["objective"] = "binary:logistic"
    return XGBClassifier(**params)


def build_models(random_state: int, multiclass: bool = False) -> dict[str, BaseEstimator]:
    """Build candidate models."""

    models: dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    xgb = optional_xgb(random_state, multiclass)
    if xgb is not None:
        models["XGBoost"] = xgb

    if not multiclass:
        estimators = [(name.lower().replace(" ", "_"), clone(model)) for name, model in models.items()]
        models["Soft Voting Hybrid"] = VotingClassifier(
            estimators=estimators,
            voting="soft",
            n_jobs=-1,
        )
    return models


def evaluate_binary(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    models: dict[str, BaseEstimator],
    random_state: int,
) -> pd.DataFrame:
    """Evaluate binary normal-vs-arrhythmia prediction."""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        print(f"Binary CV: {name}")
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(x_data)),
                ("classifier", clone(model)),
            ]
        )
        scores = cross_validate(pipeline, x_data, y_data, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "task": "Normal vs Arrhythmia prediction",
                "model": name,
                "accuracy": scores["test_accuracy"].mean(),
                "balanced_accuracy": scores["test_balanced_accuracy"].mean(),
                "precision": scores["test_precision"].mean(),
                "recall": scores["test_recall"].mean(),
                "f1": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["roc_auc", "f1"], ascending=False)


def make_grouped_target(target: pd.Series) -> pd.Series:
    """Group rare arrhythmia classes for stable multiclass classification."""

    counts = target.value_counts()
    major_classes = counts[counts >= 20].index
    return target.where(target.isin(major_classes), other=99).astype(int)


def evaluate_multiclass(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    models: dict[str, BaseEstimator],
    random_state: int,
) -> pd.DataFrame:
    """Evaluate grouped multiclass arrhythmia classification."""

    label_encoder = LabelEncoder()
    encoded_y = pd.Series(label_encoder.fit_transform(y_data), index=y_data.index)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        print(f"Multiclass CV: {name}")
        fold_rows = []
        for train_index, test_index in cv.split(x_data, encoded_y):
            x_train = x_data.iloc[train_index]
            x_test = x_data.iloc[test_index]
            y_train = encoded_y.iloc[train_index]
            y_test = encoded_y.iloc[test_index]
            pipeline = Pipeline(
                [
                    ("preprocessor", build_preprocessor(x_train)),
                    ("classifier", clone(model)),
                ]
            )
            pipeline.fit(x_train, y_train)
            prediction = pipeline.predict(x_test)
            fold_rows.append(
                {
                    "accuracy": accuracy_score(y_test, prediction),
                    "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                    "macro_precision": precision_score(
                        y_test,
                        prediction,
                        average="macro",
                        zero_division=0,
                    ),
                    "macro_recall": recall_score(
                        y_test,
                        prediction,
                        average="macro",
                        zero_division=0,
                    ),
                    "macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
                    "weighted_f1": f1_score(
                        y_test,
                        prediction,
                        average="weighted",
                        zero_division=0,
                    ),
                }
            )
        fold_frame = pd.DataFrame(fold_rows)
        rows.append(
            {
                "task": "Grouped arrhythmia class classification",
                "model": name,
                **fold_frame.mean().to_dict(),
            }
        )
    return pd.DataFrame(rows).sort_values(["macro_f1", "balanced_accuracy"], ascending=False)


def evaluate_holdout(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    model: BaseEstimator,
    random_state: int,
) -> pd.DataFrame:
    """Evaluate a representative binary holdout split."""

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        stratify=y_data,
        random_state=random_state,
    )
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(x_train)),
            ("classifier", clone(model)),
        ]
    )
    pipeline.fit(x_train, y_train)
    probability = pipeline.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return pd.DataFrame(
        [
            {
                "task": "Binary holdout",
                "model": type(model).__name__,
                "accuracy": accuracy_score(y_test, prediction),
                "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                "precision": precision_score(y_test, prediction, zero_division=0),
                "recall": recall_score(y_test, prediction, zero_division=0),
                "f1": f1_score(y_test, prediction, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probability),
            }
        ]
    )


def main() -> None:
    """Run UCI Arrhythmia experiments."""

    project_root = Path.cwd()
    config = ArrhythmiaConfig(
        project_root=project_root,
        output_dir=project_root / "outputs" / "uci_arrhythmia",
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_arrhythmia_data(project_root / "data" / "uci_arrhythmia" / "arrhythmia.data")
    x_data = data.drop(columns=["arrhythmia_class", "binary_target"])
    y_binary = data["binary_target"]
    y_grouped = make_grouped_target(data["arrhythmia_class"])

    dataset_summary = pd.DataFrame(
        [
            {
                "rows": data.shape[0],
                "feature_count": x_data.shape[1],
                "original_class_count": data["arrhythmia_class"].nunique(),
                "grouped_class_count": y_grouped.nunique(),
                "missing_cells": int(data.isna().sum().sum()),
            }
        ]
    )
    class_distribution = (
        data["arrhythmia_class"].value_counts().sort_index().rename_axis("class").reset_index(name="count")
    )
    grouped_distribution = (
        y_grouped.value_counts().sort_index().rename_axis("grouped_class").reset_index(name="count")
    )

    binary_results = evaluate_binary(
        x_data,
        y_binary,
        build_models(config.random_state, multiclass=False),
        config.random_state,
    )
    multiclass_results = evaluate_multiclass(
        x_data,
        y_grouped,
        build_models(config.random_state, multiclass=True),
        config.random_state,
    )
    holdout_results = evaluate_holdout(
        x_data,
        y_binary,
        clone(build_models(config.random_state, multiclass=False)["Soft Voting Hybrid"]),
        config.random_state,
    )

    dataset_summary.to_csv(config.output_dir / "dataset_summary.csv", index=False)
    class_distribution.to_csv(config.output_dir / "class_distribution.csv", index=False)
    grouped_distribution.to_csv(config.output_dir / "grouped_class_distribution.csv", index=False)
    binary_results.to_csv(config.output_dir / "binary_prediction_results.csv", index=False)
    multiclass_results.to_csv(config.output_dir / "grouped_multiclass_results.csv", index=False)
    holdout_results.to_csv(config.output_dir / "binary_holdout_results.csv", index=False)

    print("Dataset summary")
    print(dataset_summary.to_string(index=False))
    print("Binary results")
    print(binary_results.to_string(index=False))
    print("Grouped multiclass results")
    print(multiclass_results.to_string(index=False))


if __name__ == "__main__":
    main()
