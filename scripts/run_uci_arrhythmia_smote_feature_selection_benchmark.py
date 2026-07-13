"""SMOTE and feature-selection benchmark for UCI Arrhythmia."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ResamplingConfig:
    """Configuration for resampling benchmark."""

    project_root: Path
    output_dir: Path
    random_state: int = 42
    folds: int = 5


def load_data(project_root: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Arrhythmia binary prediction data."""

    path = project_root / "data" / "uci_arrhythmia" / "arrhythmia.data"
    data = pd.read_csv(path, header=None, na_values="?")
    data.columns = [*[f"feature_{index:03d}" for index in range(data.shape[1] - 1)], "target"]
    data["target"] = data["target"].astype(int)
    return data.drop(columns=["target"]), (data["target"] != 1).astype(int)


def build_preprocessor(x_data: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing transformer."""

    categorical_columns = [
        column
        for column in x_data.columns
        if x_data[column].nunique(dropna=True) <= 8
        and pd.api.types.is_numeric_dtype(x_data[column])
    ]
    numeric_columns = [column for column in x_data.columns if column not in categorical_columns]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
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


def optional_catboost(random_state: int, **kwargs: Any) -> BaseEstimator | None:
    """Build CatBoost classifier if installed."""

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None
    params: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
    }
    params.update(kwargs)
    return CatBoostClassifier(**params)


def optional_xgboost(random_state: int, **kwargs: Any) -> BaseEstimator | None:
    """Build XGBoost classifier if installed."""

    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return XGBClassifier(**params)


def build_classifiers(random_state: int) -> dict[str, BaseEstimator]:
    """Create classifiers used after resampling and feature selection."""

    classifiers: dict[str, BaseEstimator] = {
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=900,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=700,
            max_depth=10,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    catboost = optional_catboost(
        random_state,
        iterations=520,
        depth=4,
        learning_rate=0.025,
        l2_leaf_reg=7,
        random_strength=0.8,
        bagging_temperature=0.5,
    )
    xgboost = optional_xgboost(
        random_state,
        n_estimators=480,
        max_depth=2,
        learning_rate=0.02,
        subsample=0.82,
        colsample_bytree=0.78,
        min_child_weight=2,
        reg_lambda=6.0,
    )
    if catboost is not None:
        classifiers["CatBoost"] = catboost
    if xgboost is not None:
        classifiers["XGBoost"] = xgboost
    voting_estimators = [
        (name.lower().replace(" ", "_"), clone(model))
        for name, model in classifiers.items()
        if name in {"CatBoost", "XGBoost", "Extra Trees"}
    ]
    if len(voting_estimators) >= 2:
        classifiers["SMOTE Feature Voting Hybrid"] = VotingClassifier(
            voting_estimators,
            voting="soft",
            weights=[3, 2, 2][: len(voting_estimators)],
            n_jobs=-1,
        )
    return classifiers


def build_selectors(random_state: int) -> dict[str, BaseEstimator]:
    """Create feature selectors."""

    return {
        "MI-64": SelectKBest(
            score_func=lambda x_values, y_values: mutual_info_classif(
                x_values,
                y_values,
                random_state=random_state,
            ),
            k=64,
        ),
        "MI-96": SelectKBest(
            score_func=lambda x_values, y_values: mutual_info_classif(
                x_values,
                y_values,
                random_state=random_state,
            ),
            k=96,
        ),
        "ANOVA-96": SelectKBest(score_func=f_classif, k=96),
        "ExtraTrees-SFM": SelectFromModel(
            ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
            ),
            threshold="median",
        ),
    }


def find_best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    """Optimize threshold using out-of-fold probabilities."""

    best_threshold = 0.5
    best_metrics: dict[str, float] = {}
    best_score = -np.inf
    for threshold in np.linspace(0.2, 0.8, 121):
        predictions = (probabilities >= threshold).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_true, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
            "precision": precision_score(y_true, predictions, zero_division=0),
            "recall": recall_score(y_true, predictions, zero_division=0),
            "f1": f1_score(y_true, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_true, probabilities),
        }
        score = metrics["balanced_accuracy"] + 0.15 * metrics["f1"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def evaluate_pipeline(
    name: str,
    pipeline: ImbPipeline,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    cv: StratifiedKFold,
) -> tuple[dict[str, float | str], dict[str, float | str]]:
    """Evaluate a single imbalanced-learning pipeline."""

    scores = cross_validate(
        pipeline,
        x_data,
        y_data,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        n_jobs=-1,
    )
    cv_row: dict[str, float | str] = {
        "model": name,
        **{
            key.replace("test_", ""): float(value.mean())
            for key, value in scores.items()
            if key.startswith("test_")
        },
    }
    probabilities = cross_val_predict(
        pipeline,
        x_data,
        y_data,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold, metrics = find_best_threshold(y_data, probabilities)
    threshold_row: dict[str, float | str] = {
        "model": name,
        "best_threshold": threshold,
        **metrics,
    }
    return cv_row, threshold_row


def main() -> None:
    """Run SMOTE and feature-selection benchmark."""

    root = Path.cwd()
    config = ResamplingConfig(
        project_root=root,
        output_dir=root / "outputs" / "uci_arrhythmia_resampling",
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    x_data, y_data = load_data(config.project_root)
    preprocessor = build_preprocessor(x_data)
    classifiers = build_classifiers(config.random_state)
    selectors = build_selectors(config.random_state)
    samplers = {
        "SMOTE": SMOTE(random_state=config.random_state, k_neighbors=5),
        "ADASYN": ADASYN(random_state=config.random_state, n_neighbors=5),
    }
    cv = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.random_state)
    cv_rows: list[dict[str, float | str]] = []
    threshold_rows: list[dict[str, float | str]] = []

    for sampler_name, sampler in samplers.items():
        for selector_name, selector in selectors.items():
            for classifier_name, classifier in classifiers.items():
                name = f"{sampler_name} + {selector_name} + {classifier_name}"
                print(f"Evaluating {name}")
                pipeline = ImbPipeline(
                    [
                        ("preprocessor", clone(preprocessor)),
                        ("selector", clone(selector)),
                        ("sampler", clone(sampler)),
                        ("classifier", clone(classifier)),
                    ]
                )
                try:
                    cv_row, threshold_row = evaluate_pipeline(name, pipeline, x_data, y_data, cv)
                except ValueError as exc:
                    print(f"Skipping {name}: {exc}")
                    continue
                cv_rows.append(cv_row)
                threshold_rows.append(threshold_row)

    cv_results = pd.DataFrame(cv_rows).sort_values(["roc_auc", "f1"], ascending=False)
    threshold_results = pd.DataFrame(threshold_rows).sort_values(
        ["balanced_accuracy", "f1"],
        ascending=False,
    )
    cv_results.to_csv(config.output_dir / "resampling_feature_selection_cv_results.csv", index=False)
    threshold_results.to_csv(
        config.output_dir / "resampling_feature_selection_threshold_results.csv",
        index=False,
    )
    print(cv_results.head(15).to_string(index=False))
    print(threshold_results.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
