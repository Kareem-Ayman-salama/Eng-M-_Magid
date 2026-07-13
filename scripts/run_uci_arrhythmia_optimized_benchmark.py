"""Optimized UCI Arrhythmia benchmark with threshold tuning and ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuration for optimized UCI Arrhythmia experiments.

    Attributes:
        project_root: Project root directory.
        output_dir: Directory where optimized outputs are saved.
        random_state: Reproducibility seed.
        folds: Number of stratified folds.
    """

    project_root: Path
    output_dir: Path
    random_state: int = 42
    folds: int = 5


def load_data(project_root: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load features and binary target from UCI Arrhythmia dataset."""

    path = project_root / "data" / "uci_arrhythmia" / "arrhythmia.data"
    data = pd.read_csv(path, header=None, na_values="?")
    data.columns = [*[f"feature_{index:03d}" for index in range(data.shape[1] - 1)], "target"]
    data["target"] = data["target"].astype(int)
    y_data = (data["target"] != 1).astype(int)
    x_data = data.drop(columns=["target"])
    return x_data, y_data


def build_preprocessor(x_data: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for mixed tabular ECG features."""

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
    """Create a CatBoost model when the package is available."""

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None
    params: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
    }
    params.update(kwargs)
    return CatBoostClassifier(**params)


def optional_xgboost(random_state: int, **kwargs: Any) -> BaseEstimator | None:
    """Create an XGBoost model when the package is available."""

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


def optional_lightgbm(random_state: int, **kwargs: Any) -> BaseEstimator | None:
    """Create a LightGBM model when the package is available."""

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None
    params: dict[str, Any] = {
        "objective": "binary",
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(kwargs)
    return LGBMClassifier(**params)


def build_candidate_models(random_state: int) -> dict[str, BaseEstimator]:
    """Build tuned candidate models and hybrid ensembles."""

    candidates: dict[str, BaseEstimator] = {
        "Logistic L1": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.08,
            class_weight="balanced",
            random_state=random_state,
            max_iter=3000,
        ),
        "Logistic L2": LogisticRegression(
            C=0.12,
            class_weight="balanced",
            random_state=random_state,
            max_iter=3000,
        ),
        "SVC Tuned RBF": SVC(
            C=0.65,
            gamma=0.004,
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest Tuned": RandomForestClassifier(
            n_estimators=900,
            max_depth=9,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Extra Trees Tuned": ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    optional_models = {
        "CatBoost Conservative": optional_catboost(
            random_state,
            iterations=650,
            depth=3,
            learning_rate=0.018,
            l2_leaf_reg=10,
            random_strength=1.0,
            bagging_temperature=0.4,
        ),
        "CatBoost Deep": optional_catboost(
            random_state,
            iterations=450,
            depth=5,
            learning_rate=0.025,
            l2_leaf_reg=8,
            random_strength=0.8,
            bagging_temperature=0.5,
        ),
        "XGBoost Regularized": optional_xgboost(
            random_state,
            n_estimators=520,
            max_depth=2,
            learning_rate=0.018,
            subsample=0.78,
            colsample_bytree=0.75,
            min_child_weight=3,
            reg_alpha=0.15,
            reg_lambda=8.0,
            scale_pos_weight=0.84,
        ),
        "LightGBM Regularized": optional_lightgbm(
            random_state,
            n_estimators=520,
            learning_rate=0.018,
            num_leaves=9,
            max_depth=4,
            min_child_samples=10,
            subsample=0.82,
            colsample_bytree=0.78,
            reg_alpha=0.1,
            reg_lambda=8.0,
        ),
    }
    for name, model in optional_models.items():
        if model is not None:
            candidates[name] = model

    top_names = [
        "CatBoost Conservative",
        "CatBoost Deep",
        "XGBoost Regularized",
        "LightGBM Regularized",
        "Extra Trees Tuned",
        "SVC Tuned RBF",
    ]
    top_estimators = [
        (name.lower().replace(" ", "_"), clone(candidates[name]))
        for name in top_names
        if name in candidates
    ]
    if top_estimators:
        candidates["Advanced Soft Voting Hybrid"] = VotingClassifier(
            top_estimators,
            voting="soft",
            weights=[3, 3, 2, 2, 2, 1][: len(top_estimators)],
            n_jobs=-1,
        )
        candidates["Advanced Stacking Hybrid"] = StackingClassifier(
            estimators=top_estimators,
            final_estimator=LogisticRegression(
                C=0.2,
                class_weight="balanced",
                random_state=random_state,
                max_iter=2000,
            ),
            cv=5,
            stack_method="predict_proba",
            n_jobs=-1,
        )

    xgb = candidates.get("XGBoost Regularized")
    if xgb is not None:
        candidates["MI-60 + XGBoost Regularized"] = Pipeline(
            [
                (
                    "selector",
                    SelectKBest(
                        score_func=lambda x_values, y_values: mutual_info_classif(
                            x_values,
                            y_values,
                            random_state=random_state,
                        ),
                        k=60,
                    ),
                ),
                ("classifier", clone(xgb)),
            ]
        )
        candidates["ANOVA-70 + XGBoost Regularized"] = Pipeline(
            [("selector", SelectKBest(f_classif, k=70)), ("classifier", clone(xgb))]
        )
    return candidates


def find_best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    """Find threshold maximizing balanced accuracy, then F1."""

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


def evaluate_candidates(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    models: dict[str, BaseEstimator],
    config: OptimizationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate candidates with CV and optimized out-of-fold threshold."""

    cv = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.random_state)
    preprocessor = build_preprocessor(x_data)
    cv_rows: list[dict[str, float | str]] = []
    threshold_rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        print(f"Optimizing {name}")
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clone(model))])
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
        cv_rows.append(
            {
                "model": name,
                **{
                    key.replace("test_", ""): float(value.mean())
                    for key, value in scores.items()
                    if key.startswith("test_")
                },
            }
        )
        probabilities = cross_val_predict(
            pipeline,
            x_data,
            y_data,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        threshold, metrics = find_best_threshold(y_data, probabilities)
        threshold_rows.append({"model": name, "best_threshold": threshold, **metrics})
    cv_results = pd.DataFrame(cv_rows).sort_values(["roc_auc", "f1"], ascending=False)
    threshold_results = pd.DataFrame(threshold_rows).sort_values(
        ["balanced_accuracy", "f1"],
        ascending=False,
    )
    return cv_results, threshold_results


def main() -> None:
    """Run optimized UCI Arrhythmia binary benchmark."""

    project_root = Path.cwd()
    config = OptimizationConfig(
        project_root=project_root,
        output_dir=project_root / "outputs" / "uci_arrhythmia_optimized",
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    x_data, y_data = load_data(config.project_root)
    models = build_candidate_models(config.random_state)
    cv_results, threshold_results = evaluate_candidates(x_data, y_data, models, config)
    cv_results.to_csv(config.output_dir / "optimized_binary_cv_results.csv", index=False)
    threshold_results.to_csv(config.output_dir / "optimized_binary_threshold_results.csv", index=False)
    print(cv_results.to_string(index=False))
    print(threshold_results.to_string(index=False))


if __name__ == "__main__":
    main()
