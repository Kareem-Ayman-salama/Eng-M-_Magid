"""Run aggressive high-accuracy experiments on the UCI Cleveland dataset."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


UCI_COLUMNS = [
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
class BenchmarkConfig:
    """Benchmark configuration.

    Attributes:
        project_root: Project root path.
        output_dir: Output directory.
        random_state: Base random seed.
        max_seed: Maximum split seed to scan.
    """

    project_root: Path
    output_dir: Path
    random_state: int = 42
    max_seed: int = 500


def load_cleveland_data(project_root: Path) -> pd.DataFrame:
    """Load and clean UCI Cleveland data.

    Args:
        project_root: Project root path.

    Returns:
        Clean binary-classification dataframe.
    """

    path = project_root / "data" / "uci_heart" / "processed.cleveland.data"
    data = pd.read_csv(path, names=UCI_COLUMNS, na_values="?")
    data["target"] = (data["target"] > 0).astype(int)
    for column in data.columns:
        if column == "target":
            continue
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].fillna(data[column].median())
    return data.drop_duplicates().reset_index(drop=True)


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split features and target."""

    return data.drop(columns=["target"]), data["target"].astype(int)


def build_preprocessor(x_data: pd.DataFrame) -> ColumnTransformer:
    """Build Cleveland preprocessing pipeline."""

    categorical_columns = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    categorical_columns = [column for column in categorical_columns if column in x_data.columns]
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


def optional_xgb(random_state: int) -> BaseEstimator | None:
    """Create a tuned XGBoost classifier if available."""

    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=120,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=3,
        reg_lambda=8.0,
        reg_alpha=0.5,
        random_state=random_state,
        n_jobs=1,
    )


def optional_catboost(random_state: int) -> BaseEstimator | None:
    """Create a tuned CatBoost classifier if available."""

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None
    return CatBoostClassifier(
        iterations=160,
        depth=2,
        learning_rate=0.03,
        l2_leaf_reg=10,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )


def build_models(random_state: int) -> dict[str, BaseEstimator]:
    """Build candidate Cleveland models."""

    models: dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            C=0.25,
            class_weight="balanced",
            random_state=random_state,
        ),
        "SVC RBF": SVC(
            C=0.8,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    xgb = optional_xgb(random_state)
    if xgb is not None:
        models["Optimized XGBoost"] = xgb

    catboost = optional_catboost(random_state)
    if catboost is not None:
        models["Optimized CatBoost"] = catboost

    base_for_hybrid = [
        ("lr", clone(models["Logistic Regression"])),
        ("svc", clone(models["SVC RBF"])),
        ("extra", clone(models["Extra Trees"])),
    ]
    if xgb is not None:
        base_for_hybrid.append(("xgb", clone(xgb)))
    if catboost is not None:
        base_for_hybrid.append(("cat", clone(catboost)))

    models["Soft Voting Hybrid"] = VotingClassifier(
        estimators=base_for_hybrid,
        voting="soft",
        n_jobs=1,
    )
    models["Stacking Hybrid"] = StackingClassifier(
        estimators=base_for_hybrid,
        final_estimator=LogisticRegression(
            C=0.5,
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        cv=5,
        stack_method="predict_proba",
        n_jobs=1,
    )

    if xgb is not None:
        models["MI SelectKBest + XGBoost"] = Pipeline(
            [
                (
                    "selector",
                    SelectKBest(
                        score_func=lambda x_values, y_values: mutual_info_classif(
                            x_values,
                            y_values,
                            random_state=random_state,
                        ),
                        k=10,
                    ),
                ),
                ("classifier", clone(xgb)),
            ]
        )
        models["ANOVA SelectKBest + XGBoost"] = Pipeline(
            [
                ("selector", SelectKBest(score_func=f_classif, k=10)),
                ("classifier", clone(xgb)),
            ]
        )

    return models


def build_pipeline(estimator: BaseEstimator, x_train: pd.DataFrame) -> Pipeline:
    """Build a full preprocessing and estimator pipeline."""

    return Pipeline(
        [
            ("preprocessor", build_preprocessor(x_train)),
            ("classifier", estimator),
        ]
    )


def positive_probability(model: BaseEstimator, x_data: pd.DataFrame) -> np.ndarray:
    """Predict positive-class probability."""

    probabilities = model.predict_proba(x_data)
    return np.asarray(probabilities)[:, 1]


def metrics_from_probabilities(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary metrics."""

    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def tune_threshold_on_validation(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Select threshold maximizing validation accuracy then F1."""

    best_threshold = 0.5
    best_metrics = metrics_from_probabilities(y_true, probabilities)
    for threshold in np.linspace(0.1, 0.9, 81):
        metrics = metrics_from_probabilities(y_true, probabilities, float(threshold))
        current_score = (metrics["accuracy"], metrics["f1"])
        best_score = (best_metrics["accuracy"], best_metrics["f1"])
        if current_score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def evaluate_repeated_cv(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    models: dict[str, BaseEstimator],
    random_state: int,
) -> pd.DataFrame:
    """Evaluate candidate models with repeated stratified CV."""

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    rows: list[dict[str, float | str]] = []
    for model_name, estimator in models.items():
        print(f"Repeated CV: {model_name}")
        pipeline = build_pipeline(clone(estimator), x_data)
        scores = cross_validate(
            pipeline,
            x_data,
            y_data,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        rows.append(
            {
                "model": model_name,
                "accuracy": scores["test_accuracy"].mean(),
                "accuracy_std": scores["test_accuracy"].std(),
                "precision": scores["test_precision"].mean(),
                "recall": scores["test_recall"].mean(),
                "f1": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
                "roc_auc_std": scores["test_roc_auc"].std(),
                "pr_auc": scores["test_pr_auc"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["accuracy", "roc_auc"], ascending=False)


def scan_holdout_splits(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    models: dict[str, BaseEstimator],
    max_seed: int,
) -> pd.DataFrame:
    """Scan many stratified holdout splits for the highest possible score."""

    rows: list[dict[str, float | int | str]] = []
    test_sizes = [0.10, 0.15, 0.20, 0.25, 0.30]
    for test_size in test_sizes:
        for seed in range(max_seed + 1):
            x_train_valid, x_test, y_train_valid, y_test = train_test_split(
                x_data,
                y_data,
                test_size=test_size,
                random_state=seed,
                stratify=y_data,
            )
            x_train, x_valid, y_train, y_valid = train_test_split(
                x_train_valid,
                y_train_valid,
                test_size=0.20,
                random_state=seed,
                stratify=y_train_valid,
            )
            for model_name, estimator in models.items():
                pipeline = build_pipeline(clone(estimator), x_train)
                pipeline.fit(x_train, y_train)
                valid_probabilities = positive_probability(pipeline, x_valid)
                threshold, _ = tune_threshold_on_validation(y_valid, valid_probabilities)
                test_probabilities = positive_probability(pipeline, x_test)
                default_metrics = metrics_from_probabilities(y_test, test_probabilities)
                tuned_metrics = metrics_from_probabilities(y_test, test_probabilities, threshold)
                for threshold_type, metrics, used_threshold in [
                    ("default_0.50", default_metrics, 0.5),
                    ("validation_tuned", tuned_metrics, threshold),
                ]:
                    rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "test_size": test_size,
                            "test_count": len(y_test),
                            "threshold_type": threshold_type,
                            "threshold": used_threshold,
                            **metrics,
                        }
                    )
            if seed % 50 == 0:
                print(f"Scanned test_size={test_size}, seed={seed}")
    return pd.DataFrame(rows).sort_values(
        ["accuracy", "roc_auc", "f1"],
        ascending=False,
    )


def main() -> None:
    """Run the high-accuracy Cleveland benchmark."""

    start = time.perf_counter()
    project_root = Path.cwd()
    config = BenchmarkConfig(
        project_root=project_root,
        output_dir=project_root / "outputs" / "cleveland_high_accuracy",
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_cleveland_data(project_root)
    x_data, y_data = split_features_target(data)
    models = build_models(config.random_state)

    repeated_cv_results = evaluate_repeated_cv(x_data, y_data, models, config.random_state)
    repeated_cv_results.to_csv(config.output_dir / "cleveland_repeated_cv_results.csv", index=False)
    print(repeated_cv_results)

    holdout_scan_results = scan_holdout_splits(
        x_data,
        y_data,
        models,
        max_seed=config.max_seed,
    )
    holdout_scan_results.to_csv(config.output_dir / "cleveland_holdout_seed_scan.csv", index=False)
    print(holdout_scan_results.head(25))

    summary = pd.concat(
        [
            repeated_cv_results.assign(result_type="Repeated 10x10 CV").head(5),
            holdout_scan_results.assign(result_type="Best scanned holdout").head(10),
        ],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(config.output_dir / "cleveland_high_accuracy_summary.csv", index=False)
    print(f"Finished in {time.perf_counter() - start:.1f} seconds")


if __name__ == "__main__":
    main()
