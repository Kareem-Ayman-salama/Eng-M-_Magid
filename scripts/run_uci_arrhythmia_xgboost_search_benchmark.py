"""Randomized feature-selection and XGBoost search for UCI Arrhythmia."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


@dataclass(frozen=True)
class SearchConfig:
    """Search configuration."""

    project_root: Path
    output_dir: Path
    random_state: int = 42
    iterations: int = 80
    folds: int = 5


def load_data(project_root: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Arrhythmia binary data."""

    path = project_root / "data" / "uci_arrhythmia" / "arrhythmia.data"
    data = pd.read_csv(path, header=None, na_values="?")
    data.columns = [*[f"feature_{index:03d}" for index in range(data.shape[1] - 1)], "target"]
    data["target"] = data["target"].astype(int)
    return data.drop(columns=["target"]), (data["target"] != 1).astype(int)


def build_pipeline(random_state: int) -> Pipeline:
    """Build searchable pipeline."""

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(score_func=f_classif, k=80)),
            ("classifier", classifier),
        ]
    )


def find_best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    """Find best probability threshold."""

    best_threshold = 0.5
    best_score = -np.inf
    best_metrics: dict[str, float] = {}
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


def save_search_plot(results: pd.DataFrame, output_path: Path) -> None:
    """Save top search candidates plot."""

    frame = results.sort_values("mean_test_balanced_accuracy").tail(12)
    labels = [f"#{rank}" for rank in frame["rank_test_balanced_accuracy"]]
    plt.figure(figsize=(8.5, 4.4))
    bars = plt.barh(labels, frame["mean_test_balanced_accuracy"] * 100, color="#0891b2")
    plt.xlabel("Balanced Accuracy (%)")
    plt.title("Top XGBoost Search Candidates")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.25, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    """Run randomized XGBoost search."""

    root = Path.cwd()
    config = SearchConfig(root, root / "outputs" / "uci_arrhythmia_xgboost_search")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    x_data, y_data = load_data(config.project_root)
    cv = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.random_state)
    pipeline = build_pipeline(config.random_state)
    param_distributions = [
        {
            "selector__score_func": [f_classif],
            "selector__k": randint(30, 180),
            "classifier__n_estimators": randint(120, 650),
            "classifier__max_depth": randint(2, 7),
            "classifier__learning_rate": uniform(0.01, 0.09),
            "classifier__subsample": uniform(0.65, 0.35),
            "classifier__colsample_bytree": uniform(0.65, 0.35),
            "classifier__min_child_weight": uniform(1.0, 6.0),
            "classifier__reg_lambda": uniform(1.0, 12.0),
            "classifier__reg_alpha": uniform(0.0, 1.0),
        },
        {
            "selector__score_func": [
                lambda x_values, y_values: mutual_info_classif(
                    x_values,
                    y_values,
                    random_state=config.random_state,
                )
            ],
            "selector__k": randint(30, 180),
            "classifier__n_estimators": randint(120, 650),
            "classifier__max_depth": randint(2, 7),
            "classifier__learning_rate": uniform(0.01, 0.09),
            "classifier__subsample": uniform(0.65, 0.35),
            "classifier__colsample_bytree": uniform(0.65, 0.35),
            "classifier__min_child_weight": uniform(1.0, 6.0),
            "classifier__reg_lambda": uniform(1.0, 12.0),
            "classifier__reg_alpha": uniform(0.0, 1.0),
        },
    ]
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=config.iterations,
        scoring={
            "balanced_accuracy": "balanced_accuracy",
            "accuracy": "accuracy",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        refit="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=config.random_state,
        verbose=1,
        return_train_score=True,
    )
    search.fit(x_data, y_data)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_balanced_accuracy")
    results.to_csv(config.output_dir / "xgboost_search_cv_results.csv", index=False)
    save_search_plot(results, config.output_dir / "xgboost_search_top_candidates.png")

    best_estimator = search.best_estimator_
    scores = cross_validate(
        best_estimator,
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
    cv_summary = {
        "model": "RandomizedSearch-XGBoost",
        **{
            key.replace("test_", ""): float(value.mean())
            for key, value in scores.items()
            if key.startswith("test_")
        },
    }
    probabilities = cross_val_predict(
        best_estimator,
        x_data,
        y_data,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold, metrics = find_best_threshold(y_data, probabilities)
    threshold_summary = {"model": "RandomizedSearch-XGBoost", "best_threshold": threshold, **metrics}
    pd.DataFrame([cv_summary]).to_csv(config.output_dir / "xgboost_search_final_cv.csv", index=False)
    pd.DataFrame([threshold_summary]).to_csv(config.output_dir / "xgboost_search_threshold.csv", index=False)
    pd.DataFrame([search.best_params_]).to_csv(config.output_dir / "xgboost_search_best_params.csv", index=False)
    print(pd.DataFrame([cv_summary]).to_string(index=False))
    print(pd.DataFrame([threshold_summary]).to_string(index=False))
    print(search.best_params_)


if __name__ == "__main__":
    main()
