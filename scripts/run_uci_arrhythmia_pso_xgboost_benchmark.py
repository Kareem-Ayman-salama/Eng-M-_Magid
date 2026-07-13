"""PSO-optimized XGBoost benchmark for UCI Arrhythmia."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
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
from xgboost import XGBClassifier


@dataclass(frozen=True)
class PsoConfig:
    """Configuration for PSO-XGBoost optimization.

    Attributes:
        project_root: Project root directory.
        output_dir: Output directory.
        random_state: Reproducibility seed.
        prefilter_features: Number of MI-ranked features used by PSO.
        particles: Number of PSO particles.
        iterations: Number of PSO iterations.
        inner_folds: CV folds used inside PSO fitness.
        final_folds: CV folds used for final evaluation.
    """

    project_root: Path
    output_dir: Path
    random_state: int = 42
    prefilter_features: int = 120
    particles: int = 12
    iterations: int = 10
    inner_folds: int = 3
    final_folds: int = 5


def load_data(project_root: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Arrhythmia features and binary target."""

    path = project_root / "data" / "uci_arrhythmia" / "arrhythmia.data"
    data = pd.read_csv(path, header=None, na_values="?")
    data.columns = [*[f"feature_{index:03d}" for index in range(data.shape[1] - 1)], "target"]
    data["target"] = data["target"].astype(int)
    return data.drop(columns=["target"]), (data["target"] != 1).astype(int)


def rank_features(x_data: pd.DataFrame, y_data: pd.Series, config: PsoConfig) -> list[str]:
    """Rank features with mutual information after median imputation."""

    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(x_data)
    scores = mutual_info_classif(x_imputed, y_data, random_state=config.random_state)
    ranking = (
        pd.DataFrame({"feature": x_data.columns, "mi_score": scores})
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )
    ranking.to_csv(config.output_dir / "pso_feature_prefilter_ranking.csv", index=False)
    return ranking.head(config.prefilter_features)["feature"].tolist()


def map_particle_to_params(values: np.ndarray) -> dict[str, Any]:
    """Map normalized PSO dimensions to XGBoost hyperparameters."""

    return {
        "n_estimators": int(140 + values[0] * 360),
        "max_depth": int(2 + values[1] * 4),
        "learning_rate": float(0.01 + values[2] * 0.09),
        "subsample": float(0.65 + values[3] * 0.35),
        "colsample_bytree": float(0.65 + values[4] * 0.35),
        "min_child_weight": float(1.0 + values[5] * 5.0),
        "reg_lambda": float(1.0 + values[6] * 10.0),
        "reg_alpha": float(values[7] * 0.8),
    }


def build_xgboost(params: dict[str, Any], random_state: int) -> XGBClassifier:
    """Build an XGBoost classifier."""

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )


def selected_feature_mask(position: np.ndarray, feature_count: int) -> np.ndarray:
    """Convert particle position to a boolean feature mask."""

    mask = position[:feature_count] > 0.52
    if mask.sum() < 10:
        strongest = np.argsort(position[:feature_count])[-10:]
        mask = np.zeros(feature_count, dtype=bool)
        mask[strongest] = True
    if mask.sum() > 80:
        strongest = np.argsort(position[:feature_count])[-80:]
        mask = np.zeros(feature_count, dtype=bool)
        mask[strongest] = True
    return mask


def evaluate_candidate(
    position: np.ndarray,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    candidate_features: list[str],
    config: PsoConfig,
    cache: dict[str, float],
) -> float:
    """Evaluate one PSO particle and return fitness."""

    feature_count = len(candidate_features)
    mask = selected_feature_mask(position, feature_count)
    params_values = np.clip(position[feature_count:], 0.0, 1.0)
    params = map_particle_to_params(params_values)
    cache_key = json.dumps(
        {
            "features": np.where(mask)[0].tolist(),
            "params": {key: round(float(value), 4) for key, value in params.items()},
        },
        sort_keys=True,
    )
    if cache_key in cache:
        return cache[cache_key]

    selected_columns = [feature for feature, keep in zip(candidate_features, mask) if keep]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", build_xgboost(params, config.random_state)),
        ]
    )
    cv = StratifiedKFold(n_splits=config.inner_folds, shuffle=True, random_state=config.random_state)
    scores = cross_validate(
        model,
        x_data[selected_columns],
        y_data,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        n_jobs=-1,
    )
    balanced_accuracy = float(scores["test_balanced_accuracy"].mean())
    f1 = float(scores["test_f1"].mean())
    roc_auc = float(scores["test_roc_auc"].mean())
    penalty = 0.0004 * len(selected_columns)
    fitness = balanced_accuracy + (0.10 * f1) + (0.04 * roc_auc) - penalty
    cache[cache_key] = fitness
    return fitness


def run_pso(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    candidate_features: list[str],
    config: PsoConfig,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Run a compact PSO search."""

    rng = np.random.default_rng(config.random_state)
    dimensions = len(candidate_features) + 8
    positions = rng.uniform(0.0, 1.0, size=(config.particles, dimensions))
    velocities = rng.uniform(-0.12, 0.12, size=(config.particles, dimensions))
    personal_best = positions.copy()
    personal_scores = np.full(config.particles, -np.inf)
    global_best = positions[0].copy()
    global_score = -np.inf
    history: list[dict[str, float]] = []
    cache: dict[str, float] = {}

    for iteration in range(1, config.iterations + 1):
        for index in range(config.particles):
            score = evaluate_candidate(
                positions[index],
                x_data,
                y_data,
                candidate_features,
                config,
                cache,
            )
            if score > personal_scores[index]:
                personal_scores[index] = score
                personal_best[index] = positions[index].copy()
            if score > global_score:
                global_score = score
                global_best = positions[index].copy()

        inertia = 0.65 - (0.25 * iteration / config.iterations)
        cognitive = 1.35
        social = 1.35
        r1 = rng.uniform(0.0, 1.0, size=(config.particles, dimensions))
        r2 = rng.uniform(0.0, 1.0, size=(config.particles, dimensions))
        velocities = (
            inertia * velocities
            + cognitive * r1 * (personal_best - positions)
            + social * r2 * (global_best - positions)
        )
        velocities = np.clip(velocities, -0.25, 0.25)
        positions = np.clip(positions + velocities, 0.0, 1.0)
        selected_count = int(selected_feature_mask(global_best, len(candidate_features)).sum())
        history.append(
            {
                "iteration": iteration,
                "best_fitness": global_score,
                "selected_features": selected_count,
                "cache_size": len(cache),
            }
        )
        print(
            f"Iteration {iteration}/{config.iterations} "
            f"best={global_score:.4f} features={selected_count} cache={len(cache)}"
        )
    return global_best, history


def find_best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    """Find threshold maximizing balanced accuracy with F1 tie-support."""

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


def final_evaluation(
    best_position: np.ndarray,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    candidate_features: list[str],
    config: PsoConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate best PSO-XGBoost candidate with final CV and threshold search."""

    feature_count = len(candidate_features)
    mask = selected_feature_mask(best_position, feature_count)
    selected_columns = [feature for feature, keep in zip(candidate_features, mask) if keep]
    params = map_particle_to_params(np.clip(best_position[feature_count:], 0.0, 1.0))
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", build_xgboost(params, config.random_state)),
        ]
    )
    cv = StratifiedKFold(n_splits=config.final_folds, shuffle=True, random_state=config.random_state)
    scores = cross_validate(
        model,
        x_data[selected_columns],
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
    cv_row = {
        "model": "PSO-XGBoost",
        "selected_features": len(selected_columns),
        **{
            key.replace("test_", ""): float(value.mean())
            for key, value in scores.items()
            if key.startswith("test_")
        },
    }
    probabilities = cross_val_predict(
        model,
        x_data[selected_columns],
        y_data,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold, metrics = find_best_threshold(y_data, probabilities)
    threshold_row = {
        "model": "PSO-XGBoost",
        "best_threshold": threshold,
        "selected_features": len(selected_columns),
        **metrics,
    }
    selected_features = pd.DataFrame({"feature": selected_columns})
    params_frame = pd.DataFrame([params])
    return pd.DataFrame([cv_row]), pd.DataFrame([threshold_row]), pd.concat(
        [selected_features, params_frame.reindex(selected_features.index)],
        axis=1,
    )


def save_comparison_plot(comparison: pd.DataFrame, output_path: Path) -> None:
    """Save comparison chart."""

    frame = comparison.sort_values("accuracy", ascending=True)
    plt.figure(figsize=(8.5, 4.2))
    bars = plt.barh(frame["model"], frame["accuracy"] * 100, color="#7c3aed")
    plt.xlabel("Accuracy (%)")
    plt.title("PSO-XGBoost vs Current and Published Results")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    """Run PSO-XGBoost benchmark."""

    root = Path.cwd()
    config = PsoConfig(project_root=root, output_dir=root / "outputs" / "uci_arrhythmia_pso_xgboost")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    x_data, y_data = load_data(config.project_root)
    candidate_features = rank_features(x_data, y_data, config)
    best_position, history = run_pso(x_data, y_data, candidate_features, config)
    cv_results, threshold_results, selected_features = final_evaluation(
        best_position,
        x_data,
        y_data,
        candidate_features,
        config,
    )
    history_frame = pd.DataFrame(history)
    baseline = pd.read_csv(root / "outputs" / "uci_arrhythmia_optimized" / "optimized_binary_threshold_results.csv")
    best_current = baseline.sort_values(["balanced_accuracy", "f1"], ascending=False).iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "model": "Current CatBoost Deep",
                "accuracy": float(best_current["accuracy"]),
                "balanced_accuracy": float(best_current["balanced_accuracy"]),
                "f1": float(best_current["f1"]),
                "roc_auc": float(best_current["roc_auc"]),
                "source": "our previous CV threshold result",
            },
            {
                "model": "PSO-XGBoost",
                "accuracy": float(threshold_results.loc[0, "accuracy"]),
                "balanced_accuracy": float(threshold_results.loc[0, "balanced_accuracy"]),
                "f1": float(threshold_results.loc[0, "f1"]),
                "roc_auc": float(threshold_results.loc[0, "roc_auc"]),
                "source": "our new PSO experiment",
            },
            {
                "model": "Published PSO-XGBoost 2025",
                "accuracy": 0.9524,
                "balanced_accuracy": 0.9481,
                "f1": 0.9630,
                "roc_auc": np.nan,
                "source": "Dhanka and Maini 2025",
            },
            {
                "model": "Published MBAR+SMOTE+CatBoost 2022",
                "accuracy": 0.8633,
                "balanced_accuracy": np.nan,
                "f1": np.nan,
                "roc_auc": np.nan,
                "source": "A.P. and David 2022",
            },
        ]
    )
    cv_results.to_csv(config.output_dir / "pso_xgboost_cv_results.csv", index=False)
    threshold_results.to_csv(config.output_dir / "pso_xgboost_threshold_results.csv", index=False)
    selected_features.to_csv(config.output_dir / "pso_xgboost_selected_features_and_params.csv", index=False)
    history_frame.to_csv(config.output_dir / "pso_xgboost_search_history.csv", index=False)
    comparison.to_csv(config.output_dir / "pso_xgboost_comparison.csv", index=False)
    save_comparison_plot(comparison, config.output_dir / "pso_xgboost_comparison.png")
    print(cv_results.to_string(index=False))
    print(threshold_results.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
