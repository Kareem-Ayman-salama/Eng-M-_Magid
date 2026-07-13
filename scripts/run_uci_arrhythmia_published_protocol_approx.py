"""Approximate the published PSO-XGBoost protocol on UCI Arrhythmia."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


@dataclass(frozen=True)
class PublishedProtocolConfig:
    """Configuration for the published-protocol approximation.

    Attributes:
        project_root: Project root directory.
        output_dir: Directory for result files.
        random_state: Random seed.
        particles: PSO particle count.
        iterations: PSO iteration count.
        folds: Stratified K-Fold count.
    """

    project_root: Path
    output_dir: Path
    random_state: int = 42
    particles: int = 18
    iterations: int = 16
    folds: int = 10


def load_data(project_root: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI Arrhythmia data as binary normal-vs-arrhythmia."""

    path = project_root / "data" / "uci_arrhythmia" / "arrhythmia.data"
    data = pd.read_csv(path, header=None, na_values="?")
    data.columns = [*[f"feature_{index:03d}" for index in range(data.shape[1] - 1)], "target"]
    data["target"] = data["target"].astype(int)
    return data.drop(columns=["target"]), (data["target"] != 1).astype(int)


def params_from_particle(values: np.ndarray) -> dict[str, Any]:
    """Map normalized particle values to XGBoost parameters."""

    return {
        "n_estimators": int(100 + values[0] * 700),
        "max_depth": int(2 + values[1] * 7),
        "learning_rate": float(0.005 + values[2] * 0.145),
        "subsample": float(0.55 + values[3] * 0.45),
        "colsample_bytree": float(0.55 + values[4] * 0.45),
        "min_child_weight": float(0.5 + values[5] * 8.0),
        "gamma": float(values[6] * 3.0),
        "reg_lambda": float(0.2 + values[7] * 15.0),
        "reg_alpha": float(values[8] * 2.0),
    }


def build_model(params: dict[str, Any], random_state: int) -> Pipeline:
    """Build imputer + XGBoost pipeline."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                    tree_method="hist",
                    **params,
                ),
            ),
        ]
    )


def evaluate_particle(
    particle: np.ndarray,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    config: PublishedProtocolConfig,
    cache: dict[str, float],
) -> float:
    """Evaluate particle fitness using accuracy-focused Stratified K-Fold."""

    params = params_from_particle(np.clip(particle, 0.0, 1.0))
    key = str({name: round(float(value), 5) for name, value in params.items()})
    if key in cache:
        return cache[key]
    model = build_model(params, config.random_state)
    cv = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.random_state)
    scores = cross_validate(
        model,
        x_data,
        y_data,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1": "f1", "balanced_accuracy": "balanced_accuracy"},
        n_jobs=-1,
    )
    accuracy = float(scores["test_accuracy"].mean())
    f1 = float(scores["test_f1"].mean())
    balanced = float(scores["test_balanced_accuracy"].mean())
    fitness = accuracy + (0.05 * f1) + (0.03 * balanced)
    cache[key] = fitness
    return fitness


def run_pso(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    config: PublishedProtocolConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run PSO over XGBoost hyperparameters only."""

    rng = np.random.default_rng(config.random_state)
    dimensions = 9
    positions = rng.uniform(0.0, 1.0, size=(config.particles, dimensions))
    velocities = rng.uniform(-0.15, 0.15, size=(config.particles, dimensions))
    personal_best = positions.copy()
    personal_scores = np.full(config.particles, -np.inf)
    global_best = positions[0].copy()
    global_score = -np.inf
    cache: dict[str, float] = {}
    history: list[dict[str, float]] = []

    for iteration in range(1, config.iterations + 1):
        for index in range(config.particles):
            score = evaluate_particle(positions[index], x_data, y_data, config, cache)
            if score > personal_scores[index]:
                personal_scores[index] = score
                personal_best[index] = positions[index].copy()
            if score > global_score:
                global_score = score
                global_best = positions[index].copy()

        inertia = 0.72 - (0.28 * iteration / config.iterations)
        r1 = rng.uniform(0.0, 1.0, size=(config.particles, dimensions))
        r2 = rng.uniform(0.0, 1.0, size=(config.particles, dimensions))
        velocities = (
            inertia * velocities
            + 1.45 * r1 * (personal_best - positions)
            + 1.45 * r2 * (global_best - positions)
        )
        velocities = np.clip(velocities, -0.28, 0.28)
        positions = np.clip(positions + velocities, 0.0, 1.0)
        history.append({"iteration": iteration, "best_fitness": global_score, "cache_size": len(cache)})
        print(f"Iteration {iteration}/{config.iterations} best={global_score:.4f} cache={len(cache)}")
    return global_best, pd.DataFrame(history)


def final_evaluation(
    best_particle: np.ndarray,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    config: PublishedProtocolConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the optimized XGBoost model."""

    params = params_from_particle(np.clip(best_particle, 0.0, 1.0))
    model = build_model(params, config.random_state)
    cv = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.random_state)
    scores = cross_validate(
        model,
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
    probabilities = cross_val_predict(model, x_data, y_data, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    tuned_row = {
        "model": "Published-Protocol PSO-XGBoost",
        "accuracy": accuracy_score(y_data, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_data, predictions),
        "precision": precision_score(y_data, predictions, zero_division=0),
        "recall": recall_score(y_data, predictions, zero_division=0),
        "f1": f1_score(y_data, predictions, zero_division=0),
    }
    cv_row = {
        "model": "Published-Protocol PSO-XGBoost",
        **{
            key.replace("test_", ""): float(value.mean())
            for key, value in scores.items()
            if key.startswith("test_")
        },
    }
    params_row = {"model": "Published-Protocol PSO-XGBoost", **params}
    return pd.DataFrame([cv_row, tuned_row]), pd.DataFrame([params_row])


def save_comparison(results: pd.DataFrame, output_path: Path) -> None:
    """Save comparison plot."""

    plt.figure(figsize=(8.5, 4.2))
    frame = results.sort_values("accuracy", ascending=True)
    bars = plt.barh(frame["model"], frame["accuracy"] * 100, color="#dc2626")
    plt.xlabel("Accuracy (%)")
    plt.title("Published Protocol Approximation")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    """Run the published-protocol approximation."""

    root = Path.cwd()
    config = PublishedProtocolConfig(root, root / "outputs" / "uci_arrhythmia_published_protocol")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    x_data, y_data = load_data(root)
    best_particle, history = run_pso(x_data, y_data, config)
    results, params = final_evaluation(best_particle, x_data, y_data, config)
    comparison = pd.DataFrame(
        [
            {
                "model": "Current CatBoost Deep",
                "accuracy": 0.8451327433628318,
                "balanced_accuracy": 0.841407867494824,
                "f1": 0.825,
                "source": "our strict CV threshold result",
            },
            {
                "model": "Published-Protocol PSO-XGBoost",
                "accuracy": float(results.iloc[1]["accuracy"]),
                "balanced_accuracy": float(results.iloc[1]["balanced_accuracy"]),
                "f1": float(results.iloc[1]["f1"]),
                "source": "our protocol approximation",
            },
            {
                "model": "Published PSO-XGBoost 2025",
                "accuracy": 0.9524,
                "balanced_accuracy": 0.9481,
                "f1": 0.963,
                "source": "Dhanka and Maini 2025",
            },
        ]
    )
    history.to_csv(config.output_dir / "published_protocol_pso_history.csv", index=False)
    results.to_csv(config.output_dir / "published_protocol_pso_xgboost_results.csv", index=False)
    params.to_csv(config.output_dir / "published_protocol_best_params.csv", index=False)
    comparison.to_csv(config.output_dir / "published_protocol_comparison.csv", index=False)
    save_comparison(comparison, config.output_dir / "published_protocol_comparison.png")
    print(results.to_string(index=False))
    print(params.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
