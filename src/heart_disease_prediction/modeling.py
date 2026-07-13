"""Model training, evaluation, and blending utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin


@dataclass(frozen=True)
class ModelMetrics:
    """Classification metrics for a fitted model.

    Attributes:
        accuracy: Accuracy score.
        precision: Positive-class precision.
        recall: Positive-class recall.
        f1: Positive-class F1-score.
        roc_auc: ROC-AUC score.
        pr_auc: Average precision score.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float


@dataclass(frozen=True)
class BlendResult:
    """Result of validation-set probability blending.

    Attributes:
        alpha: Weight assigned to the primary model probability.
        metrics: Validation metrics for the blended probabilities.
    """

    alpha: float
    metrics: ModelMetrics


@dataclass(frozen=True)
class ThresholdResult:
    """Result of validation-set threshold tuning.

    Attributes:
        threshold: Probability threshold used for positive classification.
        metrics: Validation metrics at the selected threshold.
    """

    threshold: float
    metrics: ModelMetrics


@dataclass(frozen=True)
class HybridSelection:
    """Selected pairwise probability blend.

    Attributes:
        primary_name: Name of the first blended model.
        complementary_name: Name of the second blended model.
        blend_result: Validation-tuned blend result.
        threshold_result: Validation-tuned threshold result.
    """

    primary_name: str
    complementary_name: str
    blend_result: BlendResult
    threshold_result: ThresholdResult


def build_model_pipelines(
    preprocessor: Any,
    random_state: int,
) -> dict[str, Pipeline]:
    """Build baseline and optional advanced model pipelines.

    Args:
        preprocessor: Fitted-compatible preprocessing transformer.
        random_state: Seed used by stochastic models.

    Returns:
        Mapping of model names to sklearn pipelines.
    """

    models: dict[str, ClassifierMixin] = {
        "logistic_regression": LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            l2_regularization=0.01,
            random_state=random_state,
        ),
    }

    xgb_model = _build_xgboost_model(random_state)
    if xgb_model is not None:
        models["xgboost"] = xgb_model

    lightgbm_model = _build_lightgbm_model(random_state)
    if lightgbm_model is not None:
        models["lightgbm"] = lightgbm_model

    catboost_model = _build_catboost_model(random_state)
    if catboost_model is not None:
        models["catboost"] = catboost_model

    return {
        name: Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )
        for name, model in models.items()
    }


def fit_models(
    models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    """Fit multiple model pipelines.

    Args:
        models: Mapping of model names to unfitted pipelines.
        x_train: Training features.
        y_train: Training labels.

    Returns:
        Mapping of model names to fitted pipelines.
    """

    fitted_models: dict[str, Pipeline] = {}
    for name, model in models.items():
        fitted_models[name] = model.fit(x_train, y_train)
    return fitted_models


def evaluate_models(
    models: dict[str, BaseEstimator],
    x_data: pd.DataFrame,
    y_data: pd.Series,
) -> pd.DataFrame:
    """Evaluate fitted models on a labeled dataset.

    Args:
        models: Mapping of model names to fitted estimators.
        x_data: Evaluation features.
        y_data: Evaluation labels.

    Returns:
        Metrics dataframe sorted by ROC-AUC then F1.
    """

    rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        probabilities = predict_positive_proba(model, x_data)
        metrics = compute_metrics(y_data, probabilities)
        rows.append({"model": name, **metrics.__dict__})

    return (
        pd.DataFrame(rows)
        .sort_values(["roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> ModelMetrics:
    """Compute binary classification metrics.

    Args:
        y_true: Ground-truth labels.
        probabilities: Positive-class probabilities.
        threshold: Probability threshold for class labels.

    Returns:
        ModelMetrics object.
    """

    predictions = (probabilities >= threshold).astype(int)
    return ModelMetrics(
        accuracy=accuracy_score(y_true, predictions),
        precision=precision_score(y_true, predictions, zero_division=0),
        recall=recall_score(y_true, predictions, zero_division=0),
        f1=f1_score(y_true, predictions, zero_division=0),
        roc_auc=roc_auc_score(y_true, probabilities),
        pr_auc=average_precision_score(y_true, probabilities),
    )


def predict_positive_proba(
    model: BaseEstimator,
    x_data: pd.DataFrame,
) -> np.ndarray:
    """Predict positive-class probabilities.

    Args:
        model: Fitted estimator.
        x_data: Feature dataframe.

    Returns:
        Positive-class probability array.
    """

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_data)
        return np.asarray(probabilities)[:, 1]

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x_data))
        return 1 / (1 + np.exp(-scores))

    return np.asarray(model.predict(x_data), dtype=float)


def tune_probability_blend(
    primary_probabilities: np.ndarray,
    complementary_probabilities: np.ndarray,
    y_valid: pd.Series,
    alpha_grid: np.ndarray | None = None,
) -> BlendResult:
    """Tune weighted probability blending on a validation set.

    Args:
        primary_probabilities: Probabilities from the primary model.
        complementary_probabilities: Probabilities from the complementary model.
        y_valid: Validation labels.
        alpha_grid: Candidate weights for the primary model.

    Returns:
        Best blend result based on validation F1 then ROC-AUC.
    """

    if alpha_grid is None:
        alpha_grid = np.linspace(0, 1, 21)

    best_result: BlendResult | None = None
    for alpha in alpha_grid:
        blended_probabilities = (
            alpha * primary_probabilities
            + (1 - alpha) * complementary_probabilities
        )
        metrics = compute_metrics(y_valid, blended_probabilities)
        candidate = BlendResult(alpha=float(alpha), metrics=metrics)
        if best_result is None:
            best_result = candidate
            continue

        candidate_key = (candidate.metrics.f1, candidate.metrics.roc_auc)
        best_key = (best_result.metrics.f1, best_result.metrics.roc_auc)
        if candidate_key > best_key:
            best_result = candidate

    if best_result is None:
        raise ValueError("No alpha candidates were evaluated.")

    return best_result


def select_best_pairwise_blend(
    models: dict[str, BaseEstimator],
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    candidate_names: list[str] | None = None,
    alpha_grid: np.ndarray | None = None,
) -> HybridSelection:
    """Select the best pairwise probability blend on validation data.

    Args:
        models: Mapping of fitted model names to estimators.
        x_valid: Validation features.
        y_valid: Validation labels.
        candidate_names: Optional subset of model names to evaluate.
        alpha_grid: Candidate blend weights.

    Returns:
        Best pairwise blend selected by tuned-threshold F1, then ROC-AUC.
    """

    if alpha_grid is None:
        alpha_grid = np.linspace(0, 1, 101)

    names = candidate_names or list(models)
    probabilities = {
        name: predict_positive_proba(models[name], x_valid)
        for name in names
        if name in models
    }

    best_selection: HybridSelection | None = None
    model_names = list(probabilities)
    for first_index, primary_name in enumerate(model_names):
        for complementary_name in model_names[first_index + 1 :]:
            blend_result = tune_probability_blend(
                probabilities[primary_name],
                probabilities[complementary_name],
                y_valid,
                alpha_grid=alpha_grid,
            )
            blended_probabilities = blend_probabilities(
                probabilities[primary_name],
                probabilities[complementary_name],
                blend_result.alpha,
            )
            threshold_result = tune_classification_threshold(
                blended_probabilities,
                y_valid,
                objective="f1",
            )
            selection = HybridSelection(
                primary_name=primary_name,
                complementary_name=complementary_name,
                blend_result=blend_result,
                threshold_result=threshold_result,
            )
            if best_selection is None:
                best_selection = selection
                continue

            candidate_key = (
                selection.threshold_result.metrics.f1,
                selection.blend_result.metrics.roc_auc,
            )
            best_key = (
                best_selection.threshold_result.metrics.f1,
                best_selection.blend_result.metrics.roc_auc,
            )
            if candidate_key > best_key:
                best_selection = selection

    if best_selection is None:
        raise ValueError("At least two fitted models are required.")

    return best_selection


def tune_classification_threshold(
    probabilities: np.ndarray,
    y_valid: pd.Series,
    threshold_grid: np.ndarray | None = None,
    objective: str = "f1",
) -> ThresholdResult:
    """Tune a classification threshold on validation probabilities.

    Args:
        probabilities: Positive-class probabilities.
        y_valid: Validation labels.
        threshold_grid: Candidate probability thresholds.
        objective: Metric to maximize, either ``f1`` or ``accuracy``.

    Returns:
        Best threshold result.

    Raises:
        ValueError: If the objective is unsupported.
    """

    if threshold_grid is None:
        threshold_grid = np.linspace(0.2, 0.8, 61)

    if objective not in {"f1", "accuracy"}:
        raise ValueError("objective must be either 'f1' or 'accuracy'.")

    best_result: ThresholdResult | None = None
    for threshold in threshold_grid:
        metrics = compute_metrics(y_valid, probabilities, threshold=threshold)
        candidate = ThresholdResult(float(threshold), metrics)
        if best_result is None:
            best_result = candidate
            continue

        candidate_score = getattr(candidate.metrics, objective)
        best_score = getattr(best_result.metrics, objective)
        if candidate_score > best_score:
            best_result = candidate

    if best_result is None:
        raise ValueError("No threshold candidates were evaluated.")

    return best_result


def blend_probabilities(
    primary_probabilities: np.ndarray,
    complementary_probabilities: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend two probability vectors using a primary-model weight.

    Args:
        primary_probabilities: Probabilities from the primary model.
        complementary_probabilities: Probabilities from the complementary model.
        alpha: Weight assigned to the primary model.

    Returns:
        Blended probability array.
    """

    return (
        alpha * primary_probabilities
        + (1 - alpha) * complementary_probabilities
    )


def _build_xgboost_model(random_state: int) -> ClassifierMixin | None:
    """Build an XGBoost classifier if the package is installed."""

    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def _build_catboost_model(random_state: int) -> ClassifierMixin | None:
    """Build a CatBoost classifier if the package is installed."""

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None

    return CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.04,
        l2_leaf_reg=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
    )


def _build_lightgbm_model(random_state: int) -> ClassifierMixin | None:
    """Build a LightGBM classifier if the package is installed."""

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None

    return LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=100,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
