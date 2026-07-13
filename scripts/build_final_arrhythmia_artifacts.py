"""Build final artifacts for arrhythmia prediction and classification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FinalArtifactConfig:
    """Final artifact build configuration.

    Attributes:
        project_root: Project root directory.
        artifact_dir: Directory where model artifacts are saved.
        output_dir: Directory where evaluation outputs are saved.
        random_state: Reproducibility seed.
        binary_threshold: Optimized binary decision threshold.
    """

    project_root: Path
    artifact_dir: Path
    output_dir: Path
    random_state: int = 42
    binary_threshold: float = 0.49


def load_uci_arrhythmia(project_root: Path) -> pd.DataFrame:
    """Load the UCI Arrhythmia dataset."""

    data_path = project_root / "data" / "uci_arrhythmia" / "arrhythmia.data"
    data = pd.read_csv(data_path, header=None, na_values="?")
    data.columns = [*[f"feature_{index:03d}" for index in range(data.shape[1] - 1)], "arrhythmia_class"]
    data["arrhythmia_class"] = data["arrhythmia_class"].astype(int)
    data["binary_target"] = (data["arrhythmia_class"] != 1).astype(int)
    return data


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return feature column names."""

    return [column for column in data.columns if column.startswith("feature_")]


def grouped_multiclass_target(target: pd.Series) -> pd.Series:
    """Group rare arrhythmia classes into class 99."""

    counts = target.value_counts()
    keep_classes = counts[counts >= 20].index
    return target.where(target.isin(keep_classes), other=99).astype(int)


def grouped_subtype_target(target: pd.Series) -> pd.Series:
    """Create grouped subtype target for arrhythmia-only rows."""

    arrhythmia_target = target[target != 1]
    counts = arrhythmia_target.value_counts()
    keep_classes = counts[counts >= 20].index
    return arrhythmia_target.where(arrhythmia_target.isin(keep_classes), other=99).astype(int)


def build_preprocessor(x_data: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing transformer."""

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


def build_binary_model(random_state: int) -> Pipeline:
    """Build the final binary CatBoost Deep pipeline."""

    classifier = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        iterations=450,
        depth=5,
        learning_rate=0.025,
        l2_leaf_reg=8,
        random_strength=0.8,
        bagging_temperature=0.5,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
    return Pipeline([("preprocessor", "passthrough"), ("classifier", classifier)])


def build_multiclass_model(random_state: int) -> Pipeline:
    """Build grouped multiclass CatBoost pipeline."""

    classifier = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        auto_class_weights="Balanced",
        iterations=350,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=6,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
    return Pipeline([("preprocessor", "passthrough"), ("classifier", classifier)])


def probability_positive(model: Pipeline, x_data: pd.DataFrame) -> np.ndarray:
    """Return positive class probabilities."""

    return np.asarray(model.predict_proba(x_data))[:, 1]


def compute_binary_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute binary metrics."""

    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def prepare_pipeline(model: Pipeline, x_data: pd.DataFrame) -> Pipeline:
    """Attach a fresh preprocessor to a model pipeline."""

    model.set_params(preprocessor=build_preprocessor(x_data))
    return model


def save_binary_plots(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> None:
    """Save binary evaluation plots."""

    predictions = (probabilities >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, display_labels=["Normal", "Arrhythmia"])
    plt.title("Binary Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "binary_confusion_matrix.png", dpi=180)
    plt.close()

    RocCurveDisplay.from_predictions(y_true, probabilities)
    plt.title("Binary ROC Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "binary_roc_curve.png", dpi=180)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, probabilities)
    plt.title("Binary Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "binary_pr_curve.png", dpi=180)
    plt.close()

    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=8, strategy="quantile")
    plt.figure(figsize=(5.8, 4.2))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "binary_calibration_curve.png", dpi=180)
    plt.close()


def save_feature_importance(model: Pipeline, output_dir: Path) -> pd.DataFrame:
    """Save feature importance table and chart."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.get_feature_importance()
    frame = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    frame.to_csv(output_dir / "binary_feature_importance.csv", index=False)

    top_frame = frame.head(20).sort_values("importance")
    plt.figure(figsize=(8.5, 5.2))
    plt.barh(top_frame["feature"], top_frame["importance"], color="#2563eb")
    plt.xlabel("Importance")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "binary_feature_importance.png", dpi=180)
    plt.close()
    return frame


def save_shap_bar(model: Pipeline, x_sample: pd.DataFrame, output_dir: Path) -> None:
    """Save SHAP bar chart for the binary CatBoost model."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(x_sample)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)
    values = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
    mean_abs = np.abs(values).mean(axis=0)
    frame = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(20)
        .sort_values("mean_abs_shap")
    )
    plt.figure(figsize=(8.5, 5.2))
    plt.barh(frame["feature"], frame["mean_abs_shap"], color="#0f766e")
    plt.xlabel("Mean absolute SHAP")
    plt.title("SHAP Global Explanation")
    plt.tight_layout()
    plt.savefig(output_dir / "binary_shap_bar.png", dpi=180)
    plt.close()


def save_multiclass_plot(
    y_true: pd.Series,
    predictions: np.ndarray,
    output_dir: Path,
    title: str,
    filename: str,
) -> None:
    """Save multiclass confusion matrix."""

    ConfusionMatrixDisplay.from_predictions(y_true, predictions, xticks_rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=180)
    plt.close()


def build_published_comparison(output_dir: Path) -> pd.DataFrame:
    """Build published-work comparison table."""

    frame = pd.DataFrame(
        [
            {
                "study": "Our work - CatBoost Deep",
                "task": "Binary normal vs arrhythmia",
                "accuracy": 0.8451,
                "notes": "5-fold OOF threshold optimization",
            },
            {
                "study": "Chittoria et al. 2022",
                "task": "Binary normal vs arrhythmia",
                "accuracy": 0.8400,
                "notes": "UCI Arrhythmia with ML and mRMR",
            },
            {
                "study": "Anuradha and David 2022",
                "task": "Binary normal vs arrhythmia",
                "accuracy": 0.8633,
                "notes": "SMOTE + MBAR + CatBoost",
            },
            {
                "study": "Our work - Grouped CatBoost",
                "task": "Grouped disease classification",
                "accuracy": 0.7301,
                "notes": "Rare classes grouped for reliability",
            },
            {
                "study": "Chittoria et al. 2022",
                "task": "Multiclass arrhythmia classification",
                "accuracy": 0.8148,
                "notes": "Published multiclass benchmark",
            },
        ]
    )
    frame.to_csv(output_dir / "published_comparison_table.csv", index=False)
    return frame


def train_and_save_artifacts(config: FinalArtifactConfig) -> None:
    """Train final models and save all app/report artifacts."""

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_uci_arrhythmia(config.project_root)
    feature_columns = get_feature_columns(data)
    x_data = data[feature_columns]
    y_binary = data["binary_target"]
    y_grouped = grouped_multiclass_target(data["arrhythmia_class"])

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_binary,
        test_size=0.2,
        stratify=y_binary,
        random_state=config.random_state,
    )
    binary_model = prepare_pipeline(build_binary_model(config.random_state), x_train)
    binary_model.fit(x_train, y_train)
    probabilities = probability_positive(binary_model, x_test)
    metrics = compute_binary_metrics(y_test, probabilities, config.binary_threshold)
    pd.DataFrame([metrics]).to_csv(config.output_dir / "final_binary_holdout_metrics.csv", index=False)
    save_binary_plots(y_test, probabilities, config.binary_threshold, config.output_dir)
    save_feature_importance(binary_model, config.output_dir)
    save_shap_bar(binary_model, x_test.sample(min(120, len(x_test)), random_state=config.random_state), config.output_dir)

    final_binary_model = prepare_pipeline(build_binary_model(config.random_state), x_data)
    final_binary_model.fit(x_data, y_binary)
    joblib.dump(final_binary_model, config.artifact_dir / "binary_catboost_pipeline.joblib")

    x_multi_train, x_multi_test, y_multi_train, y_multi_test = train_test_split(
        x_data,
        y_grouped,
        test_size=0.2,
        stratify=y_grouped,
        random_state=config.random_state,
    )
    multiclass_model = prepare_pipeline(build_multiclass_model(config.random_state), x_multi_train)
    multiclass_model.fit(x_multi_train, y_multi_train)
    multi_predictions = multiclass_model.predict(x_multi_test).reshape(-1)
    save_multiclass_plot(
        y_multi_test,
        multi_predictions,
        config.output_dir,
        "Grouped Disease Classification Confusion Matrix",
        "grouped_multiclass_confusion_matrix.png",
    )
    multi_metrics = {
        "accuracy": accuracy_score(y_multi_test, multi_predictions),
        "balanced_accuracy": balanced_accuracy_score(y_multi_test, multi_predictions),
        "macro_f1": f1_score(y_multi_test, multi_predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_multi_test, multi_predictions, average="weighted", zero_division=0),
    }
    pd.DataFrame([multi_metrics]).to_csv(config.output_dir / "final_grouped_multiclass_holdout_metrics.csv", index=False)
    final_multiclass_model = prepare_pipeline(build_multiclass_model(config.random_state), x_data)
    final_multiclass_model.fit(x_data, y_grouped)
    joblib.dump(final_multiclass_model, config.artifact_dir / "grouped_multiclass_catboost_pipeline.joblib")

    arrhythmia_mask = data["arrhythmia_class"] != 1
    x_subtype = x_data.loc[arrhythmia_mask]
    y_subtype = grouped_subtype_target(data["arrhythmia_class"])
    subtype_model = prepare_pipeline(build_multiclass_model(config.random_state), x_subtype)
    subtype_model.fit(x_subtype, y_subtype)
    joblib.dump(subtype_model, config.artifact_dir / "hierarchical_subtype_catboost_pipeline.joblib")

    comparison = build_published_comparison(config.output_dir)
    metadata: dict[str, Any] = {
        "binary_threshold": config.binary_threshold,
        "feature_columns": feature_columns,
        "binary_labels": {"0": "Normal", "1": "Arrhythmia"},
        "grouped_class_note": "Rare classes are grouped as 99.",
        "hierarchical_strategy": "First predict Normal vs Arrhythmia, then classify arrhythmia subtype for positive cases.",
        "final_binary_holdout_metrics": metrics,
        "published_comparison_rows": len(comparison),
    }
    (config.artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(config)]).astype(str).to_csv(config.output_dir / "artifact_build_config.csv", index=False)


def main() -> None:
    """Build all final artifacts."""

    root = Path.cwd()
    train_and_save_artifacts(
        FinalArtifactConfig(
            project_root=root,
            artifact_dir=root / "artifacts" / "arrhythmia",
            output_dir=root / "outputs" / "final_artifacts",
        )
    )


if __name__ == "__main__":
    main()
