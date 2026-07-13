"""Streamlit app for arrhythmia prediction and disease classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "arrhythmia"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_artifacts"


@st.cache_resource
def load_artifacts() -> dict[str, Any]:
    """Load trained models and metadata."""

    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    return {
        "metadata": metadata,
        "binary_model": joblib.load(ARTIFACT_DIR / "binary_catboost_pipeline.joblib"),
        "multiclass_model": joblib.load(ARTIFACT_DIR / "grouped_multiclass_catboost_pipeline.joblib"),
        "subtype_model": joblib.load(ARTIFACT_DIR / "hierarchical_subtype_catboost_pipeline.joblib"),
    }


def normalize_uploaded_data(data: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series | None]:
    """Normalize uploaded data to the expected feature schema.

    Args:
        data: Uploaded dataframe.
        feature_columns: Required feature columns.

    Returns:
        Tuple of feature dataframe and optional target labels.
    """

    working = data.copy()
    target: pd.Series | None = None
    if "binary_target" in working.columns:
        target = working["binary_target"].astype(int)
        working = working.drop(columns=["binary_target"])
    elif "arrhythmia_class" in working.columns:
        target = (working["arrhythmia_class"].astype(int) != 1).astype(int)
        working = working.drop(columns=["arrhythmia_class"])
    elif "target" in working.columns:
        raw_target = working["target"].astype(int)
        target = raw_target if set(raw_target.unique()).issubset({0, 1}) else (raw_target != 1).astype(int)
        working = working.drop(columns=["target"])

    missing = [column for column in feature_columns if column not in working.columns]
    if missing and working.shape[1] >= len(feature_columns):
        working = working.iloc[:, : len(feature_columns)].copy()
        working.columns = feature_columns

    missing = [column for column in feature_columns if column not in working.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:10]}")
    return working[feature_columns].copy(), target


def classify_confidence(probability: float) -> str:
    """Map probability to confidence label."""

    confidence = max(probability, 1 - probability)
    if confidence >= 0.8:
        return "High"
    if confidence >= 0.65:
        return "Medium"
    return "Low"


def classify_risk(probability: float) -> str:
    """Map arrhythmia probability to risk level."""

    if probability >= 0.75:
        return "High Risk"
    if probability >= 0.49:
        return "Moderate Risk"
    return "Low Risk"


def build_binary_predictions(
    model: Any,
    x_data: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Build binary prediction output table."""

    probabilities = np.asarray(model.predict_proba(x_data))[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return pd.DataFrame(
        {
            "patient_id": np.arange(1, len(x_data) + 1),
            "predicted_class": np.where(predictions == 1, "Arrhythmia", "Normal"),
            "probability_arrhythmia": probabilities,
            "confidence": [classify_confidence(float(value)) for value in probabilities],
            "risk_level": [classify_risk(float(value)) for value in probabilities],
            "model_name": "CatBoost Deep",
        }
    )


def build_multiclass_predictions(model: Any, x_data: pd.DataFrame) -> pd.DataFrame:
    """Build grouped disease classification output table."""

    probabilities = np.asarray(model.predict_proba(x_data))
    predictions = model.predict(x_data).reshape(-1)
    confidence = probabilities.max(axis=1)
    return pd.DataFrame(
        {
            "patient_id": np.arange(1, len(x_data) + 1),
            "predicted_group": predictions.astype(str),
            "group_probability": confidence,
            "confidence": [classify_confidence(float(value)) for value in confidence],
            "model_name": "Grouped CatBoost",
        }
    )


def build_hierarchical_predictions(artifacts: dict[str, Any], x_data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Build hierarchical prediction output table."""

    binary = build_binary_predictions(artifacts["binary_model"], x_data, threshold)
    subtype_predictions = artifacts["subtype_model"].predict(x_data).reshape(-1).astype(str)
    binary["predicted_subtype_group"] = np.where(
        binary["predicted_class"] == "Arrhythmia",
        subtype_predictions,
        "Normal",
    )
    return binary


def binary_metrics(y_true: pd.Series, predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute binary metrics for uploaded labeled data."""

    probabilities = predictions["probability_arrhythmia"].to_numpy()
    y_pred = (predictions["predicted_class"] == "Arrhythmia").astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities) if y_true.nunique() == 2 else np.nan,
    }
    return pd.DataFrame([metrics])


def render_reference_outputs() -> None:
    """Render saved evaluation outputs."""

    plots = [
        ("Confusion Matrix", OUTPUT_DIR / "binary_confusion_matrix.png"),
        ("ROC Curve", OUTPUT_DIR / "binary_roc_curve.png"),
        ("Precision-Recall Curve", OUTPUT_DIR / "binary_pr_curve.png"),
        ("Calibration Curve", OUTPUT_DIR / "binary_calibration_curve.png"),
        ("Feature Importance", OUTPUT_DIR / "binary_feature_importance.png"),
        ("SHAP Explanation", OUTPUT_DIR / "binary_shap_bar.png"),
    ]
    for title, path in plots:
        if path.exists():
            st.subheader(title)
            st.image(str(path))


def main() -> None:
    """Run Streamlit app."""

    st.set_page_config(page_title="Arrhythmia Prediction", layout="wide")
    artifacts = load_artifacts()
    metadata = artifacts["metadata"]
    feature_columns = metadata["feature_columns"]
    threshold = float(metadata["binary_threshold"])

    st.title("Arrhythmia Prediction and Disease Classification")
    task_mode = st.sidebar.radio(
        "Task",
        ["Binary Prediction", "Grouped Disease Classification", "Hierarchical Prediction"],
    )
    threshold = st.sidebar.slider("Decision threshold", 0.2, 0.8, threshold, 0.01)
    uploaded_file = st.file_uploader("CSV file", type=["csv"])

    tab_upload, tab_results, tab_evaluation, tab_explainability = st.tabs(
        ["Upload", "Results", "Evaluation", "Explainability"]
    )

    with tab_upload:
        st.metric("Required features", len(feature_columns))
        st.metric("Binary threshold", f"{threshold:.2f}")
        if uploaded_file is None:
            st.info("Upload a CSV file to run predictions.")
            return
        uploaded = pd.read_csv(uploaded_file)
        try:
            x_data, target = normalize_uploaded_data(uploaded, feature_columns)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success("Schema compatible")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rows", len(x_data))
        col_b.metric("Features", x_data.shape[1])
        col_c.metric("Missing cells", int(x_data.isna().sum().sum()))
        st.dataframe(x_data.head(), use_container_width=True)

    with tab_results:
        if task_mode == "Binary Prediction":
            predictions = build_binary_predictions(artifacts["binary_model"], x_data, threshold)
        elif task_mode == "Grouped Disease Classification":
            predictions = build_multiclass_predictions(artifacts["multiclass_model"], x_data)
        else:
            predictions = build_hierarchical_predictions(artifacts, x_data, threshold)

        st.dataframe(predictions, use_container_width=True)
        csv_bytes = predictions.to_csv(index=False).encode("utf-8")
        st.download_button("Download results", csv_bytes, "arrhythmia_predictions.csv", "text/csv")

        if "probability_arrhythmia" in predictions.columns:
            fig = px.histogram(predictions, x="probability_arrhythmia", nbins=20, title="Arrhythmia Probability")
            st.plotly_chart(fig, use_container_width=True)
            counts = predictions["predicted_class"].value_counts().reset_index()
            counts.columns = ["class", "count"]
            st.plotly_chart(px.pie(counts, names="class", values="count", title="Prediction Split"), use_container_width=True)

    with tab_evaluation:
        if uploaded_file is not None and target is not None and "probability_arrhythmia" in predictions.columns:
            metrics = binary_metrics(target, predictions)
            st.dataframe(metrics.style.format("{:.3f}"), use_container_width=True)
        st.subheader("Saved model evaluation")
        reference_metrics = pd.read_csv(OUTPUT_DIR / "final_binary_holdout_metrics.csv")
        st.dataframe(reference_metrics.style.format("{:.3f}"), use_container_width=True)
        render_reference_outputs()

    with tab_explainability:
        importance_path = OUTPUT_DIR / "binary_feature_importance.csv"
        if importance_path.exists():
            importance = pd.read_csv(importance_path).head(20)
            st.dataframe(importance, use_container_width=True)
            st.plotly_chart(
                px.bar(importance, x="importance", y="feature", orientation="h", title="Top Features"),
                use_container_width=True,
            )
        shap_path = OUTPUT_DIR / "binary_shap_bar.png"
        if shap_path.exists():
            st.image(str(shap_path))


if __name__ == "__main__":
    main()
