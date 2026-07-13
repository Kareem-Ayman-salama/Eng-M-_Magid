"""Run optional external validation for saved arrhythmia models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def normalize_data(data: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Normalize an external validation dataframe."""

    working = data.copy()
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
    else:
        raise ValueError("External validation file must include binary_target, arrhythmia_class, or target.")

    if not set(feature_columns).issubset(working.columns):
        if working.shape[1] >= len(feature_columns):
            working = working.iloc[:, : len(feature_columns)].copy()
            working.columns = feature_columns
        else:
            missing = [column for column in feature_columns if column not in working.columns]
            raise ValueError(f"Missing feature columns: {missing[:10]}")
    return working[feature_columns], target


def evaluate_file(csv_path: Path, project_root: Path) -> dict[str, float | str]:
    """Evaluate one external validation CSV file."""

    artifact_dir = project_root / "artifacts" / "arrhythmia"
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(artifact_dir / "binary_catboost_pipeline.joblib")
    x_data, y_true = normalize_data(pd.read_csv(csv_path), metadata["feature_columns"])
    probabilities = np.asarray(model.predict_proba(x_data))[:, 1]
    predictions = (probabilities >= float(metadata["binary_threshold"])).astype(int)
    return {
        "file": csv_path.name,
        "rows": len(x_data),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities) if y_true.nunique() == 2 else np.nan,
    }


def main() -> None:
    """Evaluate any CSV files placed under data/external_validation."""

    project_root = Path.cwd()
    input_dir = project_root / "data" / "external_validation"
    output_dir = project_root / "outputs" / "external_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for csv_path in sorted(input_dir.glob("*.csv")):
        rows.append(evaluate_file(csv_path, project_root))
    output = pd.DataFrame(rows)
    output.to_csv(output_dir / "external_validation_results.csv", index=False)
    if output.empty:
        print(f"No CSV files found in {input_dir}")
    else:
        print(output.to_string(index=False))


if __name__ == "__main__":
    main()
