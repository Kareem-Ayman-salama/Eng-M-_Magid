"""Generate Kaggle notebook for AF inter-dataset experiments."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def markdown_cell(text: str) -> nbf.NotebookNode:
    """Create a markdown cell."""

    return nbf.v4.new_markdown_cell(text.strip())


def code_cell(code: str) -> nbf.NotebookNode:
    """Create a code cell."""

    return nbf.v4.new_code_cell(code.strip())


def build_notebook() -> nbf.NotebookNode:
    """Build the Kaggle AF inter-dataset notebook."""

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    cells = [
        markdown_cell(
            """
            # AF Inter-Dataset Prediction and Classification

            **Goal:** keep the previous UCI tabular work as a baseline, and add a new ECG-signal experiment on Kaggle using AF datasets such as SHDB-AF, MIT-BIH Atrial Fibrillation, and Long-Term AF.

            This notebook is self-contained for Kaggle. It discovers datasets under `/kaggle/input`, reads WFDB-compatible records when possible, extracts rhythm/beat-level features from annotations, and runs:

            - Binary prediction: AF vs non-AF.
            - Rhythm classification: AFIB/AFL/AT/Other when labels are available.
            - Inter-dataset validation: train on one dataset and test on another.
            """
        ),
        markdown_cell(
            """
            ## 1. Reproducibility and Dependencies

            If `wfdb` is missing in the Kaggle image, the first cell installs it. Internet must be enabled in Kaggle settings for installation.
            """
        ),
        code_cell(
            """
            import importlib.util
            import os
            import random
            import subprocess
            import sys
            from dataclasses import dataclass
            from pathlib import Path
            from typing import Any

            SEED = 42
            random.seed(SEED)
            os.environ["PYTHONHASHSEED"] = str(SEED)

            if importlib.util.find_spec("wfdb") is None:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wfdb"])

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import wfdb
            from sklearn.base import clone
            from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                accuracy_score,
                balanced_accuracy_score,
                classification_report,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )
            from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_validate, train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import LabelEncoder, StandardScaler

            try:
                from xgboost import XGBClassifier
            except Exception:
                XGBClassifier = None

            try:
                from catboost import CatBoostClassifier
            except Exception:
                CatBoostClassifier = None

            print("Python:", sys.version)
            print("wfdb:", wfdb.__version__)
            print("Input root exists:", Path("/kaggle/input").exists())
            """
        ),
        markdown_cell(
            """
            ## 2. Discover Kaggle Input Datasets

            Add these Kaggle datasets from the Notebook UI:

            - `yahiahedna/shdb-af-atrial-fibrillation`
            - `sahilshahare12/mit-bih-atrial-fibrillation`
            - `sahilshahare12/long-term-af-database`

            The notebook does not assume fixed folder names. It scans `/kaggle/input` recursively for `.hea`, `.atr`, and `.qrs` files.
            """
        ),
        code_cell(
            """
            INPUT_ROOT = Path("/kaggle/input")

            def summarize_inputs(input_root: Path = INPUT_ROOT) -> pd.DataFrame:
                rows = []
                for dataset_dir in sorted(input_root.iterdir()) if input_root.exists() else []:
                    if not dataset_dir.is_dir():
                        continue
                    files = list(dataset_dir.rglob("*"))
                    rows.append(
                        {
                            "dataset_dir": dataset_dir.name,
                            "total_files": sum(path.is_file() for path in files),
                            "hea_files": len(list(dataset_dir.rglob("*.hea"))),
                            "atr_files": len(list(dataset_dir.rglob("*.atr"))),
                            "qrs_files": len(list(dataset_dir.rglob("*.qrs"))),
                            "dat_files": len(list(dataset_dir.rglob("*.dat"))),
                            "approx_mb": sum(path.stat().st_size for path in files if path.is_file()) / (1024**2),
                        }
                    )
                return pd.DataFrame(rows).sort_values("dataset_dir")

            input_summary = summarize_inputs()
            input_summary
            """
        ),
        markdown_cell(
            """
            ## 3. WFDB Record Discovery

            A record is identified by a `.hea` file. Annotation files (`.atr`, `.qrs`) are matched by the same stem.
            """
        ),
        code_cell(
            """
            @dataclass(frozen=True)
            class RecordInfo:
                dataset: str
                record_name: str
                record_path: Path
                has_atr: bool
                has_qrs: bool
                has_dat: bool


            def discover_records(input_root: Path = INPUT_ROOT) -> pd.DataFrame:
                rows: list[dict[str, Any]] = []
                for hea_path in sorted(input_root.rglob("*.hea")):
                    dataset_name = hea_path.relative_to(input_root).parts[0]
                    stem = hea_path.with_suffix("")
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "record_name": hea_path.stem,
                            "record_path": str(stem),
                            "has_atr": stem.with_suffix(".atr").exists(),
                            "has_qrs": stem.with_suffix(".qrs").exists(),
                            "has_dat": stem.with_suffix(".dat").exists(),
                        }
                    )
                return pd.DataFrame(rows)

            records = discover_records()
            print(records.shape)
            records.head()
            """
        ),
        markdown_cell(
            """
            ## 4. Annotation Feature Extraction

            We start with annotation-derived features because they are much lighter than loading full 24-hour ECG signals. This is a strong first baseline for Kaggle:

            - RR interval statistics from QRS/beat annotations.
            - Rhythm labels from annotation aux notes where available.
            - Binary AF label and multiclass rhythm label.
            """
        ),
        code_cell(
            """
            AF_LABELS = {"AFIB", "(AFIB", "AF", "AFL", "(AFL"}
            RHYTHM_ALIASES = {
                "(AFIB": "AFIB",
                "AFIB": "AFIB",
                "(AFL": "AFL",
                "AFL": "AFL",
                "(AT": "AT",
                "AT": "AT",
                "(N": "OTHER",
                "N": "OTHER",
                "": "OTHER",
            }


            def clean_aux_label(value: object) -> str:
                label = str(value).strip().replace("\\x00", "")
                label = label.replace("\\\\x00", "")
                return RHYTHM_ALIASES.get(label, label.replace("(", "") or "OTHER")


            def safe_read_annotation(record_path: str, extension: str) -> wfdb.Annotation | None:
                try:
                    return wfdb.rdann(record_path, extension)
                except Exception:
                    return None


            def rr_features(samples: np.ndarray, fs: float = 200.0) -> dict[str, float]:
                if len(samples) < 3:
                    return {
                        "beat_count": float(len(samples)),
                        "rr_mean": np.nan,
                        "rr_std": np.nan,
                        "rr_min": np.nan,
                        "rr_max": np.nan,
                        "rr_rmssd": np.nan,
                        "hr_mean": np.nan,
                    }
                rr = np.diff(np.asarray(samples, dtype=float)) / fs
                rr = rr[(rr > 0.2) & (rr < 3.0)]
                if len(rr) == 0:
                    return {
                        "beat_count": float(len(samples)),
                        "rr_mean": np.nan,
                        "rr_std": np.nan,
                        "rr_min": np.nan,
                        "rr_max": np.nan,
                        "rr_rmssd": np.nan,
                        "hr_mean": np.nan,
                    }
                return {
                    "beat_count": float(len(samples)),
                    "rr_mean": float(np.mean(rr)),
                    "rr_std": float(np.std(rr)),
                    "rr_min": float(np.min(rr)),
                    "rr_max": float(np.max(rr)),
                    "rr_rmssd": float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else 0.0,
                    "hr_mean": float(60.0 / np.mean(rr)),
                }


            def extract_record_features(row: pd.Series) -> dict[str, Any] | None:
                record_path = row["record_path"]
                atr = safe_read_annotation(record_path, "atr") if row["has_atr"] else None
                qrs = safe_read_annotation(record_path, "qrs") if row["has_qrs"] else None
                annotation = atr or qrs
                if annotation is None:
                    return None

                features = rr_features(annotation.sample)
                aux_labels = [clean_aux_label(value) for value in getattr(annotation, "aux_note", [])]
                rhythm_labels = [label for label in aux_labels if label and label != "OTHER"]
                primary_rhythm = rhythm_labels[0] if rhythm_labels else "OTHER"
                is_af = int(any(label in {"AFIB", "AFL"} for label in aux_labels))
                features.update(
                    {
                        "dataset": row["dataset"],
                        "record_name": row["record_name"],
                        "primary_rhythm": primary_rhythm,
                        "binary_af": is_af,
                        "unique_aux_labels": "|".join(sorted(set(aux_labels)))[:200],
                    }
                )
                return features


            feature_rows = []
            for _, record in records.iterrows():
                result = extract_record_features(record)
                if result is not None:
                    feature_rows.append(result)

            features = pd.DataFrame(feature_rows)
            print(features.shape)
            display(features.head())
            display(features["dataset"].value_counts())
            display(features["binary_af"].value_counts(dropna=False))
            display(features["primary_rhythm"].value_counts(dropna=False).head(20))
            """
        ),
        markdown_cell(
            """
            ## 5. Save Extracted Feature Table
            """
        ),
        code_cell(
            """
            OUTPUT_DIR = Path("/kaggle/working/af_inter_dataset_outputs")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            features.to_csv(OUTPUT_DIR / "af_record_level_features.csv", index=False)
            print(OUTPUT_DIR / "af_record_level_features.csv")
            """
        ),
        markdown_cell(
            """
            ## 6. Binary Prediction: AF vs non-AF

            This baseline uses record-level annotation/RR features. If every record in a selected dataset is AF-positive, binary prediction will not be meaningful for that dataset alone, but inter-dataset evaluation still remains useful when labels vary.
            """
        ),
        code_cell(
            """
            FEATURE_COLUMNS = ["beat_count", "rr_mean", "rr_std", "rr_min", "rr_max", "rr_rmssd", "hr_mean"]


            def build_binary_models(seed: int = SEED) -> dict[str, Any]:
                models: dict[str, Any] = {
                    "Logistic Regression": Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
                        ]
                    ),
                    "Random Forest": Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("classifier", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=-1)),
                        ]
                    ),
                    "Extra Trees": Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("classifier", ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)),
                        ]
                    ),
                }
                if XGBClassifier is not None:
                    models["XGBoost"] = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "classifier",
                                XGBClassifier(
                                    objective="binary:logistic",
                                    eval_metric="logloss",
                                    n_estimators=250,
                                    max_depth=3,
                                    learning_rate=0.03,
                                    subsample=0.85,
                                    colsample_bytree=0.85,
                                    random_state=seed,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    )
                if CatBoostClassifier is not None:
                    models["CatBoost"] = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "classifier",
                                CatBoostClassifier(
                                    iterations=300,
                                    depth=4,
                                    learning_rate=0.03,
                                    loss_function="Logloss",
                                    auto_class_weights="Balanced",
                                    random_seed=seed,
                                    verbose=False,
                                ),
                            ),
                        ]
                    )
                return models


            def evaluate_binary_cv(data: pd.DataFrame) -> pd.DataFrame:
                valid = data.dropna(subset=["binary_af"]).copy()
                if valid["binary_af"].nunique() < 2 or len(valid) < 10:
                    print("Not enough class diversity for binary CV.")
                    return pd.DataFrame()
                x_data = valid[FEATURE_COLUMNS]
                y_data = valid["binary_af"].astype(int)
                cv = StratifiedKFold(n_splits=min(5, y_data.value_counts().min()), shuffle=True, random_state=SEED)
                rows = []
                for name, model in build_binary_models().items():
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
                    rows.append(
                        {
                            "model": name,
                            **{key.replace("test_", ""): value.mean() for key, value in scores.items() if key.startswith("test_")},
                        }
                    )
                return pd.DataFrame(rows).sort_values(["roc_auc", "f1"], ascending=False)


            binary_cv_results = evaluate_binary_cv(features)
            binary_cv_results
            """
        ),
        markdown_cell(
            """
            ## 7. Rhythm Classification

            Multiclass classification is attempted only for labels with enough samples. Rare labels are grouped as `OTHER_RARE`.
            """
        ),
        code_cell(
            """
            def grouped_rhythm_labels(labels: pd.Series, min_count: int = 5) -> pd.Series:
                counts = labels.value_counts()
                keep = set(counts[counts >= min_count].index)
                return labels.where(labels.isin(keep), "OTHER_RARE")


            def evaluate_rhythm_cv(data: pd.DataFrame) -> pd.DataFrame:
                valid = data.copy()
                valid["rhythm_group"] = grouped_rhythm_labels(valid["primary_rhythm"].fillna("OTHER"))
                if valid["rhythm_group"].nunique() < 2:
                    print("Not enough rhythm classes.")
                    return pd.DataFrame()
                min_class = valid["rhythm_group"].value_counts().min()
                if min_class < 2:
                    print("Not enough samples per class.")
                    return pd.DataFrame()
                encoder = LabelEncoder()
                x_data = valid[FEATURE_COLUMNS]
                y_data = encoder.fit_transform(valid["rhythm_group"])
                cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=SEED)
                models = {
                    "Random Forest": Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("classifier", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=SEED, n_jobs=-1)),
                        ]
                    ),
                    "Extra Trees": Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("classifier", ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=-1)),
                        ]
                    ),
                }
                if CatBoostClassifier is not None:
                    models["CatBoost"] = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "classifier",
                                CatBoostClassifier(
                                    iterations=300,
                                    depth=4,
                                    learning_rate=0.03,
                                    loss_function="MultiClass",
                                    auto_class_weights="Balanced",
                                    random_seed=SEED,
                                    verbose=False,
                                ),
                            ),
                        ]
                    )
                rows = []
                for name, model in models.items():
                    scores = cross_validate(
                        model,
                        x_data,
                        y_data,
                        cv=cv,
                        scoring={
                            "accuracy": "accuracy",
                            "balanced_accuracy": "balanced_accuracy",
                            "macro_f1": "f1_macro",
                            "weighted_f1": "f1_weighted",
                        },
                        n_jobs=-1,
                    )
                    rows.append(
                        {
                            "model": name,
                            "classes": ",".join(encoder.classes_),
                            **{key.replace("test_", ""): value.mean() for key, value in scores.items() if key.startswith("test_")},
                        }
                    )
                return pd.DataFrame(rows).sort_values(["macro_f1", "balanced_accuracy"], ascending=False)


            rhythm_cv_results = evaluate_rhythm_cv(features)
            rhythm_cv_results
            """
        ),
        markdown_cell(
            """
            ## 8. Inter-Dataset Validation

            Train on one dataset and test on another. This is the strongest part scientifically because it measures generalization across different ECG databases.
            """
        ),
        code_cell(
            """
            def evaluate_inter_dataset(data: pd.DataFrame) -> pd.DataFrame:
                rows = []
                datasets = sorted(data["dataset"].dropna().unique())
                for train_dataset in datasets:
                    for test_dataset in datasets:
                        if train_dataset == test_dataset:
                            continue
                        train = data[data["dataset"] == train_dataset].copy()
                        test = data[data["dataset"] == test_dataset].copy()
                        if train["binary_af"].nunique() < 2 or test["binary_af"].nunique() < 2:
                            continue
                        x_train, y_train = train[FEATURE_COLUMNS], train["binary_af"].astype(int)
                        x_test, y_test = test[FEATURE_COLUMNS], test["binary_af"].astype(int)
                        for name, model in build_binary_models().items():
                            fitted = clone(model).fit(x_train, y_train)
                            if hasattr(fitted, "predict_proba"):
                                prob = fitted.predict_proba(x_test)[:, 1]
                            else:
                                pred_raw = fitted.predict(x_test)
                                prob = pred_raw.astype(float)
                            pred = (prob >= 0.5).astype(int)
                            rows.append(
                                {
                                    "train_dataset": train_dataset,
                                    "test_dataset": test_dataset,
                                    "model": name,
                                    "accuracy": accuracy_score(y_test, pred),
                                    "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                                    "precision": precision_score(y_test, pred, zero_division=0),
                                    "recall": recall_score(y_test, pred, zero_division=0),
                                    "f1": f1_score(y_test, pred, zero_division=0),
                                    "roc_auc": roc_auc_score(y_test, prob) if y_test.nunique() == 2 else np.nan,
                                    "train_rows": len(train),
                                    "test_rows": len(test),
                                }
                            )
                return pd.DataFrame(rows).sort_values(["roc_auc", "f1"], ascending=False) if rows else pd.DataFrame()


            inter_dataset_results = evaluate_inter_dataset(features)
            inter_dataset_results.head(20)
            """
        ),
        markdown_cell(
            """
            ## 9. Visualizations and Export
            """
        ),
        code_cell(
            """
            def save_results() -> None:
                if not binary_cv_results.empty:
                    binary_cv_results.to_csv(OUTPUT_DIR / "binary_cv_results.csv", index=False)
                if not rhythm_cv_results.empty:
                    rhythm_cv_results.to_csv(OUTPUT_DIR / "rhythm_classification_results.csv", index=False)
                if not inter_dataset_results.empty:
                    inter_dataset_results.to_csv(OUTPUT_DIR / "inter_dataset_results.csv", index=False)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                sns.countplot(data=features, x="dataset", ax=axes[0])
                axes[0].set_title("Records per Dataset")
                axes[0].tick_params(axis="x", rotation=45)
                sns.countplot(data=features, x="binary_af", ax=axes[1])
                axes[1].set_title("Binary AF Label Count")
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / "dataset_label_summary.png", dpi=180)
                plt.show()

            save_results()
            print("Saved outputs to:", OUTPUT_DIR)
            """
        ),
        markdown_cell(
            """
            ## 10. Research Interpretation Template

            Use this wording after running the notebook:

            - The UCI Arrhythmia tabular experiment remains as the baseline.
            - This notebook adds signal-based AF prediction/classification using Kaggle ECG datasets.
            - If inter-dataset results are stable, this becomes the main research contribution because it tests generalization across ECG databases.
            - If one dataset has only AF-positive records, use it for rhythm/subtype classification or external validation, not standalone binary CV.
            """
        ),
    ]
    notebook["cells"] = cells
    return notebook


def main() -> None:
    """Write notebook to disk."""

    output_path = Path.cwd() / "notebooks" / "01_kaggle_af_inter_dataset_experiments.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), output_path)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
