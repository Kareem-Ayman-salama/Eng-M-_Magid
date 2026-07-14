"""Generate a Kaggle notebook for true PAF onset prediction on AFPDB."""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    """Create a markdown cell."""

    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    """Create a code cell."""

    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_notebook() -> nbf.NotebookNode:
    """Build the AFPDB true prediction notebook."""

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = [
        md(
            """
            # True Paroxysmal AF Onset Prediction - AFPDB

            This notebook adds a genuine prediction task to the project.

            Current multi-dataset ECG work is **AF detection and rhythm classification** because the AF rhythm is inside the classified window.
            This notebook is different: it predicts whether a sinus-rhythm ECG segment immediately precedes a future paroxysmal AF episode.

            Dataset:

            - Official name: `PAF Prediction Challenge Database (AFPDB)`.
            - PhysioNet slug: `afpdb`.
            - Kaggle dataset to search/add: `Paraoxymal Atrial Fibrillation Prediction Database`.
            - Likely Kaggle input path: `/kaggle/input/paraoxymal-atrial-fibrillation-prediction-database`.

            Core task:

            - Positive: pre-onset ECG excerpts immediately preceding PAF.
            - Negative: ECG excerpts distant from PAF or from non-PAF subjects.
            - Evaluation uses group-aware splits to avoid subject/record leakage.
            """
        ),
        md("## 1. Setup and Reproducibility"),
        code(
            """
            import importlib.util
            import os
            import random
            import subprocess
            import sys
            import warnings
            from dataclasses import dataclass
            from pathlib import Path
            from typing import Any

            SEED = 42
            random.seed(SEED)
            os.environ["PYTHONHASHSEED"] = str(SEED)
            warnings.filterwarnings("ignore")
            warnings.filterwarnings("ignore", message="is_sparse is deprecated", category=DeprecationWarning)

            if importlib.util.find_spec("wfdb") is None:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wfdb"])

            import joblib
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import wfdb
            from scipy import stats
            from sklearn.base import clone
            from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                ConfusionMatrixDisplay,
                PrecisionRecallDisplay,
                RocCurveDisplay,
                accuracy_score,
                balanced_accuracy_score,
                classification_report,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )
            from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            try:
                from xgboost import XGBClassifier
            except Exception:
                XGBClassifier = None

            try:
                from lightgbm import LGBMClassifier
            except Exception:
                LGBMClassifier = None

            try:
                from catboost import CatBoostClassifier
            except Exception:
                CatBoostClassifier = None

            OUTPUT_DIR = Path("/kaggle/working/afpdb_true_prediction_outputs")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            print("Python:", sys.version)
            print("WFDB:", wfdb.__version__)
            print("Output:", OUTPUT_DIR)
            """
        ),
        md("## 2. Configuration and Data Location"),
        code(
            """
            @dataclass(frozen=True)
            class Config:
                candidate_input_paths: tuple[Path, ...]
                fallback_download_dir: Path
                output_dir: Path
                window_minutes: tuple[int, ...] = (2, 5, 10, 30)
                min_beats_per_window: int = 10
                random_state: int = SEED


            CONFIG = Config(
                candidate_input_paths=(
                    Path("/kaggle/input/paraoxymal-atrial-fibrillation-prediction-database"),
                    Path("/kaggle/input/paf-prediction-challenge-database"),
                    Path("/kaggle/input/afpdb"),
                    Path("/kaggle/input/paf-prediction-challenge"),
                ),
                fallback_download_dir=Path("/kaggle/working/afpdb"),
                output_dir=OUTPUT_DIR,
            )


            def locate_or_download_afpdb(config: Config) -> Path:
                for path in config.candidate_input_paths:
                    if path.exists() and any(path.rglob("*.hea")):
                        print("Using Kaggle input:", path)
                        return path
                print("Kaggle input was not found. Downloading AFPDB from PhysioNet into /kaggle/working ...")
                config.fallback_download_dir.mkdir(parents=True, exist_ok=True)
                wfdb.dl_database("afpdb", dl_dir=str(config.fallback_download_dir))
                return config.fallback_download_dir


            DATA_ROOT = locate_or_download_afpdb(CONFIG)
            print("Data root:", DATA_ROOT)
            """
        ),
        md("## 3. Record Discovery and True Prediction Labels"),
        code(
            """
            def discover_records(root: Path) -> pd.DataFrame:
                rows: list[dict[str, Any]] = []
                for hea_path in sorted(root.rglob("*.hea")):
                    record = hea_path.stem
                    if record.endswith("c"):
                        # Continuation records may contain the onset/event and are excluded from true pre-onset prediction.
                        continue
                    if not record[0].lower() in {"p", "n", "t"}:
                        continue
                    rows.append(
                        {
                            "record": record,
                            "record_path": str(hea_path.with_suffix("")),
                            "prefix": record[0].lower(),
                            "number": int(record[1:]),
                            "has_qrs": hea_path.with_suffix(".qrs").exists(),
                            "has_dat": hea_path.with_suffix(".dat").exists(),
                        }
                    )
                return pd.DataFrame(rows)


            def load_event2_answers(root: Path) -> dict[str, int]:
                answer_files = list(root.rglob("event-2-answers*"))
                if not answer_files:
                    return {}
                tokens = answer_files[0].read_text(errors="ignore").split()
                labels = {}
                for record, label in zip(tokens[0::2], tokens[1::2]):
                    labels[record] = int(label.upper() == "A")
                return labels


            def label_record(record: str, event2_answers: dict[str, int]) -> tuple[int | None, str, str]:
                prefix = record[0].lower()
                number = int(record[1:])
                if prefix == "n":
                    return 0, "learning", f"n{(number + 1) // 2:02d}"
                if prefix == "p":
                    # PhysioNet AFPDB documentation: for p records, the second/even-numbered record in each pair
                    # immediately precedes PAF onset; the odd-numbered companion is distant from PAF.
                    return int(number % 2 == 0), "learning", f"p{(number + 1) // 2:02d}"
                if prefix == "t" and record in event2_answers:
                    return event2_answers[record], "challenge_test", f"t{(number + 1) // 2:02d}"
                return None, "unlabeled_test", f"{prefix}{(number + 1) // 2:02d}"


            records = discover_records(DATA_ROOT)
            event2_answers = load_event2_answers(DATA_ROOT)
            label_rows = records["record"].apply(lambda value: label_record(str(value), event2_answers))
            records["target"] = label_rows.map(lambda value: value[0])
            records["split_source"] = label_rows.map(lambda value: value[1])
            records["subject_group"] = label_rows.map(lambda value: value[2])
            records = records.dropna(subset=["target"]).copy()
            records["target"] = records["target"].astype(int)

            print(records.shape)
            display(records.groupby(["split_source", "target"]).size().unstack(fill_value=0))
            display(records.head())
            records.to_csv(OUTPUT_DIR / "afpdb_record_inventory.csv", index=False)
            """
        ),
        md("## 4. Feature Extraction"),
        code(
            """
            FEATURE_CACHE = OUTPUT_DIR / "afpdb_true_prediction_features.csv"


            def read_record_header(record_path: str) -> tuple[float, int, int]:
                header = wfdb.rdheader(record_path)
                return float(header.fs), int(header.sig_len), int(header.n_sig)


            def read_qrs_samples(record_path: str) -> np.ndarray:
                try:
                    annotation = wfdb.rdann(record_path, "qrs")
                    return np.asarray(annotation.sample, dtype=int)
                except Exception:
                    return np.asarray([], dtype=int)


            def rr_features(samples: np.ndarray, fs: float) -> dict[str, float]:
                if len(samples) < 4:
                    base = [
                        "beat_count", "rr_mean", "rr_std", "rr_min", "rr_max", "rr_iqr", "rr_rmssd",
                        "pnn50", "hr_mean", "rr_cv", "rr_range", "rr_mad", "rr_entropy",
                        "irregularity_index", "poincare_sd1", "poincare_sd2", "apb_proxy_rate",
                    ]
                    return {name: np.nan for name in base} | {"beat_count": float(len(samples))}
                rr = np.diff(samples.astype(float)) / fs
                rr = rr[(rr > 0.25) & (rr < 2.5)]
                if len(rr) < 3:
                    return rr_features(np.asarray([], dtype=int), fs)
                diff_rr = np.diff(rr)
                rr_mean = float(np.mean(rr))
                rr_std = float(np.std(rr))
                rr_rmssd = float(np.sqrt(np.mean(diff_rr**2)))
                rr_hist, _ = np.histogram(rr, bins=12, density=True)
                rr_hist = rr_hist[rr_hist > 0]
                sd1 = float(np.sqrt(np.var(diff_rr) / 2.0))
                sd2_term = max(0.0, 2.0 * np.var(rr) - 0.5 * np.var(diff_rr))
                sd2 = float(np.sqrt(sd2_term))
                short_rr_threshold = np.percentile(rr, 20) * 0.85
                apb_proxy_rate = float(np.mean(rr < short_rr_threshold))
                return {
                    "beat_count": float(len(samples)),
                    "rr_mean": rr_mean,
                    "rr_std": rr_std,
                    "rr_min": float(np.min(rr)),
                    "rr_max": float(np.max(rr)),
                    "rr_iqr": float(np.percentile(rr, 75) - np.percentile(rr, 25)),
                    "rr_rmssd": rr_rmssd,
                    "pnn50": float(np.mean(np.abs(diff_rr) > 0.05)),
                    "hr_mean": float(60.0 / rr_mean),
                    "rr_cv": float(rr_std / rr_mean) if rr_mean else np.nan,
                    "rr_range": float(np.max(rr) - np.min(rr)),
                    "rr_mad": float(np.mean(np.abs(rr - rr_mean))),
                    "rr_entropy": float(stats.entropy(rr_hist)) if len(rr_hist) else np.nan,
                    "irregularity_index": float((rr_std + rr_rmssd) / rr_mean) if rr_mean else np.nan,
                    "poincare_sd1": sd1,
                    "poincare_sd2": sd2,
                    "apb_proxy_rate": apb_proxy_rate,
                }


            def signal_features(record_path: str, start: int, end: int, n_sig: int) -> dict[str, float]:
                features: dict[str, float] = {}
                try:
                    record = wfdb.rdrecord(record_path, sampfrom=start, sampto=end, channels=list(range(min(2, n_sig))))
                    signal = np.asarray(record.p_signal, dtype=float)
                except Exception:
                    signal = np.empty((0, 0))
                for channel in range(2):
                    if signal.size == 0 or channel >= signal.shape[1]:
                        for stat in ["mean", "std", "iqr", "energy", "diff_mean", "skew", "kurtosis"]:
                            features[f"ch{channel}_{stat}"] = np.nan
                        continue
                    values = signal[:, channel]
                    values = values[np.isfinite(values)]
                    diff_values = np.diff(values)
                    features.update(
                        {
                            f"ch{channel}_mean": float(np.mean(values)),
                            f"ch{channel}_std": float(np.std(values)),
                            f"ch{channel}_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                            f"ch{channel}_energy": float(np.mean(values**2)),
                            f"ch{channel}_diff_mean": float(np.mean(np.abs(diff_values))) if len(diff_values) else 0.0,
                            f"ch{channel}_skew": float(stats.skew(values)) if len(values) > 2 else 0.0,
                            f"ch{channel}_kurtosis": float(stats.kurtosis(values)) if len(values) > 3 else 0.0,
                        }
                    )
                return features


            def extract_record_windows(row: pd.Series, window_minutes: int) -> list[dict[str, Any]]:
                fs, sig_len, n_sig = read_record_header(row["record_path"])
                qrs = read_qrs_samples(row["record_path"])
                window = int(window_minutes * 60 * fs)
                if window <= 0 or sig_len < window:
                    return []
                rows = []
                starts = list(range(0, sig_len - window + 1, window))
                for index, start in enumerate(starts):
                    end = start + window
                    beats = qrs[(qrs >= start) & (qrs < end)]
                    if len(beats) < CONFIG.min_beats_per_window:
                        continue
                    features = rr_features(beats, fs)
                    features.update(signal_features(row["record_path"], start, end, n_sig))
                    features.update(
                        {
                            "record": row["record"],
                            "subject_group": row["subject_group"],
                            "split_source": row["split_source"],
                            "target": int(row["target"]),
                            "window_minutes": int(window_minutes),
                            "window_index": int(index),
                            "start_sample": int(start),
                            "end_sample": int(end),
                            "fs": float(fs),
                        }
                    )
                    rows.append(features)
                return rows


            if FEATURE_CACHE.exists():
                features = pd.read_csv(FEATURE_CACHE, dtype={"record": str, "subject_group": str})
                print("Loaded cached features:", features.shape)
            else:
                rows = []
                for index, record in records.iterrows():
                    if index % 20 == 0:
                        print(f"Processing {index + 1}/{len(records)}: {record['record']}")
                    for window_minutes in CONFIG.window_minutes:
                        rows.extend(extract_record_windows(record, window_minutes))
                features = pd.DataFrame(rows)
                features.to_csv(FEATURE_CACHE, index=False)
                print("Saved:", FEATURE_CACHE)

            print(features.shape)
            display(features.groupby(["split_source", "window_minutes", "target"]).size().unstack(fill_value=0))
            display(features.head())
            """
        ),
        md("## 5. Models and Evaluation Helpers"),
        code(
            """
            META_COLUMNS = {
                "record", "subject_group", "split_source", "target", "window_minutes",
                "window_index", "start_sample", "end_sample", "fs",
            }
            NUMERIC_FEATURES = [c for c in features.columns if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(features[c])]
            print("numeric features:", len(NUMERIC_FEATURES))


            def build_models(seed: int = SEED) -> dict[str, Any]:
                models: dict[str, Any] = {
                    "Logistic Regression": Pipeline(
                        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))]
                    ),
                    "SVC RBF": Pipeline(
                        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("classifier", SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed))]
                    ),
                    "Random Forest": Pipeline(
                        [("imputer", SimpleImputer(strategy="median")), ("classifier", RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1))]
                    ),
                    "Extra Trees": Pipeline(
                        [("imputer", SimpleImputer(strategy="median")), ("classifier", ExtraTreesClassifier(n_estimators=700, class_weight="balanced", random_state=seed, n_jobs=-1))]
                    ),
                }
                if XGBClassifier is not None:
                    models["XGBoost"] = Pipeline(
                        [("imputer", SimpleImputer(strategy="median")), ("classifier", XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=450, max_depth=3, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=-1, tree_method="hist"))]
                    )
                if LGBMClassifier is not None:
                    models["LightGBM"] = Pipeline(
                        [("imputer", SimpleImputer(strategy="median")), ("classifier", LGBMClassifier(objective="binary", n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.9, colsample_bytree=0.9, class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1))]
                    )
                if CatBoostClassifier is not None:
                    models["CatBoost"] = Pipeline(
                        [("imputer", SimpleImputer(strategy="median")), ("classifier", CatBoostClassifier(iterations=500, depth=4, learning_rate=0.03, loss_function="Logloss", auto_class_weights="Balanced", random_seed=seed, verbose=False, allow_writing_files=False))]
                    )
                voting_estimators = [(name.lower().replace(" ", "_"), clone(model)) for name, model in models.items() if name in {"Random Forest", "Extra Trees", "XGBoost", "LightGBM", "CatBoost"}]
                if len(voting_estimators) >= 2:
                    models["Soft Voting Hybrid"] = VotingClassifier(voting_estimators, voting="soft", n_jobs=-1)
                    models["Stacking Hybrid"] = StackingClassifier(
                        estimators=voting_estimators,
                        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
                        cv=3,
                        stack_method="predict_proba",
                        n_jobs=-1,
                    )
                return models


            def binary_metrics(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray | None) -> dict[str, float]:
                return {
                    "accuracy": accuracy_score(y_true, pred),
                    "balanced_accuracy": balanced_accuracy_score(y_true, pred),
                    "precision": precision_score(y_true, pred, zero_division=0),
                    "recall": recall_score(y_true, pred, zero_division=0),
                    "f1": f1_score(y_true, pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_true, proba) if proba is not None and y_true.nunique() == 2 else np.nan,
                }


            def group_cv_splits(y: pd.Series, groups: pd.Series, max_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
                folds = min(max_folds, groups.nunique())
                if folds >= 2:
                    return list(GroupKFold(n_splits=folds).split(np.zeros(len(y)), y, groups))
                return list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
            """
        ),
        md("## 6. Subject-Level Cross-Validation"),
        code(
            """
            def evaluate_cv(data: pd.DataFrame, source_filter: str = "learning") -> pd.DataFrame:
                rows = []
                source_data = data[data["split_source"] == source_filter].copy()
                for window_minutes in sorted(source_data["window_minutes"].unique()):
                    working = source_data[source_data["window_minutes"] == window_minutes].copy()
                    if working["target"].nunique() < 2:
                        continue
                    x_data = working[NUMERIC_FEATURES]
                    y_data = working["target"].astype(int)
                    groups = working["subject_group"].astype(str)
                    splits = group_cv_splits(y_data, groups)
                    for name, model in build_models().items():
                        fold_rows = []
                        for train_idx, test_idx in splits:
                            try:
                                fitted = clone(model).fit(x_data.iloc[train_idx], y_data.iloc[train_idx])
                                proba = fitted.predict_proba(x_data.iloc[test_idx])[:, 1] if hasattr(fitted, "predict_proba") else None
                                pred = (proba >= 0.5).astype(int) if proba is not None else fitted.predict(x_data.iloc[test_idx])
                                fold_rows.append(binary_metrics(y_data.iloc[test_idx], pred, proba))
                            except Exception as exc:
                                print(f"Skipped {name} {window_minutes}m fold: {exc}")
                        if fold_rows:
                            rows.append({"window_minutes": window_minutes, "model": name, **pd.DataFrame(fold_rows).mean().to_dict()})
                return pd.DataFrame(rows).sort_values(["balanced_accuracy", "f1", "roc_auc"], ascending=False) if rows else pd.DataFrame()


            cv_results = evaluate_cv(features, "learning")
            display(cv_results)
            cv_results.to_csv(OUTPUT_DIR / "afpdb_subject_level_cv_results.csv", index=False)
            """
        ),
        md("## 7. Challenge-Test Evaluation"),
        code(
            """
            def evaluate_challenge_test(data: pd.DataFrame) -> pd.DataFrame:
                train = data[data["split_source"] == "learning"].copy()
                test = data[data["split_source"] == "challenge_test"].copy()
                if test.empty:
                    return pd.DataFrame()
                rows = []
                for window_minutes in sorted(set(train["window_minutes"]).intersection(set(test["window_minutes"]))):
                    train_w = train[train["window_minutes"] == window_minutes].copy()
                    test_w = test[test["window_minutes"] == window_minutes].copy()
                    if train_w["target"].nunique() < 2 or test_w["target"].nunique() < 2:
                        continue
                    x_train, y_train = train_w[NUMERIC_FEATURES], train_w["target"].astype(int)
                    x_test, y_test = test_w[NUMERIC_FEATURES], test_w["target"].astype(int)
                    for name, model in build_models().items():
                        try:
                            fitted = clone(model).fit(x_train, y_train)
                            proba = fitted.predict_proba(x_test)[:, 1] if hasattr(fitted, "predict_proba") else None
                            pred = (proba >= 0.5).astype(int) if proba is not None else fitted.predict(x_test)
                            rows.append({"window_minutes": window_minutes, "model": name, **binary_metrics(y_test, pred, proba)})
                        except Exception as exc:
                            print(f"Skipped test {name} {window_minutes}m: {exc}")
                return pd.DataFrame(rows).sort_values(["balanced_accuracy", "f1", "roc_auc"], ascending=False) if rows else pd.DataFrame()


            challenge_results = evaluate_challenge_test(features)
            display(challenge_results)
            challenge_results.to_csv(OUTPUT_DIR / "afpdb_challenge_test_results.csv", index=False)
            """
        ),
        md("## 8. Final Holdout, Curves, and Feature Importance"),
        code(
            """
            def final_holdout(data: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
                if cv_results.empty:
                    raise ValueError("No CV results available.")
                best = cv_results.iloc[0]
                best_window = int(best["window_minutes"])
                best_name = str(best["model"])
                models = build_models()
                working = data[(data["split_source"] == "learning") & (data["window_minutes"] == best_window)].copy()
                x_data = working[NUMERIC_FEATURES]
                y_data = working["target"].astype(int)
                groups = working["subject_group"].astype(str)
                splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
                train_idx, test_idx = next(splitter.split(x_data, y_data, groups=groups))
                model = clone(models[best_name]).fit(x_data.iloc[train_idx], y_data.iloc[train_idx])
                proba = model.predict_proba(x_data.iloc[test_idx])[:, 1] if hasattr(model, "predict_proba") else None
                pred = (proba >= 0.5).astype(int) if proba is not None else model.predict(x_data.iloc[test_idx])
                metrics = binary_metrics(y_data.iloc[test_idx], pred, proba)
                metrics.update({"model": best_name, "window_minutes": best_window})

                ConfusionMatrixDisplay.from_predictions(y_data.iloc[test_idx], pred, display_labels=["far/non-PAF", "pre-onset"])
                plt.title(f"AFPDB Holdout Confusion Matrix - {best_name} ({best_window} min)")
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / "afpdb_holdout_confusion_matrix.png", dpi=180)
                plt.show()

                if proba is not None:
                    RocCurveDisplay.from_predictions(y_data.iloc[test_idx], proba)
                    plt.title("AFPDB Holdout ROC Curve")
                    plt.tight_layout()
                    plt.savefig(OUTPUT_DIR / "afpdb_holdout_roc_curve.png", dpi=180)
                    plt.show()

                    PrecisionRecallDisplay.from_predictions(y_data.iloc[test_idx], proba)
                    plt.title("AFPDB Holdout Precision-Recall Curve")
                    plt.tight_layout()
                    plt.savefig(OUTPUT_DIR / "afpdb_holdout_precision_recall_curve.png", dpi=180)
                    plt.show()

                fitted_full = clone(models[best_name]).fit(x_data, y_data)
                joblib.dump(fitted_full, OUTPUT_DIR / "best_afpdb_prediction_model.joblib")
                classifier = fitted_full.named_steps.get("classifier") if hasattr(fitted_full, "named_steps") else fitted_full
                if hasattr(classifier, "feature_importances_"):
                    importance = pd.DataFrame({"feature": NUMERIC_FEATURES, "importance": classifier.feature_importances_}).sort_values("importance", ascending=False)
                    importance.to_csv(OUTPUT_DIR / "afpdb_feature_importance.csv", index=False)
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=importance.head(15), y="feature", x="importance")
                    plt.title("Top AFPDB Prediction Features")
                    plt.tight_layout()
                    plt.savefig(OUTPUT_DIR / "afpdb_feature_importance.png", dpi=180)
                    plt.show()
                return fitted_full, pd.DataFrame([metrics])


            final_model, holdout_metrics = final_holdout(features)
            display(holdout_metrics)
            holdout_metrics.to_csv(OUTPUT_DIR / "afpdb_holdout_metrics.csv", index=False)
            """
        ),
        md("## 9. Visuals and Summary"),
        code(
            """
            sns.set_theme(style="whitegrid")

            plt.figure(figsize=(10, 5))
            sns.countplot(data=features, x="window_minutes", hue="target")
            plt.title("AFPDB Window Counts by Horizon")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "afpdb_window_label_distribution.png", dpi=180)
            plt.show()

            if not cv_results.empty:
                plt.figure(figsize=(10, 5))
                sns.barplot(data=cv_results.head(12), y="model", x="balanced_accuracy", hue="window_minutes")
                plt.title("Top Subject-Level CV Results")
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / "afpdb_cv_results_plot.png", dpi=180)
                plt.show()

            if not challenge_results.empty:
                plt.figure(figsize=(10, 5))
                sns.barplot(data=challenge_results.head(12), y="model", x="balanced_accuracy", hue="window_minutes")
                plt.title("Top Challenge-Test Results")
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / "afpdb_challenge_results_plot.png", dpi=180)
                plt.show()

            summary = {
                "rows": len(features),
                "records": features["record"].nunique(),
                "subjects": features["subject_group"].nunique(),
                "numeric_features": len(NUMERIC_FEATURES),
                "best_cv_model": None if cv_results.empty else cv_results.iloc[0]["model"],
                "best_cv_window_minutes": None if cv_results.empty else int(cv_results.iloc[0]["window_minutes"]),
                "best_cv_balanced_accuracy": None if cv_results.empty else float(cv_results.iloc[0]["balanced_accuracy"]),
                "best_cv_accuracy": None if cv_results.empty else float(cv_results.iloc[0]["accuracy"]),
                "best_challenge_model": None if challenge_results.empty else challenge_results.iloc[0]["model"],
                "best_challenge_balanced_accuracy": None if challenge_results.empty else float(challenge_results.iloc[0]["balanced_accuracy"]),
                "holdout_accuracy": float(holdout_metrics.iloc[0]["accuracy"]),
                "holdout_balanced_accuracy": float(holdout_metrics.iloc[0]["balanced_accuracy"]),
                "holdout_roc_auc": float(holdout_metrics.iloc[0]["roc_auc"]),
            }
            pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "afpdb_experiment_summary.csv", index=False)
            summary
            """
        ),
        md(
            """
            ## 10. What This Adds to the Research

            This notebook should be reported separately from AF detection:

            - Current ECG project: **AF detection and rhythm classification**.
            - This notebook: **true paroxysmal AF onset prediction**.

            Suggested paper wording:

            > Alongside multi-database AF detection and rhythm classification, we add a true onset-prediction task on AFPDB. The model receives sinus-rhythm ECG excerpts and predicts whether the excerpt immediately precedes a future PAF episode, using subject-level validation to reduce leakage.

            Key outputs:

            - `afpdb_subject_level_cv_results.csv`
            - `afpdb_challenge_test_results.csv`
            - `afpdb_holdout_metrics.csv`
            - `afpdb_holdout_confusion_matrix.png`
            - `afpdb_holdout_roc_curve.png`
            - `afpdb_feature_importance.png`
            - `afpdb_experiment_summary.csv`
            """
        ),
    ]
    return nb


def main() -> None:
    """Write notebook."""

    output_path = Path.cwd() / "notebooks" / "03_afpdb_true_onset_prediction.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), output_path)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
