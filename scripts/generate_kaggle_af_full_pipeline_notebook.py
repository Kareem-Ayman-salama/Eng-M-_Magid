"""Generate a full Kaggle AF inter-dataset ML notebook."""

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
    """Build a complete Kaggle notebook."""

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = [
        md(
            """
            # AF Prediction and Rhythm Classification - Full Kaggle Pipeline

            This notebook uses the three Kaggle inputs:

            - `/kaggle/input/long-term-af-database`
            - `/kaggle/input/mit-bih-atrial-fibrillation`
            - `/kaggle/input/shdb-af-atrial-fibrillation/shdb-af-a-japanese-holter-ecg-database-of-atrial-fibrillation-1.0.0`

            It builds a complete signal/annotation-based workflow:

            - WFDB record discovery.
            - Window-level preprocessing.
            - RR interval and ECG signal feature extraction.
            - Binary prediction: AF-related rhythm vs non-AF.
            - Multiclass rhythm classification.
            - Classical, boosting, and hybrid models.
            - Within-dataset, pooled, and inter-dataset validation.
            - Output tables and plots saved to `/kaggle/working/af_full_pipeline_outputs`.
            """
        ),
        md("## 1. Setup and Reproducibility"),
        code(
            """
            import importlib.util
            import os
            import random
            import re
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
                accuracy_score,
                balanced_accuracy_score,
                classification_report,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )
            from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, cross_validate
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import LabelEncoder, StandardScaler
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

            OUTPUT_DIR = Path("/kaggle/working/af_full_pipeline_outputs")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            print("Python:", sys.version)
            print("WFDB:", wfdb.__version__)
            print("Output:", OUTPUT_DIR)
            """
        ),
        md("## 2. Configuration"),
        code(
            """
            @dataclass(frozen=True)
            class Config:
                input_paths: dict[str, Path]
                output_dir: Path
                window_seconds: int = 60
                stride_seconds: int = 60
                max_windows_per_record: int = 80
                max_records_per_dataset: int | None = None
                min_beats_per_window: int = 5
                random_state: int = SEED


            CONFIG = Config(
                input_paths={
                    "long_term_af": Path("/kaggle/input/long-term-af-database"),
                    "mit_bih_af": Path("/kaggle/input/mit-bih-atrial-fibrillation"),
                    "shdb_af": Path("/kaggle/input/shdb-af-atrial-fibrillation/shdb-af-a-japanese-holter-ecg-database-of-atrial-fibrillation-1.0.0"),
                },
                output_dir=OUTPUT_DIR,
            )

            for name, path in CONFIG.input_paths.items():
                print(name, path, "exists:", path.exists())
            """
        ),
        md("## 3. Discover Records"),
        code(
            """
            def discover_records(config: Config) -> pd.DataFrame:
                rows: list[dict[str, Any]] = []
                for dataset_name, root in config.input_paths.items():
                    if not root.exists():
                        continue
                    hea_files = sorted(root.rglob("*.hea"))
                    if config.max_records_per_dataset is not None:
                        hea_files = hea_files[: config.max_records_per_dataset]
                    for hea_path in hea_files:
                        stem = hea_path.with_suffix("")
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "record": hea_path.stem,
                                "record_path": str(stem),
                                "has_dat": stem.with_suffix(".dat").exists(),
                                "has_atr": stem.with_suffix(".atr").exists(),
                                "has_qrs": stem.with_suffix(".qrs").exists(),
                                "has_ari": stem.with_suffix(".ari").exists(),
                            }
                        )
                return pd.DataFrame(rows)


            records = discover_records(CONFIG)
            print("records:", records.shape)
            display(records.groupby("dataset").agg(records=("record", "count"), atr=("has_atr", "sum"), qrs=("has_qrs", "sum"), dat=("has_dat", "sum")))
            display(records.head())
            records.to_csv(OUTPUT_DIR / "record_inventory.csv", index=False)
            """
        ),
        md("## 4. Annotation Parsing and Feature Engineering"),
        code(
            """
            AF_LABELS = {"AFIB", "AFL", "AF"}
            AF_DATASET_DEFAULTS = {"mit_bih_af": "AFIB"}
            CLINICAL_RHYTHM_LABELS = {"AFIB", "AFL", "AT", "PAT", "SVTA", "NOD", "NORMAL"}
            LABEL_MAP = {
                "(AFIB": "AFIB",
                "AFIB": "AFIB",
                "(AFL": "AFL",
                "AFL": "AFL",
                "(AF": "AFIB",
                "AF": "AFIB",
                "(AT": "AT",
                "AT": "AT",
                "(PAT": "PAT",
                "PAT": "PAT",
                "(SVTA": "SVTA",
                "SVTA": "SVTA",
                "(NOD": "NOD",
                "NOD": "NOD",
                "(N": "NORMAL",
                "N": "NORMAL",
                "(SBR": "OTHER",
                "(B": "OTHER",
                "(T": "OTHER",
                "AUX": "",
                "NOTE": "",
            }


            def read_header(record_path: str) -> tuple[float, int, int]:
                try:
                    header = wfdb.rdheader(record_path)
                    return float(header.fs), int(header.sig_len), int(header.n_sig)
                except Exception:
                    return 250.0, 0, 1


            def infer_signal_length(annotation: wfdb.Annotation | None, sig_len: int) -> int:
                if sig_len > 0:
                    return int(sig_len)
                if annotation is None or len(annotation.sample) == 0:
                    return 0
                return int(np.max(annotation.sample))


            def read_annotation_sources(record_path: str, extensions: list[tuple[str, bool]]) -> list[tuple[str, wfdb.Annotation]]:
                annotations: list[tuple[str, wfdb.Annotation]] = []
                for extension, enabled in extensions:
                    if not enabled:
                        continue
                    try:
                        annotations.append((extension, wfdb.rdann(record_path, extension)))
                    except Exception:
                        continue
                return annotations


            def read_annotation(record_path: str, extensions: list[tuple[str, bool]]) -> tuple[str, wfdb.Annotation] | None:
                annotations = read_annotation_sources(record_path, extensions)
                return annotations[0] if annotations else None


            def annotation_label_counts(annotation: wfdb.Annotation) -> pd.Series:
                labels = [normalize_label(value) for value in getattr(annotation, "aux_note", [])]
                labels = [label for label in labels if label]
                return pd.Series(labels, dtype="object").value_counts()


            def rhythm_source_score(annotation: wfdb.Annotation) -> float:
                counts = annotation_label_counts(annotation)
                if counts.empty:
                    return 0.0
                clinical_count = float(counts[counts.index.isin(CLINICAL_RHYTHM_LABELS)].sum())
                af_count = float(counts[counts.index.isin({"AFIB", "AFL", "AT", "PAT", "SVTA", "NOD"})].sum())
                normal_count = float(counts[counts.index == "NORMAL"].sum())
                return clinical_count + (2.0 * af_count) + (0.25 * normal_count)


            def read_rhythm_annotation(record_path: str, has_atr: bool, has_qrs: bool, has_ari: bool) -> tuple[str, wfdb.Annotation] | None:
                sources = read_annotation_sources(record_path, [("atr", has_atr), ("ari", has_ari), ("qrs", has_qrs)])
                if not sources:
                    return None
                scored_sources = [(source, annotation, rhythm_source_score(annotation)) for source, annotation in sources]
                scored_sources = [item for item in scored_sources if item[2] > 0]
                if not scored_sources:
                    return None
                best_source, best_annotation, _ = max(scored_sources, key=lambda item: item[2])
                return best_source, best_annotation


            def read_beat_annotation(record_path: str, has_atr: bool, has_qrs: bool, has_ari: bool) -> tuple[str, wfdb.Annotation] | None:
                return read_annotation(record_path, [("qrs", has_qrs), ("atr", has_atr), ("ari", has_ari)])


            def normalize_label(value: object) -> str:
                label = str(value).strip().replace("\\x00", "").replace("\\\\x00", "")
                label = re.sub(r"[^\x20-\x7E]", "", label).strip()
                label = label.replace("(", "").replace(")", "").strip().upper()
                if not label:
                    return ""
                return LABEL_MAP.get(label, label)


            def rhythm_segments(annotation: wfdb.Annotation, sig_len: int, default_label: str = "UNKNOWN") -> list[tuple[int, int, str]]:
                samples = np.asarray(annotation.sample, dtype=int)
                aux_notes = [normalize_label(value) for value in getattr(annotation, "aux_note", [])]
                rhythm_points = [(int(sample), label) for sample, label in zip(samples, aux_notes) if label in CLINICAL_RHYTHM_LABELS]
                if not rhythm_points:
                    return [(0, int(sig_len), default_label)] if sig_len else []
                segments = []
                for index, (start, label) in enumerate(rhythm_points):
                    end = rhythm_points[index + 1][0] if index + 1 < len(rhythm_points) else int(sig_len or samples[-1])
                    if end > start:
                        segments.append((start, end, label))
                return segments


            def beat_samples_in_window(annotation: wfdb.Annotation, start: int, end: int) -> np.ndarray:
                samples = np.asarray(annotation.sample, dtype=int)
                return samples[(samples >= start) & (samples < end)]


            def beat_based_windows(annotation: wfdb.Annotation, fs: float, sig_len: int, config: Config, label: str) -> list[tuple[int, int, str]]:
                samples = np.asarray(annotation.sample, dtype=int)
                if len(samples) < config.min_beats_per_window:
                    return []
                window = int(config.window_seconds * fs)
                stride = int(config.stride_seconds * fs)
                start = max(0, int(samples[0]) - window // 2)
                end_limit = int(sig_len or samples[-1])
                windows = []
                while start + window <= end_limit:
                    windows.append((start, start + window, label))
                    start += stride
                if not windows:
                    windows.append((max(0, int(samples[0]) - window // 2), min(end_limit, int(samples[-1]) + window // 2), label))
                return windows


            def rr_window_features(beat_samples: np.ndarray, fs: float) -> dict[str, float]:
                if len(beat_samples) < 3:
                    return {
                        "beat_count": float(len(beat_samples)),
                        "rr_mean": np.nan,
                        "rr_std": np.nan,
                        "rr_min": np.nan,
                        "rr_max": np.nan,
                        "rr_iqr": np.nan,
                        "rr_rmssd": np.nan,
                        "pnn50": np.nan,
                        "hr_mean": np.nan,
                    }
                rr = np.diff(beat_samples.astype(float)) / fs
                rr = rr[(rr > 0.2) & (rr < 3.0)]
                if len(rr) < 2:
                    return rr_window_features(np.array([], dtype=int), fs)
                diff_rr = np.diff(rr)
                return {
                    "beat_count": float(len(beat_samples)),
                    "rr_mean": float(np.mean(rr)),
                    "rr_std": float(np.std(rr)),
                    "rr_min": float(np.min(rr)),
                    "rr_max": float(np.max(rr)),
                    "rr_iqr": float(np.percentile(rr, 75) - np.percentile(rr, 25)),
                    "rr_rmssd": float(np.sqrt(np.mean(diff_rr**2))),
                    "pnn50": float(np.mean(np.abs(diff_rr) > 0.05)),
                    "hr_mean": float(60.0 / np.mean(rr)),
                }


            def signal_window_features(record_path: str, start: int, end: int, n_sig: int) -> dict[str, float]:
                features: dict[str, float] = {}
                try:
                    record = wfdb.rdrecord(record_path, sampfrom=start, sampto=end, channels=list(range(min(2, n_sig))))
                    signal = np.asarray(record.p_signal, dtype=float)
                except Exception:
                    signal = np.empty((0, 0))
                if signal.size == 0:
                    for channel in range(2):
                        for stat_name in ["mean", "std", "min", "max", "iqr", "energy", "diff_mean"]:
                            features[f"ch{channel}_{stat_name}"] = np.nan
                    return features
                for channel in range(min(2, signal.shape[1])):
                    values = signal[:, channel]
                    values = values[np.isfinite(values)]
                    if len(values) == 0:
                        continue
                    diff_values = np.diff(values)
                    features.update(
                        {
                            f"ch{channel}_mean": float(np.mean(values)),
                            f"ch{channel}_std": float(np.std(values)),
                            f"ch{channel}_min": float(np.min(values)),
                            f"ch{channel}_max": float(np.max(values)),
                            f"ch{channel}_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                            f"ch{channel}_energy": float(np.mean(values**2)),
                            f"ch{channel}_diff_mean": float(np.mean(np.abs(diff_values))) if len(diff_values) else 0.0,
                        }
                    )
                return features
            """
        ),
        md("## 5. Build or Load Window-Level Dataset"),
        code(
            """
            PREPROCESSING_VERSION = "v4_best_rhythm_source_prediction_classification"
            FEATURE_CACHE = OUTPUT_DIR / f"af_window_features_{PREPROCESSING_VERSION}.csv"


            def make_windows_for_segment(start: int, end: int, fs: float, config: Config) -> list[tuple[int, int]]:
                window = int(config.window_seconds * fs)
                stride = int(config.stride_seconds * fs)
                if end - start < max(window // 2, 1):
                    return []
                windows = []
                current = start
                while current + window <= end:
                    windows.append((current, current + window))
                    current += stride
                if not windows and end > start:
                    windows.append((start, end))
                return windows


            def extract_features_for_record(row: pd.Series, config: Config) -> list[dict[str, Any]]:
                record_path = row["record_path"]
                fs, sig_len, n_sig = read_header(record_path)
                rhythm_source = read_rhythm_annotation(record_path, bool(row["has_atr"]), bool(row["has_qrs"]), bool(row["has_ari"]))
                beat_source = read_beat_annotation(record_path, bool(row["has_atr"]), bool(row["has_qrs"]), bool(row["has_ari"]))
                rhythm_annotation = rhythm_source[1] if rhythm_source is not None else None
                rhythm_source_name = rhythm_source[0] if rhythm_source is not None else "fallback"
                beat_annotation = beat_source[1] if beat_source is not None else None
                beat_source_name = beat_source[0] if beat_source is not None else "none"
                sig_len = infer_signal_length(beat_annotation or rhythm_annotation, sig_len)
                if beat_annotation is None or sig_len <= 0:
                    return []
                default_label = AF_DATASET_DEFAULTS.get(str(row["dataset"]), "UNKNOWN")
                segments = rhythm_segments(rhythm_annotation or beat_annotation, sig_len, default_label=default_label)
                record_rows = []
                rng = np.random.default_rng(abs(hash((row["dataset"], row["record"]))) % (2**32))
                candidate_windows = []
                for start, end, label in segments:
                    for window_start, window_end in make_windows_for_segment(start, end, fs, config):
                        candidate_windows.append((window_start, window_end, label))
                if not candidate_windows:
                    candidate_windows = beat_based_windows(beat_annotation, fs, sig_len, config, default_label)
                if len(candidate_windows) > config.max_windows_per_record:
                    indices = rng.choice(len(candidate_windows), size=config.max_windows_per_record, replace=False)
                    candidate_windows = [candidate_windows[index] for index in sorted(indices)]
                for window_index, (start, end, label) in enumerate(candidate_windows):
                    beats = beat_samples_in_window(beat_annotation, start, end)
                    if len(beats) < config.min_beats_per_window:
                        continue
                    features = rr_window_features(beats, fs)
                    features.update(signal_window_features(record_path, start, end, n_sig))
                    normalized = normalize_label(label)
                    rhythm_label = normalized if normalized not in {"UNKNOWN", ""} else "OTHER"
                    binary_af = int(rhythm_label in {"AFIB", "AFL", "AF"})
                    features.update(
                        {
                            "dataset": row["dataset"],
                            "record": row["record"],
                            "window_id": f"{row['dataset']}__{row['record']}__{window_index}",
                            "start_sample": int(start),
                            "end_sample": int(end),
                            "fs": float(fs),
                            "rhythm_label": rhythm_label,
                            "binary_af": binary_af,
                            "rhythm_source": rhythm_source_name,
                            "beat_source": beat_source_name,
                        }
                    )
                    record_rows.append(features)
                if not record_rows and candidate_windows:
                    fallback_windows = beat_based_windows(beat_annotation, fs, sig_len, config, default_label)
                    if len(fallback_windows) > config.max_windows_per_record:
                        indices = rng.choice(len(fallback_windows), size=config.max_windows_per_record, replace=False)
                        fallback_windows = [fallback_windows[index] for index in sorted(indices)]
                    for window_index, (start, end, label) in enumerate(fallback_windows):
                        beats = beat_samples_in_window(beat_annotation, start, end)
                        if len(beats) < config.min_beats_per_window:
                            continue
                        features = rr_window_features(beats, fs)
                        features.update(signal_window_features(record_path, start, end, n_sig))
                        rhythm_label = normalize_label(label) or "OTHER"
                        if rhythm_label == "UNKNOWN":
                            rhythm_label = "OTHER"
                        features.update(
                            {
                                "dataset": row["dataset"],
                                "record": row["record"],
                                "window_id": f"{row['dataset']}__{row['record']}__fallback__{window_index}",
                                "start_sample": int(start),
                                "end_sample": int(end),
                                "fs": float(fs),
                                "rhythm_label": rhythm_label,
                                "binary_af": int(rhythm_label in {"AFIB", "AFL", "AF"}),
                                "rhythm_source": rhythm_source_name,
                                "beat_source": beat_source_name,
                            }
                        )
                        record_rows.append(features)
                return record_rows


            if FEATURE_CACHE.exists():
                windows = pd.read_csv(FEATURE_CACHE)
                print("Loaded cached features:", windows.shape)
            else:
                rows = []
                for index, record in records.iterrows():
                    if index % 10 == 0:
                        print(f"Processing {index + 1}/{len(records)}: {record['dataset']} {record['record']}")
                    rows.extend(extract_features_for_record(record, CONFIG))
                windows = pd.DataFrame(rows)
                windows.to_csv(FEATURE_CACHE, index=False)
                print("Saved:", FEATURE_CACHE)

            print(windows.shape)
            if windows.empty:
                raise ValueError("No windows were extracted. Check WFDB annotation extensions and Kaggle input paths.")
            display(windows.head())
            display(windows.groupby("dataset").agg(windows=("window_id", "count"), records=("record", "nunique"), af_rate=("binary_af", "mean")))
            display(windows.groupby(["dataset", "rhythm_source", "beat_source"]).size().reset_index(name="windows").sort_values(["dataset", "windows"], ascending=[True, False]))
            display(windows["rhythm_label"].value_counts().head(20))
            """
        ),
        md("## 6. Data Cleaning and Feature Matrix"),
        code(
            """
            META_COLUMNS = {"dataset", "record", "window_id", "start_sample", "end_sample", "fs", "rhythm_label", "binary_af", "rhythm_source", "beat_source"}
            NUMERIC_FEATURES = [column for column in windows.columns if column not in META_COLUMNS and pd.api.types.is_numeric_dtype(windows[column])]

            clean_windows = windows.copy()
            clinical_label_map = {
                "NORMAL": "NORMAL",
                "AFIB": "AFIB",
                "AF": "AFIB",
                "AFL": "AFL",
                "AT": "ATRIAL_TACHYCARDIA",
                "PAT": "ATRIAL_TACHYCARDIA",
                "SVTA": "ATRIAL_TACHYCARDIA",
            }
            clean_windows["rhythm_group"] = (
                clean_windows["rhythm_label"]
                .fillna("OTHER")
                .replace({"": "OTHER", "UNKNOWN": "OTHER"})
                .map(lambda value: clinical_label_map.get(str(value).upper(), "OTHER"))
            )
            counts = clean_windows["rhythm_group"].value_counts()
            keep = counts[counts >= 20].index
            clean_windows["rhythm_group"] = clean_windows["rhythm_group"].where(clean_windows["rhythm_group"].isin(keep), "OTHER_RARE")
            clean_windows = clean_windows.dropna(subset=["binary_af"])

            print("features:", len(NUMERIC_FEATURES))
            display(clean_windows.groupby(["dataset", "binary_af"]).size().unstack(fill_value=0))
            display(clean_windows["rhythm_group"].value_counts())
            """
        ),
        md("## 7. Models"),
        code(
            """
            def build_binary_models(seed: int = SEED) -> dict[str, Any]:
                models: dict[str, Any] = {
                    "Logistic Regression": Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
                        ]
                    ),
                    "SVC RBF": Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("classifier", SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)),
                        ]
                    ),
                    "Random Forest": Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("classifier", RandomForestClassifier(n_estimators=500, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=-1)),
                        ]
                    ),
                    "Extra Trees": Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("classifier", ExtraTreesClassifier(n_estimators=600, class_weight="balanced", random_state=seed, n_jobs=-1)),
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
                                    n_estimators=350,
                                    max_depth=3,
                                    learning_rate=0.03,
                                    subsample=0.85,
                                    colsample_bytree=0.85,
                                    random_state=seed,
                                    n_jobs=-1,
                                    tree_method="hist",
                                ),
                            ),
                        ]
                    )
                if LGBMClassifier is not None:
                    models["LightGBM"] = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "classifier",
                                LGBMClassifier(
                                    objective="binary",
                                    n_estimators=350,
                                    max_depth=-1,
                                    learning_rate=0.03,
                                    subsample=0.85,
                                    colsample_bytree=0.85,
                                    class_weight="balanced",
                                    random_state=seed,
                                    n_jobs=-1,
                                    verbose=-1,
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
                                    iterations=400,
                                    depth=4,
                                    learning_rate=0.03,
                                    loss_function="Logloss",
                                    auto_class_weights="Balanced",
                                    random_seed=seed,
                                    verbose=False,
                                    allow_writing_files=False,
                                ),
                            ),
                        ]
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


            def build_multiclass_models(seed: int = SEED) -> dict[str, Any]:
                models: dict[str, Any] = {
                    "Random Forest": Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("classifier", RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)),
                        ]
                    ),
                    "Extra Trees": Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("classifier", ExtraTreesClassifier(n_estimators=600, class_weight="balanced", random_state=seed, n_jobs=-1)),
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
                                    objective="multi:softprob",
                                    eval_metric="mlogloss",
                                    n_estimators=350,
                                    max_depth=3,
                                    learning_rate=0.03,
                                    subsample=0.85,
                                    colsample_bytree=0.85,
                                    random_state=seed,
                                    n_jobs=-1,
                                    tree_method="hist",
                                ),
                            ),
                        ]
                    )
                if LGBMClassifier is not None:
                    models["LightGBM"] = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "classifier",
                                LGBMClassifier(
                                    objective="multiclass",
                                    n_estimators=350,
                                    learning_rate=0.03,
                                    subsample=0.85,
                                    colsample_bytree=0.85,
                                    class_weight="balanced",
                                    random_state=seed,
                                    n_jobs=-1,
                                    verbose=-1,
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
                                    iterations=400,
                                    depth=4,
                                    learning_rate=0.03,
                                    loss_function="MultiClass",
                                    auto_class_weights="Balanced",
                                    random_seed=seed,
                                    verbose=False,
                                    allow_writing_files=False,
                                ),
                            ),
                        ]
                    )
                return models
            """
        ),
        md("## 8. Group-Aware Evaluation Helpers"),
        code(
            """
            def safe_group_folds(y: pd.Series, groups: pd.Series, max_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
                unique_groups = groups.nunique()
                folds = min(max_folds, unique_groups)
                if folds < 2:
                    stratified_folds = int(min(3, y.value_counts().min()))
                    if stratified_folds < 2:
                        return []
                    splitter = StratifiedKFold(n_splits=stratified_folds, shuffle=True, random_state=SEED)
                    return list(splitter.split(np.zeros(len(y)), y))
                splitter = GroupKFold(n_splits=folds)
                return list(splitter.split(np.zeros(len(y)), y, groups))


            def binary_metrics(y_true: pd.Series, y_pred: np.ndarray, proba: np.ndarray | None = None) -> dict[str, float]:
                metrics = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                }
                metrics["roc_auc"] = roc_auc_score(y_true, proba) if proba is not None and y_true.nunique() == 2 else np.nan
                return metrics


            def evaluate_binary_group_cv(data: pd.DataFrame, dataset_filter: str | None = None) -> pd.DataFrame:
                working = data.copy()
                if dataset_filter is not None:
                    working = working[working["dataset"] == dataset_filter].copy()
                if working["binary_af"].nunique() < 2 or len(working) < 20:
                    return pd.DataFrame()
                x_data = working[NUMERIC_FEATURES]
                y_data = working["binary_af"].astype(int)
                groups = working["record"].astype(str)
                splits = safe_group_folds(y_data, groups)
                if not splits:
                    return pd.DataFrame()
                rows = []
                for name, model in build_binary_models().items():
                    fold_rows = []
                    for train_idx, test_idx in splits:
                        try:
                            fitted = clone(model).fit(x_data.iloc[train_idx], y_data.iloc[train_idx])
                            pred = fitted.predict(x_data.iloc[test_idx])
                            proba = fitted.predict_proba(x_data.iloc[test_idx])[:, 1] if hasattr(fitted, "predict_proba") else None
                            fold_rows.append(binary_metrics(y_data.iloc[test_idx], pred, proba))
                        except Exception as exc:
                            print(f"Skipped {name} fold in {dataset_filter or 'pooled'}: {exc}")
                    if fold_rows:
                        rows.append({"scope": dataset_filter or "pooled", "model": name, **pd.DataFrame(fold_rows).mean().to_dict()})
                return pd.DataFrame(rows).sort_values(["balanced_accuracy", "f1", "roc_auc"], ascending=False) if rows else pd.DataFrame()


            def evaluate_multiclass_group_cv(data: pd.DataFrame, dataset_filter: str | None = None) -> pd.DataFrame:
                working = data.copy()
                if dataset_filter is not None:
                    working = working[working["dataset"] == dataset_filter].copy()
                if working["rhythm_group"].nunique() < 2 or len(working) < 20:
                    return pd.DataFrame()
                min_count = working["rhythm_group"].value_counts().min()
                if min_count < 2:
                    return pd.DataFrame()
                encoder = LabelEncoder()
                y_data = pd.Series(encoder.fit_transform(working["rhythm_group"]), index=working.index)
                x_data = working[NUMERIC_FEATURES]
                groups = working["record"].astype(str)
                splits = safe_group_folds(y_data, groups)
                if not splits:
                    return pd.DataFrame()
                rows = []
                for name, model in build_multiclass_models().items():
                    fold_rows = []
                    for train_idx, test_idx in splits:
                        try:
                            fitted = clone(model).fit(x_data.iloc[train_idx], y_data.iloc[train_idx])
                            pred = fitted.predict(x_data.iloc[test_idx])
                            truth = y_data.iloc[test_idx]
                            fold_rows.append(
                                {
                                    "accuracy": accuracy_score(truth, pred),
                                    "balanced_accuracy": balanced_accuracy_score(truth, pred),
                                    "macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
                                    "weighted_f1": f1_score(truth, pred, average="weighted", zero_division=0),
                                }
                            )
                        except Exception as exc:
                            print(f"Skipped {name} fold in {dataset_filter or 'pooled'}: {exc}")
                    if fold_rows:
                        rows.append({"scope": dataset_filter or "pooled", "model": name, "classes": "|".join(encoder.classes_), **pd.DataFrame(fold_rows).mean().to_dict()})
                return pd.DataFrame(rows).sort_values(["macro_f1", "balanced_accuracy"], ascending=False) if rows else pd.DataFrame()
            """
        ),
        md("## 9. Run Binary Prediction Experiments"),
        code(
            """
            binary_results = []
            binary_results.append(evaluate_binary_group_cv(clean_windows))
            for dataset_name in sorted(clean_windows["dataset"].unique()):
                result = evaluate_binary_group_cv(clean_windows, dataset_name)
                if not result.empty:
                    binary_results.append(result)

            binary_results = pd.concat([frame for frame in binary_results if not frame.empty], ignore_index=True) if binary_results else pd.DataFrame()
            if not binary_results.empty:
                binary_results = binary_results.sort_values(["scope", "balanced_accuracy", "f1", "roc_auc"], ascending=[True, False, False, False])
            display(binary_results)
            binary_results.to_csv(OUTPUT_DIR / "binary_prediction_results.csv", index=False)
            """
        ),
        md("## 10. Run Rhythm Classification Experiments"),
        code(
            """
            classification_results = []
            classification_results.append(evaluate_multiclass_group_cv(clean_windows))
            for dataset_name in sorted(clean_windows["dataset"].unique()):
                result = evaluate_multiclass_group_cv(clean_windows, dataset_name)
                if not result.empty:
                    classification_results.append(result)

            classification_results = pd.concat([frame for frame in classification_results if not frame.empty], ignore_index=True) if classification_results else pd.DataFrame()
            if not classification_results.empty:
                classification_results = classification_results.sort_values(["scope", "macro_f1", "balanced_accuracy"], ascending=[True, False, False])
            display(classification_results)
            classification_results.to_csv(OUTPUT_DIR / "rhythm_classification_results.csv", index=False)
            """
        ),
        md("## 11. Inter-Dataset Validation"),
        code(
            """
            def evaluate_inter_dataset_binary(data: pd.DataFrame) -> pd.DataFrame:
                rows = []
                datasets = sorted(data["dataset"].unique())
                for train_dataset in datasets:
                    for test_dataset in datasets:
                        if train_dataset == test_dataset:
                            continue
                        train = data[data["dataset"] == train_dataset].copy()
                        test = data[data["dataset"] == test_dataset].copy()
                        if train["binary_af"].nunique() < 2 or test["binary_af"].nunique() < 2:
                            continue
                        x_train, y_train = train[NUMERIC_FEATURES], train["binary_af"].astype(int)
                        x_test, y_test = test[NUMERIC_FEATURES], test["binary_af"].astype(int)
                        for name, model in build_binary_models().items():
                            try:
                                fitted = clone(model).fit(x_train, y_train)
                                pred = fitted.predict(x_test)
                                proba = fitted.predict_proba(x_test)[:, 1] if hasattr(fitted, "predict_proba") else None
                                rows.append(
                                    {
                                        "train_dataset": train_dataset,
                                        "test_dataset": test_dataset,
                                        "model": name,
                                        "train_windows": len(train),
                                        "test_windows": len(test),
                                        **binary_metrics(y_test, pred, proba),
                                    }
                                )
                            except Exception as exc:
                                print(f"Skipped {name} {train_dataset}->{test_dataset}: {exc}")
                return pd.DataFrame(rows).sort_values(["roc_auc", "f1"], ascending=False) if rows else pd.DataFrame()


            inter_dataset_results = evaluate_inter_dataset_binary(clean_windows)
            display(inter_dataset_results)
            inter_dataset_results.to_csv(OUTPUT_DIR / "inter_dataset_binary_results.csv", index=False)
            """
        ),
        md("## 12. Final Model, Confusion Matrix, and Feature Importance"),
        code(
            """
            def fit_best_binary_model(data: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
                if binary_results.empty:
                    raise ValueError("No binary results available.")
                pooled_results = binary_results[binary_results["scope"] == "pooled"].copy()
                ranking_source = pooled_results if not pooled_results.empty else binary_results.copy()
                ranking_source = ranking_source.sort_values(["balanced_accuracy", "f1", "roc_auc"], ascending=False)
                best_name = ranking_source.iloc[0]["model"]
                models = build_binary_models()
                model = clone(models[best_name])
                x_data = data[NUMERIC_FEATURES]
                y_data = data["binary_af"].astype(int)
                splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                train_idx, test_idx = next(splitter.split(x_data, y_data, groups=data["record"].astype(str)))
                train_groups = data.iloc[train_idx]["record"].astype(str)
                threshold = 0.5
                if train_groups.nunique() >= 3:
                    tune_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
                    subtrain_rel, valid_rel = next(tune_splitter.split(x_data.iloc[train_idx], y_data.iloc[train_idx], groups=train_groups))
                    subtrain_idx = np.asarray(train_idx)[subtrain_rel]
                    valid_idx = np.asarray(train_idx)[valid_rel]
                    tune_model = clone(models[best_name]).fit(x_data.iloc[subtrain_idx], y_data.iloc[subtrain_idx])
                    if hasattr(tune_model, "predict_proba") and y_data.iloc[valid_idx].nunique() == 2:
                        valid_proba = tune_model.predict_proba(x_data.iloc[valid_idx])[:, 1]
                        candidates = np.linspace(0.1, 0.9, 81)
                        scores = [
                            balanced_accuracy_score(y_data.iloc[valid_idx], (valid_proba >= candidate).astype(int))
                            for candidate in candidates
                        ]
                        threshold = float(candidates[int(np.argmax(scores))])
                model.fit(x_data.iloc[train_idx], y_data.iloc[train_idx])
                proba = model.predict_proba(x_data.iloc[test_idx])[:, 1] if hasattr(model, "predict_proba") else None
                pred = (proba >= threshold).astype(int) if proba is not None else model.predict(x_data.iloc[test_idx])
                metrics = binary_metrics(y_data.iloc[test_idx], pred, proba)
                metrics["threshold"] = threshold
                print("Best model:", best_name, metrics)
                ConfusionMatrixDisplay.from_predictions(y_data.iloc[test_idx], pred, display_labels=["non-AF", "AF"])
                plt.title(f"Holdout Confusion Matrix - {best_name}")
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / "binary_holdout_confusion_matrix.png", dpi=180)
                plt.show()

                fitted_full = clone(models[best_name]).fit(x_data, y_data)
                joblib.dump(fitted_full, OUTPUT_DIR / "best_binary_model.joblib")
                return fitted_full, pd.DataFrame([{"model": best_name, **metrics}])


            if not binary_results.empty:
                best_binary_model, holdout_metrics = fit_best_binary_model(clean_windows)
                display(holdout_metrics)
                holdout_metrics.to_csv(OUTPUT_DIR / "binary_holdout_metrics.csv", index=False)
            """
        ),
        md("## 13. Plots and Summary"),
        code(
            """
            sns.set_theme(style="whitegrid")

            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            sns.countplot(data=clean_windows, x="dataset", ax=axes[0])
            axes[0].set_title("Windows per Dataset")
            axes[0].tick_params(axis="x", rotation=25)
            sns.countplot(data=clean_windows, x="binary_af", ax=axes[1])
            axes[1].set_title("Binary AF Label")
            top_labels = clean_windows["rhythm_group"].value_counts().head(8).reset_index()
            top_labels.columns = ["rhythm_group", "count"]
            sns.barplot(data=top_labels, y="rhythm_group", x="count", ax=axes[2])
            axes[2].set_title("Rhythm Groups")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "dataset_summary_plots.png", dpi=180)
            plt.show()

            if not binary_results.empty:
                top_binary = binary_results.head(10).copy()
                plt.figure(figsize=(10, 5))
                sns.barplot(data=top_binary, y="model", x="balanced_accuracy", hue="scope")
                plt.title("Top Binary Prediction Results")
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / "binary_results_plot.png", dpi=180)
                plt.show()

            if not classification_results.empty:
                top_multi = classification_results.head(10).copy()
                plt.figure(figsize=(10, 5))
                sns.barplot(data=top_multi, y="model", x="macro_f1", hue="scope")
                plt.title("Top Rhythm Classification Results")
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / "classification_results_plot.png", dpi=180)
                plt.show()

            summary = {
                "windows": len(clean_windows),
                "records": clean_windows["record"].nunique(),
                "datasets": clean_windows["dataset"].nunique(),
                "numeric_features": len(NUMERIC_FEATURES),
                "best_binary_model": None if binary_results.empty else binary_results.iloc[0]["model"],
                "best_binary_balanced_accuracy": None if binary_results.empty else float(binary_results.iloc[0]["balanced_accuracy"]),
                "best_classification_model": None if classification_results.empty else classification_results.iloc[0]["model"],
                "best_classification_macro_f1": None if classification_results.empty else float(classification_results.iloc[0]["macro_f1"]),
            }
            pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False)
            summary
            """
        ),
        md(
            """
            ## 14. What To Report

            After running the notebook once, use these files from `/kaggle/working/af_full_pipeline_outputs`:

            - `af_window_features.csv`: extracted window-level dataset.
            - `binary_prediction_results.csv`: AF prediction results.
            - `rhythm_classification_results.csv`: rhythm classification results.
            - `inter_dataset_binary_results.csv`: generalization between datasets.
            - `binary_holdout_confusion_matrix.png`: final confusion matrix.
            - `best_binary_model.joblib`: saved best model.
            - `experiment_summary.csv`: concise summary.

            Research positioning:

            - UCI Arrhythmia remains the tabular baseline.
            - This notebook adds signal/annotation-based ECG prediction and classification.
            - Inter-dataset validation is the strongest contribution because it tests generalization across different ECG databases.
            """
        ),
    ]
    return nb


def main() -> None:
    """Write the generated notebook."""

    output_path = Path.cwd() / "notebooks" / "02_kaggle_af_full_pipeline.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), output_path)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
