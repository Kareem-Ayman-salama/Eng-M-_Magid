"""Advanced UCI Arrhythmia prediction and classification benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class AdvancedConfig:
    """Advanced benchmark configuration."""

    project_root: Path
    output_dir: Path
    random_state: int = 42


class FeatureTokenizerTransformer(nn.Module):
    """Small FT-Transformer for tabular binary classification."""

    def __init__(self, n_features: int, token_dim: int = 24, n_heads: int = 4) -> None:
        """Initialize the model."""

        super().__init__()
        self.feature_weight = nn.Parameter(torch.empty(n_features, token_dim))
        self.feature_bias = nn.Parameter(torch.zeros(n_features, token_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 3,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(token_dim, 1),
        )
        nn.init.xavier_uniform_(self.feature_weight)

    def forward(self, x_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""

        tokens = x_values.unsqueeze(-1) * self.feature_weight + self.feature_bias
        cls = self.cls_token.expand(x_values.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0]).squeeze(-1)


def load_data(project_root: Path) -> pd.DataFrame:
    """Load UCI Arrhythmia dataset."""

    path = project_root / "data" / "uci_arrhythmia" / "arrhythmia.data"
    data = pd.read_csv(path, header=None, na_values="?")
    data.columns = [*[f"feature_{index:03d}" for index in range(data.shape[1] - 1)], "arrhythmia_class"]
    data["arrhythmia_class"] = data["arrhythmia_class"].astype(int)
    data["binary_target"] = (data["arrhythmia_class"] != 1).astype(int)
    return data.copy()


def split_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return feature dataframe."""

    return data.drop(columns=["arrhythmia_class", "binary_target"])


def grouped_target(target: pd.Series) -> pd.Series:
    """Group rare classes into class 99."""

    counts = target.value_counts()
    keep = counts[counts >= 20].index
    return target.where(target.isin(keep), other=99).astype(int)


def build_preprocessor(x_data: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline."""

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
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
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


def optional_xgb(random_state: int, multiclass: bool) -> BaseEstimator | None:
    """Build XGBoost if available."""

    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return XGBClassifier(
        objective="multi:softprob" if multiclass else "binary:logistic",
        eval_metric="mlogloss" if multiclass else "logloss",
        n_estimators=350,
        max_depth=3,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=4.0,
        random_state=random_state,
        n_jobs=-1,
    )


def optional_lgbm(random_state: int, multiclass: bool) -> BaseEstimator | None:
    """Build LightGBM if available."""

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None
    return LGBMClassifier(
        objective="multiclass" if multiclass else "binary",
        n_estimators=350,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=5,
        min_child_samples=8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def optional_catboost(random_state: int, multiclass: bool) -> BaseEstimator | None:
    """Build CatBoost if available."""

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None
    return CatBoostClassifier(
        loss_function="MultiClass" if multiclass else "Logloss",
        eval_metric="TotalF1" if multiclass else "AUC",
        iterations=350,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=6,
        auto_class_weights="Balanced",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )


def build_models(random_state: int, multiclass: bool = False) -> dict[str, BaseEstimator]:
    """Build advanced candidate models."""

    models: dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=600,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=700,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if not multiclass:
        models["SVC RBF"] = SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=random_state)

    for name, model in [
        ("XGBoost", optional_xgb(random_state, multiclass)),
        ("LightGBM", optional_lgbm(random_state, multiclass)),
        ("CatBoost", optional_catboost(random_state, multiclass)),
    ]:
        if model is not None:
            models[name] = model

    if not multiclass:
        base = [(name.lower().replace(" ", "_"), clone(model)) for name, model in models.items()]
        models["Soft Voting Hybrid"] = VotingClassifier(base, voting="soft", n_jobs=-1)
        models["Stacking Hybrid"] = StackingClassifier(
            estimators=base,
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
            cv=5,
            stack_method="predict_proba",
            n_jobs=-1,
        )
        xgb = models.get("XGBoost")
        if xgb is not None:
            models["MI SelectKBest + XGBoost"] = Pipeline(
                [
                    (
                        "selector",
                        SelectKBest(
                            score_func=lambda x_values, y_values: mutual_info_classif(
                                x_values,
                                y_values,
                                random_state=random_state,
                            ),
                            k=80,
                        ),
                    ),
                    ("classifier", clone(xgb)),
                ]
            )
            models["ANOVA SelectKBest + XGBoost"] = Pipeline(
                [("selector", SelectKBest(f_classif, k=80)), ("classifier", clone(xgb))]
            )
    return models


def encode_if_needed(y_data: pd.Series, multiclass: bool) -> tuple[pd.Series, LabelEncoder | None]:
    """Encode multiclass labels for libraries that require contiguous labels."""

    if not multiclass:
        return y_data.astype(int), None
    encoder = LabelEncoder()
    return pd.Series(encoder.fit_transform(y_data), index=y_data.index), encoder


def evaluate_models(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    models: dict[str, BaseEstimator],
    random_state: int,
    multiclass: bool,
) -> pd.DataFrame:
    """Evaluate models with cross-validation."""

    y_eval, _ = encode_if_needed(y_data, multiclass)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        print(f"Evaluating {name}")
        pipeline = Pipeline([("preprocessor", build_preprocessor(x_data)), ("classifier", clone(model))])
        if not multiclass:
            scores = cross_validate(
                pipeline,
                x_data,
                y_eval,
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
            rows.append({"task": "Binary prediction", "model": name, **{k.replace("test_", ""): v.mean() for k, v in scores.items() if k.startswith("test_")}})
        else:
            fold_rows = []
            for train_idx, test_idx in cv.split(x_data, y_eval):
                pipeline.fit(x_data.iloc[train_idx], y_eval.iloc[train_idx])
                pred = pipeline.predict(x_data.iloc[test_idx])
                truth = y_eval.iloc[test_idx]
                fold_rows.append(
                    {
                        "accuracy": accuracy_score(truth, pred),
                        "balanced_accuracy": balanced_accuracy_score(truth, pred),
                        "macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
                        "weighted_f1": f1_score(truth, pred, average="weighted", zero_division=0),
                    }
                )
            rows.append({"task": "Grouped multiclass classification", "model": name, **pd.DataFrame(fold_rows).mean().to_dict()})
    sort_columns = ["roc_auc", "f1"] if not multiclass else ["macro_f1", "balanced_accuracy"]
    return pd.DataFrame(rows).sort_values(sort_columns, ascending=False)


def train_ft_transformer_binary(
    x_data: pd.DataFrame,
    y_data: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    """Train a small FT-Transformer binary holdout baseline."""

    torch.manual_seed(random_state)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        stratify=y_data,
        random_state=random_state,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state,
    )
    preprocessor = build_preprocessor(x_train)
    train_x = np.asarray(preprocessor.fit_transform(x_train), dtype=np.float32)
    valid_x = np.asarray(preprocessor.transform(x_valid), dtype=np.float32)
    test_x = np.asarray(preprocessor.transform(x_test), dtype=np.float32)
    train_y = y_train.to_numpy(dtype=np.float32)

    loader = DataLoader(
        TensorDataset(torch.tensor(train_x), torch.tensor(train_y)),
        batch_size=32,
        shuffle=True,
    )
    model = FeatureTokenizerTransformer(train_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    pos_weight = torch.tensor([(len(train_y) - train_y.sum()) / max(train_y.sum(), 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    epochs = 0
    for epoch in range(1, 80):
        epochs = epoch
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_prob = torch.sigmoid(model(torch.tensor(valid_x))).numpy()
        valid_auc = roc_auc_score(y_valid, valid_prob)
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 12:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_prob = torch.sigmoid(model(torch.tensor(test_x))).numpy()
    pred = (test_prob >= 0.5).astype(int)
    return pd.DataFrame(
        [
            {
                "task": "Binary prediction holdout",
                "model": "Feature Tokenizer Transformer",
                "epochs": epochs,
                "best_valid_auc": best_auc,
                "accuracy": accuracy_score(y_test, pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, test_prob),
                "pr_auc": average_precision_score(y_test, test_prob),
            }
        ]
    )


def main() -> None:
    """Run advanced UCI Arrhythmia benchmark."""

    project_root = Path.cwd()
    config = AdvancedConfig(project_root, project_root / "outputs" / "uci_arrhythmia_advanced")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(project_root)
    x_data = split_features(data)
    y_binary = data["binary_target"]
    y_grouped = grouped_target(data["arrhythmia_class"])

    summary = pd.DataFrame(
        [
            {
                "rows": len(data),
                "features": x_data.shape[1],
                "original_classes": data["arrhythmia_class"].nunique(),
                "grouped_classes": y_grouped.nunique(),
                "missing_cells": int(data.isna().sum().sum()),
            }
        ]
    )
    binary_results = evaluate_models(x_data, y_binary, build_models(config.random_state, False), config.random_state, False)
    multiclass_results = evaluate_models(x_data, y_grouped, build_models(config.random_state, True), config.random_state, True)
    transformer_results = train_ft_transformer_binary(x_data, y_binary, config.random_state)

    summary.to_csv(config.output_dir / "dataset_summary.csv", index=False)
    binary_results.to_csv(config.output_dir / "advanced_binary_results.csv", index=False)
    multiclass_results.to_csv(config.output_dir / "advanced_grouped_multiclass_results.csv", index=False)
    transformer_results.to_csv(config.output_dir / "ft_transformer_binary_holdout.csv", index=False)
    print(summary.to_string(index=False))
    print(binary_results.to_string(index=False))
    print(multiclass_results.to_string(index=False))
    print(transformer_results.to_string(index=False))


if __name__ == "__main__":
    main()
