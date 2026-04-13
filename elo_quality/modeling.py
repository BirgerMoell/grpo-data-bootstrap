from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _make_regressor(name: str, seed: int):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "elasticnet":
        return ElasticNet(random_state=seed, alpha=0.001, l1_ratio=0.2, max_iter=10000)
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "gbr":
        return GradientBoostingRegressor(random_state=seed)
    raise ValueError(f"Unsupported algorithm: {name}")


def make_pipeline(
    categorical_cols: list[str],
    numeric_cols: list[str],
    algorithm: str,
    seed: int,
) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return Pipeline(steps=[("preprocess", preprocess), ("model", _make_regressor(algorithm, seed))])


@dataclass
class TrainingResult:
    model: Pipeline
    metrics: dict[str, Any]


def train(
    df: pd.DataFrame,
    model_col: str,
    target_col: str,
    categorical_cols: list[str],
    numeric_cols: list[str],
    algorithm: str = "ridge",
    test_size: float = 0.2,
    seed: int = 42,
) -> TrainingResult:
    if target_col not in df.columns:
        raise ValueError(f"target column missing: {target_col}")

    if model_col not in categorical_cols:
        categorical_cols = [model_col] + [c for c in categorical_cols if c != model_col]

    X = df[categorical_cols + numeric_cols].copy()
    y = df[target_col].astype(float).to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True
    )
    pipe = make_pipeline(categorical_cols, numeric_cols, algorithm, seed)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_val)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
        "mae": float(mean_absolute_error(y_val, pred)),
        "r2": float(r2_score(y_val, pred)),
    }
    try:
        metrics["spearman"] = float(pd.Series(y_val).corr(pd.Series(pred), method="spearman"))
    except Exception:
        metrics["spearman"] = float("nan")
    metrics["n_train"] = int(len(X_train))
    metrics["n_val"] = int(len(X_val))
    metrics["algorithm"] = algorithm
    return TrainingResult(model=pipe, metrics=metrics)


def predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def save_model(model: Pipeline, out_path: str | Path, metadata: dict[str, Any]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, out_path)


def load_model(path: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    payload = joblib.load(path)
    return payload["model"], payload.get("metadata", {})
