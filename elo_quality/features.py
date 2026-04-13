from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

from .io_utils import add_length_features


@dataclass(frozen=True)
class FeatureConfig:
    model_col: str = "model_id"
    prompt_col: str | None = "prompt"
    response_col: str | None = "response"
    include_model_id: bool = True
    extra_numeric: tuple[str, ...] = ()
    extra_categorical: tuple[str, ...] = ()


def build_feature_frame(
    df: pd.DataFrame,
    config: FeatureConfig,
    include_text_features: bool = True,
) -> tuple[pd.DataFrame, List[str], List[str]]:
    work = df.copy()
    text_cols = [c for c in [config.prompt_col, config.response_col] if c]
    if include_text_features:
        work = add_length_features(work, text_cols, "len")

    cat_cols: List[str] = []
    num_cols: List[str] = []

    if config.include_model_id:
        cat_cols.append(config.model_col)
    cat_cols.extend([c for c in config.extra_categorical if c and c in work.columns])

    if include_text_features and config.prompt_col in work.columns:
        num_cols.extend([f"len_{config.prompt_col}_chars", f"len_{config.prompt_col}_words"])
    if include_text_features and config.response_col in work.columns:
        num_cols.extend([f"len_{config.response_col}_chars", f"len_{config.response_col}_words"])

    num_cols.extend([c for c in config.extra_numeric if c and c in work.columns])

    return work, cat_cols, num_cols
