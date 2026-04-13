from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV/JSON/JSONL data into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    ext = path.suffix.lower()
    if ext in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if ext == ".tsv" else ",")
    if ext == ".jsonl":
        return pd.read_json(path, lines=True, orient="records")
    if ext == ".json":
        return pd.read_json(path)
    if ext in {".parquet"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
        return
    if ext == ".parquet":
        df.to_parquet(path, index=False)
        return
    if ext in {".json", ".ndjson"}:
        df.to_json(path, orient="records", force_ascii=False)
        return
    df.to_csv(path, index=False)


@dataclass(frozen=True)
class TextColumnStats:
    char_length_col: str
    word_length_col: str


def add_length_features(df: pd.DataFrame, columns: Iterable[str], prefix: str) -> pd.DataFrame:
    """Add char and word count features for the selected text columns."""
    for col in columns:
        if col not in df.columns:
            continue
        df[f"{prefix}_{col}_chars"] = df[col].fillna("").astype(str).str.len()
        df[f"{prefix}_{col}_words"] = df[col].fillna("").astype(str).str.split().map(len)
    return df
