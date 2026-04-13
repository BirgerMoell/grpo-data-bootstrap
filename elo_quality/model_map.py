from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


def normalize_model_id(value: object) -> str:
    return str(value).strip().lower()


@dataclass(frozen=True)
class ModelEloRecord:
    model_id: str
    elo: float
    snapshot_at: Optional[str] = None


def build_model_map(
    source_path: str | Path,
    model_col: str,
    elo_col: str,
    snapshot_col: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(source_path) if str(source_path).endswith(".csv") else pd.read_json(source_path)
    if model_col not in df.columns:
        raise ValueError(f"model column '{model_col}' not present in source")
    if elo_col not in df.columns:
        raise ValueError(f"elo column '{elo_col}' not present in source")

    df = df[[model_col, elo_col] + ([snapshot_col] if snapshot_col else [])].copy()
    df["model_id_norm"] = df[model_col].map(normalize_model_id)

    # If snapshot exists, use latest snapshot per model_id (most recent date).
    if snapshot_col and snapshot_col in df.columns:
        df[snapshot_col] = pd.to_datetime(df[snapshot_col], utc=True, errors="coerce")
        df = df.sort_values([ "model_id_norm", snapshot_col])
        keep_idx = df.groupby("model_id_norm")[snapshot_col].transform(max) == df[snapshot_col]
        df = df[keep_idx]
    else:
        df["snapshot_at"] = None

    out = df.drop_duplicates("model_id_norm", keep="last").copy()
    out["elo"] = pd.to_numeric(out[elo_col], errors="coerce")
    out = out.dropna(subset=["elo", "model_id_norm"])
    out = out[["model_id_norm", "elo", "snapshot_at"]].rename(columns={"snapshot_at": "snapshot_at_iso"})
    return out.reset_index(drop=True)


def attach_elo(df: pd.DataFrame, model_map_path: str | Path, model_col: str) -> pd.DataFrame:
    model_map = pd.read_csv(model_map_path) if str(model_map_path).endswith(".csv") else pd.read_json(model_map_path)
    if "model_id" not in model_map.columns:
        raise ValueError("model map must include 'model_id'")
    if "model_id" in model_map.columns and "model_id_norm" not in model_map.columns:
        model_map["model_id_norm"] = model_map["model_id"].map(normalize_model_id)
    else:
        model_map["model_id_norm"] = model_map["model_id_norm"].map(normalize_model_id)

    if "elo" not in model_map.columns:
        raise ValueError("model map must include 'elo'")

    out = df.copy()
    out["model_id_norm"] = out[model_col].map(normalize_model_id)
    out = out.merge(
        model_map[["model_id_norm", "elo"]].drop_duplicates("model_id_norm", keep="first"),
        on="model_id_norm",
        how="left",
    )
    return out
