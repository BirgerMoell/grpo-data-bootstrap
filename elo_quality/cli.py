from __future__ import annotations

import argparse
import json
from pathlib import Path

from .features import FeatureConfig, build_feature_frame
from .io_utils import read_table, write_table
from .model_map import build_model_map, attach_elo, normalize_model_id
from .modeling import load_model, predict as run_predict, save_model, train


def cmd_build_map(args: argparse.Namespace) -> None:
    model_map = build_model_map(
        source_path=args.source,
        model_col=args.model_col,
        elo_col=args.elo_col,
        snapshot_col=args.snapshot_col,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model_map = model_map.rename(columns={"model_id_norm": "model_id"})
    model_map.to_csv(out, index=False)
    print(f"Wrote model map: {out} (rows: {len(model_map)})")


def cmd_train(args: argparse.Namespace) -> None:
    df = read_table(args.samples)
    if args.model_col not in df.columns:
        raise ValueError(f"Missing model column: {args.model_col}")

    if args.target_col:
        if args.target_col not in df.columns:
            raise ValueError(f"Missing target column: {args.target_col}")
    else:
        if not args.model_map:
            raise ValueError("Either --target-col or --model-map is required")
        df = attach_elo(df, args.model_map, args.model_col)
        if "elo" not in df.columns:
            raise ValueError("No elo column found after attaching map")
        df[args.target_col_generated] = df["elo"]

    target_col = args.target_col or args.target_col_generated
    df = df.dropna(subset=[args.model_col, target_col]).copy()

    feature_config = FeatureConfig(
        model_col=args.model_col,
        prompt_col=args.prompt_col if args.prompt_col else None,
        response_col=args.response_col if args.response_col else None,
        include_model_id=True,
        extra_numeric=tuple(args.numeric_features or []),
        extra_categorical=tuple(args.categorical_features or []),
    )
    feature_df, cat_cols, num_cols = build_feature_frame(
        df,
        feature_config,
        include_text_features=not args.no_text_features,
    )
    feature_df = feature_df[[c for c in cat_cols + num_cols if c in feature_df.columns]]
    if len(df) == 0:
        raise ValueError("No rows available after feature filtering and target mapping")

    result = train(
        df=feature_df.join(df[[target_col]]),
        model_col=args.model_col,
        target_col=target_col,
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        algorithm=args.algorithm,
        test_size=args.test_size,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = out_dir / "elo_model.joblib"
    save_model(
        result.model,
        artifact,
        {
            "model_col": args.model_col,
            "prompt_col": args.prompt_col,
            "response_col": args.response_col,
            "categorical_cols": cat_cols,
            "numeric_cols": num_cols,
            "algorithm": args.algorithm,
            "target_col": target_col,
        },
    )

    (out_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    print(f"Training done. metrics: {result.metrics}")
    print(f"Saved model: {artifact}")


def cmd_score(args: argparse.Namespace) -> None:
    model, metadata = load_model(args.model)
    df = read_table(args.samples)
    model_col = args.model_col or metadata.get("model_col")
    if model_col is None:
        raise ValueError("Model column is not known; pass --model-col")
    if model_col not in df.columns:
        raise ValueError(f"Missing model column: {model_col}")

    feature_config = FeatureConfig(
        model_col=model_col,
        prompt_col=args.prompt_col or metadata.get("prompt_col"),
        response_col=args.response_col or metadata.get("response_col"),
        include_model_id=True,
        extra_numeric=tuple(args.numeric_features or metadata.get("numeric_cols", [])),
        extra_categorical=tuple(args.categorical_features or metadata.get("categorical_cols", [])),
    )
    feature_df, cat_cols, num_cols = build_feature_frame(
        df,
        feature_config,
        include_text_features=not args.no_text_features,
    )

    X = feature_df[[c for c in cat_cols + num_cols if c in feature_df.columns]]
    preds = run_predict(model, X)
    out = df.copy()
    out[args.score_col] = preds
    write_table(out, args.out)

    print(f"Scored {len(out)} rows. Output: {args.out}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .modeling import train
    df = read_table(args.samples)
    if args.target_col not in df.columns:
        raise ValueError(f"target column missing: {args.target_col}")
    model, metadata = load_model(args.model)

    model_col = args.model_col or metadata.get("model_col")
    if model_col is None:
        raise ValueError("Model column is not known; pass --model-col")
    if model_col not in df.columns:
        raise ValueError(f"Missing model column: {model_col}")

    feature_config = FeatureConfig(
        model_col=model_col,
        prompt_col=args.prompt_col or metadata.get("prompt_col"),
        response_col=args.response_col or metadata.get("response_col"),
        include_model_id=True,
        extra_numeric=tuple(args.numeric_features or metadata.get("numeric_cols", [])),
        extra_categorical=tuple(args.categorical_features or metadata.get("categorical_cols", [])),
    )
    feature_df, cat_cols, num_cols = build_feature_frame(
        df,
        feature_config,
        include_text_features=not args.no_text_features,
    )

    X = feature_df[[c for c in cat_cols + num_cols if c in feature_df.columns]]
    y = df[args.target_col].astype(float).to_numpy()
    pred = run_predict(model, X)

    metrics = {
        "rmse": float(((y - pred) ** 2).mean() ** 0.5),
        "mae": float(abs(y - pred).mean()),
    }
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="grpo-elo", description="ELO proxy quality utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_map = sub.add_parser("build-model-map", help="Normalize and persist LMArena model->ELO mapping")
    p_map.add_argument("--source", required=True, help="Input CSV/JSON source with model + elo columns")
    p_map.add_argument("--model-col", required=True, default="model", help="Model id column")
    p_map.add_argument("--elo-col", required=True, default="elo", help="ELO column")
    p_map.add_argument("--snapshot-col", default=None, help="Optional snapshot/timestamp column")
    p_map.add_argument("--out", required=True, help="Output CSV path")
    p_map.set_defaults(func=cmd_build_map)

    p_train = sub.add_parser("train", help="Train regression model from samples")
    p_train.add_argument("--samples", required=True, help="Input CSV/JSON/JSONL with samples")
    p_train.add_argument("--model-col", default="model_id", help="Model id column")
    p_train.add_argument("--target-col", default=None, help="Numeric target column (if absent, uses model-map)")
    p_train.add_argument("--target-col-generated", default="elo_from_model_map", help="Generated target name from model map")
    p_train.add_argument("--model-map", default=None, help="Model map CSV/JSON with model_id + elo")
    p_train.add_argument("--prompt-col", default="prompt", help="Optional prompt column")
    p_train.add_argument("--response-col", default="response", help="Optional response column")
    p_train.add_argument("--numeric-features", nargs="*", default=[], help="Additional numeric features")
    p_train.add_argument("--categorical-features", nargs="*", default=[], help="Additional categorical features")
    p_train.add_argument("--algorithm", default="ridge", choices=["ridge", "elasticnet", "rf", "gbr"])
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--no-text-features", action="store_true", help="Disable auto-generated prompt/response length features")
    p_train.add_argument("--out-dir", required=True, help="Output directory for model artifacts")
    p_train.set_defaults(func=cmd_train)

    p_score = sub.add_parser("score", help="Score rows with trained model")
    p_score.add_argument("--model", required=True, help="Path to trained model artifact")
    p_score.add_argument("--samples", required=True, help="Input samples")
    p_score.add_argument("--out", required=True, help="Output file")
    p_score.add_argument("--model-col", default=None, help="Optional model column override")
    p_score.add_argument("--prompt-col", default=None, help="Optional prompt column override")
    p_score.add_argument("--response-col", default=None, help="Optional response column override")
    p_score.add_argument("--numeric-features", nargs="*", default=[], help="Additional numeric features")
    p_score.add_argument("--categorical-features", nargs="*", default=[], help="Additional categorical features")
    p_score.add_argument("--no-text-features", action="store_true", help="Disable auto-generated prompt/response length features")
    p_score.add_argument("--score-col", default="elo_pred", help="Output score column name")
    p_score.set_defaults(func=cmd_score)

    p_eval = sub.add_parser("evaluate", help="Evaluate trained model on a labeled file")
    p_eval.add_argument("--model", required=True, help="Path to trained model artifact")
    p_eval.add_argument("--samples", required=True, help="Input samples with labels")
    p_eval.add_argument("--target-col", required=True, help="Label column")
    p_eval.add_argument("--model-col", default=None, help="Optional model column override")
    p_eval.add_argument("--prompt-col", default=None, help="Optional prompt column override")
    p_eval.add_argument("--response-col", default=None, help="Optional response column override")
    p_eval.add_argument("--numeric-features", nargs="*", default=[], help="Additional numeric features")
    p_eval.add_argument("--categorical-features", nargs="*", default=[], help="Additional categorical features")
    p_eval.add_argument("--no-text-features", action="store_true", help="Disable auto-generated prompt/response length features")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
