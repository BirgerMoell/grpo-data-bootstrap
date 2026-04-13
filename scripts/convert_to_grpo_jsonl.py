#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict

from datasets import load_dataset


def parse_row(row: Dict[str, Any], prompt_key: str, response_key: str, reward_key: str | None):
    prompt = str(row.get(prompt_key, "")).strip()
    response = str(row.get(response_key, "")).strip()
    reward = 0.0
    if reward_key and reward_key in row:
        try:
            reward = float(row.get(reward_key))
        except (TypeError, ValueError):
            reward = 0.0

    if not prompt or not response:
        return None

    return {
        "prompt": prompt,
        "response": response,
        "reward": reward,
        "meta": {
            "source_row": {k: str(v) for k, v in row.items() if k not in {prompt_key, response_key, reward_key}},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HF datasets into generic GRPO JSONL format.")
    parser.add_argument("--dataset", required=True, help="HF dataset id, e.g. open-r1/mixture-of-thoughts")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--prompt-col", required=True, help="Column containing prompt / task / question")
    parser.add_argument("--response-col", required=True, help="Column containing generated answer/completion")
    parser.add_argument("--reward-col", default="", help="Optional reward/score column")
    parser.add_argument("--out", required=True, help="Output jsonl path")
    parser.add_argument("--limit", type=int, default=0, help="Optional row cap; 0 = no cap")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    reward_col = args.reward_col if args.reward_col else None

    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            item = parse_row(row, args.prompt_col, args.response_col, reward_col)
            if not item:
                continue
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")
            count += 1
            if args.limit and count >= args.limit:
                break

    print(f"Wrote {count} rows to {args.out}")


if __name__ == "__main__":
    main()
