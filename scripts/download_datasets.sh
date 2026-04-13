#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw

echo "Checking for huggingface-cli..."
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Install with: pip install -U huggingface_hub"
  exit 1
fi

declare -a DATASETS=(
  "open-r1/mixture-of-thoughts"
  "open-r1/OpenR1-Math-220k"
  "open-r1/codeforces-cots"
  "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
)

for ds in "${DATASETS[@]}"; do
  safe_name="$(echo "$ds" | tr '/' '__')"
  echo "Downloading $ds to data/raw/$safe_name ..."
  huggingface-cli download "$ds" --repo-type dataset --local-dir "data/raw/$safe_name" --local-dir-use-symlinks False
done

echo "Done. Update scripts/convert_to_grpo_jsonl.py with your chosen columns."
