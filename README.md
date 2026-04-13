# GRPO Data Bootstrap Kit

This repo is a practical starter for building a **reasoning-focused GRPO training
pipeline** with the OpenEuroLLM stack (`oellm-autoexp`), especially for language
models where you want to improve reasoning with reinforcement learning.

It includes:

- A curated list of strong open reasoning and reward datasets
- A concrete `oellm-autoexp` config template for GRPO
- Minimal scripts to stage data and convert common data formats into a GRPO-friendly JSONL
- A staged curriculum plan you can follow directly

## Repository structure

- `manifests/reasoning_dataset_catalogue.yaml` – candidate reasoning data sources
- `manifests/reward_dataset_catalogue.yaml` – reward / preference data sources
- `configs/oellm_autoexp_grpo_example.yaml` – starter GRPO config
- `scripts/download_datasets.sh` – fetch HF datasets (requires `huggingface-cli`)
- `scripts/convert_to_grpo_jsonl.py` – normalize dataset rows into generic format

## Quick start

1. Install prerequisites

```bash
pip install datasets huggingface_hub pyyaml
```

2. Download source datasets (optional but recommended)

```bash
cd /Users/birger/ai_sweden/grpo-data-bootstrap
bash scripts/download_datasets.sh
```

3. Convert one source into GRPO format

```bash
python scripts/convert_to_grpo_jsonl.py \
  --dataset open-r1/mixture-of-thoughts \
  --split train \
  --prompt-col prompt \
  --response-col completion \
  --out data/processed/open_r1_mixture.jsonl
```

4. Point `oellm-autoexp` to your processed data and the config:

```bash
oellm --backend megatron launch \
  --config configs/oellm_autoexp_grpo_example.yaml \
  --data.train_paths data/processed/open_r1_mixture.jsonl
```

Replace the command above with your project’s actual launcher syntax if different.

## Recommended workflow

1. Start with synthetic reasoning traces for policy warm-up (SFT-style stage, optional)
2. Add verifier/judge reward in GRPO stage
3. Add preference-based reward signals and increase group size gradually
4. Keep evaluation separate (`MATH`, `GPQA`, `LiveCodeBench`, etc.)
5. Expand to code-reasoning only after math stability is stable

## Notes on data usage

- This repo intentionally separates:
  - **Reasoning traces** (what the model should generate)
  - **Reward signals** (how quality is scored)
- Avoid using the same prompts in both train and eval to prevent leakage.

## License / compliance

Use these datasets within their individual dataset licenses and terms of service.
`data/` directories are excluded in git by default; only manifests and scripts are
stored in this repository.
