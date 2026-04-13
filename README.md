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

## New: ELO proxy CLI library

`grpo-elo` is a new CLI utility to turn model-level ELO scores into a trainable quality proxy.

It supports:

- model map normalization from an LMArena-like source table (`build-model-map`)
- training a regression model from samples + model IDs and/or target columns (`train`)
- scoring any file of samples (`score`)
- evaluating a saved model (`evaluate`)

Installed via local package, this becomes a reusable CLI:

```bash
cd /Users/birger/ai_sweden/grpo-data-bootstrap
pip install -e .
grpo-elo --help
```

Example end-to-end:

1) Build model map:

```bash
grpo-elo build-model-map \
  --source manifests/model_elo_input.csv \
  --model-col model_name \
  --elo-col elo \
  --snapshot-col snapshot_date \
  --out manifests/model_elo_map.csv
```

`manifests/model_elo_input.csv` should include one row per model snapshot, for example:

`model_name,elo,snapshot_date`

2) Train model-quality predictor:

```bash
grpo-elo train \
  --samples data/processed/open_r1_mixture.jsonl \
  --model-col model_id \
  --model-map manifests/model_elo_map.csv \
  --prompt-col prompt \
  --response-col response \
  --numeric-features response_score reward_raw \
  --algorithm ridge \
  --out-dir models/elo-regressor
```

This creates:

- `models/elo-regressor/elo_model.joblib`
- `models/elo-regressor/metrics.json`

3) Score new samples and use `elo_pred` as a prior for training data filtering:

```bash
grpo-elo score \
  --model models/elo-regressor/elo_model.joblib \
  --samples data/processed/open_r1_mixture.jsonl \
  --out data/processed/open_r1_mixture_with_elo_pred.jsonl \
  --prompt-col prompt \
  --response-col response
```

Suggested integration with your GRPO flow:

- Use `elo_pred` for pre-sampling and filtering.
- Combine with verifier/judge score when scoring quality for policy optimization.
- Keep ELO-based score as a prior, not as the only reward.

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
