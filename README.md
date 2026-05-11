# bonsai-training

Training and evaluation scaffolding for reasoning-focused LoRA experiments on PrismML's unpacked Bonsai 1.7B model.

This repo is currently organized around a conservative workflow:

- keep the Bonsai base model unchanged
- prepare a filtered chat-format SFT dataset
- train a small external LoRA adapter
- compare base-model and adapter outputs on a fixed prompt set

The repo also includes project notes that document the current training strategy, dataset audit, and replication research.

## Repository Contents

- `scripts/prepare_reasoning_sft.py`: downloads and normalizes supported Hugging Face datasets into chat JSONL.
- `scripts/train_lora_sft.py`: trains a PEFT LoRA adapter against the unpacked Bonsai checkpoint.
- `scripts/evaluate_prompts.py`: generates outputs for a fixed prompt file from the base model or a base-plus-adapter pair.
- `configs/reasoning_lora_pilot.json`: pilot defaults for the first local and rented-GPU LoRA runs.
- `data/eval/reasoning_smoke_prompts.jsonl`: small fixed prompt set for smoke evaluation.
- `runs/data/`: example train and eval JSONL splits already produced by the prep script.
- `FINETUNE_BONSAI_WITHOUT_PERFORMANCE_REGRESSION_PLAN.md`: fine-tuning approach and guardrails.
- `REPLICATE_OR_EXCEED_BONSAI_TERNARY_PLAN.md`: longer-term ternary reproduction and recovery plan.
- `DATASET_AUDIT_AND_PILOT_MIX.md`: dataset selection notes for the initial SFT branch.

## Current Approach

The working assumption in this repo is that generic full fine-tuning followed by ordinary low-bit requantization is high risk for Bonsai. The current path instead uses:

1. `prism-ml/Ternary-Bonsai-1.7B-unpacked` as the trainable Hugging Face model
2. LoRA adapters over a frozen base
3. reasoning-heavy SFT data with aggressive filtering
4. fixed prompt regression checks before and after training

## Prerequisites

- Python 3.11+ is recommended
- `uv` is the preferred Python package manager
- enough local disk for model weights and dataset cache
- enough GPU memory for the run you choose
  - local smoke settings in `configs/reasoning_lora_pilot.json` assume a constrained setup
  - the first fuller run is intended for a 24 GB to 48 GB rented GPU

There is no dependency lockfile in the repo yet, so install the libraries used by the scripts directly:

```bash
uv venv
source .venv/bin/activate
uv pip install torch transformers peft datasets accelerate
```

## Quick Start

### 1. Prepare a smoke dataset

```bash
uv run python scripts/prepare_reasoning_sft.py \
  --source gpt54-xhigh-2000 \
  --limit 200 \
  --eval-size 20 \
  --max-chars 20000 \
  --train-output runs/data/gpt54_xhigh_train_smoke.jsonl \
  --eval-output runs/data/gpt54_xhigh_eval_smoke.jsonl
```

Supported `--source` values are defined in `scripts/prepare_reasoning_sft.py` and currently include GPT-5.4 Xhigh reasoning sets, Bespoke Stratos, Mixture-of-Thoughts subsets, and a few optional comparison datasets.

### 2. Run a LoRA smoke train

```bash
uv run python scripts/train_lora_sft.py \
  --model prism-ml/Ternary-Bonsai-1.7B-unpacked \
  --train-jsonl runs/data/gpt54_xhigh_train_smoke.jsonl \
  --eval-jsonl runs/data/gpt54_xhigh_eval_smoke.jsonl \
  --output-dir runs/lora/gpt54_xhigh_smoke \
  --max-seq-length 2048 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-steps 20 \
  --learning-rate 2e-5 \
  --fp16
```

### 3. Generate prompt outputs for regression checks

Base model only:

```bash
uv run python scripts/evaluate_prompts.py \
  --model prism-ml/Ternary-Bonsai-1.7B-unpacked \
  --output runs/eval/base_reasoning_smoke.jsonl \
  --fp16
```

Base model plus adapter:

```bash
uv run python scripts/evaluate_prompts.py \
  --model prism-ml/Ternary-Bonsai-1.7B-unpacked \
  --adapter runs/lora/gpt54_xhigh_smoke \
  --output runs/eval/adapter_reasoning_smoke.jsonl \
  --fp16
```

## Data Format

The training and eval scripts expect JSONL rows shaped like:

```json
{
  "source": "gpt54-xhigh-2000",
  "messages": [
    {"role": "user", "content": "Question text"},
    {"role": "assistant", "content": "Answer text"}
  ]
}
```

The prep script normalizes multiple source schemas into this `messages` format and rejects rows that are empty, too long, malformed, duplicated, or matched by the identity-filter regex.

## Evaluation Inputs

`data/eval/reasoning_smoke_prompts.jsonl` contains a small fixed prompt set spanning:

- arithmetic and logic
- code generation
- strict output formatting
- instruction following around quantization and adapters

This is intended as a fast regression check, not as a benchmark.

## Notes

- `scripts/train_lora_sft.py` masks the prompt portion of each example so the loss is focused on the assistant response.
- `scripts/evaluate_prompts.py` can load an external PEFT adapter with the base model for side-by-side output checks.
- `configs/reasoning_lora_pilot.json` is the current source of truth for the initial pilot hyperparameters.
- The planning documents should be read before broadening this repo into merged-weight fine-tunes or new quantization experiments.
