# Progress

Date: 2026-05-04

## Completed

- Confirmed this directory contains only tokenizer/config files and ONNX inference artifacts, not trainable PyTorch/HF weights.
- Reviewed the public Bonsai 1.7B packed and unpacked model cards.
- Reviewed the PrismML Bonsai whitepaper sections on Qwen3-derived architecture, packed low-bit format, and custom runtime kernels.
- Reviewed BitNet public references for native ternary/1.58-bit training context.
- Created `REPLICATE_OR_EXCEED_BONSAI_TERNARY_PLAN.md`.
- Created `FINETUNE_BONSAI_WITHOUT_PERFORMANCE_REGRESSION_PLAN.md`.
- Audited candidate reasoning datasets for the first external LoRA run.
- Created `DATASET_AUDIT_AND_PILOT_MIX.md`.
- Added initial data preparation, LoRA SFT training, and prompt generation scripts under `scripts/`.
- Audited Hugging Face `gpt-5.4` search results and promoted canonical GPT-5.4 Xhigh reasoning data as the first frontier-distill SFT pilot.
- Generated `runs/data/gpt54_xhigh_train_smoke.jsonl` with 200 train rows and `runs/data/gpt54_xhigh_eval_smoke.jsonl` with 20 eval rows.

## Current Conclusion

Plain fine-tuning followed by generic Q1/Q2 quantization is high risk. The safer path is adapter-based fine-tuning over an unchanged packed Bonsai base. The path to replicating or exceeding Bonsai requires ternary-aware recovery/training plus a runtime that uses packed ternary kernels directly.

## Current Dataset Decision

- Use `vanty120/Gpt-5.4-Xhigh-Reasoning-2000x` as the first SFT pilot dataset.
- Use `vanty120/Gpt-5.4-Xhigh-Reasoning-750x` as a smaller secondary source.
- Use `Bespoke-Stratos-17k` and `open-r1/Mixture-of-Thoughts` as comparison baselines or fallbacks rather than the first-choice run.
- Quarantine the recursive GPT-5.5/Versper datasets from the initial SFT mix unless filtered because samples are dominated by self-improvement/persona behaviors rather than benchmarkable reasoning.
- User verified `Bialy17/databricks-dolly-1k-GPT5.5` with the maintainer as Apache-2.0; use it only as optional general instruction replay, not as the primary reasoning dataset.

## Next Step

Download `prism-ml/Ternary-Bonsai-1.7B-unpacked`, then run a short LoRA smoke test locally or on a rented GPU if local VRAM is insufficient.
