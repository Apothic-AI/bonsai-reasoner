# Plan: Fine-Tune Bonsai Without Degrading Its Performance

Date: 2026-05-04

## Short Answer

The lowest-risk plan is to keep the packed Bonsai base frozen and train a small adapter. Do not full-fine-tune the unpacked model and then run generic Q1/Q2 quantization as the default path. That is likely to degrade quality and may also lose the runtime properties that make Bonsai useful.

The guiding rule is simple: preserve the compressed base exactly unless measurements prove a merged-and-recompressed model is better.

## Objectives

- Improve a target behavior or domain skill.
- Preserve Bonsai's baseline general quality.
- Preserve the small packed base as the deployment anchor.
- Keep adapter overhead small enough that memory and decode speed remain close to the original.
- Avoid any step that requires PrismML's proprietary compression unless it is replaced by a measured public equivalent.

## Recommended Strategy

Use PEFT/LoRA against the unpacked HF model for training, but deploy against the unchanged packed Bonsai base plus an adapter if the runtime supports it.

Reasoning:

- The unpacked model is the trainable HF-compatible representation.
- The packed model is the performance artifact.
- Merging LoRA into the base and then quantizing to Q1/Q2 is the risky step.
- A small FP16/BF16 adapter adds memory and compute, but should be much less destructive than re-quantizing all base weights.

## Workflow

### Stage 0: Establish Baselines

Before training anything:

1. Run the packed Bonsai model on a fixed prompt set.
2. Run the unpacked FP16 model on the same prompt set.
3. Measure task-specific quality on the target domain.
4. Measure general quality with a compact regression suite.
5. Measure runtime:
   - model size
   - resident memory
   - prompt tokens/sec
   - decode tokens/sec
   - energy/token if available

These measurements become the no-regression gate.

### Stage 1: Data Design

Use three validation sets:

- Target validation set: proves the fine-tune helped.
- General regression set: catches broad model degradation.
- Format regression set: catches chat-template, tool-calling, JSON, and instruction-following failures.

Training data should include:

- High-quality target examples.
- Negative or correction examples only if they reflect real target failures.
- General replay samples from broad instruction/code/math text to reduce catastrophic forgetting.
- Exact Qwen3/Bonsai chat formatting.

Avoid:

- Large volumes of low-quality synthetic data.
- Rewriting the model's whole assistant personality unless that is the product goal.
- Training on outputs with malformed `<think>` or chat markers.

### Stage 2: Adapter Training

Start with conservative LoRA:

- Rank: 4, 8, and 16 as an ablation.
- Alpha: 2x rank as a starting point.
- Dropout: 0 to 0.05.
- Learning rate: start around `1e-5` to `5e-5` for narrow domain adaptation; only test higher rates if the target gain is too small.
- Epochs: keep low, prefer early stopping.
- Gradient clipping: enabled.
- Weight decay: low or zero for LoRA parameters.

Target modules:

- First pass: attention projections only (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
- Second pass: add MLP projections (`gate_proj`, `up_proj`, `down_proj`) only if attention-only adapters underfit.
- Avoid embeddings and LM head unless the domain requires new tokens or strong terminology shifts.

Loss:

- Supervised cross-entropy on target data.
- KL regularization against the original Bonsai/unpacked model on general replay prompts.
- Optional teacher KL from a larger model for target-domain reasoning.

### Stage 3: Quantization-Aware Adapter Training

If possible, train the adapter with the frozen base represented as it will be deployed:

- Base weights frozen.
- Forward path simulates the packed ternary base or uses dequantized ternary weights.
- LoRA weights remain FP16/BF16.
- Only adapter parameters receive gradients.

This exposes the adapter to the base model's quantization behavior during training without changing the base.

### Stage 4: Deployment Options

Preferred option: packed base plus external adapter.

- Keep the original Bonsai packed model unchanged.
- Load the trained LoRA adapter at runtime.
- Measure the adapter overhead directly.

Acceptable option: ONNX Runtime GenAI adapter path.

- Export an ONNX-compatible base and adapter with Olive if the target runtime supports the needed model and adapter format.
- Use `.onnx_adapter` for runtime adapter switching.
- Treat this as a separate deployment target from PrismML's packed MLX artifact unless the backend can use the same compressed base.

Risky option: merge adapter and re-quantize.

- Only attempt after the adapter path proves the target-domain gain.
- Use a ternary-aware recovery process, not plain Q1/Q2 PTQ.
- Require side-by-side quality and speed measurements before accepting it.
- Reject the merge if general quality drops or speed/size no longer beats the baseline.

## No-Regression Gates

Do not accept a fine-tune unless:

- Target eval improves by a predefined amount.
- General regression eval remains within tolerance.
- Format/tool-calling tests do not regress.
- Runtime memory increase is acceptable.
- Decode speed remains close enough to the packed baseline for the intended use.
- The adapter does not cause repeated refusal, verbosity, style, or reasoning-format drift.

Suggested initial tolerances:

- General benchmark drop: no more than 1 to 2 percentage points on small evals.
- Perplexity increase: no more than 2% on general held-out text.
- Decode speed loss: no more than 5% to 15%, depending on adapter rank and runtime.
- Memory overhead: LoRA adapter should stay small relative to the 0.45 GiB packed base.

## Experiments To Run

Run these as a grid:

| Experiment | Base | Trainable Params | Deployment | Purpose |
| --- | --- | --- | --- | --- |
| A | Unpacked Bonsai | LoRA attention r=4 | Packed base + adapter | Lowest overhead |
| B | Unpacked Bonsai | LoRA attention r=8 | Packed base + adapter | Default candidate |
| C | Unpacked Bonsai | LoRA attention+MLP r=8 | Packed base + adapter | More capacity |
| D | Unpacked Bonsai | LoRA attention r=16 | Packed base + adapter | Capacity ceiling |
| E | Unpacked Bonsai | Merged LoRA | Re-ternary recovery | Risk assessment only |
| F | Qwen3-1.7B | LoRA or QAT | New ternary model | Independent reproduction track |

## When Not To Fine-Tune

Prefer RAG, prompt engineering, tools, or constrained decoding if:

- The target data changes frequently.
- The task mostly requires private/domain facts.
- The target failures are formatting failures that can be solved with decoding constraints.
- Runtime speed and memory are more important than a small behavior gain.

## Implementation Notes

- Keep exact model revisions in every run log.
- Save data hashes and prompt templates.
- Use deterministic eval scripts where possible.
- Generate a short sample report for every checkpoint.
- Never compare a packed runtime to a standard HF runtime and call it a model-quality improvement; separate quality from runtime.

## Final Recommendation

Start with LoRA r=8 on attention projections only, trained against the unpacked Bonsai HF checkpoint with general replay and KL regularization. Deploy it as an external adapter over the unchanged packed Bonsai model if the runtime supports that path. Only consider merging and re-quantizing after the adapter has proven useful and a ternary-aware recovery pipeline exists.

## Sources

- PrismML Ternary-Bonsai-1.7B MLX model card: https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit
- PrismML Ternary-Bonsai-1.7B unpacked model card: https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-unpacked
- ONNX Runtime GenAI LoRA adapter tutorial: https://onnxruntime.ai/docs/genai/tutorials/finetune.html
- BitNet b1.58 paper: https://arxiv.org/abs/2402.17764
- Hugging Face BitNet docs: https://huggingface.co/docs/transformers/model_doc/bitnet
