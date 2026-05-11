# Plan: Replicate or Exceed Bonsai-Style Ternary Performance

Date: 2026-05-04

## Short Answer

Yes, a normal fine-tune followed by generic Q1/Q2 post-training quantization is likely to lose a material part of Bonsai's quality result. It may still produce a small and runnable model, but it should not be expected to preserve the same quality/size/speed point unless the compression stage includes ternary-aware recovery and the runtime has kernels that exploit the packed format.

The likely reason is that Bonsai's public materials describe more than ordinary quantization:

- The 1.7B MLX release uses ternary weights `{-1, 0, +1}` with group size 128 and FP16 group scaling across embeddings, attention projections, MLP projections, and the LM head.
- The unpacked model is FP16 safetensors of the ternary model for stock Hugging Face tooling, not a claim that the model was originally trained and deployed as ordinary FP16.
- The 8B whitepaper says the architecture is Qwen3-derived and unchanged, while the novelty is in end-to-end low-bit storage, runtime format, and custom kernels.
- BitNet b1.58 shows that high-quality 1.58-bit LLMs generally rely on quantization-aware training from the start or during recovery, not simple after-the-fact rounding.

The design below assumes we do not have PrismML's proprietary compression, so we need to recreate the important public behaviors with measurable stages.

## Goal

Produce a Qwen3-derived 1.7B-class model that matches or beats Bonsai on:

- General quality at fixed size: perplexity, instruction following, math, coding, tool use.
- Packed footprint: target about 0.45 to 0.55 GiB for 1.7B ternary g128-style storage.
- Decoding speed: target Bonsai-class speed on the selected backend, measured by tokens/sec at fixed prompt/output sizes.
- Energy per generated token where hardware permits measurement.
- Reproducibility: every result tied to exact model commit, data mix, quantizer, runtime, and eval harness.

## Baselines First

Do not begin with a new training recipe. Establish the floor and identify the real gap.

1. Evaluate the local ONNX model.
   - Measure perplexity on a stable held-out corpus.
   - Measure task quality on a small but representative suite.
   - Measure CPU/GPU/Apple Silicon throughput if the target hardware is available.

2. Evaluate the Bonsai MLX 2-bit release.
   - Use PrismML's recommended runtime path.
   - Record packed size, prompt processing speed, token generation speed, and output quality.

3. Evaluate the unpacked FP16 Bonsai model.
   - This gives the quality of the expanded ternary weights under normal HF execution.
   - Expect speed and memory to be much worse than packed Bonsai; this is a sanity check, not a deployment target.

4. Evaluate generic PTQ attempts.
   - Try standard 2-bit/1-bit ONNX, llama.cpp, MLX, or bitsandbytes-compatible flows.
   - Keep these as negative controls. If they collapse quality, that verifies the need for ternary-aware recovery.

## Architecture Choice

Start with Qwen3-1.7B or the Bonsai unpacked model depending on the experiment:

- Use Qwen3-1.7B if the goal is to recreate the full compression pipeline independently.
- Use `prism-ml/Ternary-Bonsai-1.7B-unpacked` if the goal is to recover or adapt from Prism's already-compressed distribution.
- Do not alter the Transformer architecture until a same-architecture baseline is reproduced. Architecture changes add too many unknowns.

The first target should keep the Qwen3 layer structure and only replace large matrix operations with ternary-aware training and packed inference.

## Quantization Format

Implement a public ternary g128 format:

- Weight codebook: `{-1, 0, +1}`.
- Group size: 128 contiguous weights along the innermost matmul dimension unless kernel layout proves otherwise.
- Metadata: FP16 or BF16 scale per group.
- Optional metadata: learned threshold per tensor, channel, or group if it improves reconstruction.
- Stored representation: 2-bit packed ternary codes initially, because that is practical on standard hardware. A true entropy-coded log2(3) format can be a later optimization.

For each group, start with an initialization that solves:

`min_s,q ||W - s*q||_2`, where `q in {-1,0,+1}`.

Then improve it with block reconstruction and quantization-aware training.

## Training Recipe

Use a staged recipe. Each stage must beat the previous one on held-out metrics before moving on.

### Stage 1: Ternary PTQ Baseline

Implement a deterministic ternary quantizer:

- Per-group scale.
- Search threshold values for zero assignment.
- Keep norm/RMSNorm parameters in FP16/BF16.
- Quantize embeddings and LM head only after projection layers are stable.

Expected result: strong degradation. This is a diagnostic baseline.

### Stage 2: Block Reconstruction

Run layer/block-wise reconstruction before full QAT:

- Freeze all other layers.
- Optimize group scales and optional thresholds against captured activations.
- Use MSE on block outputs, plus KL divergence on logits for final blocks.
- Reconstruct attention and MLP projections separately, then jointly within each Transformer block.
- Use calibration data from pretraining-like text, instruction data, code, and math.

Expected result: much better than PTQ but probably still below Bonsai quality.

### Stage 3: Ternary-Aware Continued Training

Train with FP32/BF16 shadow weights and ternary forward weights:

- Forward pass uses ternary weights and group scales.
- Backward pass updates shadow weights through a straight-through estimator.
- Scales are trainable, with constraints to avoid runaway magnitudes.
- Quantization is applied to all large linear weights from the beginning of this stage.
- Embeddings and LM head can be ramped in after the model stabilizes.

Loss:

- Next-token cross-entropy on mixed general text/code/math/instruction data.
- KL distillation from Qwen3-1.7B and/or the Bonsai unpacked model.
- Optional intermediate representation matching for fragile layers.
- Optional auxiliary sparsity target to keep the zero ratio stable and kernel-friendly.

This is the first stage likely to reproduce Bonsai-like quality.

### Stage 4: Instruction Recovery

Run supervised fine-tuning after ternary-aware recovery:

- Keep ternary forward simulation enabled.
- Use the exact Qwen3/Bonsai chat template.
- Mix task data with general replay data to avoid overfitting.
- Use KL-to-base regularization so instruction tuning does not destroy general behavior.
- Validate on IFEval-style constraints, math, code, tool-calling, and long-context probes.

### Stage 5: Preference and Safety Alignment

Only run DPO/ORPO/KTO if SFT retains quality.

- Keep preference training short.
- Add KL regularization to the recovered ternary SFT checkpoint.
- Track regressions aggressively; small low-bit models are easy to over-align.

## Exceeding Bonsai

The best chance to exceed the published result is not a clever one-shot quantizer. It is a better training and evaluation loop.

Possible improvements:

- Use stronger teacher distillation from a larger Qwen3 model for math, coding, and tool use.
- Train the ternary model on more high-quality code/math data than the original base received.
- Use selective higher precision only where measurements justify it, such as norms, output scale metadata, or a small residual adapter. Keep this as an ablation because it may break strict footprint comparisons.
- Add activation-aware training: W1.58A8 first, then explore a BitNet a4.8-style path with 4-bit activations and sparse 8-bit intermediate handling.
- Use architecture-neutral kernel optimizations: packed ternary matmul, scale fusion, KV cache quantization, prompt/decode-specific kernels.

## Runtime Plan

Quality and speed are separate deliverables. A high-quality ternary checkpoint does not produce Bonsai-like speed unless the runtime exploits it.

1. Reference runtime:
   - PyTorch/HF module with fake ternary quantization for correctness and training.
   - Slow but debuggable.

2. Packed inference runtime:
   - MLX if Apple Silicon is the primary deployment target.
   - llama.cpp or bitnet.cpp-style kernels if CPU/NVIDIA/Metal is the target.
   - ONNX Runtime only if `MatMulNBits` or a custom operator can represent the exact ternary format efficiently.

3. Kernel requirements:
   - Decode ternary codes inside matmul.
   - Apply group scales without materializing FP16 weights.
   - Use separate prompt-processing and token-generation paths.
   - Benchmark memory bandwidth, not just arithmetic throughput.

## Evaluation Harness

Track these metrics per checkpoint:

- Perplexity: WikiText-like, C4-like, code corpus, domain corpus if applicable.
- Instruction following: IFEval or similar rule-based checks.
- Math: GSM8K, MATH-500, or a smaller local subset for iteration.
- Code: HumanEval+/MBPP+ if sandboxing is available.
- Tool use: BFCL-style AST/execution checks if tool calling matters.
- Long context: retrieval and needle tests at 8K, 16K, and the advertised context length.
- Regression prompts: a fixed local prompt set with exact decoded outputs or judge rubrics.
- Runtime: model size, resident memory, prompt tokens/sec, decode tokens/sec, energy/token.

Acceptance should be comparative:

- Must beat generic PTQ by a large margin.
- Must approach Bonsai's quality before runtime work is considered successful.
- Must preserve size and speed together; one without the other is not the Bonsai result.

## Risk Register

- Biggest quality risk: ternary PTQ without recovery will damage the model.
- Biggest speed risk: using a runtime that stores small weights but dequantizes to FP16 before matmul.
- Biggest reproducibility risk: hidden data and unlogged eval prompts.
- Biggest product risk: optimizing for benchmark average while harming the target user workflow.
- Biggest engineering risk: kernel work may dominate the schedule after model quality is solved.

## Decision Gates

Stop or change direction if:

- Stage 2 reconstruction cannot recover most of the PTQ loss.
- Stage 3 QAT is unstable after scale/threshold changes and lower learning rates.
- Packed runtime speed is not better than a strong 4-bit baseline.
- The target task can be solved better with RAG, tools, or a small adapter on the original Bonsai model.

## Sources

- PrismML Ternary-Bonsai-1.7B MLX model card: https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit
- PrismML Ternary-Bonsai-1.7B unpacked model card: https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-unpacked
- PrismML Bonsai whitepaper PDF: https://github.com/PrismML-Eng/Bonsai-demo/blob/d2079f77a64f649da6fafdb3fc6af5799035b195/1-bit-bonsai-8b-whitepaper.pdf
- BitNet b1.58 paper: https://arxiv.org/abs/2402.17764
- BitNet a4.8 paper: https://arxiv.org/abs/2411.04965
- Hugging Face BitNet docs: https://huggingface.co/docs/transformers/model_doc/bitnet
