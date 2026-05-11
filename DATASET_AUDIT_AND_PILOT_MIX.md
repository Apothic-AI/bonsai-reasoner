# Dataset Audit and Pilot Mix

Date: 2026-05-04

## Recommendation

Start the external LoRA work with a frontier-distill SFT branch centered on GPT-5.4 Xhigh reasoning datasets. The strongest signal in the current Hugging Face search results is not the model label alone; it is the combination of:

- explicit reasoning fields
- English instruction-following format
- enough examples for a meaningful adapter run
- low risk of teaching a self-referential persona

Initial SFT mix:

- Primary pilot: `vanty120/Gpt-5.4-Xhigh-Reasoning-2000x`
- Secondary pilot: `vanty120/Gpt-5.4-Xhigh-Reasoning-750x`
- Optional replay: `Bialy17/databricks-dolly-1k-GPT5.5`
- Optional specialized branch: `wenyupapa/BIRD-Verified-CoT-2462-GPT5.4` for SQL/text-to-SQL

Use `bespokelabs/Bespoke-Stratos-17k` and `open-r1/Mixture-of-Thoughts` as comparison baselines, not as the first-choice training data, unless the GPT-5.4/GPT-5.5 data fails quality checks.

## Candidate Decisions

| Dataset | Evidence Observed | Decision |
| --- | --- | --- |
| `vanty120/Gpt-5.4-Xhigh-Reasoning-2000x` | Apache-2.0, English, 2,752 rows observed locally, schema has `instruction`, `thinking`, and `response`; samples are direct reasoning/instruction examples. Some rows are extremely long, so token/character filtering is required. | Primary SFT pilot. |
| `vanty120/Gpt-5.4-Xhigh-Reasoning-750x` | Apache-2.0, English, same `instruction`/`thinking`/`response` schema; samples include hard security and jailbreak-defense reasoning. | Secondary SFT pilot with a smaller quota because it is narrower. |
| `Nettoov/Gpt-5.4-Xhigh-Reasoning-2000x`, `ansulev/Gpt-5.4-Xhigh-Reasoning-2000x`, `PhantomG27249/Gpt-5.4-Xhigh-Reasoning-2000x` | Same schema and identical first rows observed against the canonical Xhigh 2000x dataset. | Treat as likely mirrors. Do not include until deduplication proves unique rows. |
| `dalisoft/gpt-5.4-xhigh-reasoning-550x` | Apache-2.0 metadata, 550-ish examples, schema is chat `messages`; assistant content uses `<thinking>` XML-style traces. | Candidate code/general reasoning supplement after normalization checks. |
| `Roman1111111/gpt-5.4-step-by-step-reasoning` | MIT, 1K-10K examples, `question`/`answer` schema with subject/topic/difficulty metadata. | Useful secondary data, lower priority than Xhigh because it lacks a separate thinking field. |
| `wenyupapa/BIRD-Verified-CoT-2462-GPT5.4` | 2,462 GPT-5.4 SQL reasoning examples with validation metadata and long database schemas. | Specialized SQL branch, not general pilot data. |
| `Rexhaif/MLEM-reasoning-gpt5.4` | 100K-1M examples; samples are evaluator/judgment tasks rather than normal assistant reasoning. | Do not use for general SFT; possible future evaluator model data. |
| `WithinUsAI/GPT5.5_thinking_max_distill_god_seed_25K` | Apache-2.0, 25k JSONL, fields are `id`, `category`, `difficulty`, `instruction`, `input`, `output`, `tags`; samples focus on recursive self-improvement, model identity, and speculative capability-upgrade plans. | Secondary candidate only with identity/persona filtering. |
| `versperai/Versper-V1-Evo-ORPO-GPT5.5-Think-Recursive-25k` | Apache-2.0, 25k preference triplets with `prompt`, `chosen`, `rejected`; samples overlap heavily with the recursive-agent style and rejected answers are often truncated fragments. | Prefer later ORPO/DPO, not first SFT. |
| `Bialy17/databricks-dolly-1k-GPT5.5` | JSON with `instruction` and `output`; user verified Apache-2.0 status with the maintainer. It appears Dolly-like rather than reasoning-trace data. | Allow as a small general instruction replay set, not as primary reasoning-distillation data. |
| `bespokelabs/Bespoke-Stratos-17k` | Apache-2.0, 17k examples, DeepSeek-R1 SFT distillation, schema is `system` plus `conversations`, card describes rejection sampling for correctness. | Comparison baseline or fallback. |
| `open-r1/Mixture-of-Thoughts` | 349,317 examples, fields are `messages`, `num_tokens`, and `source`; card says 350k verified DeepSeek-R1 traces across math/code/science. | Scale baseline or fallback. |

## Pilot Training Mix

Phase 1 smoke test:

- 200 to 1,000 examples from `vanty120/Gpt-5.4-Xhigh-Reasoning-2000x`
- Filter long examples aggressively; the first smoke split used `--max-chars 20000`
- Max sequence length: 2,048 locally, 4,096 to 8,192 on a rented GPU
- Adapter: LoRA rank 8, attention projections only
- Goal: prove the preprocessing, chat template, loss masking, checkpoint saving, and prompt eval all work

Phase 2 first real adapter:

- `vanty120/Gpt-5.4-Xhigh-Reasoning-2000x` as the base
- Add a smaller quota from `vanty120/Gpt-5.4-Xhigh-Reasoning-750x`
- Add a small general replay set before increasing epochs; `Bialy17/databricks-dolly-1k-GPT5.5` is acceptable for this role
- Eval before and after on fixed math/code/instruction prompts
- Stop if outputs drift toward self-referential model identity, excessive verbosity, or malformed `<think>` markers

Phase 3 scale-up:

- Add `Roman1111111/gpt-5.4-step-by-step-reasoning` if the Xhigh adapter improves reasoning without style regression
- Add `wenyupapa/BIRD-Verified-CoT-2462-GPT5.4` only for a SQL/tool-use adapter branch
- Compare against a DeepSeek-R1 data adapter using `Bespoke-Stratos-17k` or `open-r1/Mixture-of-Thoughts`

## Data Quality Gates

Reject examples that:

- Contain training-target identity strings such as `I am GPT-5.5`, `GPT-5.5 Thinking Max Distill`, `Versper-V1-Evo`, or `Recursive Seed AI`
- Center the response on recursive self-improvement or autonomous self-modification unless that is the explicit target product behavior
- Have empty prompt or empty assistant response
- Exceed the active context budget after tokenization
- Include malformed chat markers
- Duplicate validation or regression prompts

## Next Concrete Step

Prepare a small SFT JSONL from the canonical GPT-5.4 Xhigh 2000x dataset, download the unpacked Bonsai HF weights, and run a short LoRA smoke test. If the local RTX 4050 cannot fit the run, move the exact command to a rented GPU rather than changing the training recipe.
