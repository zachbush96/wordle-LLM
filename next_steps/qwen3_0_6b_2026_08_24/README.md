# Qwen3-0.6B Wordle experiment track

This folder is the self-contained 2026-08-24/25 Qwen3-0.6B follow-up to the Gemma work in `next_steps/chatgpt_2026_08_23`.

The experiment pins `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca` and keeps `WORDLE-PROTOCOL-002`, the 32-game development split, 128 fixed-state diagnostics, 200 retention probes, natural generation, and the locked-test boundary unchanged. The canonical Qwen chat mode is non-thinking; the native-thinking baseline is retained as a separate ablation.

## What ran

- Native-thinking and non-thinking base benchmarks.
- Disjoint 32-state general and singleton memorization LoRAs.
- Matched rank-16 LoRA, rsLoRA, and DoRA balanced-002 curves at steps 150/300/450/600.
- A one-pass, 4,096-unique-state LoRA coverage curve.
- Constraint-first sampled multi-label SFT.
- Structured feedback/constraint/singleton/full-policy microtasks.
- Full-parameter BF16 balanced-002 tuning at steps 150/300/450/600.
- A repetition-penalty 1.05 decoder ablation on the strongest canonical full checkpoint.
- Q-SFT, DPO/ORPO, SFT-to-GRPO, and GRPO/AVSPO eligibility checks.

The locked test remained closed. See [RESULTS_AND_HANDOFF.md](RESULTS_AND_HANDOFF.md) for findings, [experiment_matrix.json](experiment_matrix.json) for compact metrics/accounting, [results_manifest.json](results_manifest.json) for the content-hash inventory, and `results/` for raw development traces.

Local downloaded weights and checkpoints are intentionally ignored by Git. The runner stores them under `models/base/Qwen--Qwen3-0.6B` and `artifacts/qwen3-0.6b-2026-08-24`.

## Reproduction

```powershell
py -m next_steps.qwen3_0_6b_2026_08_24.qwen3_experiment verify-model
py -m next_steps.qwen3_0_6b_2026_08_24.qwen3_experiment benchmark
py -m next_steps.qwen3_0_6b_2026_08_24.qwen3_experiment train --dataset balanced --adapter rslora
py -m next_steps.qwen3_0_6b_2026_08_24.qwen3_experiment evaluate --dataset balanced --adapter rslora --checkpoint step-000300
py -m next_steps.qwen3_0_6b_2026_08_24.collect_results
py -m pytest -q
```

Available datasets are `balanced`, `coverage4096`, `constraint`, `structured`, `tiny-general`, and `tiny-singleton`. Available trainable scopes are `lora`, `rslora`, `dora`, and `full` (full is restricted to balanced-002).
