# Adapter and objective registry implementation note

Date: 2026-08-21

## Research purpose

This change separates parameter-efficient adapter configuration from the Wordle training objective. The distinction matters for later comparisons: SFT, DPO, and ORPO are learning objectives, while LoRA, rank-stabilized LoRA, and DoRA are adapter parameterizations. Treating these as separate factors prevents an apparent objective improvement from being confounded with a changed trainable-parameter architecture.

## Implemented contract

- A normalized adapter schema accepts the historical top-level `lora` block and a new `adapter` block.
- Historical defaults remain LoRA rank 16, alpha 32, dropout 0.05, no bias, and the seven attention/MLP projection targets used by existing runs.
- LoRA, rank-stabilized LoRA, and DoRA have separate launchers while sharing one validated PEFT schema. Explicit `use_rslora` and `use_dora` switches are supported by the installed PEFT version.
- SFT, DPO, ORPO, GRPO, and Q-SFT have serializable metadata describing objective family, training signal, warm-start requirements, reference-policy requirements, and trainer entrypoint.
- Objective config validation accepts scalar or grid-valued hyperparameters and rejects method mismatches, invalid positive parameters, and warm-start declarations that conflict with the registry.
- SFT now records normalized adapter configuration and trainable-target validation in `accounting.json`.

## Controls and interpretation

The adapter registry does not change `WORDLE-PROTOCOL-002`, datasets, prompts, generation, parsing, rewards, or evaluation. Existing legacy SFT specs normalize to the same PEFT LoRA configuration, so this is intended to be behavior-preserving for historical recipes. Any future adapter comparison must hold model, data manifest, objective, seed, optimizer/token budget, and evaluation decoder fixed and must report trainable parameter count and peak VRAM.

Trainable-target validation checks that every declared target module produces at least one trainable parameter tensor. It does not prove that two model families expose semantically identical projection modules, so Gemma/Qwen comparisons remain cross-family evidence unless architecture is otherwise matched.

## Verification evidence

Focused tests cover legacy normalization, PEFT configuration construction, invalid configuration rejection, objective-registry semantics, grid validation, and declared trainable-target coverage. GPU training was intentionally not required for these unit tests.

## Remaining limitations

- LoRA-family parameterizations are registered; prompt tuning, IA3, AdaLoRA, and other PEFT families still require separate schemas and trainability tests.
- DPO, ORPO, GRPO, and Q-SFT are registered but are not yet routed through a single shared generic experiment runner.
- Existing saved run IDs do not include a newly normalized `adapter` block; backward compatibility is preserved, but historical specs should not be rewritten.
- Runtime target validation confirms name coverage, not mathematical equivalence or adapter quality.
