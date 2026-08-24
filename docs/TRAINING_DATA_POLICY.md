# Training data publication policy

The GitHub repository contains code, deterministic manifests, compact experiment evidence, and **three representative rows per training type** under `examples/training_data/`. Full generated training/dev corpora, model weights, checkpoints, and raw run directories remain local and are ignored by Git.

This boundary applies to:

- matched direct single-step, multi-turn, and visible-reasoning representations;
- the Unsloth instruction/input/output variants;
- balanced-002 state categories;
- structured feedback, constraint, candidate, singleton, and full-policy microtasks;
- constraint-first and frozen-target Q-SFT rows;
- tiny overfit fixtures;
- coverage-max/growth turn-2, low-posterior, singleton, and later-game states.

The sample manifest records each local source path, complete-corpus hash and row count at export time, the selection rule, and the committed sample hash. Training-only answers can appear in examples; locked-test answers never do.

## Regenerate full corpora locally

Run these commands from the repository root in PowerShell. Generated JSONL files remain ignored.

```powershell
# Matched 4,096-row direct, multi-turn, and reasoning comparison.
py scripts/build_training_data.py --states 4096 --dev-states 512 --seed 2026 --force

# Matched 2,000-row Unsloth/Gemma representation bundle.
py scripts/build_unsloth_training_data.py --examples-per-variety 2000 --dev-states 512 --seed 2026 --force

# Balanced-002 source curriculum used by the full-parameter comparisons.
py -m wordle_lab.experiments.common_curriculum prepare-balanced `
  --universe-size 128 --train-secrets 96 --states 512 --seed 2026 --force

# Tiny memorization, structured microtasks, constraint-first, and Q-SFT bundles.
py -m next_steps.chatgpt_2026_08_23.tiny_overfit build --force
py -m next_steps.chatgpt_2026_08_23.structured_microtasks_experiment build --force
py -m next_steps.chatgpt_2026_08_23.constraint_first_policy build --force
py -m next_steps.chatgpt_2026_08_23.q_sft_frozen build

# Coverage-max, disjoint legality phase, and the 10K-20K ladder source bundle.
py -m next_steps.chatgpt_2026_08_23.coverage_max_experiment build --force
py -m next_steps.chatgpt_2026_08_23.coverage_legality_extension build --force
py -m next_steps.chatgpt_2026_08_23.coverage_growth_ladder build --force
```

Refresh the committed examples only after rebuilding the local sources:

```powershell
py scripts/export_training_samples.py --count 3
```

Do not use `git add -f` on ignored data or artifact directories. Review the staged payload before every push with:

```powershell
git diff --cached --stat
git diff --cached --name-only
```
