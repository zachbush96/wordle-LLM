# Tiny LLM Wordle Lab (local)

This is the local, CUDA-enabled version of the original Colab notebook export.
It benchmarks `google/gemma-3-270m-it`, can generate the oracle datasets, and can
train/evaluate the original LoRA experiments.

## Setup

```powershell
python -m pip install -r requirements.txt
```

The requirements select the official PyTorch 2.10 CUDA 12.8 wheel for Windows;
the NVIDIA driver may report a newer CUDA capability and still run this wheel.

The model is gated. Accept the Gemma license on Hugging Face, then authenticate
once with `hf auth login`, or set `HF_TOKEN` in your shell. Never put a token in
this source file.

## Run

```powershell
# Download once (later runs reuse models/base/google--gemma-3-270m-it)
python tiny_llm_wordle_lab.py --download-model

# Rules/parser/GPU/model preflight
python tiny_llm_wordle_lab.py --self-test

# Quick baseline; change to 1000 for the frozen full benchmark
python tiny_llm_wordle_lab.py --baseline --games 25
python tiny_llm_wordle_lab.py --baseline --games 1000

# Compare five decoding strategies and save per-turn prompts/raw outputs
python tiny_llm_wordle_lab.py --decoding-sweep --games 25

# Full baseline + oracle data + LoRA training + paired evaluation
python tiny_llm_wordle_lab.py --experiment SFT-001
```

All models, data, checkpoints, and results persist under this directory. Set
`WORDLE_LAB_DIR` to move the complete artifact tree elsewhere.

The current `BASELINE-003` protocol uses deterministic greedy decoding and
requires one generated token before either Gemma terminal token may stop the
response. This prevents an empty first-token `<end_of_turn>` response without
forcing a five-letter token, supplying a candidate list, or selecting a guess
for the model. The prompt also states the ordinary Wordle rule not to repeat a
previous guess. The decoder settings are saved in every baseline summary.

## Reasoning-capable post-training study

`tiny_llm_wordle_lab.py` is preserved as the historical exact-word reference.
The experiment in `EXPERIMENT_DESIGN.md` lives in the `wordle_lab` package and
uses the separately hashed `WORDLE-PROTOCOL-002` contract.

```powershell
python -m wordle_lab.cli prepare-data --train-states 2048 --dev-states 512
python -m wordle_lab.cli validate
python -m wordle_lab.cli baseline --split dev --games 25
python -m wordle_lab.cli reference --policy random_allowed --split dev --games 25
python -m wordle_lab.cli reference --policy oracle --split dev --games 25
python -m wordle_lab.cli train-sft --representation state_direct --seed 1337 --max-steps 20 --train-limit 512 --dev-games 25
python -m wordle_lab.cli retention --run-id base

# The full new baseline is the only unconditional locked-test evaluation.
python -m wordle_lab.cli baseline --split test
```

Trained checkpoints cannot access the 1,000-answer test until the explicit
study-level `select` transition. Run artifacts are content-addressed under
`artifacts/runs/`; dose checkpoints are saved at 25/50/75/100%.

### NotebookLM post-training technique stack

The implementation inventory from the shared NotebookLM conversation is
available without starting a GPU run:

```powershell
python -m wordle_lab.cli techniques
python -m pytest -q
```

The stack includes validated LoRA configuration, SFT, DPO, ORPO, GRPO,
Q-SFT-direct, ACR/AVSPO stability support, and a gated SFT-to-GRPO pipeline.
The preregistered matched study is in
`configs/studies/notebooklm_methods.yaml`; detailed source and manuscript notes
are under `research_notes/`. NotebookLM's hardcoded opening guess is
intentionally rejected because harness-selected guesses violate natural model
generation. Its structured character-list prompt is catalogued but requires a
new frozen protocol and matched baseline rather than silently changing
`WORDLE-PROTOCOL-002`.

## Development curriculum and intervention sweeps

The original 2,500-answer protocol sampled secrets from the full allowed-word
list and proved too diffuse for the 270M model. The development-only common-word
curriculum tests whether the model can first learn valid, state-dependent play
without changing the generative evaluator or constraining its logits.

```powershell
# Reproduce the successful 128-word curriculum pilot.
python -m wordle_lab.experiments.common_curriculum prepare --universe-size 128 --train-secrets 96 --states 512 --seed 2026
python -m wordle_lab.experiments.common_curriculum train --universe-size 128 --train-secrets 96 --states 512 --steps 600 --dev-games 25 --seed 2026 --learning-rate 0.00005

# Evaluate saved doses and honest decoder interventions.
python -m wordle_lab.experiments.common_curriculum evaluate --run-id sft-common-explicit-repeat-s2026-56f010458f --checkpoint final --decoder greedy_rep105 --dev-games 25

# Repeat-focused ORPO continuation from the successful SFT parent.
python -m wordle_lab.experiments.common_preference --parent-run-id sft-common-explicit-repeat-s2026-56f010458f --steps 100 --learning-rate 0.000005 --lambda-or 0.1 --dev-games 25
python -m wordle_lab.experiments.common_curriculum evaluate --run-id orpo-common-repeat-s2026-35b37fb67a --checkpoint step-000050 --decoder greedy_rep105 --dev-games 25
```

### Balanced policy diagnostics (development only)

The plan-driven u128 path removes the duplicate exposure floor, caps state and
target frequency, supports an action-token-weighted loss, and evaluates all 32
held-out development secrets. It never reads the locked protocol test split.

```powershell
# Build the versioned balanced data and run the balanced/word-focused cell of
# the fixed-budget 2x2 ablation. Use weight 1 for ordinary completion loss and
# dataset-version current for either corresponding control cell.
python -m wordle_lab.experiments.common_curriculum prepare-balanced --universe-size 128 --train-secrets 96 --states 512 --seed 2026
python -m wordle_lab.experiments.common_curriculum train --universe-size 128 --train-secrets 96 --states 512 --steps 600 --dev-games 32 --seed 2026 --learning-rate 0.00005 --dataset-version balanced --word-token-weight 8
```

The completed 2x2 screen confirmed that balanced coverage, not weighting by
itself, caused the improvement. Curriculum 003 and the strict-prompt controls
can be reproduced with:

```powershell
# Audited targeted data: true singleton, varied turn-2, low-posterior, and
# format-anchor buckets. This remains development-only and training-secret-only.
python -m wordle_lab.experiments.common_curriculum prepare-targeted --universe-size 128 --train-secrets 96 --states 1024 --seed 2026
python -m wordle_lab.experiments.common_curriculum train --universe-size 128 --train-secrets 96 --states 1024 --steps 600 --dev-games 32 --seed 2026 --learning-rate 0.00005 --dataset-version targeted --word-token-weight 8

# Exact balanced-002 states with strict output wording and declared root anchors.
python -m wordle_lab.experiments.common_curriculum prepare-balanced-strict-anchored --universe-size 128 --train-secrets 96 --states 512 --seed 2026
python -m wordle_lab.experiments.common_curriculum train --universe-size 128 --train-secrets 96 --states 512 --steps 600 --dev-games 32 --seed 2026 --learning-rate 0.00005 --dataset-version balanced_strict_anchored --word-token-weight 8

# Prompt-only evaluations are stored separately and never overwrite the
# curriculum-matched checkpoint evaluation.
python -m wordle_lab.experiments.common_curriculum evaluate --run-id RUN_ID --checkpoint final --decoder greedy_rep105 --prompt-variant strict --dev-games 32
```

These strict-data experiments improved format in isolation but did not beat
the 8/32 balanced-002 parent or pass the strategic gates. See
`PLAN_RESULTS.md` before launching another run.

Every evaluation now writes fixed-state diagnostic JSONL and aggregate JSON by
turn and posterior size. DAgger collection/training helpers live in
`wordle_lab.experiments.on_policy_recovery`; mixed 50/25/25 preference pairs
can be built by passing its `recovery.jsonl` artifact to:

```powershell
python -m wordle_lab.experiments.common_preference --parent-run-id RUN_ID --recovery-jsonl RECOVERY_JSONL --preference-pairs 512 --steps 100 --dev-games 32
```

These are curriculum results on held-out common-word development secrets.
They are deliberately not mixed with `WORDLE-PROTOCOL-002` or used to open its
locked test gate. See `EXPERIMENT_REPORT.md` for accepted and rejected results.

## Current model and matched-data policy (2026-08-21)

All current training and evaluation entrypoints are hard-pinned to the local
`google/gemma-3-270m-it` revision
`ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`. `WORDLE_MODEL_DIR` is no longer
accepted, and the loader fails closed for any other architecture, size, or
revision. Historical Qwen artifacts remain readable as provenance only.

The current representation comparison uses 4,096 identical source states in
each of three separate files: reasoning single-step, non-reasoning single-step,
and non-reasoning multi-step. Build and independently audit them with:

```powershell
py scripts/build_training_data.py --force
py scripts/audit_training_data.py --token-lengths
py scripts/train_sft.py --partition reasoning_single_step --dry-run
```

Each technique has its own launcher under `scripts/`: `train_sft.py`,
`train_dpo.py`, `train_orpo.py`, `train_grpo.py`, `train_grpo_stable.py`,
`train_q_sft.py`, and `train_sft_grpo.py`. Adapter ablations likewise have
`train_lora.py`, `train_rslora.py`, and `train_dora.py`. Q-SFT deliberately requires an
independently frozen Bellman snapshot joined by `build_q_sft_data.py`; it will
not invent targets from evaluator or locked-test information.

## Historical cross-family 1.5B capacity experiment (2026-08-20)

At the time, the base model was selected at process start with
`WORDLE_MODEL_DIR`. That historical mechanism has now been removed. It was used
for a matched balanced-curriculum run with the public
`Qwen/Qwen2.5-1.5B-Instruct` checkpoint because Gemma 3 1B requires a
Hugging Face license login that was not available locally.

Run `sft-common-balanced-word-s2026-b2c325bec6` reached 14/32 held-out
development wins with greedy decoding, versus 8/32 for the matched 270M
condition. Repeats fell from 27.4% to 2.0%, diagnostic action-target accuracy
rose from 14.1% to 41.3%, and turn-2 violations fell from 79.3% to 35.1%.
It still failed the promotion gates, especially singleton accuracy (7/74), so
the locked 1,000-answer test remains closed.

On-policy recovery collection now removes duplicate unchanged states before
mixing. This prevents repeated invalid generations at the same history from
recreating the exposure skew the balanced curriculum was designed to remove.

A matched Qwen2.5 3B condition used micro-batch 2 with accumulation 2 to keep
the effective batch at 4. It reached 15/32 at the final checkpoint and 16/32
at step 450, but gameplay compliance was only 71.2% and 74.6% respectively.
Those checkpoints were rejected; more parameters improved turn-2 decisions
while worsening reliable terminal behavior.
