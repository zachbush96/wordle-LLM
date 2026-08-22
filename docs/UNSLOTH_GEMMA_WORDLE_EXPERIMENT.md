# Unsloth Gemma Wordle Fine-Tuning Experiment

Date: 2026-08-22

Experiment: `UNSLOTH-GEMMA-SFT-001`

Run: `unsloth-sft-non_reasoning_single_step-s2026-b04e5f76a5`

Decision: **Wordle play did not improve; do not promote or open the locked test.**

## Executive answer

Unsloth successfully fine-tuned the exact pinned Gemma 3 270M IT model and produced a portable 16-bit LoRA adapter. Training was technically successful and efficient: 300 optimizer steps and 619,673 tokens completed in 292.4 seconds with 3.22 GB peak allocated VRAM.

The model did **not** get better at Wordle. The matched base and every trained checkpoint scored 0/32 held-out development wins. Fine-tuning did teach the output contract—terminal compliance and valid-word output rose from 0% to 100%—but it did not teach feedback-conditioned decisions. The final checkpoint still violated the Wordle posterior on 83.3% of valid gameplay turns, scored 0% on singleton states, and reduced retention from 30% to 12.5%.

That distinction is the main result: Unsloth made the training path work, but a faster backend did not change the learning objective or overcome the 270M model's state-tracking/generalization bottleneck.

## Question and hypothesis

The experiment asked whether Unsloth LoRA training on the project's audited 4,096-state Gemma-only bundle could improve unassisted Wordle play under the frozen protocol.

The working hypothesis was that a larger, balanced, inference-shaped state bundle plus an efficient Unsloth backend might move Gemma beyond learning only the `Final answer: WORD` envelope. The falsification criteria were held-out wins, posterior consistency, singleton accuracy, action-target accuracy, and retention—not training loss or output fluency.

## Frozen boundaries

- Protocol: `WORDLE-PROTOCOL-002`.
- Model: `google/gemma-3-270m-it` at revision `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`.
- Evaluation: natural greedy generation through the unchanged Transformers/PEFT harness.
- Forbidden: candidate injection, vocabulary masking, reranking, repeat bans, output repair, harness-selected guesses, and held-out-secret training.
- Split: 96 training secrets and 32 held-out development secrets in the fixed u128 universe.
- Locked 1,000-answer test: never read or evaluated.

The comparison bundle audit passed all ten checks, including held-out-secret separation, feedback recomputation, oracle-fact verification, matched targets, multi-turn history fidelity, and evaluation-only development probes.

## Why this Unsloth setup

Unsloth's current official documentation supports direct Windows installation and Gemma 3 fine-tuning. Its Gemma guidance describes BF16 activation handling and FP16/BF16-friendly kernels; the project used the code-based API rather than Studio. See the [official Unsloth repository](https://github.com/unslothai/unsloth) and [Gemma 3 fine-tuning guide](https://docs.unsloth.ai/basics/tutorial-how-to-run-and-fine-tune-gemma-3).

The runtime was isolated under `.cache/unsloth-venv` so its dependency pins did not alter the historical experiment environment.

| Component | Version / choice |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Ti, 16 GB |
| Unsloth | 2026.8.19 |
| Unsloth Zoo | 2026.8.13 |
| PyTorch | 2.10.0+cu128 |
| Transformers in training runtime | 5.5.0 |
| Triton Windows | 3.7.1.post27 |
| Precision | BF16, no weight quantization |
| Adapter | LoRA rank 16, alpha 32, dropout 0 |
| Targets | q/k/v/o plus gate/up/down projections |
| Gradient checkpointing | Unsloth |

No 4-bit quantization was used because the 270M model fits comfortably in memory and a quantization change would add another experimental factor.

## Data and recipe

The selected representation was `non_reasoning_single_step`: each row contains the complete visible Wordle history and a direct terminal answer. This minimized sequence length and avoided asking a 270M model to learn both visible rationale and action selection at once.

| Item | Value |
| --- | --- |
| Training rows | 4,096 |
| Source partition SHA-256 | `51c51c47097024dc829fd733d668cf70ae1d7ccb878c5b0fee4b3c88831731ad` |
| Selected rows SHA-256 | `60ecc5807f8e28e6dafe5920cc9fe855fa5a65fb81c25b09f6c02f2de976740c` |
| Seed | 2026 |
| Learning rate | 5e-5 with warmup and cosine decay |
| Optimizer | AdamW |
| Effective batch | 16 |
| Maximum length | 320 tokens |
| Loss | Completion-only causal LM loss |
| Checkpoints | 75, 150, 225, and 300 steps |

An initial batch-4 × accumulation-4 attempt was stopped before its first checkpoint after live measurements showed only about 5.5 GB total GPU use and poor launch efficiency. The completed run used batch-16 × accumulation-1. This preserved the effective optimizer batch and all learning hyperparameters while reducing redundant launches. An 8-step/128-row smoke run then verified memory use, checkpoint portability, and saving before the primary run started.

## Training outcome

| Metric | Result |
| --- | ---: |
| Optimizer steps | 300 |
| Optimizer tokens | 619,673 |
| Wall time | 292.38 s (0.081 GPU-hours elapsed) |
| Peak allocated VRAM | 3,221,700,608 bytes (3.00 GiB) |
| Trainable parameters | 3,796,992 (1.396%) |
| Total parameters with adapter | 271,895,168 |
| Final adapter SHA-256 | `3b2c57fa44b8516cfaf6f680a4ee2b31481801ef7b541659685df2670876c1e3` |

Training loss fell smoothly:

| Step | Loss | Cumulative tokens |
| ---: | ---: | ---: |
| 1 | 3.88 | 2,083 |
| 25 | 1.30 | 51,484 |
| 75 | 0.91 | 154,616 |
| 150 | 0.81 | 308,900 |
| 225 | 0.80 | 464,126 |
| 300 | 0.75 | 619,673 |

The falling loss confirms that optimization worked. It does not establish held-out policy learning.

## Development results

Every checkpoint was loaded outside Unsloth through the normal Transformers/PEFT path. Each saw the same 32 held-out games, 512 fixed-state development probes, and 200 retention prompts.

| Condition | Wins | Terminal compliance | Invalid guesses | Gameplay repeats | Gameplay constraint violations | Fixed-state violations | Singleton accuracy | Action-target accuracy | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched frozen base | 0/32 | 0% | 100% | n/a | n/a | n/a | 0% | n/a | 30.0% |
| Step 75 | 0/32 | 100% | 0% | 83.3% | 83.3% | 98.2% | 0% | 0.59% | 15.0% |
| Step 150 | 0/32 | 100% | 0% | 75.0% | 83.3% | 98.0% | 0% | 0.59% | 12.5% |
| Step 225 | 0/32 | 100% | 0% | 65.6% | 83.3% | 97.9% | 0% | 1.17% | 12.5% |
| Step 300 | 0/32 | 100% | 0% | 54.2% | 83.3% | 98.4% | 0% | 0.20% | 12.5% |

The base produces no parseable guesses, so posterior/repeat rates for it are undefined rather than zero.

For historical context, the earlier balanced-002, word-focused Gemma run reached 8/32 wins, 89.2% compliance, 79.3% turn-2 violations, 4/74 singleton accuracy, and 14.1% action-target accuracy. It used a different training representation and diagnostic probe set, so it is context—not a matched Unsloth backend comparison. The new Unsloth run did not displace it.

## What changed in the model

The checkpoints show a recognizable failure progression:

- Step 75 emitted `SHARE` on all 192 gameplay calls.
- Step 150 emitted mostly `SHARE`, with limited `ORDER` and `CHECK` variation.
- Step 225 switched mostly to `LEAVE`.
- Step 300 used more words and reduced repeats, but its most common guess was `LEAST` and it still failed singleton states.

This is distributional imitation, not reliable state tracking. More surface diversity reduced repeats but did not improve posterior consistency or wins.

## Did it improve?

### Yes, narrowly

- The model learned the exact output format: 0% to 100% terminal compliance.
- It moved from no valid guesses to 100% valid-word outputs.
- The final adapter is portable across Unsloth and the ordinary PEFT evaluator.
- The Unsloth path trained the full 4,096-row run locally in under five minutes with low VRAM use.

### No, on the research question

- Win rate remained 0/32.
- Singleton accuracy remained 0%, the clearest test of feedback-to-answer mapping.
- Fixed-state posterior violations stayed between 97.9% and 98.4%.
- The small action-target signal peaked at step 225 and then regressed.
- Retention fell by 17.5 percentage points at the final checkpoint.
- The trained model remained far below the historical 8/32 Gemma result.

Therefore the defensible answer is **no Wordle improvement**. Calling this successful because formatting improved would confuse task compliance with strategy.

## Why it did not improve

1. **Unsloth is an execution backend, not a better Wordle objective.** It accelerated and reduced the memory cost of LoRA, but the supervision was still ordinary next-token imitation.
2. **The model minimized common completion patterns instead of learning the transition rule.** The loss curve improved while singleton/action accuracy did not—a direct sign that the objective rewarded easier token regularities more than feedback-conditioned decisions.
3. **Held-out histories generalize poorly at 270M.** Only 7.8% of fixed development histories exactly matched a training history; the adapter did not infer the reusable green/yellow/gray transformation for unseen histories.
4. **Formatting and strategy compete for limited capacity.** The adapter perfectly retained the short terminal envelope but lost general retention and failed posterior constraints.
5. **More of the same dose was not supported by the checkpoint curve.** Step 225 had the best action-target accuracy, while step 300 had lower repeat rate but worse diagnostic accuracy. Doubling the same run would be optimization churn, not a new hypothesis.

## How to push further

Priority order:

1. **Run a matched Unsloth reproduction of balanced-002 with the historical 8× action-token loss.** This separates the backend question from the data/objective question and tests whether Unsloth can reproduce the strongest Gemma recipe. It requires enabling logits explicitly so action-token weighting remains exact.
2. **Run the preregistered 1,024/2,048/4,096-state learning curves before another objective.** Use the same representation, seed policy, token accounting, and all four dose checkpoints. Stop if singleton/action accuracy remains flat.
3. **Test frozen-target Q-SFT on the audited training-only Bellman snapshots.** This is the most direct objective change for delayed Wordle action value, while canonical evaluation remains natural generation.
4. **Use a same-family Gemma capacity ablation.** Gemma 3 1B is the cleanest next model if its gated checkpoint can be provisioned. The prior Qwen 1.5B result suggests capacity matters, but it confounds family and size.
5. **Replicate only a genuinely promising parent across seeds 1337, 7331, and 2026.** Promotion still requires at least 99% terminal compliance, under 30% turn-2 violations, materially positive singleton accuracy, and three-seed consistency.
6. **Keep the locked test closed.** No checkpoint here approaches the promotion gates.

The strongest immediate experiment is item 1: use Unsloth to reproduce the known 8/32 balanced/word-focused recipe under matched settings. If it reproduces the behavior while reducing cost, Unsloth is validated as the backend; if not, the backend-specific numerical path needs investigation before more ambitious objectives.

## Reproduction

Create an isolated Unsloth environment using the current [official installation guidance](https://github.com/unslothai/unsloth#-get-started) and `requirements-unsloth.txt`, then from the repository root run:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& '.\.cache\unsloth-venv\Scripts\python.exe' scripts\train_unsloth_sft.py `
  --partition non_reasoning_single_step `
  --steps 300 `
  --seed 2026 `
  --learning-rate 0.00005

py scripts\evaluate_unsloth_sft.py `
  --run-dir artifacts\runs\unsloth-sft-non_reasoning_single_step-s2026-b04e5f76a5 `
  --checkpoint step-000300 `
  --dev-games 32 `
  --diagnostic-items 512

py scripts\evaluate_gemma_comparison_base.py --dev-games 32 --diagnostic-items 512
```

Local checkpoints, raw game JSONL, fixed-state outputs, retention rows, and training logs remain in the ignored `artifacts/runs/` tree. The compact, tracked result record is [research/unsloth_gemma_wordle_results.json](research/unsloth_gemma_wordle_results.json).

## Verification record

- Comparison data audit: passed.
- Locked-test access: false throughout.
- Adapter loaded through ordinary PEFT evaluation: passed.
- Compilation: `py -m compileall -q wordle_lab scripts tests`.
- Regression suite after implementation: 76 passed before documentation finalization; rerun at handoff.
- Final adapter model SHA-256: `3b2c57fa44b8516cfaf6f680a4ee2b31481801ef7b541659685df2670876c1e3`.
- Training metrics SHA-256: `7baa9b26394f812125ee7e4015bdb10c50c5ad37c0c06c30cc0c1b4a9656cb4c`.
