# Qwen3-0.6B Wordle results and handoff

Date: 2026-08-25

## Executive result

Qwen3-0.6B learned substantially from post-training, but no checkpoint is promotable.

The base model produced 0/32 wins and 0% terminal compliance in both native-thinking and non-thinking modes. The training stack itself is sound: both disjoint 32-state LoRAs reached 32/32 natural exact recall with rank-1 targets and about 0.99997 mean target probability.

On held-out Wordle play, the strongest canonical result was full-parameter tuning at step 450: **16/32 wins**, 94.1% compliance, 31.6% turn-2 posterior violations, 7/74 singleton accuracy, 41.7% action-target accuracy, and zero retention. A repetition-penalty decoder reached 17/32 but reduced compliance to 89.9%, so it is a rejected decoder tradeoff rather than the selected policy. The best adapter-only result was rsLoRA at step 300 with 15/32 wins, but only 78.9% compliance and 3/74 singleton accuracy.

This is real development improvement over the base and ordinary LoRA, and it closely approaches the historical Qwen2.5-1.5B strategic metrics with a smaller model. It is not a reliable Wordle strategy: formatting, singleton solving, retention, and seed replication remain inadequate. The locked 1,000-answer test was never opened.

## Protocol

- Model: `Qwen/Qwen3-0.6B`
- Revision: `c1899de289a04d12100db370d81485cdf75e47ca`
- Parameters: 596,049,920 base parameters
- GPU: NVIDIA GeForce RTX 4060 Ti, 16 GB
- Protocol: `WORDLE-PROTOCOL-002`
- Development gameplay: 32 frozen balanced-002 answers
- Fixed-state diagnostics: 128 states, including 74 singletons and 58 turn-2 states
- Retention: 200 frozen probes
- Seed: 2026
- Canonical generation: greedy, Qwen non-thinking chat template, 128 new-token limit
- Forbidden throughout: candidate injection, vocabulary masking, reranking, repeat bans, output repair, harness-selected guesses, and locked-test access

Native thinking was evaluated separately. It exhausted the bounded output on internal reasoning, scoring 0/32, 0% compliance, and 0% retention. Non-thinking also scored 0/32 and 0% compliance but retained 44%; it became the declared canonical mode.

## Main development results

| Condition | Wins | Compliance | Invalid | Repeat | Turn-2 violation | Singleton | Action target | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base, non-thinking | 0/32 | 0.0% | 100.0% | 0.0% | undefined | 0/74 | undefined | 44.0% |
| LoRA, step 300 | 8/32 | 81.9% | 18.6% | 24.8% | 72.7% | 4/74 | 17.1% | 34.0% |
| LoRA, step 450 | 12/32 | 70.1% | 29.9% | 12.3% | 56.9% | 2/74 | 27.0% | 32.5% |
| rsLoRA, step 300 | **15/32** | 78.9% | 21.1% | 7.2% | 49.1% | 3/74 | 32.0% | 36.0% |
| rsLoRA, step 600 | 11/32 | 56.8% | 46.8% | 9.3% | **28.1%** | 3/74 | 43.5% | 35.5% |
| DoRA, step 450 | 12/32 | 69.8% | 30.2% | 14.2% | 54.4% | 4/74 | 31.1% | 28.5% |
| Coverage LoRA, 4,096 examples | 8/32 | 95.6% | 4.4% | 19.7% | 80.7% | 4/74 | 6.4% | 27.0% |
| Constraint-first, step 600 | 12/32 | 93.9% | 11.1% | 19.4% | 73.7% | 8/74 | 8.7% | 18.0% |
| Full tune, step 300 | 14/32 | 95.0% | 5.0% | 20.4% | 50.0% | 5/74 | 31.3% | 0.0% |
| Full tune, step 450 | **16/32** | **94.1%** | 13.1% | 13.5% | **31.6%** | **7/74** | **41.7%** | 0.0% |
| Full tune, step 600 | 15/32 | 94.3% | 12.7% | 13.8% | 31.6% | 7/74 | 42.5% | 0.0% |
| Full step 450, repetition penalty 1.05 | 17/32 | 89.9% | 17.0% | 10.6% | 31.6% | 7/74 | 42.1% | 0.0% |

The rsLoRA endpoint is a useful warning: it passed the isolated turn-2 threshold, but only by collapsing terminal reliability. No dose passed all required gates.

## Memorization diagnostics

Both tiny LoRA cells completed 400 steps with 8x action-token weighting.

| Cell | Natural exact | Terminal compliance | Mean target rank | Mean target probability |
| --- | ---: | ---: | ---: | ---: |
| General, 32 disjoint states | 32/32 | 100% | 1.0 | 0.999957 |
| Singleton, 32 disjoint states | 32/32 | 100% | 1.0 | 0.999974 |

Qwen can fit exact feedback-conditioned outputs. Held-out singleton failure is therefore a transfer/objective problem, not a broken tokenizer, adapter, optimizer, or inability to memorize the requested mapping.

## Adapter and full-tune interpretation

Ordinary LoRA peaked at 12 wins. rsLoRA materially improved held-out actions and peaked at 15 wins, while DoRA did not improve the reliability frontier. Increasing rsLoRA dose reduced turn-2 violations but destroyed formatting, showing that legality and terminal control were not learned together.

Full tuning produced the best balanced result. Compared with the step-450 ordinary LoRA, full tuning added four wins, raised compliance by 24.1 percentage points, lowered turn-2 violations by 25.3 points, and increased singleton accuracy from 2/74 to 7/74. This supports trainable scope as a real bottleneck, as it did for Gemma. It does not solve the objective mismatch, and retention falling to zero makes the checkpoint unsuitable for deployment or further preference/RL training.

## Coverage and objective exercises

The one-pass 4,096-state LoRA did not reproduce Gemma's coverage gain. It peaked at 9/32 around 3,072 examples and ended at 8/32. Compliance improved with dose, but posterior legality, singleton behavior, action matching, and retention did not.

Constraint-first sampled multi-label SFT reached 12/32 at step 600 and its best singleton result was 8/74, but it still had 73.7% turn-2 violations. Like the Gemma result, sampled legal labels did not behave like a true set-normalized legal-action loss.

The structured microtask run used every one of the 1,216 training records exactly once after Qwen tokenization required a declared 512-token limit. The original effective-batch-16 attempt was stopped before any checkpoint because padding-heavy native training could not reach step 150 within the bounded window; the completed one-pass cell is explicitly not a matched exposure replication.

| Held-out microtask | Accuracy | Strict-format coverage |
| --- | ---: | ---: |
| Feedback decode | 7/128 = 5.5% | 96.9% |
| Constraint merge | 0/128 = 0.0% | 55.5% |
| Candidate validity | 34/96 = 35.4% | 100% |
| Singleton solve | 1/128 = 0.8% | 71.1% |
| Full policy | 0/128 = 0.0% | 85.9% |

Overall accuracy was 42/608 (6.9%) with 80.9% format coverage. Schema learning again failed to transfer into symbolic constraint operations.

## Why post-SFT preference and RL methods did not run

The best canonical parent, full step 450, fails every prerequisite:

| Gate | Required | Observed |
| --- | ---: | ---: |
| Terminal compliance | at least 99% | 94.1% |
| Turn-2 posterior violation | below 30% | 31.6% |
| Singleton accuracy | at least 80% | 9.5% |

Q-SFT therefore records `blocked_prerequisite_legality_gate_failed`. SFT-to-GRPO and GRPO/AVSPO record ineligible warm starts; AVSPO also lacks the required passing baseline-GRPO ablation. DPO and ORPO were not run from a parent whose strategy and retention are already unreliable. This follows the same fail-closed contract used for Gemma: running additional objectives from a bad parent would spend compute without answering a valid causal question.

No three-seed replication was launched because no single-seed parent passed promotion gates. The locked test remained closed.

## Did it improve?

Yes, on development learning and capacity:

- Tiny memorization moved from malformed output to 32/32 exact recall in two disjoint cells.
- rsLoRA reached 15/32, well above ordinary LoRA's 12/32 peak.
- Full tuning reached 16/32 canonically and materially improved legality/action metrics over LoRA.
- The 0.6B full-tune result is competitive with the historical Qwen2.5-1.5B adapter's 14/32, 94.1% compliance, 35.1% turn-2 violations, 7/74 singleton accuracy, and 41.3% action accuracy.

No, on promotable strategy:

- The best canonical Qwen checkpoint still misses format and turn-2 gates.
- It solves only 7/74 singleton diagnostics.
- Retention is zero.
- The apparent 17-win decoder result sacrifices compliance.
- There is no three-seed replication or locked-test evidence.

Against the strongest Gemma coverage result (19/32 at 15,360 examples), Qwen3-0.6B full tuning has fewer wins but much lower turn-2 violations (31.6% versus 53.4%). Both have zero retention and weak singleton behavior. This is a useful cross-family tradeoff, not evidence that either checkpoint is generally better.

## Next materially different hypothesis

Do not extend these SFT curves or add another sampled-label curriculum. The useful next experiment is a true set-normalized legal-action objective with explicit retention replay/regularization, evaluated from the base or a jointly trained parent rather than applied after catastrophic forgetting. It should optimize probability mass over the complete legal action set and separately preserve terminal-format examples. Run three matched seeds only if a development parent first reaches 99% compliance, sub-30% turn-2 violations, and materially improved singleton accuracy.

## Verification and evidence

- Full repository: `262 passed in 117.83s`.
- Qwen-focused tests: `3 passed`.
- `compileall` passed for `wordle_lab`, `scripts`, and this track.
- Model manifest pins and hashes the exact downloaded snapshot.
- `experiment_matrix.json` declares 27 evaluated gameplay conditions plus tiny/structured results and local training accounting.
- `results_manifest.json` hashes 143 committed result files.
- Raw per-game, per-fixed-state, retention, memorization, and structured outputs are under `results/`.
- Local model weights and checkpoints remain ignored; no locked-test payload is present in this bundle.
