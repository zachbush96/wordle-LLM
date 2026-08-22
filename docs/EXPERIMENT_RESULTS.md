# Development Experiment Results

Date: 2026-08-15

This is the detailed development-results ledger through the 2026-08-20 capacity
experiments. For the current summary and the planned Gemma-only comparison, start
with the root [README](../README.md). Earlier, checkpoint-level pilots remain in
[history/PILOT_REPORT.md](history/PILOT_REPORT.md).

No locked-test answer or trajectory was read. Training labels use only the 96 u128
training secrets; the 32 development secrets are evaluation-only. Generation stays
unconstrained: Gemma receives no candidate list, trie mask, reranker, or hidden-answer
access.

## Implemented

- Fixed-state diagnostics with per-item JSONL and aggregate metrics by turn and
  posterior size.
- `COMMON-WORD-CURRICULUM-002`: distinct inference-shaped states, four-copy
  state cap, eight-example target cap, explicit composition/provenance, and no
  synthetic `Rejected:` conversation.
- Configurable word-token-weighted SFT loss and weighted-token accounting.
- Training-secret-only on-policy recovery collection and a hashed 50/50
  static/recovery DAgger round.
- Preference builder with 50% model constraint violations, 25% repeats, and
  25% posterior-consistent strategic negatives.

## One-seed diagnostic results

| Metric | Current u128 SFT | Balanced + word loss | + DAgger round 1 |
| --- | ---: | ---: | ---: |
| Held-out wins | 1/32 | 8/32 | 9/32 |
| Gameplay terminal compliance | 95.4% | 89.2% | 75.0% |
| Gameplay repeat rate | 17.3% | 27.4% | 0.0% |
| Diagnostic action-target accuracy | 1.6% | 14.1% | 11.5% |
| Diagnostic posterior violations | 91.4% | 82.8% | 84.4% |
| Turn-2 posterior violations | 84.5% | 79.3% | 78.6% |
| Singleton accuracy | 0/74 | 4/74 | 2/74 |

The balanced, word-focused cell is an encouraging direction—not a method winner.
It used 363,758 active optimizer tokens versus 435,691 in the historical parent,
ran on one seed, and missed every strategic promotion gate except fixed-state
terminal compliance. DAgger removed repeats but hurt formatting and state-level
accuracy, so we stopped further DAgger rounds and preference training.

The next valid step is to complete the four ablation cells with matched token
accounting, then tune the balanced cell for format retention before collecting a
second recovery round. The locked test stays closed.

## 2026-08-19 continuation

The fixed-seed 2x2 ablation is now complete. All cells used 600 optimizer
steps, seed 2026, learning rate 5e-5, the same 96/32 u128 split, and natural
generation. The locked test remained closed.

| Dataset | Loss | Wins | Gameplay compliance | Turn-2 violations | Singleton | Target accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Current 001 | completion | 1/32 | 90.0% | 87.9% | 0/74 | 1.6% |
| Current 001 | word-focused | 0/32 | 87.6% | 87.7% | 0/74 | 2.4% |
| Balanced 002 | completion | 6/32 | 81.6% | 81.0% | 2/74 | 10.9% |
| Balanced 002 | word-focused | 8/32 | 89.2% | 79.3% | 4/74 | 14.1% |

This points to balanced state/target coverage as the main improvement. Word
weighting helped only with balanced data; it did not rescue the duplicate-heavy
curriculum. Re-evaluating the best balanced adapter with `greedy_rep105` kept
8/32 wins and 89.2% compliance while reducing gameplay repeats from 27.4% to
1.9%.

### New data experiments

`COMMON-WORD-CURRICULUM-003` added 1,024 audited examples: 307 genuine
singleton states covering all 96 training targets, 359 varied turn-2 states,
256 posterior-size 2-4 states, 51 format anchors, and 51 later broad states.
It contains 974 unique source histories and 126/128 action words. Its rendered
hash is `238d713f7579e8767ed6f42a258351f2732d9a6826fead484cfb921bf096d60f`.
No held-out episode secret appears in its labels.

The from-scratch 003 run achieved 100% gameplay compliance with no invalid
guesses. That shows the format can be learned, but not the strategy: it fell to
3/32 wins, 92.7% turn-2 violations, 0/74 singleton accuracy, and 0.8% target
accuracy. A 150-step low-rate continuation from the 8/32 parent peaked at 5/32
wins and 97.8% compliance at step 38, then declined with additional training.

`COMMON-WORD-CURRICULUM-004` rerendered the exact balanced-002 states with the
strict prompt. With only four root examples it reached 3/32 wins and 63.4%
compliance. `COMMON-WORD-CURRICULUM-005` added an explicit, declared 51-copy
root-format anchor while preserving every non-root state. It reached 6/32 and
95.0% compliance with greedy decoding. Its repetition-penalty checkpoint curve
peaked at 7/32, but only after compliance fell to 88.5%; the 100%-compliant
checkpoints scored 0-3/32.

Applying the strict prompt to the original balanced adapter without training
was also rejected: it produced only 3-4/32 wins and 37-40% compliance because
the prompt distribution no longer matched its training data.

### Decision

The original balanced/word-focused adapter remains the strongest strategic
development run, but it misses every promotion gate. Curriculum 003 shows that
the 270M model can learn the exact output envelope; it also shows that more
training-only singleton examples do not yield held-out constraint solving. More
SFT curriculum churn is not justified. The next useful test is a seed-matched
capacity ablation with an approximately 1B Gemma checkpoint under curriculum
002 and the same diagnostics. No larger Gemma checkpoint is present locally, so
that test requires model provisioning first.

## 2026-08-20 cross-family capacity continuation

A public Qwen2.5 1.5B Instruct checkpoint was run on the exact balanced-002
data recipe: 600 steps, seed 2026, learning rate 5e-5, and the 32-secret
development split. It achieved 14/32 wins, 94.1% gameplay compliance, 2.0%
repeats, 41.3% diagnostic action-target accuracy, 35.1% turn-2 violations, and
7/74 singleton accuracy. That is a strong capacity/family signal over the
matched 270M result, but it is neither a clean Gemma-family ablation nor a
promotable model.

Targeted-003 training tied the 14/32 win count while regressing diagnostic
accuracy and posterior consistency. A repetition-penalty checkpoint reached
16/32 but was rejected for 20.6% invalid guesses and 7.2% repeats.
Deduplicated DAgger improved singleton accuracy to 15.4% but fell to 13/32 and
regressed overall constraint consistency. Strict anchored continuation fell
to 11/32. The balanced Qwen adapter therefore remains the best multi-metric
development artifact, and the locked test remains closed.

A final matched Qwen2.5 3B capacity condition preserved the effective batch
with micro-batch 2 and accumulation 2. The final checkpoint reached 15/32 and
crossed the turn-2 violation gate at 29.8%, but gameplay compliance collapsed
to 71.2% with 37.4% invalid guesses. Step 450 reached 16/32 but only 74.6%
compliance. Every 3B dose was rejected. More parameters alone did not solve
terminal reliability or singleton recovery.

## 2026-08-22 Unsloth Gemma continuation

The first audited Gemma-comparison training cell used Unsloth 2026.8.19 with
the pinned Gemma 3 270M model and all 4,096 `non_reasoning_single_step` rows.
The 16-bit LoRA run completed 300 steps and 619,673 optimizer tokens in 292.4
seconds, with 3.22 GB peak allocated VRAM. The locked test remained closed.

All 75/150/225/300-step checkpoints scored 0/32 held-out development wins.
Terminal compliance improved from the matched base's 0% to 100%, but the final
checkpoint had 83.3% gameplay constraint violations, 98.4% fixed-state
posterior violations, 0% singleton accuracy, 0.2% action-target accuracy, and
12.5% retention. The matched base retained 30%.

The run therefore improved formatting, not Wordle play. Falling training loss
and increasing guess diversity did not transfer to feedback-conditioned action
selection. The complete recipe, dose table, hashes, and next-step decisions are
in [UNSLOTH_GEMMA_WORDLE_EXPERIMENT.md](UNSLOTH_GEMMA_WORDLE_EXPERIMENT.md).
