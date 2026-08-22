# Plan implementation results (development only)

Date: 2026-08-15

No locked-test answer or trajectory was read. Training labels use only the 96
u128 training secrets. The 32 development secrets are used only for evaluation.
Generation remains unconstrained: no candidate list, trie mask, reranker, or
hidden-answer access is provided to Gemma.

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

The balanced word-focused cell is a useful directional result, not a method
winner: it used 363,758 active optimizer tokens versus 435,691 in the historical
parent, only one seed was run, and it fails all strategic promotion gates except
fixed-state terminal compliance. DAgger eliminated repeats but degraded format
and state-level accuracy, so further DAgger rounds and preference training were
stopped.

The next valid experiment is to complete the four ablation cells with matched
token accounting, then tune the balanced cell for format retention before
collecting a second recovery round. No locked test should be opened.

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

This isolates balanced state/target coverage as the main improvement. Word
weighting is useful only on the balanced data; it does not rescue the
duplicate-heavy curriculum. Re-evaluating the best balanced adapter with
`greedy_rep105` kept 8/32 wins and 89.2% compliance while reducing gameplay
repeats from 27.4% to 1.9%.

### New data experiments

`COMMON-WORD-CURRICULUM-003` added 1,024 audited examples: 307 genuine
singleton states covering all 96 training targets, 359 varied turn-2 states,
256 posterior-size 2-4 states, 51 format anchors, and 51 later broad states.
It contains 974 unique source histories and 126/128 action words. Its rendered
hash is `238d713f7579e8767ed6f42a258351f2732d9a6826fead484cfb921bf096d60f`.
No held-out episode secret appears in its labels.

The from-scratch 003 run achieved 100% gameplay compliance and zero invalid
guesses, proving that format can be trained, but regressed to 3/32 wins,
92.7% turn-2 violations, 0/74 singleton accuracy, and 0.8% target accuracy.
A 150-step low-rate continuation from the 8/32 parent peaked at 5/32 wins and
97.8% compliance at step 38; later doses degraded further.

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

The original balanced/word-focused adapter remains the best strategic
development run, but it fails every promotion gate. Curriculum 003 proves that
the 270M model can learn the exact output envelope while simultaneously showing
that broader training-only singleton coverage does not produce held-out
constraint-solving. More SFT curriculum churn is not justified. The next
useful experiment is a seed-matched capacity ablation with a roughly 1B Gemma
checkpoint under curriculum 002, followed by the same diagnostics. No larger
Gemma checkpoint is currently present in the local model directory, so that
experiment requires provisioning the model before training.

## 2026-08-20 cross-family capacity continuation

A public Qwen2.5 1.5B Instruct checkpoint was run on the exact balanced-002
data recipe, 600-step budget, seed 2026, learning rate 5e-5, and 32-secret
development split. It achieved 14/32 wins, 94.1% gameplay compliance, 2.0%
repeats, 41.3% diagnostic action-target accuracy, 35.1% turn-2 violations, and
7/74 singleton accuracy. This is a strong capacity/family signal over the
matched 270M result, but not a clean Gemma-family ablation and not promotable.

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
compliance. Every 3B dose was rejected, confirming that raw capacity alone did
not solve terminal reliability or singleton recovery.
