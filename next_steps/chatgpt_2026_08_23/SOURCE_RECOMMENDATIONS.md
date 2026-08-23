# Source recommendations: “Suggest Next Steps”

## Source

- ChatGPT conversation title: `Suggest Next Steps`
- Conversation URL: <https://chatgpt.com/c/6a8a52f1-c4f4-83ea-a03a-d722f66f3c68>
- Discussion date: 2026-08-22
- Date note: the title and URL were read directly in Chrome. The discussion date is inferred from the user's description of it as “last night” on 2026-08-23.
- Captured for implementation: 2026-08-23

## How to read this document

This is a faithful, structured paraphrase of the next steps recommended in the source conversation. It is an implementation specification, not an experiment-results report.

For every item below:

- **Recommendation** states what the source conversation proposed.
- **Implementation requirement** states what must be built, controlled, or measured for that proposal to be tested.
- **Outcome boundary** states what the recommendation itself did not prove.

Passing code tests, producing a dataset, or successfully launching training would establish implementation readiness only. Strategic improvement must be supported separately by completed runs and their measured results.

## Complete recommendation set

### 1. Run two tiny 32-state memorization diagnostics first

**Recommendation**

Before interpreting another large fine-tune, test whether the model can memorize a deliberately tiny supervised mapping. Run two separate cells:

1. 32 general training states; and
2. 32 training-only singleton states whose visible feedback leaves exactly one answer.

The cells should be distinct so the singleton diagnostic is not simply a relabeling of the general set.

**Implementation requirement**

- Use exactly 32 examples in each cell, with training secrets only.
- Keep the general and singleton source-state IDs disjoint.
- Audit every feedback row, target, source split, and content hash.
- Evaluate the untrained parent before fine-tuning and the trained checkpoint after fine-tuning on the exact same 32 states.
- Preserve ordinary, natural model generation and exact terminal-answer accuracy.
- For every state, record the target word's exact rank and normalized probability before and after training, not merely loss or top-1 output.
- Preserve enough detail to compare rank movement, probability movement, exact recall, top-1 rank accuracy, and aggregate rank statistics separately for the general and singleton cells.
- Treat this as a training-set memorization diagnostic, not a held-out generalization result.

**Outcome boundary**

The source conversation recommended this diagnostic; it did not establish that either cell would be memorized or that memorization would transfer to Wordle play.

### 2. Reproduce balanced-002 with 8× action-token weighting through Unsloth

**Recommendation**

Reproduce the strongest historical Gemma 270M balanced-data recipe using the current Unsloth backend. The discriminating condition is `COMMON-WORD-CURRICULUM-002` with 8× weighting on the answer/action token.

**Implementation requirement**

- Use the exact balanced-002 training rows and their recorded hash.
- Match the historical seed, step count, learning rate, effective batch, prompt representation, LoRA targets, checkpoint schedule, and 8× action-token objective as closely as the backend permits.
- Record any backend-specific difference rather than silently treating it as matched.
- Evaluate saved doses with the frozen development protocol and natural generation.
- Report terminal compliance, invalid guesses, repeats, overall and turn-2 posterior violations, singleton accuracy, action-target accuracy, retention, and wins together.
- Keep the locked test closed.

The point of this cell is to separate an Unsloth/backend effect from the data-and-objective change in the recent direct-SFT runs.

**Outcome boundary**

The recommendation did not claim that Unsloth would recover the historical result or improve it. A completed matched run is required.

### 3. Compare matched LoRA against full-parameter fine-tuning

**Recommendation**

Test whether the low-rank adapter itself is limiting the 270M model by comparing LoRA with full-parameter fine-tuning.

**Implementation requirement**

- Hold the base model revision, training data, ordering, objective, seed, prompt, step/token budget, evaluation split, and decoding contract fixed.
- Make trainable parameter scope the intended experimental difference: rank-16 LoRA versus all model parameters.
- Match optimizer behavior and effective batch where technically possible, and document unavoidable memory-driven deviations.
- Run a memory/preflight check before allocating a full training run.
- Save comparable checkpoints, accounting, raw evaluation outputs, and aggregate metrics.
- Preserve the same leakage and locked-test boundaries in both cells.

**Outcome boundary**

The source conversation posed adapter capacity as a hypothesis. It did not establish that full fine-tuning is better, feasible within available VRAM, or strategically sufficient.

### 4. Add structured feedback and legality microtasks, then mix them with full-policy data

**Recommendation**

Teach and measure the feedback-to-constraint transformation directly instead of relying only on an end-to-end next-word loss. The requested microtasks are:

1. decode one feedback row into fixed positions, forbidden positions, minimum letter counts, maximum letter counts, and excluded letters;
2. merge constraints correctly across multiple turns;
3. classify proposed candidates as valid or invalid;
4. solve singleton states; and
5. mix these records with ordinary full-policy examples in a shared curriculum.

**Implementation requirement**

- Handle duplicate letters with exact Wordle accounting. A gray duplicate after a green/yellow copy establishes an upper count; it does not necessarily exclude the letter.
- Make multi-turn merging deterministic and fail on contradictory constraints.
- Build candidate-validity data that is exactly 50% valid and 50% invalid.
- Balance invalid records across the six requested causes: green-position violation, yellow-position violation, missing required letter, gray/excluded letter, duplicate-count violation, and repeated guess.
- Retain all detected violations while assigning a deterministic primary label.
- Derive singleton targets from visible feedback and a training-only answer universe, not by copying a hidden answer into the model input.
- Define one record schema that can carry the microtasks and full-policy rows without exposing secrets or posterior candidate lists.
- Include provenance, record hashes, source-state hashes, split membership, and label-recomputation audits.
- Evaluate per-task accuracy and per-invalid-reason accuracy, not only a combined score.

**Outcome boundary**

The recommendation identified a targeted training/evaluation layer. Creating correct microtask data does not itself show that a fine-tuned model learns the tasks or improves gameplay.

### 5. Test a constraint-first full-policy objective

**Recommendation**

Train the policy to prioritize clue legality and posterior consistency before optimizing exact agreement with one oracle action. Where several actions are consistent with the feedback, use diverse legal targets rather than treating every non-oracle word as equally wrong.

**Implementation requirement**

- Generate targets from the training-only posterior and exclude repeated guesses.
- Preserve multiple acceptable, feedback-consistent actions where the state permits them.
- Give singleton states explicit emphasis because they have one unambiguous legal answer.
- Use natural generation at evaluation time; do not inject candidates, mask vocabulary, rerank, ban repeats in the harness, or repair output.
- Select checkpoints using legality/posterior consistency with singleton correctness as a mandatory diagnostic, alongside format, retention, and gameplay metrics.

**Outcome boundary**

The source conversation proposed constraint-first learning as a better-targeted objective. It did not establish that it will outperform balanced word-focused SFT.

### 6. Consider Q-SFT only after basic legality is learned

**Recommendation**

Q-SFT remains a possible way to teach delayed action value, but it should come after the model demonstrates basic feedback legality and constraint tracking. It is not the first response to the current failure mode.

**Implementation requirement**

- Gate Q-SFT on a parent checkpoint that first meets declared development thresholds for format, legality, and singleton behavior.
- Freeze Bellman/likelihood targets before optimization.
- Build every snapshot from training-only states and declared behavior support.
- Do not derive targets from hidden secrets, development/test answers, evaluator internals, or post-hoc outcomes unavailable to the policy.
- Preserve parent checkpoint identity, source-state IDs, behavior probabilities, target hashes, and objective accounting.
- Stop if the prerequisite legality gate is not met.

**Outcome boundary**

The recommendation did not claim that Q-SFT is currently ready to run or that its value targets will improve Wordle. Its priority is conditional on the earlier legality result.

### 7. Run a clean same-family Gemma 1B capacity condition

**Recommendation**

Test a larger Gemma checkpoint under a matched recipe so model capacity can be separated from the earlier cross-family Qwen signal.

**Implementation requirement**

- Use a Gemma instruction checkpoint near 1B parameters and pin its exact model revision before running.
- Match balanced-002, 8× action-token weighting, seed, optimizer schedule, effective batch, checkpoint doses, prompt, and evaluation protocol to the 270M condition.
- Make model scale the intended difference and record any hardware-forced deviation.
- Perform a license/authentication/local-snapshot preflight without recording credentials.
- Do not start training until the exact pinned snapshot is available.
- Keep the locked test closed and use the same multi-metric development gates.

**Outcome boundary**

The source conversation recommended a same-family capacity test. It did not establish access to the gated checkpoint, successful provisioning, or a capacity benefit.

## Explicitly deprioritized work

The source conversation explicitly placed the following directions behind the diagnostics and controlled comparisons above. This list is paraphrased rather than quoted:

- more verbose-reasoning/rationale experiments;
- larger random training corpora;
- additional decoder sweeps;
- more DAgger rounds; and
- GRPO at the current stage.

These were not declared permanently useless. They were deprioritized because they add data, decoding, or optimization complexity before the project has shown that the model can memorize tiny mappings, decode feedback reliably, obey legality constraints, and recover singleton actions.

## Shared interpretation and evidence boundary

Across the recommendations, the implementation should preserve `WORDLE-PROTOCOL-002`, training/development separation, natural generation, raw outputs, manifests, hashes, and the closed locked-test gate. No recommendation authorizes candidate injection, vocabulary masking, reranking, repeat suppression in the harness, malformed-output repair, or hidden-answer access.

This document records what the source conversation asked to test. Actual implementation status, run logs, blockers, and measured outcomes belong in separate experiment reports so a built test is never mistaken for a successful result.
