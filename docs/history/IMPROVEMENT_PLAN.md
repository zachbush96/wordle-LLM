# Gemma Wordle Improvement Plan

## Purpose and scope

This historical plan lays out the next development-only experiments for
`google/gemma-3-270m-it` on Wordle. At the time, the evidence suggested the
model had learned the terminal-output envelope and a fixed opening word, but
not how to choose a feedback-dependent, posterior-consistent next guess.

This plan applies only to the `COMMON-WORD-CURRICULUM-001` u128 pilot. It does
not change `WORDLE-PROTOCOL-002`, its parser, its frozen evaluator, or its
locked 1,000-answer test gate. Any prompt- or tool-assisted evaluation needs a
separate label and cannot be compared as an unassisted policy result.

## Evidence motivating the work

The key comparison is the u128 SFT parent against its selected ORPO
continuation, both evaluated with `greedy_rep105` on 25 held-out secrets.

| Metric | SFT step 600 | ORPO step 50 |
| --- | ---: | ---: |
| Wins | 1/25 | 1/25 |
| Strict terminal compliance | 100.0% | 100.0% |
| Repeat rate | 18.5% | 10.3% |
| Posterior-constraint violation rate | 82.2% | 82.2% |
| New, posterior-consistent actions | 17.8% | 17.8% |
| Singleton-posterior states solved | 0/89 | 0/96 |

The evidence pointed to a few likely causes:

- `SHARE` is emitted for all 25 first moves. Formatting and the opener are not
  the immediate bottleneck.
- At turn 2, 18/25 evaluated histories are exact canonical training states,
  but the SFT parent matches the teacher action in only 1/18 and violates the
  posterior constraints in 24/25 cases.
- Once the model makes that first bad action, none of its later runtime states
  appear in the static canonical training corpus. This is an off-policy
  recovery problem.
- The u128 rendered set has 1,604 rows, including 738 repeat-correction rows.
  One root state is copied 128 times and two turn-5 states account for 256
  rows. The five most frequent target words make up 29.1% of labels. The
  exposure floor is therefore distorting the action-label distribution.
- Repeat-only ORPO improved only the behavior represented by its 511/512
  repeat negatives. It did not improve constraint use, and longer ORPO doses
  degraded formatting (6.5% failure at step 75, 20.6% at step 100).

## Implementation order

### 1. Start with focused diagnostics

Create a development-only diagnostic suite that runs outside the gameplay loop.
It should evaluate held-out common-word states for:

1. proposed-word legality against the complete feedback history;
2. ability to emit the sole candidate when the posterior has size one;
3. exact oracle-action match for states already seen in the training set;
4. posterior-consistency, repeat, format, and action-target accuracy by turn;
5. train-state coverage for actual model rollout states.

Persist per-item JSONL and an aggregate JSON summary next to the run artifact.
The metrics must distinguish ordinary valid English words from actions that
conflict with the project’s posterior-consistency policy.

Suggested acceptance gates for promoting a recipe beyond one-seed diagnostics:

- terminal compliance at least 99%;
- turn-2 posterior-constraint violation below 30%;
- singleton-answer accuracy materially above zero (target: at least 80%);
- improvement reproduced across three seeds before any method winner claim.

Evaluate all 32 u128 held-out secrets rather than the current alphabetical
prefix of 25. Do not use this selection process to access the locked test.

### 2. Build a balanced, word-focused SFT dataset

Add a versioned u128 curriculum builder. Keep the public 128-word action
vocabulary and the existing train/development secret separation, but replace
the duplicate-heavy exposure floor with more distinct training states.

Requirements:

- cap copies of any canonical state and any exact target word at a small,
  declared value (for example, four);
- use a declared mixture that emphasizes the post-opener decision and recovery
  states, for example: 10% root, 40% turn 2, 30% later on-policy states, and
  20% recovery/singleton states;
- generate the additional states from training secrets only, including varied
  legal and model-like prior guesses; never generate labelled trajectories
  from held-out dev secrets;
- store the source state, state type, turn, posterior size, target word, and
  target frequency in the dataset manifest;
- keep the inference prompt shape aligned with the training prompt. Do not
  use a synthetic `Rejected:` conversation as the main recovery representation.

Add a loss option that either masks the invariant `Final answer:` prefix after
a format warmup or weights the target-word tokens 5–10x more heavily. The goal
is to stop the constant terminal envelope from dominating the completion loss.

Run this fixed-budget 2x2 ablation before increasing model size or vocabulary:

| Dataset | Loss |
| --- | --- |
| Current u128 curriculum | Current completion loss |
| Balanced u128 curriculum | Current completion loss |
| Current u128 curriculum | Word-focused loss |
| Balanced u128 curriculum | Word-focused loss |

Use the same optimizer-step and token budgets as the current 600-step u128
pilot for the first screen. Report data composition, target-frequency
distribution, optimizer tokens, and all diagnostics above.

### 3. Add on-policy recovery training (DAgger-style)

Starting with the best balanced SFT parent, roll the model out on training
secrets only. Record real inference contexts where it produces a constraint
violation, repeat, malformed output, or singleton miss. For each context,
create a supervised target from a posterior-consistent oracle action.

Implementation requirements:

- use exactly the same prompt renderer as evaluation;
- retain model-produced histories verbatim, including earlier bad but valid
  guesses and their real feedback;
- mix approximately 50% static oracle states and 50% on-policy recovery
  states to preserve the learned opener and avoid catastrophic drift;
- run two or three data-collection/training rounds, logging the source parent,
  rollout seed, error type, and state coverage gained per round;
- prioritize singleton posterior states because the current model solves none
  of them after it has drifted off trajectory.

The first comparison should be static balanced SFT against balanced SFT plus
DAgger with matched tokens and decoding. The main measure of success is fewer
late-turn out-of-distribution states and constraint violations—not merely a
noisy game win count.

### 4. Replace repeat-only preference pairs

Begin this only after a stable SFT/DAgger parent passes the diagnostic gates.
Build preference pairs from the same inference-shaped contexts with a declared
negative mix, initially:

- 50% model-generated posterior-constraint violations;
- 25% prior repeats;
- 25% posterior-consistent but strategically inferior actions.

Keep chosen and rejected completions identical apart from the decision when
possible. In particular, use the model’s observed bad second-turn actions as
hard negatives. Preserve the exact `Final answer:` envelope on both sides so
the objective targets action quality rather than formatting.

Compare continued SFT, ORPO, and optionally DPO only with seed-matched parents
and equalized preference-pair budgets. Save and evaluate checkpoints at 25,
50, 75, and 100% of the continuation budget. The current data suggests that
25–50 steps are the likely useful ORPO range; checkpoint selection must use
development diagnostics and formatting compliance.

### 5. Scale cautiously after the u128 policy works

Do not treat the failed u512 run as a clean capacity result: it also changed
vocabulary difficulty and destabilized output formatting. First build a nested
u128-to-u256 curriculum with a fixed global holdout that never becomes training
data. Replay u128 states during u256 continuation and use a lower continuation
learning rate.

Once the corrected u128 recipe has positive, replicated strategic metrics,
run a capacity ablation with the same curriculum, token budget, and decoder:

- Gemma 270M at the current LoRA capacity;
- 270M with a modest LoRA-rank/capacity variant;
- a roughly 1B Gemma condition.

Report quality versus wall time, VRAM, and optimizer/generated tokens. Do not
attribute improvement to capacity if the data recipe differs.

### 6. Defer GRPO and label augmented agents separately

Do not make GRPO the next main experiment. Its current shaped reward lacks an
explicit constraint-violation component and mixes realized information gain with
oracle-regret terms. Before using it, add and unit-test:

- an explicit posterior-consistency penalty;
- normalized expected, rather than only realized, information-gain terms;
- a multi-turn rollout reward in addition to static-state rewards;
- logs for every reward component and its scale.

Candidate-list prompting, constrained decoding, trie masking, and exhaustive
candidate reranking may be useful for tool-augmented Wordle agents. They need a
separate benchmark label and can never stand in for the unassisted Gemma result.

## Recommended implementation touchpoints

- `wordle_lab/experiments/common_curriculum.py`: versioned balanced dataset
  builder, composition manifest, and nested curriculum support.
- `wordle_lab/methods/sft.py`: word-focused label weighting/masking and
  accounting for weighted tokens.
- `wordle_lab/experiments/common_preference.py`: constraint-negative and
  model-generated preference pairs.
- New experiment module, for example
  `wordle_lab/experiments/on_policy_recovery.py`: rollout collection and
  DAgger-style dataset construction.
- New diagnostics module, for example
  `wordle_lab/analysis/state_diagnostics.py`: state-level probe generation,
  evaluation, JSONL output, and aggregate metrics.
- `wordle_lab/methods/rewards.py` and `wordle_lab/methods/grpo.py`: only after
  the SFT/DAgger gate has been met.

## Verification and reporting requirements

- Add deterministic unit tests for every new dataset builder, loss mask, and
  preference-negative type.
- Assert that no dev or locked-test secret is used to generate labelled
  training episodes.
- Verify prompt-renderer identity between each training representation and its
  evaluation condition.
- Make every artifact content-addressed and include source parent/checkpoint,
  dataset hashes, error-type counts, and data-composition statistics.
- Save per-turn and per-posterior-size metrics; a single win-rate scalar is
  insufficient for selecting the next method.
- Treat one-seed, 32-secret screens as diagnostics. Use three seed-matched runs
  for any claim of improvement, and keep the locked test closed until the
  existing study-level selection rule is satisfied.
