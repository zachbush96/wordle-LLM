# Q-SFT: Implementation Notes and Boundaries

## Scope and provenance

Q-SFT means **Q-learning via Supervised Fine-Tuning**. It comes from Hong,
Dragan, and Levine’s *Q-SFT: Q-Learning for Language Models via Supervised
Fine-Tuning* (ICLR 2025). The primary sources are
[arXiv:2411.05193](https://arxiv.org/abs/2411.05193) and
[OpenReview v4MTnPiYXY](https://openreview.net/forum?id=v4MTnPiYXY).

In plain terms, Q-SFT treats multi-turn Q-learning as weighted supervised fine-
tuning on a fixed offline dataset. Instead of adding and training a separate
value head, it expresses action values through model probabilities and trains
with weighted cross entropy. The shared NotebookLM description makes those
points, and Sections 4.1 and 4.3 of the paper support them.

## Objective implemented

For a transition `(s, a, r, s')`, Q-SFT defines the empirical Bellman-likelihood
target

`B_hat p_bar(a|s) = r + gamma max_a' [p_bar(a'|s') / pi_beta(a'|s')]`.

Equation 3 assigns that target probability to the observed action and distributes the remaining mass uniformly over all other actions:

`L = -[B_hat p_bar log p_theta(a|s) + sum_(a' != a) ((1 - B_hat p_bar)/(|A|-1)) log p_theta(a'|s)]`.

`q_sft_soft_cross_entropy` applies this formula across the tokenizer vocabulary
without building a dense soft-label tensor. `bellman_likelihood_target` computes
the behavior-corrected backup. Targets outside `[0, 1]` are rejected rather than
silently clipped: the paper’s Assumption 4.1 requires bounded discounted returns,
so a Wordle reward with negative step costs must be documented and normalized
first.

The training entry accepts frozen, offline Bellman-target snapshots. That keeps
training deterministic and prevents access to evaluation secrets or a live
environment. A target-refresh/data-generation pass may recompute snapshots from
a frozen target and behavior model, but it must stay train-only and record its
provenance. This repository is therefore a controlled approximation of Algorithm
1, which updates target parameters during optimization.

## Frozen-protocol adaptation and material limitation

The paper extracts a policy at inference time using
`pi_hat(a|s) proportional to pi_beta(a|s) exp(beta p_theta(a|s))`.
`WORDLE-PROTOCOL-002` forbids harness-side candidate injection, vocabulary
masking, guess reranking, and selection by a second policy. Canonical evaluation
therefore uses ordinary deterministic generation from the trained Q-SFT adapter.
That tests the learning objective, but **does not reproduce the paper’s policy-
extraction algorithm**. Label the result `Q-SFT-direct`, not a full reproduction.

The implementation rejects rows that contain answer, secret, candidate-list,
posterior-list, or locked-test fields. It neither reads nor alters the frozen
evaluator; the standard parser and six-valid-guess rules stay unchanged.

## NotebookLM-reported results (not reproduced)

The shared conversation reported **13.8% win rate**, **4.72 average guesses for
solved games**, **1.9% format failures**, **24.8% constraint violations**, and
**4.5 GPU-hours on 100 historical targets**. These are NotebookLM-reported
figures only. They have not been reproduced, and the model, split, reward
normalization, seeds, confidence intervals, and policy-extraction setting were
not supplied. Do not compare them directly with `WORDLE-PROTOCOL-002` until
those details are known.

The primary paper's Table 1 reports an average Wordle score of `-2.11` for Q-SFT across 100 evaluations on its separate 20K-trajectory LMRL dataset, using a task reward of `-1` per incorrect guess. That is a literature result on a different protocol and is not evidence for this adapter.

## Experiment record required for the paper

For every run, retain the source-data hash, behavior-policy checkpoint,
target-snapshot checkpoint, reward transformation, discount, target summary
statistics, base model, parent adapter, LoRA settings, seed, optimizer
steps/tokens, wall time, peak VRAM, and checkpoint hashes. Report wins alongside
format validity, invalid guesses, repeats, constraint violations, singleton
accuracy, and costs. Compare seed-matched SFT and Q-SFT-direct using the same
train-only dataset and frozen development evaluation. Keep the 1,000-answer test
closed until the existing promotion gates pass.

## Open questions

- Determine and preregister a monotone reward normalization satisfying bounded returns.
- Measure sensitivity to `gamma`, target refresh cadence, and target/behavior probability floors without clipping away data errors.
- Separate gains from the Q-SFT soft-label objective from gains due only to label smoothing.
- If a noncanonical policy-extraction study is ever run, label it exploratory and never mix it with frozen-protocol scores.
