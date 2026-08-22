# Q-SFT implementation notes

## Scope and provenance

Q-SFT means **Q-learning via Supervised Fine-Tuning**, from Hong, Dragan, and Levine, *Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning*, ICLR 2025. Primary sources: [arXiv:2411.05193](https://arxiv.org/abs/2411.05193) and [OpenReview v4MTnPiYXY](https://openreview.net/forum?id=v4MTnPiYXY).

The shared NotebookLM conversation described Q-SFT as multi-turn Q-learning cast as weighted SFT over a static offline dataset. It emphasized that action values are represented directly by model probabilities, using weighted cross entropy rather than Q-value regression, with no separately initialized value head. Those statements agree with Sections 4.1 and 4.3 of the primary paper.

## Objective implemented

For transition `(s, a, r, s')`, Q-SFT defines the empirical Bellman-likelihood target

`B_hat p_bar(a|s) = r + gamma max_a' [p_bar(a'|s') / pi_beta(a'|s')]`.

Equation 3 assigns that target probability to the observed action and distributes the remaining mass uniformly over all other actions:

`L = -[B_hat p_bar log p_theta(a|s) + sum_(a' != a) ((1 - B_hat p_bar)/(|A|-1)) log p_theta(a'|s)]`.

`q_sft_soft_cross_entropy` implements this formula over the tokenizer vocabulary without constructing a dense soft-label tensor. `bellman_likelihood_target` implements the behavior-corrected backup. Targets outside `[0, 1]` are rejected, not silently clipped, because the paper's Assumption 4.1 requires bounded discounted returns; any Wordle reward scheme with negative step costs must first be documented and normalized.

The training entry accepts frozen, offline Bellman-target snapshots. This makes the training stage deterministic and prevents access to evaluation secrets or a live environment. A target-refresh/data-generation pass may recompute snapshots from a frozen target model and behavior model, but it must remain train-only and record its provenance. This repository implementation is therefore a controlled approximation to Algorithm 1, whose target parameters are updated during optimization.

## Frozen-protocol adaptation and material limitation

The paper extracts a policy at inference time using `pi_hat(a|s) proportional to pi_beta(a|s) exp(beta p_theta(a|s))`. WORDLE-PROTOCOL-002 forbids harness-side candidate injection, vocabulary masking, guess reranking, or selection by a second policy. Consequently, canonical evaluation uses ordinary deterministic generation from the trained Q-SFT adapter. This tests the learning objective, but **does not reproduce the paper's policy-extraction algorithm** and must be identified as `Q-SFT-direct`, not claimed as a full reproduction.

The implementation also rejects rows containing answer, secret, candidate-list, posterior-list, or locked-test fields. It does not read or alter the frozen evaluator. The standard parser and six-valid-guess rules remain unchanged.

## NotebookLM-reported results (not reproduced)

The shared conversation reported the following Wordle figures: **13.8% win rate**, **4.72 average guesses among solved games**, **1.9% format failures**, **24.8% constraint violations**, and **4.5 GPU-hours on 100 historical targets**. These are NotebookLM-reported numbers only. They have not been reproduced, and their exact model, split, reward normalization, seeds, confidence intervals, and policy-extraction setting were not supplied. They must not be compared directly with WORDLE-PROTOCOL-002 results until those details are resolved.

The primary paper's Table 1 reports an average Wordle score of `-2.11` for Q-SFT across 100 evaluations on its separate 20K-trajectory LMRL dataset, using a task reward of `-1` per incorrect guess. That is a literature result on a different protocol and is not evidence for this adapter.

## Experiment record required for the paper

For every run, retain the source data hash, behavior-policy checkpoint, target-snapshot checkpoint, reward transformation, discount, target summary statistics, base model, parent adapter, LoRA settings, seed, optimizer steps/tokens, wall time, peak VRAM, and checkpoint hashes. Report wins together with format validity, invalid guesses, repeats, constraint violations, singleton accuracy, and costs. Compare seed-matched SFT and Q-SFT-direct under the same train-only dataset and frozen dev evaluation. Keep the locked 1,000-answer test closed until the existing promotion gates pass.

## Open questions

- Determine and preregister a monotone reward normalization satisfying bounded returns.
- Measure sensitivity to `gamma`, target refresh cadence, and target/behavior probability floors without clipping away data errors.
- Separate gains from the Q-SFT soft-label objective from gains due only to label smoothing.
- If a noncanonical policy-extraction study is ever run, label it exploratory and never mix it with frozen-protocol scores.
