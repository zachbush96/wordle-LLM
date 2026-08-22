# Stable GRPO and SFT-to-GRPO hybrid track

## Scope and protocol boundary

This track implements a configuration and diagnostic layer for grouped policy optimization under frozen `WORDLE-PROTOCOL-002`. It does not modify the evaluator, parser, environment, oracle, word lists, or locked-test transition. It does not introduce a fixed opening word, candidate forcing, output repair, or synthetic game outcomes. No training result is claimed here; the implementation was exercised only with unit tests and compilation checks.

## Advantage Collapse Rate (ACR)

Following AVSPO (arXiv:2605.21125), for a rollout group \(g\) with scalar rewards \(r_{g,1}, \ldots, r_{g,n}\), define the group as collapsed when

\[
  \operatorname{std}_i(r_{g,i}) < \tau,
\]

Across \(G\) observed groups, the diagnostic is

\[
  \mathrm{ACR} = \frac{1}{G}\sum_{g=1}^{G} \mathbf{1}[g\text{ is collapsed}].
\]

with default \(\tau=10^{-6}\). The implementation records group size, mean, population standard deviation, range, and collapse flag. Empty and singleton groups are rejected because they do not define a grouped relative-advantage estimate. A study-level guard can stop after a declared minimum number of groups when ACR exceeds the configured ceiling. ACR is a training diagnostic, not an evaluation metric.

## AVSPO-style virtual advantage support

AVSPO triggers virtual support only when batch ACR is above its adaptive threshold and the current group is collapsed. Its count is \(K=\max(1,\min(G,\lceil G\,\mathrm{ACR}^{\alpha}\rceil))\), with \(\alpha=0.5\). For \(k=1,\ldots,K\), a positive observed maximum produces \(r_{obs}(1-k/(K+1))\). An all-zero group instead uses \(r_{anchor}(K-k+1)/K\), with anchor `0.1`. Each support record is explicitly labeled:

- `sample_type: synthetic_virtual_reward`
- `synthetic: true`
- `environment_outcome: false`
- `usage: advantage_estimation_only`

The support points participate only in the mean and standard deviation used to normalize the real rewards. The function returns trainable advantages only for real rollout rewards; it never returns a virtual trajectory or virtual policy sample. This can shift even equal real rewards relative to the augmented normalization distribution, so it introduces estimator bias; the AVSPO paper explicitly warns of bias and our paper must report this rather than describe the method as neutral numerical stabilization. Future trainer integration must preserve the separation and must not add virtual entries to episode, solve, win-rate, or reward-component logs.

The adaptive threshold begins at `0.5` and follows \(t \leftarrow t + \eta\,\operatorname{sign}(\Delta J)(\mathrm{ACR}-t)\), with \(\eta=0.01\). The source paper studies a binary-reward setting. Wordle uses multi-component shaped rewards, so transferring the rule is a hypothesis requiring an ablation, not evidence that its reported benefit transfers. In particular, the paper-provided schedules do not define an all-negative group. The implementation declines to create virtual support for that case instead of inventing an extension.

## Reward rubric

The stable configuration uses the existing `wordle-shaped-v1` component names and defaults: solve `5.0`, information gain `1.0`, oracle regret `-1.0`, repeat `-2.0`, and format `-3.0`. Validation requires the exact existing key set and finite numeric weights. This prevents misspelled or silently omitted components. It makes no changes to how real Wordle outcomes are scored.

## Staged hybrid design

The hybrid contract is exactly `SFT -> SFT dev evaluation -> GRPO`. SFT trains from the base model. GRPO must name `sft.final_checkpoint` as its parent, and orchestration passes that checkpoint explicitly. The promotion gate is dev-only and requires declared upper bounds for format failures, invalid guesses, repeats, and constraint violations. A missing checkpoint, non-dev evaluation, missing metric, or failed threshold stops the pipeline before GRPO. The spec requires `test_access: forbidden`; the orchestration API never offers a test split.

## Entropy-collapse guardrail

Completion entropy is monitored against the larger of an absolute floor and a fraction of the initial-window mean. Training is eligible to stop only after a minimum observation count and a configured number of consecutive below-floor observations. This avoids reacting to one noisy low-entropy batch. The combined stability decision reports both entropy and ACR evidence and does not itself mutate a trainer, allowing the training loop to persist the diagnostics before performing an orderly stop.

## Planned measurements for the paper

For each seed, log real reward components, real group reward vectors, ACR, completion entropy, real normalized advantages, virtual-support metadata, invalid/repeat/format/constraint rates, wins, singleton accuracy, optimizer tokens, wall time, and VRAM. Compare seed-matched SFT-only and SFT-to-GRPO runs on identical dev states. Report the exact stopping thresholds and all early stops. Locked-test access remains prohibited until the existing study-level promotion process selects a final method. The virtual-support ablation should compare disabled versus enabled support with all other settings fixed, and must explicitly state that virtual points are not simulated Wordle outcomes.

The shared notebook reports GRPO at 11.3% and SFT+GRPO at 15.6%. Those figures have not been reproduced in this repository and must appear, if cited at all, as notebook-reported background rather than our experimental result. Primary method reference: AVSPO, arXiv:2605.21125; formula locations should be replaced with precise section/equation citations during manuscript source verification.

## Trainer integration

`AVSPOGRPOTrainer` integrates the support calculation after TRL has gathered
real reward-function outputs and before its policy loss consumes advantages.
It replaces only the real-sample advantage tensor; no synthetic completion is
added. The integration targets the installed TRL 1.10 internal lifecycle, so a
TRL upgrade requires a smoke test that verifies reward capture, distributed
process slicing, and the returned `advantages` tensor before expensive runs.
