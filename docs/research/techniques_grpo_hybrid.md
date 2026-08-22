# Stable GRPO and SFT-to-GRPO Hybrid Track

## Scope and protocol boundary

This track adds configuration and diagnostics for grouped policy optimization
under frozen `WORDLE-PROTOCOL-002`. It does not change the evaluator, parser,
environment, oracle, word lists, or locked-test transition. It also avoids fixed
openers, candidate forcing, output repair, and synthetic game outcomes. There is
no training result to report here: this implementation has only been exercised
with unit tests and compilation checks.

## Advantage Collapse Rate (ACR)

Following AVSPO ([arXiv:2605.21125](https://arxiv.org/abs/2605.21125)), a rollout
group \(g\) with scalar rewards \(r_{g,1}, \ldots, r_{g,n}\) is considered
collapsed when

\[
  \operatorname{std}_i(r_{g,i}) < \tau,
\]

Across \(G\) observed groups, the diagnostic is

\[
  \mathrm{ACR} = \frac{1}{G}\sum_{g=1}^{G} \mathbf{1}[g\text{ is collapsed}].
\]

where the default is \(\tau=10^{-6}\). The implementation records group size,
mean, population standard deviation, range, and the collapse flag. Empty and
singleton groups are rejected because they cannot define a grouped relative-
advantage estimate. A study-level guard may stop after a declared minimum number
of groups when ACR exceeds its configured ceiling. ACR is a training diagnostic,
not an evaluation metric.

## AVSPO-style virtual advantage support

AVSPO adds virtual support only when batch ACR is above its adaptive threshold
and the current group has collapsed. Its count is
\(K=\max(1,\min(G,\lceil G\,\mathrm{ACR}^{\alpha}\rceil))\), with
\(\alpha=0.5\). For \(k=1,\ldots,K\), a positive observed maximum produces
\(r_{obs}(1-k/(K+1))\). An all-zero group instead uses
\(r_{anchor}(K-k+1)/K\), with anchor `0.1`. Each support record is explicitly
labeled:

- `sample_type: synthetic_virtual_reward`
- `synthetic: true`
- `environment_outcome: false`
- `usage: advantage_estimation_only`

Support points affect only the mean and standard deviation used to normalize real
rewards. The function returns trainable advantages for real rollouts only—never a
virtual trajectory or policy sample. Because adding support can shift even equal
real rewards under the expanded normalization distribution, it introduces
estimator bias. The AVSPO paper warns about that bias, and any paper using this
method must report it instead of calling the procedure neutral numerical
stabilization. Future trainer integration must keep this separation and never add
virtual entries to episode, solve, win-rate, or reward-component logs.

The adaptive threshold begins at `0.5` and follows
\(t \leftarrow t + \eta\,\operatorname{sign}(\Delta J)(\mathrm{ACR}-t)\),
with \(\eta=0.01\). The source paper studies binary rewards. Wordle uses shaped,
multi-component rewards, so carrying over this rule is a testable hypothesis—not
evidence that the paper’s benefit transfers. The supplied schedules also do not
cover all-negative groups. Rather than invent an extension, this implementation
does not create virtual support for that case.

## Reward rubric

The stable configuration keeps the existing `wordle-shaped-v1` components and
defaults: solve `5.0`, information gain `1.0`, oracle regret `-1.0`, repeat
`-2.0`, and format `-3.0`. Validation requires that exact key set and finite
numeric weights, preventing misspelled or silently omitted components. It does
not change how real Wordle outcomes are scored.

## Staged hybrid design

The hybrid path is deliberately simple: `SFT -> SFT dev evaluation -> GRPO`. SFT
starts from the base model. GRPO must name `sft.final_checkpoint` as its parent,
and orchestration passes that checkpoint explicitly. The promotion gate uses only
development data and requires declared limits for format failures, invalid
guesses, repeats, and constraint violations. A missing checkpoint, non-dev
evaluation, missing metric, or failed threshold stops the pipeline before GRPO.
The spec requires `test_access: forbidden`, and the orchestration API never
offers a test split.

## Entropy-collapse guardrail

Completion entropy is checked against whichever is higher: an absolute floor or a
fraction of the initial-window mean. Training can stop only after a minimum
number of observations and a configured run of below-floor values, avoiding a
reaction to one noisy batch. The combined stability decision reports both entropy
and ACR evidence. It does not mutate a trainer itself, so the training loop can
save diagnostics before stopping cleanly.

## Planned measurements for the paper

For each seed, log real reward components, real group reward vectors, ACR, completion entropy, real normalized advantages, virtual-support metadata, invalid/repeat/format/constraint rates, wins, singleton accuracy, optimizer tokens, wall time, and VRAM. Compare seed-matched SFT-only and SFT-to-GRPO runs on identical dev states. Report the exact stopping thresholds and all early stops. Locked-test access remains prohibited until the existing study-level promotion process selects a final method. The virtual-support ablation should compare disabled versus enabled support with all other settings fixed, and must explicitly state that virtual points are not simulated Wordle outcomes.

The shared notebook reports GRPO at 11.3% and SFT+GRPO at 15.6%. This repository
has not reproduced those figures. If cited, they belong in notebook-reported
background—not in our experimental results. The primary method reference is
AVSPO, arXiv:2605.21125; replace formula placeholders with precise section and
equation citations during manuscript source verification.

## Trainer integration

`AVSPOGRPOTrainer` calculates support after TRL gathers real reward-function
outputs and before the policy loss uses advantages. It replaces only the
real-sample advantage tensor; it adds no synthetic completion. The integration
targets the installed TRL 1.10 internal lifecycle, so any TRL upgrade needs a
smoke test for reward capture, distributed-process slicing, and the returned
`advantages` tensor before expensive runs.
