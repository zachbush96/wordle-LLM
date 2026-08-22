# Wordle LLM Post-Training Experiment Design

Status: a historical preregistration and design reference. The framework is
implemented. One Unsloth direct-SFT cell has run and was rejected; the full
matched technique comparison has not yet run.
For completed development experiments, see
[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md); the root [README](../README.md)
is the best starting point for the project’s current status.

## 1. Research questions

This study is meant to answer three distinct questions:

1. Which post-training objective most improves Wordle play?
2. Which supervision representation transfers best to the unchanged game interface?
3. What behavior changed, and what data/compute dose produced that change?

Do not collapse those questions into one big leaderboard. Method, data
representation, training dose, and random seed are separate experimental factors;
keeping them separate is what makes a result interpretable.

## 2. Frozen contract

Before the study begins, freeze a new protocol version (here,
`WORDLE-PROTOCOL-002`) and record a new full baseline. The earlier exact-word
protocol is historical context, not a comparable result; do not mix it with the
reasoning-capable protocol.

The following pieces are shared evaluation infrastructure, not experimental
treatments:

- base model and revision: `google/gemma-3-270m-it` at the pinned local revision;
- train/dev/test answer splits and word list;
- `WordleEnv`, scoring rules, reasoning-capable prompt wording, terminal-answer
  parser, and six-turn limit;
- deterministic greedy generation and the new shared reasoning token budget;
- the same ordered answer set for every paired comparison;
- the retention probe.

The evaluator must never mask logits to the vocabulary, choose a guess for the
model, ban repeated words, inject candidate words, repair malformed output, or
truncate a response into a valid guess. New instrumentation is fine only when it
observes behavior without changing it.

### Reasoning-capable output protocol

Every condition uses the same ending:

```text
<zero or more tokens of visible analysis>
Final answer: CRANE
```

The final non-empty line must match, case-insensitively,
`Final answer: <five ASCII letters>`, with optional surrounding whitespace and no
following prose. The captured word is then checked against the frozen allowed-word
list. The parser must not search the reasoning body for a convenient five-letter word
and must not choose the last dictionary word it happens to see.

Direct-output datasets contain only `Final answer: CRANE`. Reasoning datasets begin
the assistant completion with the trace and finish with that same terminal line. In
other words, the comparison changes learned reasoning behavior—not the parser or the
answer syntax.

Use the same generation ceiling for every condition (initial proposal: 128 new
tokens), preserve EOS/end-of-turn stopping, and do not programmatically stop at the
first terminal marker. A direct model may finish early; a reasoning model may spend
more of the common budget. A missing terminal line, prose after that line, or an
unfinished answer at the limit is a format failure. Record generated-token count so
any improvement can be weighed against inference cost.

The existing 25-game smoke runs have exposed 25 test answers during development.
Report the full 1,000-game result, but also publish a sensitivity analysis on the 975
answers not used by those smoke runs. Do not use any full-test result to select a
dataset, hyperparameter, checkpoint, or method.

## 3. Four post-training techniques

All techniques use the same LoRA target modules and rank unless a separately labelled
capacity ablation says otherwise.

| ID | Technique | Training signal | Role in the comparison |
|---|---|---|---|
| `sft` | Supervised fine-tuning | Oracle demonstration | Essential imitation baseline |
| `dpo` | Direct Preference Optimization | Chosen/rejected action pairs | Offline preference learning with a reference policy |
| `orpo` | Odds Ratio Preference Optimization | Chosen/rejected pairs plus chosen-response likelihood | Reference-free, monolithic preference objective |
| `grpo` | Group Relative Policy Optimization | Groups of sampled actions scored by a deterministic Wordle reward | Online exploration and reward optimization |

There are two useful—and clearly different—method comparisons:

- **Objective comparison:** start DPO, ORPO, GRPO, and a continued-SFT control from
  the same seed-matched SFT warm-start checkpoint. This estimates the marginal effect
  of the continuation objective.
- **Recipe comparison:** compare the final deployable paths `base -> SFT`,
  `base -> SFT -> DPO`, `base -> SFT -> ORPO`, and `base -> SFT -> GRPO`. This answers
  which practical recipe produces the best model, while disclosing that the latter
  three received an additional stage.

ORPO may also be run directly from the base model as a labelled exploratory ablation,
but it must not replace the matched warm-start comparison.

## 4. Four data representations

Build every representation from one canonical, versioned collection of Wordle
episodes and states. The answer is metadata; it is never shown in the model-visible
prompt.

| ID | Representation | Model-visible supervision | What it tests |
|---|---|---|---|
| `state_direct` | Single-state/direct | Current flattened game history -> terminal answer line | Minimal task-specific imitation |
| `episode_multiturn` | Multi-turn trajectory | Full sequence of user feedback and assistant guesses -> next terminal answer line | Whether temporal conversation structure improves state tracking |
| `state_rationale` | Process annotated | State -> deterministic constraint/information-gain trace -> terminal answer line | Whether explicit intermediate reasoning teaches a better policy |
| `mixed_curriculum` | Difficulty-balanced mixture | Direct states, later-turn states, failures/corrections, and a small rationale component | Whether diverse, curriculum-ordered coverage beats a homogeneous corpus |

Gemma 3 270M IT has no private reasoning channel, so `state_rationale` trains visible
reasoning as ordinary assistant tokens. Completion-only masking must train both the
trace and terminal answer, while still masking all prompt tokens. Each deterministic
teacher trace should appear in this order:

1. summarize the constraints/posterior implied by prior feedback;
2. assess one or more candidate actions using information gain and solve probability;
3. state the choice rationale;
4. emit `Final answer: WORD` as the last line.

The trace must not contain the secret answer unless the oracle-selected next guess is
the answer; the generator should only use information available in the visible game
state. Store structured trace facts beside the rendered natural-language trace so
they can be verified and regenerated. Avoid decorative preambles so the first
completion tokens carry task reasoning.

Visible-rationale training might improve decisions while increasing latency or format
failures. Report all three outcomes; the tradeoff is part of the result. A
`rationale_then_direct` curriculum may be used as a diagnostic, but not as a fifth
primary representation.

Preference views are derived from the same canonical records:

- chosen response: the oracle action;
- hard rejected response: a valid, posterior-consistent but worse information-gain
  action whenever available;
- behavioral rejected response: repeat, posterior violation, or malformed output;
- default mix: 80% hard strategic negatives, 10% behavioral negatives, 10% malformed
  negatives, recorded exactly in the manifest.

For `state_rationale` preference pairs, both chosen and rejected completions must put
visible analysis first and the terminal answer last. Prefer a shared, factually correct
constraint analysis followed by different action assessments and terminal choices;
this keeps the preference contrast focused on the decision. Any model-sampled
rejected trace must be retained verbatim and annotated for false trace facts rather
than silently repaired.

GRPO prompts use the same state distribution. Its reward is versioned and decomposed
into logged components. A recommended initial state-level reward is:

`solve bonus + information gain - oracle regret - repeat penalty - format penalty`

The primary GRPO ablation should compare that shaped reward with a sparse
environment-only reward. Reward shaping is training signal; it must never alter the
evaluation harness.

## 5. A staged experiment, not a brute-force grid

Use development answers for every selection decision. Open the locked test only after
those rules have been applied.

### Stage 0: freeze and validate

- Implement and hash the shared reasoning-capable prompt, 128-token generation
  configuration, terminal-answer parser, and rendered training envelope.
- Freeze a new baseline under this protocol on all 1,000 answers once. Keep
  `BASELINE-003` only as a historical exact-word baseline.
- Add dataset leakage checks, deterministic replay tests, parser tests, and a
  hash-based protocol lock.
- Establish a random-policy and oracle-policy ceiling in the same environment. These
  are reference lines, not model competitors.

### Stage 1: data screen under one technique

Run SFT on the four primary representations with three seeds. Keep the number of
unique canonical states fixed. Record examples, prompt tokens, completion tokens,
optimizer steps, and GPU time rather than pretending that “two epochs” is equal
exposure across representations.

Select the top two representations by the preregistered development score:

`dev win rate - 0.25 * format failure rate - 0.10 * retention loss`

Publish all component metrics so the scalar selection rule cannot conceal a tradeoff.

### Stage 2: method screen on fixed data

Using the best representation compatible with direct output and seed-matched SFT
warm starts, run continued SFT, DPO, ORPO, and GRPO with three seeds. Use a small,
predeclared development-only hyperparameter grid. One technique should not receive a
large search while the others are left at defaults.

### Stage 3: interaction check

Run the top two methods with the top two data representations. This 2x2 check tells us
whether the apparent best data format is generally useful or only works with one
objective. Reuse Stage 1/2 cells where the specifications are identical.

### Stage 4: dose-response and locked confirmation

For the two finalist recipes, save/evaluate checkpoints at 25%, 50%, 75%, and 100% of
the planned optimizer-update budget. Select one checkpoint per recipe on development
data, then evaluate those frozen checkpoints once on the 1,000-answer test.

Three seeds is the minimum primary design. If seed variance is comparable to the
observed method difference, expand the finalists to five seeds before making a winner
claim.

## 6. Fairness and accounting

No single equality constraint makes SFT, preference learning, and online RL equally
expensive. Rather than hiding that mismatch, report two views:

1. **Fixed canonical-state budget:** same number and sampling distribution of unique
   training states.
2. **Cost-effectiveness frontier:** achieved score versus optimizer tokens, generated
   tokens, wall time, peak VRAM, and GPU-hours.

Every run manifest must include:

- method, data representation, parent checkpoint, seed, git commit, package versions;
- hashes for splits, canonical records, rendered dataset, reward function, prompt,
  parser, and generation configuration;
- LoRA configuration and exact trainable parameter count;
- examples, unique states, prompt/completion/generated tokens, epochs, optimizer
  steps, effective batch size, learning rate, and checkpoint selected;
- wall-clock training/evaluation time, peak allocated VRAM, and failures/retries;
- DPO/ORPO pair counts and negative-type mixture;
- GRPO group size, rollouts, total sampled tokens, reward weights, and each reward
  component's distribution.
- reasoning tokens, total generated tokens, terminal-marker compliance, and inference
  latency per call.

## 7. Metrics and explanations

### Primary outcome

- game win rate within six valid Wordle guesses;
- paired absolute win-rate delta versus the frozen base and versus the matched SFT
  parent, with a paired bootstrap 95% confidence interval.

### Secondary outcomes

- wins by guess number and mean guesses conditional on winning;
- terminal-answer format failure, invalid extracted guess, repeated guess, and
  posterior-constraint violation rates;
- reasoning-presence rate, reasoning-token count, total generated-token count,
  terminal-marker compliance, and answer-at-end compliance;
- unique first guesses and total unique guesses;
- retention score and category-level retention delta.

### Strategy diagnostics: the “why”

For every valid guess, compute the following without affecting generation:

- posterior size before and after the guess;
- realized information gain;
- expected remaining candidates;
- oracle regret: model expected-remaining score minus the oracle's score;
- action quality percentile among legal posterior candidates;
- error transition counts, such as malformed -> valid, repeat -> novel, or valid but
  strategically weak -> high-information.

These diagnostics help separate three possible sources of improvement: better
formatting, better rule/state tracking, and better strategic word choice.

Do not claim that a fluent trace is faithful merely because the final guess is good.
Where a trace states machine-checkable facts (fixed positions, excluded letters,
posterior count, expected remaining candidates), compare them with the environment and
report trace-fact accuracy separately from game performance.

Use paired bootstrap intervals over answers, McNemar tests for paired win/loss changes,
and Holm correction for planned pairwise comparisons. Treat seeds as independent
training replicates and show every seed point; do not compute uncertainty from 1,000
games while ignoring training-seed variance.

## 8. Required charts

The final report should generate the following from tidy result tables—not
hand-entered values:

1. method leaderboard with seed points and 95% intervals;
2. data-representation leaderboard under SFT;
3. method x data interaction heatmap;
4. dose-response learning curves at 25/50/75/100%;
5. performance-versus-GPU-hours Pareto chart;
6. paired per-answer win/loss transition chart versus baseline;
7. wins-by-turn distribution;
8. behavioral error decomposition;
9. oracle-regret and information-gain distributions by turn;
10. reasoning length versus win rate, format compliance, and latency;
11. trace-fact accuracy by fact type and turn;
12. Wordle improvement versus retention-delta tradeoff;
13. training curves and GRPO reward-component curves;
14. data composition and token-budget stacked bars.

Each chart must link back to the run IDs and include the number of seeds, games, and
the uncertainty definition.

## 9. Proposed code architecture

Keep the current script as a reproducibility reference while moving new work into a
package:

```text
wordle_lab/
  protocol/
    env.py                 # frozen Wordle rules and scoring
    prompting.py           # frozen inference prompt renderer
    parsing.py             # frozen terminal-answer parser
    generation.py          # frozen evaluation decoding
    evaluator.py           # gameplay plus observation-only diagnostics
    protocol_lock.json     # hashes of all frozen components and output grammar
  data/
    canonical.py           # episode/state schema and deterministic generator
    builders.py            # four representation builders
    preferences.py         # chosen/rejected construction
    manifests.py           # hashes, leakage checks, token counts
  methods/
    base.py                # TrainerBackend interface
    sft.py
    dpo.py
    orpo.py
    grpo.py
    rewards.py             # versioned, unit-tested GRPO components
  experiments/
    schema.py              # validated ExperimentSpec
    registry.py            # YAML spec loading and run-ID generation
    runner.py              # train/eval/resume state machine
    accounting.py          # time, tokens, parameters, VRAM
  analysis/
    collect.py             # JSONL -> Parquet tidy tables
    statistics.py          # paired intervals/tests and seed aggregation
    plots.py               # the fourteen standard charts
    report.py              # Markdown/HTML experiment card
  cli.py
configs/
  protocol.yaml
  datasets/*.yaml
  methods/*.yaml
  studies/*.yaml
artifacts/
  runs/<run_id>/
    spec.json
    status.json
    dataset_manifest.json
    train_metrics.jsonl
    games.jsonl
    summary.json
    diagnostics.parquet
    accounting.json
    checkpoints/
  studies/<study_id>/
    run_index.parquet
    aggregate_metrics.parquet
    comparisons.parquet
    plots/
    report.md
```

`TrainerBackend` exposes `prepare(spec)`, `train()`, `save()`, and `load_for_eval()`.
The evaluator only accepts a model/tokenizer handle and a locked protocol object; it
does not know which trainer produced the model.

The runner is a resumable state machine:

`PLANNED -> DATA_READY -> TRAINING -> TRAINED -> DEV_EVALUATED -> SELECTED -> TEST_EVALUATED -> REPORTED`

Only a study-level selection command may transition a run to `SELECTED`. Test
evaluation refuses all other runs. Artifacts are append-only and run IDs are content
derived from the normalized specification plus a readable prefix.

## 10. Initial run inventory

Minimum primary runs before checkpoint/dose evaluations:

- Stage 1: 4 representations x 3 seeds = 12 SFT runs;
- Stage 2: 4 objectives x 3 seeds = 12 runs, with identical Stage 1 cells reused;
- Stage 3: at most 2 x 2 x 3 = 12 runs, again reusing identical cells;
- Stage 4: checkpoints from 2 finalist recipes x 3 seeds, with no retraining if the
  runner saved the scheduled dose checkpoints.

That is at most 36 primary training runs before reuse, rather than a blind
4-method x 4-data x 4-dose x 3-seed grid of 192 runs.

## 11. Decision rules

Call a technique better only if:

- its locked-test mean win-rate improvement is positive;
- the paired confidence interval and seed-to-seed results support the direction;
- the gain is not solely parser compliance unless that is explicitly the claim;
- its retention regression is within the preregistered tolerance;
- its additional data and compute are disclosed;
- strategy diagnostics provide a plausible behavioral explanation.

If the apparent best method changes with seed, compute budget, or data
representation, report a conditional conclusion rather than naming a universal
winner.
