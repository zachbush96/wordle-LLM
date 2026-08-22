# Wordle LLM Research Lab

## Executive overview

This project asks a deliberately difficult question: **can a small language model play Wordle by generating its own text, without a hidden solver choosing or constraining its guesses?**

The short answer is: not reliably yet. The strongest historical Gemma 3 270M run solved **8 of 32 development games (25%)**. A matched Qwen2.5 1.5B run reached **14 of 32 (43.75%)** and made much stronger feedback-dependent choices, but it still missed the reliability gates. Because it also changes model family, it is a useful capacity signal—not a clean Gemma result. Some Qwen2.5 3B checkpoints won 15–16 games, but too often failed to return a usable answer. That is why this project treats a win count as one important metric, not the verdict.

The active work is a stricter, Gemma-only comparison. It includes an audited 4,096-state training bundle; launchers for LoRA, rsLoRA, and DoRA; implementations of SFT, DPO, ORPO, GRPO, and Q-SFT; a gated SFT-to-GRPO path; and **76 passing tests**. The first current-study cell has now run through Unsloth: it trained efficiently and learned perfect output formatting, but scored **0/32** held-out wins and did not learn feedback-conditioned play. See the [full Unsloth experiment report](docs/UNSLOTH_GEMMA_WORDLE_EXPERIMENT.md).

One safeguard matters above all: no trained checkpoint has seen the frozen 1,000-answer test. That door stays closed until a candidate reaches at least 99% terminal compliance, fewer than 30% turn-2 posterior violations, clearly positive singleton accuracy, and the same result across three matched seeds.

## Contents

- [Where the project stands](#where-the-project-stands)
- [What we are testing](#what-we-are-testing)
- [How the project evolved](#how-the-project-evolved)
- [The Wordle engine](#the-wordle-engine)
- [The LLM harness](#the-llm-harness)
- [Experiments and results](#experiments-and-results)
- [Repository guide](#repository-guide)
- [Setup and common commands](#setup-and-common-commands)
- [What we plan to try next](#what-we-plan-to-try-next)
- [Research boundaries](#research-boundaries)

## Where the project stands

There are two different statuses worth keeping separate:

1. **Best historical development model:** Qwen2.5 1.5B with the balanced curriculum. It is the strongest multi-metric artifact produced so far, but it is not accepted as a general Wordle result.
2. **Current canonical study:** Gemma 3 270M only, pinned to one exact model revision. The data and training machinery are ready; the matched learning curves and technique comparison are still pending.

| Model / condition | Development wins | Terminal compliance | Repeat rate | Turn-2 violations | Singleton accuracy | Action-target accuracy | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Gemma 3 270M, balanced-002 + word-focused SFT | 8/32 | 89.2% | 27.4% | 79.3% | 4/74 | 14.1% | Best historical Gemma strategy result; not promotable |
| Gemma 3 270M, Unsloth direct SFT, step 300 | 0/32 | 100% | 54.2% | 96.7% | 0/326 | 0.2% | Rejected: learned format, not strategy |
| Same Gemma adapter, repetition penalty 1.05 | 8/32 | 89.2% | 1.9% | — | — | — | Useful decoder diagnostic; did not improve wins |
| Qwen2.5 1.5B, matched balanced-002 recipe | 14/32 | 94.1% | 2.0% | 35.1% | 7/74 | 41.3% | Best overall development artifact; cross-family and not promotable |
| Qwen2.5 3B, final checkpoint | 15/32 | 71.2% | — | 29.8% | — | — | Rejected: 37.4% invalid guesses |

These are results on a small, held-out **32-word common-word development split**, not a representative public Wordle win rate. The only full frozen-test result is the unconditional base-model baseline: untrained Gemma won **0/1,000** and had 0% terminal-format compliance. No trained model has seen that test.

The detailed result ledger is in [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md). Earlier pilots and dose curves are preserved in [docs/history/PILOT_REPORT.md](docs/history/PILOT_REPORT.md).

## What we are testing

The question is broader than “how many games did it win?” We want to know whether post-training teaches the model to:

- read the complete visible guess/feedback history;
- produce a legal five-letter word in a strict output envelope;
- avoid repeating guesses;
- respect every green, yellow, and gray constraint, including duplicate letters;
- choose a useful next action rather than memorize one opener;
- recover after its own imperfect earlier choices;
- solve states with only one possible answer left;
- retain general language behavior after task-specific training.

So each run is evaluated from several angles: wins, terminal compliance, invalid and repeated guesses, posterior-constraint violations, singleton accuracy, action-target accuracy, reasoning-token cost, latency, training tokens, VRAM, and retention. A checkpoint with more wins can still be rejected when it is unreliable or strategically inconsistent.

The full preregistered design and decision rules are documented in [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md).

## How the project evolved

The project began as a single local/CUDA notebook export, now preserved as [tiny_llm_wordle_lab.py](tiny_llm_wordle_lab.py). That version established the exact-word baseline, downloaded the model, ran decoding sweeps, and saved raw turn-level prompts and generations.

The code was then split into the [wordle_lab](wordle_lab) package so the game rules, model interface, data construction, training objectives, experiment state, and analysis could be tested independently. The key decisions came from observable failures:

- **Decoding was not the main problem.** Greedy, beam search, repetition penalties, and sampling all failed the 25-game baseline. Sampling created variety but also more malformed output.
- **Formatting and strategy are different skills.** Early SFT taught `Final answer: WORD` formatting and a fixed opener, but the model repeatedly emitted that opener regardless of feedback.
- **Data balance mattered more than loss weighting alone.** A fixed-seed 2×2 ablation showed that balanced state/target exposure caused the meaningful gain; an 8× target-word loss did not rescue the duplicate-heavy data.
- **More parameters helped decisions but did not guarantee reliability.** Qwen2.5 1.5B improved every strategic metric, while 3B produced more wins but severe format/validity regression.
- **A solver hidden in the harness would answer a different question.** Candidate injection, vocabulary masking, reranking, output repair, and harness-selected guesses are excluded from the canonical benchmark.

The historical implementation plan is retained in [docs/history/IMPROVEMENT_PLAN.md](docs/history/IMPROVEMENT_PLAN.md), while the results of implementing it are in [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md).

## The Wordle engine

The core environment is [wordle_lab/protocol/env.py](wordle_lab/protocol/env.py). It provides:

- exact green/yellow/gray scoring with duplicate-letter accounting;
- a six-valid-guess game loop;
- allowed-word validation;
- repeat detection;
- posterior reconstruction from the visible history;
- deterministic state transitions suitable for replay and hashing.

Invalid model calls are recorded but do not consume one of the six valid guesses. The evaluator stops after 12 total model calls so malformed output cannot loop forever. Valid repeated guesses do consume a turn, just as any accepted guess would.

### Is the engine limited?

Yes—intentionally. It is a reproducible research simulator, not a clone of the live New York Times game.

- It has no browser UI, keyboard state, daily puzzle schedule, statistics screen, or account integration.
- It does not implement a separate “hard mode” switch. Posterior consistency is measured as a strategy diagnostic, not enforced as a legality rule.
- Its answer sets and legal guesses are pinned local files. The 128-word curriculum is an artificial development benchmark and is much easier and narrower than general Wordle.
- The frozen 1,000-answer split is a project benchmark, not an official public leaderboard.
- The oracle can calculate posterior candidates and information gain for labels and diagnostics, but those candidates are never exposed to the model during canonical evaluation.

These limits keep runs deterministic, auditable, and inexpensive enough for controlled post-training comparisons. They also mean development win percentages are not claims about performance on the live game.

## The LLM harness

The evaluation harness connects a causal language model to the engine while leaving the actual guess entirely to the model.

| Component | Responsibility |
| --- | --- |
| [prompting.py](wordle_lab/protocol/prompting.py) | Replays the full visible guess/feedback history and defines the output contract |
| [generation.py](wordle_lab/protocol/generation.py) | Runs deterministic greedy generation with a shared 128-token ceiling |
| [parsing.py](wordle_lab/protocol/parsing.py) | Accepts only a final non-empty line shaped like `Final answer: CRANE` |
| [evaluator.py](wordle_lab/protocol/evaluator.py) | Runs games and records format, validity, repeats, posterior violations, tokens, and latency |
| [lock.py](wordle_lab/protocol/lock.py) | Hashes the protocol components, word list, split manifest, and retention probe |
| [runner.py](wordle_lab/experiments/runner.py) | Creates content-derived runs and refuses trained-test access before selection |
| [models.py](wordle_lab/models.py) | Fails closed unless the current study uses the exact pinned Gemma model/revision |

The model may reason in visible text, but its last non-empty line must contain exactly one five-letter answer. The parser neither hunts through the reasoning for a convenient word nor repairs malformed output. Every call receives the full visible history, and the project retains raw per-game JSONL alongside aggregate summaries in the local artifact store.

The current canonical model is [Google Gemma 3 270M IT](https://huggingface.co/google/gemma-3-270m-it), pinned to revision `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`. Historical Qwen artifacts remain documented, but current entrypoints reject them so they cannot silently contaminate a Gemma-only comparison.

## Experiments and results

### 1. Exact-word baseline and decoding sweep

Five decoding conditions—greedy, beam-4, repetition-penalized greedy, and two sampling temperatures—were tested over 25 games. All finished at **0/25**. Greedy and beam search collapsed to one guess; sampling increased diversity without producing wins and increased format/validity failures. This established greedy decoding as the canonical control.

### 2. First `WORDLE-PROTOCOL-002` representation pilots

| Condition | Training dose | Wins | Terminal compliance | What happened |
| --- | ---: | ---: | ---: | --- |
| Base Gemma | none | 0/25 | 0% | Copied the literal placeholder `WORD` |
| Direct SFT | 512 states / 100 steps | 0/25 | 100% | Learned `ARLES`, then repeated it |
| Episode multi-turn | 512 / 100 | 0/25 | 100% | Same repeat collapse |
| Mixed curriculum | 512 / 100 | 0/25 | 100% | Same collapse at higher cost |
| Visible rationale | 512 / 100 | 0/25 | 100% | Fluent but mostly unfaithful reasoning |
| Larger direct SFT | 2,048 / 300 | 0/25 | 60.2% | More variety, worse spelling/validity/retention |

Reference policies confirmed that the engine was behaving sensibly: the random-posterior and greedy-oracle references each won 25/25 development games, while uniform random allowed-word play won 0/25. The base model scored 30% on the retention probe; the larger direct adapter fell to 0%.

### 3. Common-word curriculum and ORPO dose screen

Reducing the universe to 128 common words produced the first genuine held-out solve: `SHARE` → `GBBBG` → `SENSE`. The 600-step SFT parent reached 1/25 with 100% compliance. A repetition penalty of 1.05 reduced repeats without losing that win. A short ORPO continuation reduced repeats further, but did not improve posterior consistency or wins; longer ORPO doses damaged formatting.

The complete checkpoint table is in the [pilot report](docs/history/PILOT_REPORT.md#common-word-curriculum).

### 4. Balanced-data × word-loss ablation

All four cells used seed 2026, 600 optimizer steps, learning rate 5e-5, the same 96/32 train/development split, and natural generation.

| Dataset | Loss | Wins | Compliance | Turn-2 violations | Singleton | Target accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Curriculum 001 | Completion | 1/32 | 90.0% | 87.9% | 0/74 | 1.6% |
| Curriculum 001 | Word-focused | 0/32 | 87.6% | 87.7% | 0/74 | 2.4% |
| Balanced 002 | Completion | 6/32 | 81.6% | 81.0% | 2/74 | 10.9% |
| Balanced 002 | Word-focused | 8/32 | 89.2% | 79.3% | 4/74 | 14.1% |

This is the project’s clearest causal result so far: balanced state and target exposure mattered, while target-token weighting on its own did not.

### 5. Recovery, targeted curricula, and strict-format tests

| Experiment | Best relevant result | Decision |
| --- | --- | --- |
| DAgger round 1 on Gemma | 9/32, but 75.0% compliance and 2/74 singleton | Stopped; repeats improved while core reliability regressed |
| Targeted curriculum 003 | 3/32, 100% compliance, 92.7% turn-2 violations, 0/74 singleton | Rejected; learned format rather than strategy |
| Strict-prompt curriculum 004 | 3/32, 63.4% compliance | Rejected |
| Strict anchored curriculum 005 | 6/32, 95.0% compliance | Rejected; did not beat balanced-002 |
| Strict prompt applied only at inference | 3–4/32, 37–40% compliance | Rejected as prompt-distribution mismatch |

### 6. Historical capacity experiments

The [Qwen2.5 1.5B model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) used the same balanced-002 recipe and improved the Gemma 270M result from 8/32 to 14/32. That is strong evidence that capacity and/or model family matters. It is not a clean capacity ablation because both changed.

The 3B experiment demonstrates why wins are not enough: the final model reached 15/32 and a step-450 checkpoint reached 16/32, but terminal compliance was only 71.2% and 74.6%. Both were rejected. Package provenance and loading notes are preserved in [docs/artifacts/QWEN_1_5B_ADAPTER.md](docs/artifacts/QWEN_1_5B_ADAPTER.md); the generated ZIP itself is intentionally not stored in Git.

### 7. Current Gemma-only framework validation

The current matched bundle contains 4,096 source states, three separate 4,096-row representations, and 512 evaluation-only development probes. Its manifest records hashes, split isolation, feedback recomputation, posterior checks, oracle-fact checks, and matched target IDs. See [the data manifest](data/gemma-270m-comparison-v1/u128-train96-n4096/manifest.json) and [design note](docs/research/GEMMA_270M_COMPARISON_DATA.md).

The automated suite currently covers 76 cases across:

- duplicate-letter scoring, strict parsing, prompt replay, and invalid-turn handling;
- split/provenance checks and matched comparison data;
- exact Gemma model/revision enforcement;
- LoRA, rsLoRA, and DoRA configuration validation;
- SFT, DPO, ORPO, GRPO, Q-SFT, and hybrid-pipeline contracts;
- balanced/targeted curriculum selection and target-token weighting;
- DAgger history fidelity and preference-pair composition;
- GRPO advantage-collapse and virtual-support calculations;
- rejection of secret, candidate-list, and evaluator-data injection.

Implemented and tested code is not experimental evidence. The first current-study experiment used Unsloth for 300 steps on the direct single-step partition. It reached 100% compliance but 0/32 wins, 98.4% fixed-state posterior violations, 0% singleton accuracy, and 12.5% retention, so it was rejected without opening the locked test. The remaining technique matrix is registered but not yet trained. Technique-specific notes live under [docs/research](docs/research).

## Repository guide

```text
.
├── README.md                 Project overview, status, and entry points
├── docs/                     Design, results, historical reports, and research notes
├── wordle_lab/               Tested engine, harness, data, methods, and experiment code
├── scripts/                  One-purpose data and training launchers
├── configs/                  Versioned study, objective, and adapter configurations
├── tests/                    Fast protocol and training-contract tests
├── data/                     Audited, tracked comparison data plus ignored local datasets
├── tiny_llm_wordle_lab.py    Historical all-in-one baseline/LoRA runner
└── requirements.txt          Windows/CUDA Python environment
```

Useful entry points:

- [Experiment results](docs/EXPERIMENT_RESULTS.md)
- [Experiment design](docs/EXPERIMENT_DESIGN.md)
- [Current study configuration](configs/studies/notebooklm_methods.yaml)
- [Technique registry](wordle_lab/experiments/technique_catalog.py)
- [Training launchers](scripts)
- [Test suite](tests)
- [Paper/reproducibility notes](docs/research/paper_evidence_log.md)

Models, checkpoints, raw runs, plots, and most generated datasets are local-only and ignored by Git. They remain under `models/`, `artifacts/`, `results/`, `plots/`, `experiments/`, and `checkpoints/` when created. This keeps the source repository readable without discarding local experimental evidence.

## Setup and common commands

The project is designed for Windows PowerShell with an NVIDIA CUDA GPU.

```powershell
py -m pip install -r requirements.txt
```

Gemma is gated. Accept the model license and authenticate with Hugging Face before the first download; never place an access token in this repository.

```powershell
# Download the pinned base model for local/offline reuse.
py tiny_llm_wordle_lab.py --download-model

# Run the fast regression suite.
py -m pytest -q

# Check the frozen protocol and local data prerequisites.
py -m wordle_lab.cli validate

# Rebuild and audit the current matched representation bundle.
py scripts/build_training_data.py --force
py scripts/audit_training_data.py --token-lengths

# Inspect a training cell without launching an expensive run.
py scripts/train_sft.py --partition reasoning_single_step --dry-run

# List the implemented/registered technique stack.
py -m wordle_lab.cli techniques
```

Run scripts directly from the repository root. They use [scripts/_bootstrap.py](scripts/_bootstrap.py) to make the local package importable.

## What we plan to try next

1. **Reproduce the historical balanced-002, 8× action-token recipe through Unsloth.** The direct Unsloth SFT cell learned formatting but no strategy; a matched reproduction is needed to separate backend behavior from the data/objective change.
2. **Run seed-matched Gemma learning curves at 1,024, 2,048, and 4,096 states.** This tests whether the audited representation bundle produces a real dose-response before adding a more complicated objective.
3. **Compare the three matched representations under ordinary SFT.** Reasoning single-step, direct single-step, and non-reasoning multi-step share the same source states and oracle targets, so any difference is easier to attribute to representation.
4. **Replicate promising parents across three seeds.** The current headline development numbers are one-seed diagnostics. Replication is required before a method winner or locked-test promotion.
5. **Separate adapter and objective questions.** LoRA vs. rsLoRA vs. DoRA should not be conflated with SFT vs. preference/RL objectives. Each comparison needs the same data, seed policy, token budget, and evaluation.
6. **Pursue Q-SFT only with frozen, training-only Bellman snapshots.** It is worth testing because Wordle decisions have delayed value, but the implementation must not derive targets from secrets, evaluator internals, or test data.
7. **Advance SFT → GRPO only after the SFT parent passes its development gate.** GRPO may improve multi-turn recovery, but the earlier experiments show that extra optimization can quickly destroy output compliance.
8. **Run a clean same-family capacity ablation later.** Qwen suggested that 270M may be a capacity bottleneck, but a larger Gemma run under a separately declared, matched configuration is needed to separate capacity from architecture and training history.
9. **Keep attacking singleton and late-turn recovery.** Even the best Qwen 1.5B run solved only 7/74 singleton probes. This is the clearest remaining sign that the model is not reliably translating feedback into an action.

The current stop rule is deliberate: do not churn more curricula or open the locked test simply because one checkpoint posts a higher win count. First establish reliable format, posterior consistency, singleton recovery, and seed replication.

## Research boundaries

Canonical evaluation uses natural generation. It does **not**:

- inject candidate words or reveal the posterior;
- mask logits to the legal vocabulary;
- choose or rerank the model’s guess;
- ban repeats in code;
- repair malformed output;
- expose held-out secrets during training;
- select checkpoints using the locked test.

Tool-augmented solvers may be interesting later, but they must use a separate benchmark label. They cannot be presented as the same unassisted LLM policy.

For methodological background, see the primary [LoRA paper](https://arxiv.org/abs/2106.09685) and [DPO paper](https://arxiv.org/abs/2305.18290). The repository does not link to other “LLM plays Wordle” projects or optimization exercises.
