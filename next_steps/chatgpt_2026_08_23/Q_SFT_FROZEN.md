# Frozen-target Q-SFT: implementation and verification record

## Outcome

`QSFT-BALANCED-002-FROZEN-001` is implemented as a guarded, warm-start-only Q-SFT experiment. The deterministic training bundle can be built and audited, but the frozen historical parent is not eligible to train: its hash-pinned development evidence fails every prerequisite legality threshold. Both `dry-run` and `train` therefore return `blocked_prerequisite_legality_gate_failed`; the training core is not called and no run directory is created. No model training or model evaluation was started while preparing this implementation. The locked test remained closed.

The implementation is in `q_sft_frozen.py`, its immutable study choices are in `q_sft_frozen_config.json`, and focused coverage is in `tests/test_q_sft_frozen.py`. The generated, content-addressed input bundle is in `generated/q_sft_frozen/`.

## Why this experiment

The historical balanced-002 SFT parent achieved strong output-format behavior but did not establish posterior-consistent Wordle strategy. This experiment tests whether a small, frozen value target can make the same completion examples carry a graded strategic signal, while retaining the exact Gemma parent, frozen protocol, natural generation, and development-only evaluation.

It deliberately does not implement inference-time Q-SFT policy extraction. That would rerank or otherwise select an action outside the model's natural generation and would violate `WORDLE-PROTOCOL-002`.

## Frozen target

For posterior size `P` and `k` valid guesses remaining, the target is:

```text
V_0(P) = 0
V_k(P) = 1/P + (1 - 1/P) * gamma * V_(k-1)(P)
gamma = 0.99
```

This is a conservative discounted solve probability under a weak surrogate: every attempt succeeds with probability `1/P`, and a miss is assumed not to shrink the posterior. That assumption avoids importing an oracle transition model into training.

The target has an explicit contract:

- It is always in `[0, 1]`.
- It is nonincreasing as `P` grows.
- It is nondecreasing as the number of guesses remaining grows.
- It uses only the training row's public `posterior_size` and `turn` values.

The builder exhaustively checked all `128 x 6 = 768` configured `(P, turn)` states. The contract passed, with a grid minimum of `0.0078125` and maximum of `1.0`. The 512 balanced-002 training rows actually used targets from `0.044845245811` through `1.0`.

## Data isolation and emitted schemas

The source is exactly `COMMON-WORD-CURRICULUM-002` at `data/common-curriculum-002/u128-train96/train.jsonl`. Every input row must declare `source_state.split == "common_train"`; a development or locked-test row fails closed.

The persisted snapshot schema contains exactly:

```json
{
  "comparison_id": "...",
  "source_state_id": "common_train-...",
  "behavior_policy_id": "BALANCED-002-EMPIRICAL-UNIFORM-001",
  "behavior_action": "ABOUT",
  "behavior_probability": 1.0,
  "behavior_support_size": 1,
  "behavior_support_sha256": "...",
  "posterior_size": 1,
  "turn": 2,
  "bellman_target": 0.0
}
```

The joined core-training schema contains exactly:

```json
{
  "example_id": "...",
  "comparison_id": "...",
  "source_state_id": "common_train-...",
  "behavior_policy_id": "BALANCED-002-EMPIRICAL-UNIFORM-001",
  "behavior_action": "ABOUT",
  "behavior_probability": 1.0,
  "behavior_support_size": 1,
  "behavior_support_sha256": "...",
  "posterior_size": 1,
  "turn": 2,
  "prompt": [],
  "completion": [],
  "bellman_target": 0.0
}
```

`source_state_id` is bound to an exact equality between the source row's top-level `state_id` and `source_state.state_id`. `behavior_action` is bound to both top-level `target_word` and the completion's final answer. For each stable state, the declared support is the set of distinct training actions; its sorted canonical JSON is committed by `behavior_support_sha256`, and every action receives exactly `1 / behavior_support_size`. Repeated curriculum-balancing samples must carry identical metadata and do not inflate the support.

The 512 rows contain 378 stable source states, 378 state/action pairs, and 117 globally distinct actions. Every observed state has one declared action, so the audited support-size and probability bounds are `[1, 1]` and `[1.0, 1.0]`. The aggregate state-support commitment is `515e5ee91348cbfdeba46f0f0744803230e45ef2011631904c225727ebe89694`.

`posterior_size` and `turn` are also retained so the Bellman target can be recomputed directly from each sanitized snapshot. Private `source_state` data is dropped during the join. Snapshot validation recursively rejects secret, answer, candidate, posterior-candidate, oracle, evaluator, and locked-test fields. The joined rows also pass the existing core `validate_q_sft_rows` guardrail.

## Exact provenance

The parent and source validators verify all of the following before a training call can be considered:

| Item | Pinned value |
| --- | --- |
| Protocol | `WORDLE-PROTOCOL-002` |
| Protocol component hash | `afb9884a341f51fbf9c902e07bb130c0a4d742f189aadb3dd0f9ce92fa0f681a` |
| Model | `google/gemma-3-270m-it` |
| Model revision | `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3` |
| Source manifest SHA-256 | `091681fd66f3af5b1e329fe457de6ffac0247421e83a04c8d68d95489be26889` |
| Source training SHA-256 | `8a5741e061349243bc9467ba53254fec648b83dafb5944f65c0d61ab65466e7f` |
| State manifest SHA-256 | `4ab23b5cd883d8ad9b542befadc23c2aec3a3d631b78f239bb551ca998fd6a3c` |
| Required parent run | `sft-common-balanced-word-s2026-0649b4deeb` |
| Required parent checkpoint | `checkpoints/final` |
| Parent spec SHA-256 | `655972a100f33ca26d0e7834f602de9856f88aed89107aed67e6426f9d8c95bc` |
| Parent adapter tree SHA-256 | `074f3a7fe657e34a50cb67f1bf121d61a7c5d5978234c2cc203604ba20e8b833` |

The parent validator also requires a Gemma 3 270M causal-LM LoRA adapter, the exact seed `2026`, curriculum `COMMON-WORD-CURRICULUM-002`, and historical word-token weight `8.0`. There is no implicit base-model start: `--parent-adapter` is required for dry-run, train, and evaluate.

## Prerequisite legality gate

The source recommendation makes Q-SFT conditional: it must stop until the parent demonstrates basic format, feedback legality, and singleton behavior on development data. The implementation therefore reads only four fixed development artifacts from the exact parent run and verifies their SHA-256 digests before using any metric:

| Development evidence | SHA-256 |
| --- | --- |
| `summary.json` | `5501c697996717e9a67be75e90f1ee57dbaefa90a29899b733bbdd8f0d093b9d` |
| `games.jsonl` | `f0df429c6af2a80799dedb3abdadeb64672e392bc610aeb7db6ddff198f8defc` |
| `diagnostics/7b309ade0477/summary.json` | `05079bbeed9be50efc4f48e4acc3b68756f1ff64146808dbba20ac1e0896c859` |
| `diagnostics/7b309ade0477/items.jsonl` | `a0204ff22d4376aedb8519d0c9b66c72e2e5d2ff55a4f51c984584c1863f8593` |

The evidence is also required to identify the same run, `COMMON-WORD-CURRICULUM-002`, the held-out development split, 32 games, diagnostic artifact `7b309ade0477`, all 128 diagnostic items, 58 turn-2 items, and 74 singleton items. The embedded and standalone diagnostic summaries must match. Missing, altered, inconsistent, non-finite, or out-of-range evidence fails closed.

The fixed parent's result is:

| Gate | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Terminal-marker compliance | `>= 0.99` | `0.8920454545454546` | fail |
| Turn-2 posterior-constraint violation rate | `< 0.30` | `0.7931034482758621` | fail |
| Singleton answer accuracy | `>= 0.80` | `0.05405405405405406` | fail |

The blocked response includes the exact thresholds, observed values, comparison operators, failed checks, evidence paths and hashes, coverage counts, `training_started: false`, `run_directory_created: false`, and `locked_test_access: false`. CLI `dry-run` and `train` use exit code `3` for this status. Bundle `build` and `audit` remain available because they are deterministic, training-only data operations and do not imply that the parent is eligible.

## Generated bundle results

The deterministic bundle contains 512 snapshots and 512 joined training rows.

| Artifact | SHA-256 |
| --- | --- |
| `q_sft_frozen_config.json` | `b9bb589fb207a3c272474fdd641ab421b5e73770b5797bbb3ed5a0fc250f9fdd` |
| `generated/q_sft_frozen/manifest.json` | `a1cc599b8d8f702e132b9489db4712b7abbff70c51c1c37ef12e602f3f528fd0` |
| `generated/q_sft_frozen/snapshots.jsonl` | `5a9efa4f88d7c0c07ac208dd2d57c1e69b27665627afc9617e5029003c9d8798` |
| `generated/q_sft_frozen/training_rows.jsonl` | `090e05b445a72b61e80ed33c7da552c1ec730e9863f7397cb8601ea93e328566` |

Rebuilding from the source produces byte-identical JSONL hashes. Re-auditing the existing bundle recomputes the source, protocol, target, join, row counts, file hashes, and manifest content hash.

## Training and evaluation behavior

Only after the prerequisite legality gate passes would the `train` command delegate to the existing `wordle_lab.methods.q_sft.train_q_sft` implementation. At that point it would write the resolved specification, the passing gate evidence, source-tree and Git provenance, copied dataset manifest, training metrics, accounting, and checkpoints into a new content-derived run directory. Existing run directories are never overwritten. The currently pinned parent does not reach this path.

The `evaluate` command is development-only and calls the existing protocol evaluator directly. It does not patch the generation function or add candidate injection, vocabulary masking, reranking, repeat bans, output repair, or harness-selected guesses. It records:

- 32 natural-generation development games;
- 128 deterministic fixed-state development diagnostics;
- turn-2 posterior-constraint violations;
- singleton and action-target accuracy;
- format, invalid-guess, repeat, and win metrics;
- the existing language, arithmetic, logic, and instruction retention probes;
- normalized promotion-gate metrics; and
- a required-file artifact manifest with SHA-256 provenance.

The locked-test flag must be explicitly false in the configuration, bundle, run specification, summary, gate metrics, and artifact provenance. No test split or locked-test answer input is accepted by this experiment path.

## Commands

Run from the repository root in PowerShell:

```powershell
py -m next_steps.chatgpt_2026_08_23.q_sft_frozen build

py -m next_steps.chatgpt_2026_08_23.q_sft_frozen dry-run `
  --parent-adapter artifacts/runs/sft-common-balanced-word-s2026-0649b4deeb/checkpoints/final

py -m pytest -q -p no:cacheprovider `
  next_steps/chatgpt_2026_08_23/tests/test_q_sft_frozen.py
```

For the currently pinned parent, the following training command is expected to stop with exit code `3` and `blocked_prerequisite_legality_gate_failed`. It was not run during implementation because the focused test proves that it stops before trainer invocation or run-directory creation:

```powershell
py -m next_steps.chatgpt_2026_08_23.q_sft_frozen train `
  --parent-adapter artifacts/runs/sft-common-balanced-word-s2026-0649b4deeb/checkpoints/final

```

There is consequently no Q-SFT checkpoint to evaluate. A future parent must be deliberately pinned with complete provenance and frozen development evidence meeting all three gates before training or development evaluation can proceed.

## Verification completed

- Focused Q-SFT test suite: `17 passed`.
- Live CLI dry run: `blocked_prerequisite_legality_gate_failed` with exit code `3`, as required.
- Bundle build and immediate re-audit: `passed`.
- Snapshot forbidden-field scan: no matches.
- Target contract: all 768 configured states passed boundedness and monotonicity checks.
- Training: blocked by the prerequisite legality gate and not run.
- Model development evaluation: not run because there is no trained Q-SFT checkpoint yet.
- Locked test: not opened or evaluated.

Focused tests verify deterministic behavior-support commitments, stable state/action bindings, exact uniform probabilities, target recomputation from preserved inputs, forbidden-field rejection, config/manifest provenance, exact prerequisite thresholds, all four development evidence hashes, the fixed parent's three failures, and refusal before the core trainer or output-directory creation. A separate synthetic eligible-parent lifecycle test replaces only heavyweight model calls; it confirms the downstream path passes all 512 sanitized rows and the explicit parent into the Q-SFT core, records the behavior-support commitment and passing prerequisite evidence in the run specification, and keeps development evaluation and locked-test isolation unchanged.
