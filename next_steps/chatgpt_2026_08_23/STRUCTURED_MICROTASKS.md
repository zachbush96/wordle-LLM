# Structured Wordle Microtasks

Experiment: `STRUCTURED-MICROTASKS-SFT-001`

Status: implementation, audited data build, 600-step Unsloth training, and
608-record development evaluation complete; development gates failed.

## Purpose

The earlier direct SFT run learned the terminal answer envelope without learning
feedback-conditioned play. This experiment decomposes that missing capability
into four machine-readable auxiliary tasks while retaining natural full-policy
examples:

- `feedback_decode`: convert one visible guess/feedback row into positional and
  duplicate-count constraints.
- `constraint_merge`: combine all visible rows into one constraint state.
- `candidate_validity`: classify one proposed word, with an exactly 50/50
  valid/invalid split and equal invalid counts for green, yellow,
  missing-required, gray, duplicate-count, and repeated-guess failures.
- `singleton_solve`: return the sole word implied by visible history.
- `full_policy`: produce an ordinary unassisted `Final answer: WORD` action.

The auxiliary tasks require strict JSON objects. Full-policy prompts and outputs
retain the historical explicit Wordle prompt and natural terminal-answer format.
Evaluation parses the raw model output only; it does not extract fenced JSON,
repair malformed text, mask the vocabulary, inject candidate lists, ban repeats,
or select guesses in the harness.

## Built bundle

The generated bundle is under `generated/structured_microtasks_v1` in this
folder. It contains 1,216 training rows and 608 development rows.

| Task | Train | Development |
| --- | ---: | ---: |
| feedback decode | 256 | 128 |
| constraint merge | 192 | 128 |
| candidate validity | 288 | 96 |
| singleton solve | 80 | 128 |
| natural full policy | 400 | 128 |

Training uses only the 96 balanced-002 training secrets. Development states are
deterministically generated from the separate 32-secret development split, and
any visible history matching a training history is removed. Record IDs and
history overlap are both zero. Private source-state files exist solely for label
and provenance recomputation; they are not part of the mixed SFT view.

Duplicate-heavy histories are intentionally retained at a minimum 25% rate for
feedback, merge, singleton, and full-policy selections. Candidate validity also
contains a dedicated, balanced duplicate-count failure class.

## Commands

```powershell
py next_steps\chatgpt_2026_08_23\structured_microtasks_experiment.py build
py next_steps\chatgpt_2026_08_23\structured_microtasks_experiment.py audit
py next_steps\chatgpt_2026_08_23\structured_microtasks_experiment.py token-audit
py next_steps\chatgpt_2026_08_23\structured_microtasks_experiment.py dry-run
```

The ordinary Transformers/PEFT training path is:

```powershell
py next_steps\chatgpt_2026_08_23\structured_microtasks_experiment.py train-sft
```

The isolated Unsloth completion-SFT path is:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& '.\.cache\unsloth-venv\Scripts\python.exe' `
  next_steps\chatgpt_2026_08_23\structured_microtasks_experiment.py train-unsloth
```

The isolated Unsloth command was used for run
`structured-microtasks-unsloth-s2026-30990a0c2c`. Evaluate natural generations
with:

```powershell
py next_steps\chatgpt_2026_08_23\structured_microtasks_experiment.py evaluate `
  --run-dir artifacts\runs\<run-id> `
  --checkpoint final
```

The evaluator reports total accuracy, coverage, per-task accuracy, strict-format
compliance, and accuracy for each of the six candidate-invalid reasons. Its
development gates are recorded in `structured_microtasks_config.json`. Even a
gate-passing checkpoint permits replication only; it does not open the locked
test.

## Measured result

Training reduced completion loss from `3.1793179512` at step 1 to
`0.1170825958` at step 600. The run took `550.92` seconds and peaked at
`4,955,495,424` allocated VRAM bytes.

| Development task | Correct / total | Accuracy | Strict-format compliance |
| --- | ---: | ---: | ---: |
| feedback decode | 9 / 128 | 0.0703 | 0.9766 |
| constraint merge | 1 / 128 | 0.0078 | 0.9453 |
| candidate validity | 41 / 96 | 0.4271 | 1.0000 |
| singleton solve | 0 / 128 | 0.0000 | 0.8984 |
| full policy | 0 / 128 | 0.0000 | 1.0000 |

Overall exact accuracy was `51/608 = 0.08388`; parse coverage was `0.96217`.
Candidate-invalid reason accuracy was `0.625` for missing-required, `0.125`
for duplicate-count, repeated-guess, and yellow, and `0.0` for green and gray.
Only full-policy format compliance passed its declared check. The overall gate
failed, and the locked test remained closed.

The negative result is informative: low training loss and mostly valid output
schemas did not establish duplicate-safe constraint reasoning or singleton
recovery. Raw and parsed natural outputs, the aggregate summary, training trace,
specification, and accounting are collected under `results/structured_microtasks/`.
