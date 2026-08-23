# ChatGPT Wordle next-step suite

Date: 2026-08-23

This folder implements the complete next-step program captured from the ChatGPT conversation titled **Suggest Next Steps**. The source and faithful recommendation set are in [SOURCE_RECOMMENDATIONS.md](SOURCE_RECOMMENDATIONS.md); the measured outcome and interpretation are in [RESULTS_AND_HANDOFF.md](RESULTS_AND_HANDOFF.md).

Every executable condition preserves `WORDLE-PROTOCOL-002`, natural model generation, the frozen development split, and the closed locked-test boundary. There is no candidate injection, vocabulary mask, reranker, harness repeat ban, output repair, or hidden-answer access at evaluation time.

## Completion map

| Recommendation | Implementation | Outcome |
| --- | --- | --- |
| Two 32-state memorization cells | `tiny_overfit.py`, `train_tiny_overfit.py` | Complete: both general and singleton cells reached 32/32 exact natural recall. |
| Balanced-002, 8x word loss through Unsloth | `balanced_002_unsloth.py` | Complete: 600 steps plus five development conditions; no promotion gate passed. |
| Matched LoRA versus full-parameter 270M | `full_finetune.py`, `full_finetune_experiment.py` | Complete: full tuning improved 8 to 14 development wins, but singleton and retention failed. |
| Structured legality microtasks and mixed training | `microtasks.py`, `structured_microtasks_experiment.py` | Complete: audited train/dev bundle, 600-step run, 608-record development evaluation; gates failed. |
| Constraint-first full-policy objective | `constraint_first_policy.py`, `train_constraint_first.py` | Complete for the implemented sampled multi-label SFT approximation; all four doses evaluated, none promotable. |
| Frozen-target Q-SFT | `q_sft_frozen.py` | Bundle and target audit complete; training correctly blocked because the fixed parent fails all prerequisite legality gates. |
| Same-family Gemma 3 1B | `gemma_1b_capacity.py` | Exact matched condition and preflight implemented; training unavailable because the pinned gated snapshot is absent and no Hugging Face authentication is configured. |

The Q-SFT and 1B rows are not missing zero-valued experiments. They are explicit unavailable states with evidence and no synthetic metrics.

## Folder layout

- `generated/`: deterministic, content-addressed train/dev bundles and blocker evidence.
- `results/`: compact Git-tracked run specifications, accounting, training traces, raw development outputs, evaluation summaries, and collection provenance. Model checkpoints remain outside Git.
- `tests/`: leakage, provenance, objective, evaluator, threshold, and lifecycle tests for this suite.
- [STRUCTURED_MICROTASKS.md](STRUCTURED_MICROTASKS.md): schema, duplicate-letter logic, data balance, and measured microtask results.
- [Q_SFT_FROZEN.md](Q_SFT_FROZEN.md): frozen target, parent identity, prerequisite gate, and blocked outcome.
- [results_declaration.json](results_declaration.json): explicit allowlist used to collect ignored run evidence.

## Reproduce the verification

From the repository root:

```powershell
py -m pytest -q
py -m next_steps.chatgpt_2026_08_23.collect_results validate `
  --manifest next_steps/chatgpt_2026_08_23/results_declaration.json
py -m next_steps.chatgpt_2026_08_23.collect_results collect `
  --manifest next_steps/chatgpt_2026_08_23/results_declaration.json `
  --dry-run
```

Training commands, run IDs, exact hashes, metric tables, caveats, and the recommended interpretation for the next LLM discussion are recorded in [RESULTS_AND_HANDOFF.md](RESULTS_AND_HANDOFF.md).

## Locked-test decision

No condition met the three-seed promotion requirement of at least 99% terminal compliance, less than 30% turn-2 posterior violations, and materially strong singleton accuracy (target at least 80%). This suite therefore did not open, read, or evaluate the locked 1,000-game test.
