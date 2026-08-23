# Unsloth Gemma Wordle Data and Representation Experiment

Date: 2026-08-22

Experiment: `UNSLOTH-GEMMA-ALPACA-002`

Decision: **The new data improved coverage and auditability, but none of the three fine-tunes improved Wordle strategy. Do not promote a checkpoint or open the locked test.**

## Executive result

The experiment generated exactly 2,000 correct examples for each requested variety: direct single-step, non-reasoning multi-step, and reasoning single-step. All 6,000 rows expose Unsloth-style `instruction`, `input`, and `output` fields and retain model-native `messages`, which are rendered through Gemma's tokenizer chat template during training.

All three 300-step Unsloth LoRA runs completed successfully. Every one of the 12 saved dose checkpoints was then evaluated on 40 held-out development games, 512 fixed-state probes, and 200 retention prompts using unchanged natural greedy generation.

No checkpoint won a held-out game or solved a singleton state. Direct single-step and multi-step learned perfect terminal formatting but repeated guesses on roughly 83% of gameplay turns and violated the Wordle posterior on 98.2% or more of fixed states. Reasoning was slower and less reliable: its final checkpoint generated 101.2 tokens per call on average, had 25.9% invalid guesses, and still had 98.8% fixed-state violations.

## Official Unsloth format decision

The implementation follows Unsloth's current official guidance:

- The [datasets guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide) defines instruction data semantically as `Instruction` (the task), `Input` (the user query or context), and `Output` (the expected response).
- The [chat templates guide](https://docs.unsloth.ai/basics/chat-templates) distinguishes single-turn Alpaca-style data from multi-turn conversation data and recommends applying the model's chat template.
- The [Gemma 3 guide](https://docs.unsloth.ai/basics/tutorial-how-to-run-and-fine-tune-gemma-3) is the model-specific training reference.

Each JSONL row therefore contains lowercase `instruction`, `input`, and `output` keys for inspection and interchange, plus a `messages` array. Training uses `messages` through `tokenizer.apply_chat_template`; it does not concatenate an invented generic Alpaca prompt around Gemma. For multi-step rows, `messages` preserves the alternating user/assistant episode. This is the important distinction: the semantic fields make the dataset understandable, while Gemma's own turn tokens determine the actual training sequence.

## Data generator and correctness contract

Generator: `scripts/build_unsloth_training_data.py`

Dataset: `data/gemma-270m-unsloth-alpaca-v2/u160-train120-n2000`

The 160-word study universe is the intersection of:

1. a pinned copy of the original 2,315-word Wordle answer list, SHA-256 `5209b35f823f8b80f0404f863bd80df06d6a966c6eb1016d69f38badc6eed5d0`; and
2. protocol-002's training split.

This gives actual Wordle answer words while proving that no word was sourced from the protocol development or locked-test files. The locked test was not read. The universe contains 80 common, 56 intermediate, and 24 advanced words, ranked with the pinned `wordfreq` dependency. It is split into 120 training secrets and 40 study-development secrets.

The generator creates 2,000 matched source states and renders all three representations from exactly the same state/action labels. The audit recomputes:

- every Wordle feedback pattern, including duplicate-letter accounting;
- the remaining posterior and secret membership;
- the deterministic oracle target and rationale facts;
- train/development secret separation;
- multi-turn transcript fidelity;
- final-answer agreement across all three representations; and
- Gemma messages plus instruction/input/output field agreement.

Audit result: 2,000 source states, 6,000 rendered rows, all checks passed. Maximum Gemma-rendered lengths were 152 tokens for single-step, 264 for multi-step, and 265 for reasoning, below the fixed 320-token limit.

| Partition | Rows | SHA-256 |
| --- | ---: | --- |
| Direct single-step | 2,000 | `c6c1b3d4745904df17ac8a0738cdc7f4dba6d58e07d7339368d149a5467d0451` |
| Non-reasoning multi-step | 2,000 | `306e79edd3f232b60ffb8637a1b09a7ade89e9c36befc99e92fcf3569ac950b2` |
| Reasoning single-step | 2,000 | `585b91c45de8e842aa5a911c67d5cee5fcc4ad7463714451f7d2e5f737c8ef4d` |

## Matched training recipe

All three conditions used the pinned `google/gemma-3-270m-it` revision `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`, Unsloth 2026.8.19, BF16 without quantization, seed 2026, completion-only loss, learning rate 5e-5, effective batch 16, and LoRA rank 16/alpha 32 on q/k/v/o and gate/up/down projections. Checkpoints were saved at steps 75, 150, 225, and 300.

| Variant | Optimizer tokens | Train time | Peak allocated VRAM | Final train loss |
| --- | ---: | ---: | ---: | ---: |
| Single-step | 628,514 | 288.9 s | 3.00 GiB | 0.70 |
| Multi-step | 895,300 | 288.8 s | 4.68 GiB | 0.68 |
| Reasoning | 1,058,524 | 291.7 s | 4.67 GiB | 0.20 |

Steps and examples are matched, but tokens are not: richer representations contain more tokens. This is a representation-and-dose comparison, not an equal-token ablation.

## Final checkpoint results

| Condition | Wins | Gameplay compliance | Invalid | Repeats | Gameplay violations | Fixed-state violations | Singleton | Target accuracy | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen base | 0/40 | 0.0% | 100.0% | n/a | n/a | n/a | 0.0% | n/a | 30.0% |
| Prior 4,096-row Unsloth single-step, same holdout | 0/40 | 100.0% | 0.0% | 56.2% | 100.0% | 99.8% | 0.0% | 0.0% | 12.5% |
| New 2,000-row single-step | 0/40 | 100.0% | 0.0% | 82.9% | 83.3% | 98.2% | 0.0% | 0.59% | 12.5% |
| New 2,000-row multi-step | 0/40 | 100.0% | 0.0% | 83.3% | 83.3% | 98.2% | 0.0% | 0.59% | 12.5% |
| New 2,000-row reasoning | 0/40 | 87.2% | 25.9% | 49.6% | 82.8%* | 98.8% | 0.0% | 0.41% | 17.5% |

`*` Gameplay constraint rates are conditional on parseable valid guesses; they must be read with the invalid-guess rate. The base has no parseable guesses, so posterior and repeat rates are undefined rather than zero.

The new direct data moved the prior adapter's fixed-state violation rate from 99.8% to 98.2% and target accuracy from 0% to 0.59% on this holdout, but repeats worsened by 26.7 percentage points and wins/singletons stayed at zero. That is a tiny diagnostic movement, not a strategic improvement.

## Checkpoint curve

| Variant | Step | Wins | Compliance | Fixed violations | Singleton | Target accuracy | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single | 75 | 0/40 | 100.0% | 98.8% | 0.0% | 0.20% | 25.0% |
| Single | 150 | 0/40 | 100.0% | 98.2% | 0.0% | 0.00% | 22.5% |
| Single | 225 | 0/40 | 100.0% | 98.8% | 0.0% | 0.39% | 12.5% |
| Single | 300 | 0/40 | 100.0% | 98.2% | 0.0% | 0.59% | 12.5% |
| Multi | 75 | 0/40 | 100.0% | 98.2% | 0.0% | 0.59% | 12.5% |
| Multi | 150 | 0/40 | 100.0% | 98.2% | 0.0% | 0.59% | 12.5% |
| Multi | 225 | 0/40 | 100.0% | 98.2% | 0.0% | 0.59% | 12.5% |
| Multi | 300 | 0/40 | 100.0% | 98.2% | 0.0% | 0.59% | 12.5% |
| Reasoning | 75 | 0/40 | 0.0% | 99.0% | 0.0% | 0.00% | 25.5% |
| Reasoning | 150 | 0/40 | 0.0% | 100.0% | 0.0% | 0.00% | 23.0% |
| Reasoning | 225 | 0/40 | 43.5% | 98.9% | 0.0% | 0.54% | 19.5% |
| Reasoning | 300 | 0/40 | 87.2% | 98.8% | 0.0% | 0.41% | 17.5% |

No transient checkpoint improved wins or singleton accuracy. More dose taught the reasoning output envelope but not the feedback transformation.

## Conclusion

The data itself is a meaningful improvement: it has the requested fields, three matched representations, broader verified lexical complexity, exact counts, source hashes, native Gemma chat rendering, and stronger correctness/leakage audits. The model result is negative.

The 270M adapter continues to minimize common completion patterns rather than learn state-conditioned Wordle decisions. Multi-turn history did not help. Visible rationale increased both training-token exposure and inference cost while harming reliability. The strongest historical Gemma development result remains the earlier balanced/word-focused 8/32 run; these new runs do not displace it.

No checkpoint meets any strategic promotion gate. The locked test remains closed. The next justified experiment is not more ordinary SFT data churn; it is either the preregistered frozen-target Q-SFT objective or a same-family larger Gemma capacity condition, followed by seed replication only if development gates are met.

## Saved artifacts and reproduction

Compact tracked results: `docs/research/unsloth_gemma_wordle_alpaca_v2_results.json`

Local run directories contain all specs, manifests, train metrics, four adapters, raw games, raw state outputs, retention rows, and evaluation summaries:

- `artifacts/runs/unsloth-sft-non_reasoning_single_step-s2026-b60cc18cd3`
- `artifacts/runs/unsloth-sft-non_reasoning_multi_step-s2026-46e287240c`
- `artifacts/runs/unsloth-sft-reasoning_single_step-s2026-c2e8d1797d`

Final adapter SHA-256 values are `0f1ea0d07614e2fa3bbd939aa9b5c1c6ea80c21649b02e75ab7fb2f139895b17`, `21a0751d719937ee42060fd03cc9bde7ee4f514835819e1538db09eb52ff853b`, and `27d224a6055c14babf938eeb2c024d4d88d399d4c681be4882c621b729c59cb8` respectively.

```powershell
py scripts\build_unsloth_training_data.py --force

$data = 'data\gemma-270m-unsloth-alpaca-v2\u160-train120-n2000'
$python = '.\.cache\unsloth-venv\Scripts\python.exe'
& $python scripts\train_unsloth_sft.py --partition non_reasoning_single_step --data-dir $data --steps 300 --seed 2026
& $python scripts\train_unsloth_sft.py --partition non_reasoning_multi_step --data-dir $data --steps 300 --seed 2026
& $python scripts\train_unsloth_sft.py --partition reasoning_single_step --data-dir $data --steps 300 --seed 2026
```

The locked test was never read or evaluated.
