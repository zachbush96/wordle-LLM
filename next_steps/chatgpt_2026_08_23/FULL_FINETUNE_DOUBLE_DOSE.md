# Full-parameter Gemma 270M double-dose experiment

Date: 2026-08-23

Experiment: `GEMMA-270M-FULL-FINETUNE-CONTINUATION-002`

Decision: **Doubling the strongest run from 600 to 1,200 total optimizer steps did not improve the overall Wordle policy. Keep the 600-step checkpoint as the development leader.**

## Which techniques produced meaningful results?

Three earlier conditions produced real movement rather than formatting-only gains:

1. `COMMON-WORD-CURRICULUM-002` balanced state/target coverage plus 8x action-token loss reached 8/32 development wins with Gemma 270M LoRA. The matched ablation showed that the balanced coverage, not word weighting by itself, drove the improvement.
2. Full-parameter Gemma 270M tuning on the same 512 rows and 8x objective reached 14/32 wins, 100% terminal compliance, 34.5% turn-2 violations, and 36.7% action-target accuracy. This was the strongest same-model technique and the one scaled here.
3. Qwen 1.5B with the balanced recipe reached 14/32 wins, 35.1% turn-2 violations, and 41.3% action-target accuracy. That is useful capacity evidence, but it is a cross-family model change rather than a clean Gemma training-dose comparison.

The tiny 32-example overfit cells reached 32/32 recall, but that demonstrated memorization only. Ordinary single-step, multi-turn, reasoning, structured-microtask, and sampled constraint-first SFT did not produce reliable held-out strategy.

## Double-dose design

The exact full-parameter step-600 checkpoint from `full-finetune-balanced-word-primary-s2026-57ba532ae7` was hash-authenticated and used as the parent. The continuation added another 600 optimizer steps on the same audited 512-row balanced-002 dataset with seed 2026, effective batch 4, learning rate `5e-5`, 8x action-token loss, BF16, and all 268,098,176 parameters trainable.

The parent run did not save optimizer, scheduler, or RNG state. A deterministic uninterrupted resume was therefore impossible. The continuation explicitly restarts AdamW, the 5%-warmup cosine phase, and the seed-2026 data order. This limitation is encoded in the run spec and must be considered when interpreting the result.

Run: `artifacts/runs/full-finetune-balanced-word-continuation-s2026-8d08e31818`

Training completed in 138.8 seconds, processed 363,758 continuation optimizer tokens, peaked at 4.12 GiB allocated VRAM, and reduced continuation loss from 0.04362 to 0.001287. Checkpoints were evaluated at total optimizer doses 750, 900, 1,050, and 1,200.

## Development comparison

All evaluations used the same frozen 32 development games, 128 state diagnostics including 74 singleton states and 58 turn-2 states, 200 retention probes, explicit-feedback prompt, and natural greedy generation. No candidate injection, masking, reranking, repeat suppression, or output repair was used.

| Total steps | Wins | Compliance | Invalid | Repeats | Posterior violations | Turn-2 violations | Singleton | Target accuracy | Retention |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 600 parent | **14/32** | **100.0%** | **0.0%** | 11.4% | 60.9% | 34.5% | **2/74** | 36.7% | 0.0% |
| 750 | 7/32 | 96.0% | 4.0% | 18.7% | 68.0% | 50.0% | **2/74** | 28.9% | 0.0% |
| 900 | 9/32 | 88.0% | 15.8% | 10.4% | 60.0% | 30.9% | 1/74 | 39.2% | 0.0% |
| 1,050 | 12/32 | 84.0% | 16.0% | 12.0% | **59.1%** | **28.1%** | 0/74 | **40.9%** | 0.0% |
| 1,200 | 11/32 | 84.1% | 15.9% | 12.6% | **59.1%** | **28.1%** | 0/74 | **40.9%** | 0.0% |

## Interpretation

More full-parameter training strengthened action recall on states covered by the balanced curriculum: the later checkpoints moved action-target accuracy from 36.7% to 40.9% and turn-2 violations from 34.5% to 28.1%. That is a real diagnostic movement.

It did not produce a better complete policy. Wins fell from 14 to 11 at the doubled endpoint, terminal compliance fell by 15.9 percentage points, invalid guesses rose from zero to 15.9%, singleton solving fell from 2/74 to 0/74, and retention remained zero. The 1,050-step checkpoint showed the best later-dose strategic diagnostics but still lost two games and had much worse reliability than the parent.

The near-zero training loss alongside degraded held-out gameplay is consistent with over-specialization to the covered training states. The data has no training coverage for the held-out singleton diagnostics, so extra repetition improves covered turn-2 actions while erasing what little singleton generalization and output reliability remained.

## Decision and next scale hypothesis

Do not promote any continuation checkpoint and do not open the locked test. The 600-step parent remains the best development checkpoint.

The next scaling test should increase **coverage or capacity**, not repeat the same 512 rows longer. The cleanest pending condition remains the already specified Gemma 3 1B balanced-002 run if the gated model snapshot becomes available. If staying on 270M, the next controlled experiment should add disjoint singleton/recovery coverage with retention replay or a true set-normalized legal-action objective; it should not simply add more epochs to this curriculum.

The locked 1,000-answer test remained closed.
