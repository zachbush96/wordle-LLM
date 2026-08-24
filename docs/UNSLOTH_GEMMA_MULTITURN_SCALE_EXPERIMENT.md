# Unsloth Gemma Multi-turn Data Scale Experiment

Date: 2026-08-23

Experiment: `UNSLOTH-GEMMA-MULTITURN-SCALE-003`

Decision: **Increasing the audited multi-turn set from 2,000 to 5,000 examples did not improve Wordle strategy. Stop before 8,000 examples.**

## Question and decision rule

The preceding `UNSLOTH-GEMMA-ALPACA-002` experiment trained Gemma 3 270M on 2,000 non-reasoning multi-turn examples for 300 optimizer steps. This follow-up asks whether a larger set of the same example type changes the result.

The comparison keeps the model, seed, learning rate, LoRA configuration, completion-only objective, effective batch size, data generator, development secrets, development probes, and natural greedy evaluation fixed. Training exposure remains 2.4 epochs: 2,000 rows used 300 steps at batch 16, while 5,000 rows use 750 steps at batch 16. A fixed 300-step run would draw only 4,800 examples and would not be a genuine 5,000-example scale test.

Before training, meaningful improvement was defined as preserving at least 99% terminal compliance while showing a clear strategic movement: at least 5% action-target accuracy, a reduction of at least 10 percentage points in fixed-state posterior violations, positive singleton solving, or several held-out development wins. Lower training loss or valid formatting alone would not qualify.

## Audited data

Dataset: `data/gemma-270m-unsloth-alpaca-v2/u160-train120-n5000`

The generator produced 5,000 multi-turn rows from 5,000 source records, including 4,845 unique states. It also rendered matched single-step and reasoning partitions for the existing audit contract, but only the multi-turn partition was trained. All ten correctness, provenance, history-fidelity, held-out-split, and leakage checks passed. The locked test was not read.

- Multi-turn partition SHA-256: `13a71b59ee8494f12a4cbadc69aa2de717e9787ad6b98535ecb6d51b6ae39300`
- Manifest SHA-256: `bac2b89d9e8f24097244cb13a4b190442b6752ca308e281d44803336efcc1208`
- Development-probe SHA-256: `52950b96e136f886300818ebd3eb32bc7ce12b3cba1e715121c61c3264a33d11`, exactly matching the 2,000-row study
- Maximum rendered multi-turn length: 265 tokens under the fixed 320-token limit

## Training

Run: `artifacts/runs/unsloth-sft-non_reasoning_multi_step-s2026-c083e18bf8`

The run used pinned `google/gemma-3-270m-it` revision `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`, seed 2026, BF16 LoRA rank 16/alpha 32, completion-only loss, learning rate `5e-5`, batch 16, and 750 optimizer steps. It completed in 697.4 seconds, processed 2,235,118 optimizer tokens, and peaked at 5,018,763,264 allocated VRAM bytes (4.67 GiB). Training loss fell from 3.734 to 0.787 and had already flattened near 0.78 at the first saved checkpoint.

The step-750 adapter SHA-256 is `647dd3ddcb476c58a77791e81ef73d4cc3dc290a3dc4b6a47759b4885934ed80`.

## Result

Evaluation used the same 40 held-out development games, 512 fixed-state probes, and 200 retention prompts as the 2,000-row study.

| Condition | Wins | Compliance | Gameplay violations | Gameplay repeats | Fixed-state violations | Singleton | Target accuracy | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K multi-turn, step 300 | 0/40 | 100% | 83.33% | 83.33% | 98.24% | 0% | 0.59% | 12.5% |
| 5K multi-turn, step 188 | 0/40 | 100% | 83.33% | 83.33% | 98.24% | 0% | 0.59% | 12.5% |
| 5K multi-turn, step 375 | 0/40 | 100% | 83.33% | 83.33% | 98.24% | 0% | 0.59% | 7.5% |
| 5K multi-turn, step 562 | 0/40 | 100% | 83.33% | 83.33% | 98.24% | 0% | 0.59% | 12.5% |
| 5K multi-turn, step 750 | 0/40 | 100% | 83.33% | 83.33% | 98.24% | 0% | 0.59% | 12.5% |

Every saved 5K checkpoint produced the same strategic metrics as the 2K final checkpoint. The larger set therefore did not merely fail the promotion gate; it showed no measurable strategic movement at all. The only checkpoint difference was a temporary retention regression at step 375.

## Interpretation and next decision

The model converged early to the same validly formatted but state-insensitive repeated-guess policy. Adding more ordinary multi-turn demonstrations did not teach feedback-conditioned action selection, even though it increased unique training states and more than doubled optimizer-token exposure.

The conditional 8K run was not launched because the preregistered 5K improvement condition failed. A 5K single-step run was also not justified: single-step and multi-turn were already essentially tied at 2K. A 5K reasoning run was not justified because the 2K reasoning condition was slower, less compliant, and strategically no better.

The evidence points away from another representation-scale repeat and toward changing the learning signal or capacity. The strongest already-preregistered options are the frozen-target Q-SFT objective or a same-family larger Gemma capacity condition. Those should remain separate experiments so any improvement can be attributed to the objective or model rather than mixed with this data-scale result.

The locked 1,000-answer test remains closed.
