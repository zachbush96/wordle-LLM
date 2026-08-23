# Wordle next-step results and LLM handoff

Date: 2026-08-23

Source: ChatGPT conversation **Suggest Next Steps**

Source URL: <https://chatgpt.com/c/6a8a52f1-c4f4-83ea-a03a-d722f66f3c68>

## Bottom line

The model can memorize the requested mappings, so the training stack is capable of fitting exact feedback-conditioned outputs. The failure is held-out strategy: weighted LoRA, structured microtasks, and sampled multi-label constraint-first SFT all reached low training loss without learning reliable feedback legality or singleton solving. Full-parameter 270M tuning produced the clearest improvement—14/32 development wins versus 8/32 for the exactly audited historical LoRA comparator—but still missed the turn-2 gate, achieved only 2/74 singleton answers, and reduced retention to zero.

Q-SFT was correctly stopped because its parent fails the prerequisite legality gate. The clean Gemma 3 1B capacity experiment is fully specified but could not start because the pinned gated model snapshot is unavailable without Hugging Face authentication. No locked-test file was opened.

## Shared protocol and environment

- Protocol: `WORDLE-PROTOCOL-002`
- Base model: `google/gemma-3-270m-it`, revision `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`
- GPU: NVIDIA GeForce RTX 4060 Ti, 16 GB
- Development gameplay: 32 frozen balanced-002 secrets
- Fixed-state diagnostics: 128 items, including 74 singleton items and 58 turn-2 items
- Natural generation only; no candidate injection, vocabulary masking, reranking, repeat suppression, or output repair
- Locked test: closed throughout

## 1. Tiny memorization diagnostics

### What and why

Two disjoint 32-state, training-only cells test whether the 270M model and LoRA stack can fit exact mappings before another generalization hypothesis is considered. One cell uses general states; the other uses states whose visible feedback leaves exactly one answer.

Both cells use 400 optimizer steps, 8x word-token weighting, and paired base-versus-adapter evaluation on identical states. The evaluator records natural exact recall plus the target word's full-vocabulary rank and normalized probability.

### Results

| Cell | Loss, first to final | Base exact | Adapter exact | Base mean rank | Adapter mean rank | Base mean probability | Adapter mean probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| General | 6.53653 to 0.0000254 | 0/32 | 32/32 | 63.000 | 1.000 | 0.009593 | 0.999985 |
| Singleton | 6.99988 to 0.0000185 | 0/32 | 32/32 | 56.219 | 1.000 | 0.007051 | 0.999987 |

Every target rank improved in both cells. This is a training-set memorization result, not held-out Wordle performance. It rules out a simple inability to fit 32 exact mappings and points toward generalization, representation, or objective mismatch.

Run IDs:

- `tiny-overfit-general-s2026-15451a8514`
- `tiny-overfit-singleton-s2026-0fe89d7a95`

## 2. Balanced-002 with 8x action-token loss through Unsloth

### What and why

This cell reproduces the 512-row `COMMON-WORD-CURRICULUM-002` recipe with seed 2026, effective batch 4, 600 steps, checkpoints at 150/300/450/600, rank-16 LoRA, learning rate 5e-5, and exact 8x action-token loss. The data and protocol audits recomputed every feedback row and pinned the dataset, model, allowed-word, development, and evaluator hashes.

The intentional backend difference is Unsloth's patched training implementation rather than the historical native Transformers/PEFT loop. No other recipe difference is treated as matched silently.

### Results

Training loss fell from 8.12036 to 1.29601 in 549.99 seconds. Peak allocated VRAM was 2.405 GB.

| Dose / decoder | Wins | Compliance | Invalid | Repeat | Posterior violation | Turn-2 violation | Singleton | Action target | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150 / greedy | 0/32 | 1.000 | 0.000 | 0.469 | 0.977 | 0.966 | 0.000 | 0.008 | 0.190 |
| 300 / greedy | 3/32 | 0.897 | 0.103 | 0.379 | 0.891 | 0.810 | 0.014 | 0.063 | 0.250 |
| 450 / greedy | 7/32 | 0.880 | 0.120 | 0.284 | 0.852 | 0.828 | 0.054 | 0.109 | 0.275 |
| 600 / greedy | 7/32 | 0.885 | 0.115 | 0.272 | 0.836 | 0.776 | 0.041 | 0.109 | 0.275 |
| Final / greedy, repetition penalty 1.05 | 8/32 | 0.852 | 0.148 | 0.019 | 0.836 | 0.776 | 0.041 | 0.109 | unavailable |

No condition passed the development gates. The 8-win decoder result reproduces the historical headline win count, but the combined diagnostics show that the policy remains largely posterior-inconsistent and weak on singleton states. Decoder variation changed repeats without learning legality.

Run ID: `unsloth-balanced-002-word8-s2026-467ae5da73`

## 3. Native LoRA versus full-parameter 270M tuning

### What and why

The full-parameter cell holds model revision, exact 512 rows and order, seed, learning rate, effective batch, 8x objective, optimizer schedule, prompt, checkpoint doses, and development evaluation fixed against the hash-authenticated historical native Transformers/PEFT LoRA run `sft-common-balanced-word-s2026-0649b4deeb`. The intended difference is trainable scope: 3,796,992 LoRA parameters versus all 268,098,176 base parameters.

The unavoidable precision difference is recorded: historical LoRA trainables were FP32 while full-model trainables were BF16. The comparison is a matched, single-seed diagnostic, not replicated superiority.

### Results

The memory smoke test passed before the primary run. Full training took 133.59 seconds, peaked at 4.423 GB allocated VRAM, and reduced loss from 8.07647 to 0.07689.

| Dose | Wins | Compliance | Invalid | Repeat | Posterior violation | Turn-2 violation | Singleton | Action target | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150 | 3/32 | 1.000 | 0.000 | 0.508 | 0.875 | 0.793 | 0.027 | 0.070 | 0.000 |
| 300 | 9/32 | 1.000 | 0.000 | 0.128 | 0.758 | 0.621 | 0.027 | 0.227 | 0.000 |
| 450 | 10/32 | 0.951 | 0.049 | 0.123 | 0.630 | 0.379 | 0.014 | 0.331 | 0.000 |
| 600 | 14/32 | 1.000 | 0.000 | 0.114 | 0.609 | 0.345 | 0.027 | 0.367 | 0.000 |

At step 600, full tuning versus the native LoRA comparator changed wins by +6, win rate by +0.1875, compliance by +0.1080, overall posterior violations by -0.2188, turn-2 violations by -0.4483, and action-target accuracy by +0.2266. Singleton accuracy moved backward by -0.0270, and full-model retention was 0.000; comparator retention is unavailable.

This supports trainable capacity as a real limitation, but not as a complete solution. The final turn-2 rate was still above the strict 0.30 gate, singleton accuracy was 2/74, and catastrophic retention makes the checkpoint unsuitable for promotion.

Run ID: `full-finetune-balanced-word-primary-s2026-57ba532ae7`

## 4. Structured microtasks and mixed policy training

### What and why

The deterministic mixed curriculum contains 1,216 training rows and 608 disjoint development rows for duplicate-safe feedback decoding, multi-turn merge, exactly balanced candidate validity, singleton solving, and natural full-policy generation. Candidate validity is 50/50, with each invalid cause represented equally: green, yellow, missing required, gray, duplicate count, and repeated guess.

The rank-16 Unsloth run used completion loss for 600 steps. Loss fell from 3.17932 to 0.11708 in 550.92 seconds; peak allocated VRAM was 4.955 GB.

### Results

| Development task | Accuracy | Strict-format coverage |
| --- | ---: | ---: |
| Feedback decode | 9/128 = 0.070 | 125/128 = 0.977 |
| Constraint merge | 1/128 = 0.008 | 121/128 = 0.945 |
| Candidate validity | 41/96 = 0.427 | 96/96 = 1.000 |
| Singleton solve | 0/128 = 0.000 | 115/128 = 0.898 |
| Full policy | 0/128 = 0.000 | 128/128 = 1.000 |

Overall accuracy was 51/608 = 0.0839 and coverage was 0.9622. Invalid-reason accuracy was 0.625 for missing-required, 0.125 each for duplicate-count, repeated-guess, and yellow, and 0 for green and gray. Every development gate failed except full-policy format compliance.

Low loss and mostly valid output schemas did not translate into the requested symbolic operations. The clearest failure is 0/128 singleton solving despite explicit singleton training.

Run ID: `structured-microtasks-unsloth-s2026-30990a0c2c`

## 5. Constraint-first full-policy objective

### What and why

The implemented condition emits up to four deterministic, posterior-consistent, non-repeated labels for each non-singleton training state and doubles singleton rows. The final bundle has 1,018 rows from 378 unique states: 291 non-singleton states with multiple legal labels and 87 singleton states contributing 174 rows.

This is **sampled multi-label word-focused SFT**, not a true set-normalized objective. That distinction matters: the run tests whether deterministic support diversification helps, not whether probability mass assigned anywhere inside the complete legal-action set is optimized directly.

Training used 8x word-token loss for 600 steps. Loss changed from 5.63744 to 2.66341 in 544.01 seconds; peak allocated VRAM was 2.404 GB.

### Results

| Dose | Wins | Compliance | Invalid | Repeat | Posterior violation | Turn-2 violation | Singleton | Action target | Retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150 | 0/32 | 0.000 | 1.000 | 0.000 | 0.981 | 0.966 | 0.000 | 0.000 | 0.175 |
| 300 | 0/32 | 1.000 | 0.000 | 0.807 | 0.977 | 0.966 | 0.000 | 0.000 | 0.200 |
| 450 | 1/32 | 0.916 | 0.084 | 0.286 | 0.883 | 0.793 | 0.014 | 0.023 | 0.250 |
| 600 | 1/32 | 0.945 | 0.055 | 0.189 | 0.897 | 0.804 | 0.014 | 0.024 | 0.250 |

All four doses failed the post-hoc development selection policy, so no checkpoint was selected and the locked test stayed closed. The selection thresholds and ranking were added after training began and are explicitly labeled `post_hoc_after_training_started`; they must not be represented as preregistered.

This sampled multi-label formulation did not improve the policy. At 300 steps it learned the output envelope while repeating heavily; later doses reduced repeats but remained overwhelmingly posterior-inconsistent and near-zero on singleton/action targets.

Run ID: `constraint-first-s2026-ac97761439`

## 6. Frozen-target Q-SFT

The 512-row target bundle was rebuilt and reproduced byte-identical hashes. Its conservative Bellman recurrence was checked over all 768 configured states and remained bounded and monotonic. Each sanitized snapshot preserves `comparison_id`, stable `source_state_id`, behavior policy/action/probability/support commitment, `posterior_size`, `turn`, and `bellman_target`. The audit found 378 stable state/action pairs, 117 globally distinct actions, and support size 1 for every source state in this balanced-002 source. Repeated balancing samples must carry identical metadata and cannot inflate support. The training join contains no secrets, candidate lists, evaluator data, or development/test answers.

The fixed parent's four development evidence files are hash-pinned. It fails every prerequisite:

| Gate | Required | Observed |
| --- | ---: | ---: |
| Terminal compliance | at least 0.99 | 0.8920 |
| Turn-2 posterior violation | strictly below 0.30 | 0.7931 |
| Singleton accuracy | at least 0.80 | 0.0541 |

Live dry-run result: `blocked_prerequisite_legality_gate_failed`, exit code 3, `training_started: false`, `run_directory_created: false`. Training and evaluation were intentionally not run because doing so would violate the source recommendation.

## 7. Same-family Gemma 3 1B capacity condition

The matched recipe is pinned to `google/gemma-3-1b-it` revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752`. It authenticates the 270M comparator artifacts and binds the exact balanced-002 data, 8x objective, seed, schedule, prompt, evaluator, allowed-word list, canonical diagnostics, and retention probes. The implemented `evaluate-all` path covers exact doses 150/300/450/600, validates reusable artifacts by hash, and applies the same format, legality, posterior, singleton, retention, and gameplay selection contract as the 270M comparison.

Live preflight confirmed that the remote revision matches, but the model is gated, the local snapshot is absent, and no Hugging Face authentication is configured. Provisioning therefore returned `blocked_missing_huggingface_auth_for_gated_model` with `download_attempted: false` and no credentials recorded. Training and evaluation metrics are unavailable, not zero.

## Answer to “Did it improve?”

Partly, but not enough to promote anything.

- **Yes, memorization:** both tiny cells moved from 0/32 to 32/32 and rank 1 with approximately 0.99999 target probability.
- **Yes, trainable-scope signal:** full-parameter 270M improved the matched LoRA comparator from 8 to 14 wins and sharply reduced posterior violations.
- **No, robust strategy:** the best completed 270M condition still missed the turn-2 threshold, had 2/74 singleton accuracy, and erased retention.
- **No, structured-task transfer:** low loss and compliant formatting did not produce feedback decoding, constraint merging, singleton solving, or policy accuracy.
- **No, sampled constraint-first benefit:** diversifying legal labels in ordinary SFT performed worse than the balanced baseline.
- **Unknown, 1B capacity:** the clean same-family test remains unavailable due to gated-model access.
- **Not yet eligible, Q-SFT:** the prerequisite legality gate correctly stopped the run.

The evidence suggests two separate bottlenecks: adapter/trainable capacity materially affects strategy, while the current token-level objectives still do not teach reusable constraint operations or preserve general capability. Falling loss is therefore not a sufficient selection signal.

## Most useful next hypothesis for the LLM to assess

Provision and run the already specified same-family Gemma 3 1B condition first. It is the cleanest unresolved test because full-parameter 270M produced the only substantial strategic movement. If 1B remains weak on singleton and turn-2 legality, the next controlled objective experiment should use a **true set-normalized legal-action loss** rather than the sampled multi-label approximation, with explicit retention regularization or replay. Q-SFT should remain blocked until one parent meets the existing legality and singleton gates.

Any follow-up should use at least three matched seeds before a promotion claim. None of the single-seed development results here authorizes opening the locked test.

## Verification completed

- Repository regression: `py -m pytest -q` -> `248 passed in 85.31s`.
- Focused next-step suite: `167 passed`.
- `compileall` passed for `wordle_lab`, `scripts`, and this folder.
- Tiny, balanced-002, structured-microtask, constraint-first, Q-target, model-access, protocol, leakage, and hash audits passed.
- Constraint reuse validation independently recomputed 32 games, 128 fixed-state diagnostics, and 200 retention probes at each of four doses.
- Results collection validated 58 explicitly declared files from six available runs; Q-SFT and Gemma 1B remain two explicit unavailable conditions with `metrics: null`.
- `git diff --check` passed; line-ending notices on existing Windows files were non-errors.

## Evidence map

- [SOURCE_RECOMMENDATIONS.md](SOURCE_RECOMMENDATIONS.md): captured source discussion.
- [results/collection_manifest.json](results/collection_manifest.json): content hashes and availability for collected run evidence.
- [results_declaration.json](results_declaration.json): explicit collection allowlist.
- [generated/tiny_overfit/manifest.json](generated/tiny_overfit/manifest.json): tiny-cell data and disjointness audit.
- [generated/structured_microtasks_v1/manifest.json](generated/structured_microtasks_v1/manifest.json): microtask bundle audit.
- [generated/constraint_first/manifest.json](generated/constraint_first/manifest.json): legal-label bundle audit.
- [generated/q_sft_frozen/manifest.json](generated/q_sft_frozen/manifest.json): frozen Q-target audit.
- [generated/capacity/gemma_1b_preflight.json](generated/capacity/gemma_1b_preflight.json): exact 1B blocker.
- [Q_SFT_FROZEN.md](Q_SFT_FROZEN.md): Q-SFT target, gate, and implementation details.
- [STRUCTURED_MICROTASKS.md](STRUCTURED_MICROTASKS.md): microtask definitions and data design.

All committed evidence is development or training evidence. Checkpoints remain local and ignored; no model weights or locked-test payloads are collected into Git.
