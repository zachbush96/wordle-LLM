# Gemma 3 270M Matched Training-Data Design

Date: 2026-08-21

The active study uses only `google/gemma-3-270m-it` at revision
`ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`. Runtime checks verify the model
identifier, revision, architecture, model type, 640 hidden size, and 18 layers.
Historical Qwen checkpoints are deliberately excluded from current comparisons.

The dataset contains 4,096 underlying training examples, rendered three matched
ways (12,288 rows total): reasoning single-step, non-reasoning single-step, and
non-reasoning multi-step. Each `comparison_id` keeps the same state, turn,
posterior, and deterministic oracle target in every partition. That makes the
representation—not the underlying Wordle decision—the intended variable.

The turn quotas for turns 1 through 6 are 128/1024/1024/819/614/487. There are
3,969 unique observation states. The other 127 rows are necessary root-format
anchors and remain distributed across training-secret provenance. Selected labels
cover 127 of the 128-word action universe.

Why this scale? 4,096 source states are eight times the earlier 512-state
balanced curriculum and four times the 1,024-state targeted curriculum, while
still being practical for a 270M model and three-seed ablations. “Large enough”
is an experimental assumption, not a demonstrated saturation point. Report
learning curves at 1,024, 2,048, and 4,096 before claiming sample efficiency.

The audit recomputes feedback from each training-only secret, rejects histories
that continue after a solve, confirms the secret remains in the posterior,
recomputes oracle facts, checks the held-out 96/32 secret split, and confirms
matching IDs and targets across partitions. It also confirms that only the
reasoning partition shows a rationale and that multi-step prompts replay the
complete action/feedback history. The 512 development probes are evaluation-only.
Neither the builder nor the audit reads the locked protocol test set.

Measured with the pinned Gemma tokenizer, maximum/mean full-example lengths are
263/219 tokens for reasoning single-step, 149/129 for non-reasoning single-step,
and 255/182 for non-reasoning multi-step. A max length of 320 therefore avoids
completion truncation for the current bundle.

Preference launchers derive oracle/hard-negative pairs from the same states. For
singleton states, they fall back to an objectively invalid prior-repeat negative;
that preserves all 4,096 matched pairs. Both sides use the same terminal envelope.
In the reasoning partition, neutral and structurally identical rationale wording
prevents the learner from spotting explicit “best/worse” cues. Q-SFT stays
separate: its builder joins only independently frozen, train-only Bellman
snapshots and rejects secret, candidate, evaluator, and unknown fields.

Building the data did not launch training. The next valid experiment is a
seed-matched, three-part SFT screen on development probes, followed by method
comparisons using the selected representation and seed-matched SFT parents.
Locked-test access remains forbidden until the documented gates pass. LoRA,
rsLoRA, and DoRA ablations have separate entrypoints and must hold the
representation, seed, optimizer-token budget, and development evaluation fixed.
