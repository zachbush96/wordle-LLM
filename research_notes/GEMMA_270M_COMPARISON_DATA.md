# Gemma 3 270M matched training-data design

Date: 2026-08-21

The active study is restricted to `google/gemma-3-270m-it` at revision
`ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`. Runtime checks verify the model
identifier, revision, architecture, model type, 640 hidden size, and 18 layers.
Historical Qwen checkpoints are excluded from all current comparisons.

The dataset contains 4,096 underlying training examples and three matched
renderings (12,288 rendered rows total): reasoning single-step,
non-reasoning single-step, and non-reasoning multi-step. Every `comparison_id`
has the same state, turn, posterior, and deterministic oracle target in all
three partitions. This makes representation the intended manipulated variable.
The turn quotas are 128/1024/1024/819/614/487 for turns 1 through 6. There are
3,969 unique observation states; the 127 repetitions are unavoidable root
format anchors, distributed across training-secret provenance. The selected
labels cover 127 of the 128-word action universe.

Scale rationale: 4,096 source states are eight times the earlier 512-state
balanced curriculum and four times the 1,024-state targeted curriculum, while
remaining practical for a 270M model and three-seed ablations. “Large enough”
is an experimental assumption, not a proven saturation point; learning curves
at 1,024, 2,048, and 4,096 should be reported before making a sample-efficiency
claim.

The audit recomputes feedback from each training-only secret, rejects histories
after a solved guess, proves the secret remains in the posterior, recomputes all
oracle facts, checks the held-out 96/32 secret split, and confirms exact IDs and
targets across partitions. It also checks that only the reasoning partition has
visible rationale and that multi-step prompts reproduce the full action/feedback
history. The 512 dev probe states are evaluation-only. The locked protocol test
set is not read by the builder or audit.

Measured with the pinned Gemma tokenizer, maximum/mean full-example lengths are
263/219 tokens for reasoning single-step, 149/129 for non-reasoning single-step,
and 255/182 for non-reasoning multi-step. A max length of 320 therefore avoids
completion truncation for the current bundle.

Preference launchers derive paired oracle/hard-negative actions from these same
states, falling back to an objectively invalid prior-repeat negative for
singleton states. This preserves all 4,096 matched pairs. Both sides use the
same terminal envelope. In the reasoning partition,
both sides use neutral, structurally identical rationale wording so the learner
cannot solve the preference task from explicit “best/worse” phrases. Q-SFT is
kept separate: its builder only joins independently frozen, train-only Bellman
snapshots and rejects secret, candidate, evaluator, and unknown fields.

No training run was launched as part of data construction. The next valid
experiment is a seed-matched three-part SFT screen on development probes, then
method comparisons using the same selected representation and seed-matched SFT
parents. Locked-test access remains forbidden until the documented gates pass.
LoRA, rsLoRA, and DoRA adapter ablations have separate entrypoints and must hold
the representation, seed, optimizer-token budget, and development evaluation
fixed when compared.
