# Gemma 3 270M coverage scaling experiment

Date: 2026-08-23

Experiments: `GEMMA-270M-COVERAGE-MAX-001`, `GEMMA-270M-COVERAGE-STACK-002`, `GEMMA-270M-COVERAGE-GROWTH-003`, and `GEMMA-270M-COVERAGE-FORCED-15K-004`

Decision: **Increasing unique multi-turn state coverage produced meaningful Gemma 3 270M growth. A user-requested override of the 10K stop continued to 15,360 examples and reached 19/32 wins with perfect compliance, the new gameplay development high. The 7,168 checkpoint remains better balanced on singleton, repeat, and target diagnostics. No checkpoint qualifies for locked-test promotion.**

## What was scaled

This experiment scaled the techniques that had previously produced meaningful movement:

1. balanced coverage of feedback-conditioned states rather than generic instruction rows;
2. full-parameter tuning of Gemma 3 270M rather than LoRA;
3. 8x loss weight on the five-letter action tokens;
4. exact Gemma-native chat templating and natural greedy generation.

The first phase contained 4,096 rows and 4,064 unique non-root states: 32 root-format examples, 1,280 turn-2 states, 1,024 low-posterior states, 1,536 true singleton states, and 224 later broad states. All 96 training singleton targets were covered. The model saw each row once: 1,024 optimizer steps at batch size 4, not extra epochs.

Because 4,096 examples materially improved wins and singleton generalization, a second phase added 4,096 new states disjoint from phase one: 1,024 turn-2, 1,792 low-posterior, 768 true singleton, and 512 later broad states. It continued the trained weights for one pass at a conservative learning rate of `1e-5`. Optimizer state was not available from phase one, so phase two explicitly used a fresh AdamW optimizer and scheduler. The two phases provide 8,192 distinct examples in total.

Both datasets passed feedback recomputation, prompt/completion fidelity, split, state uniqueness, target-balance, and locked-test-isolation audits. There were zero non-root development-history collisions. The locked test was never read.

## Development results

All checkpoints used the same frozen 32 development games, 128 state probes including 74 singleton and 58 turn-2 states, and 200 retention probes. There was no output repair, vocabulary masking, candidate injection, reranking, or repeat ban.

| Unique examples | Wins | Compliance | Invalid | Repeats | Posterior violations | Turn-2 violations | Singleton | Target accuracy | Retention |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prior 512-row full tune | 14/32 | 100.0% | 0.0% | **11.4%** | **60.9%** | **34.5%** | 2/74 | **36.7%** | 0.0% |
| 1,024 | 7/32 | 99.4% | 0.6% | 21.8% | 87.5% | 77.6% | 2/74 | 3.9% | 0.0% |
| 2,048 | 12/32 | 100.0% | 0.0% | 21.9% | 77.3% | 67.2% | 6/74 | 10.9% | 0.0% |
| 3,072 | 15/32 | 100.0% | 0.0% | 17.7% | 75.8% | 69.0% | 8/74 | 17.2% | 0.0% |
| 4,096 | **17/32** | 100.0% | 0.0% | 14.1% | 73.4% | 67.2% | 10/74 | 19.5% | 0.0% |
| 5,120 | 15/32 | 100.0% | 0.0% | 18.3% | 68.8% | 58.6% | **13/74** | **23.4%** | 0.0% |
| 6,144 | **17/32** | 94.6% | 5.4% | 14.3% | 70.3% | 56.9% | 9/74 | 19.5% | 0.0% |
| 7,168 | **17/32** | 100.0% | 0.0% | 16.8% | 70.3% | 60.3% | 10/74 | 21.1% | 0.0% |
| 8,192 | **17/32** | 94.6% | 5.4% | 14.4% | 69.5% | 60.3% | 11/74 | 21.9% | 0.0% |
| 10,240 | **17/32** | 100.0% | 0.0% | 18.1% | 71.1% | 56.9% | 9/74 | 18.0% | 0.0% |
| 12,288 | 18/32 | 95.3% | 4.7% | 14.2% | 70.3% | 56.9% | 10/74 | 19.5% | 0.0% |
| 15,360 | **19/32** | 100.0% | 0.0% | 18.6% | 70.3% | **53.4%** | 9/74 | 18.8% | 0.0% |

## Interpretation

Unique coverage, not more epochs, was the useful scaling axis. From the prior full-tune result to 4,096 unique rows, wins increased from 14 to 17 and singleton accuracy increased from 2/74 to 10/74 while preserving perfect compliance. The monotonic first-phase movement in wins and singleton accuracy is meaningful development evidence that broad feedback-conditioned coverage teaches behavior the 512-row repeated curriculum did not.

The gain is incomplete. Overall posterior violations worsened from 60.9% to 73.4%, turn-2 violations nearly doubled from 34.5% to 67.2%, and retention remained zero. This indicates that broader coverage improved answer recovery and game completion without learning a reliably legal posterior policy.

The legality-heavy second phase partially repaired that tradeoff but plateaued. At 5,120 examples, singleton accuracy peaked at 13/74 and turn-2 violations improved to 58.6%, but wins fell to 15. At 7,168 examples, the model matched the 4,096 checkpoint's 17 wins and 10/74 singleton accuracy while lowering overall and turn-2 violations to 70.3% and 60.3%; repeats rose to 16.8%. The full 8,192 endpoint did not add wins and failed the terminal-compliance gate.

## Conditional 10K-20K ladder

A third audited bundle contained 13,312 additional examples, all non-root and disjoint from the earlier 8,160 distinct training states and all development histories. The planned milestones were 10,240, 12,288, 15,360, and 20,480 cumulative unique examples. Because nearly all distinct turn-2 histories were already consumed, the full ladder comprised 48 remaining turn-2 states, 7,274 low-posterior states, 3,994 singleton states, and 1,996 later broad states.

Training continued from the compliant 7,168 checkpoint using one continuous fresh AdamW trajectory at `5e-6`; the parent had no optimizer state. The declared stop policy required at least 99% compliance, no win regression, bounded diagnostic regressions, and either more wins or a material singleton, legality, repeat, or target-accuracy gain.

At 10,240 examples, wins and compliance held at 17/32 and 100%. Turn-2 violations improved from 60.3% to 56.9%, only 3.4 percentage points. Meanwhile singleton accuracy fell from 10/74 to 9/74, action-target accuracy fell from 21.1% to 18.0%, overall violations rose from 70.3% to 71.1%, and repeats rose from 16.8% to 18.1%. No declared improvement threshold was met, so the conditional run stopped as specified.

## Forced 15K override

The user subsequently requested the 15K condition despite the earlier stop. The exact 10,240 checkpoint was hash-authenticated and continued over dataset rows 3,072 through 8,191, providing 5,120 additional unseen, disjoint examples. Because the stopped run did not save optimizer state, the forced continuation explicitly restarted AdamW and a 5%-warmup cosine schedule at `5e-6`. This makes the weight continuation exact but the optimizer continuation discontinuous.

At 12,288 examples, wins increased to 18/32 and singleton accuracy recovered to 10/74, but compliance fell to 95.3% with 4.7% invalid guesses. At 15,360, reliability recovered: 19/32 wins, 100% compliance, zero invalid guesses, and 53.4% turn-2 violations. This is meaningful gameplay growth over the 7,168 and 10,240 checkpoints. It is not a complete strategic improvement: singleton accuracy is 9/74, repeats are 18.6%, target accuracy is 18.8%, overall violations remain 70.3%, and retention remains zero.

## Decision

Keep three development checkpoints for distinct purposes:

- `artifacts/runs/gemma-270m-coverage-max-s2026-b452e4f6a5/checkpoints/step-001024` is the clean 4,096-example early coverage leader.
- `artifacts/runs/gemma-270m-coverage-stack-s2026-99814e6413/checkpoints/coverage-007168` is the 7,168-example balanced coverage checkpoint, with the same wins/compliance/singleton result and somewhat better posterior legality.
- `artifacts/runs/gemma-270m-coverage-forced-15k-s2026-4689310577/checkpoints/coverage-015360` is the 15,360-example gameplay leader at 19/32 wins with perfect compliance.

Do not promote any checkpoint or open the locked test: turn-2 violations remain far above 30%, singleton solving is weak, retention is zero, and the result is a single seed with a restarted optimizer. The 15K result reopens the coverage-growth hypothesis for a controlled 20K endpoint, but replication across three matched seeds remains required before any promotion claim.
