# NotebookLM Technique Inventory

Source notebook: `Group Relative Policy Optimization: Mechanics, Diagnosis, and Implementation`, read 2026-08-21 from the user-supplied shared notebook.

## Scope extracted from the conversation

The conversation separates LoRA (a parameter-efficient adapter mechanism) from
the training objective. It proposes comparisons among SFT, DPO, ORPO, GRPO,
Q-SFT, and a staged SFT-to-GRPO pipeline. It also introduces Advantage Collapse
Rate (ACR) and Adaptive Virtual Sample Policy Optimization (AVSPO). Other ideas
include partial-game-state curricula, structured character-list prompts,
multi-component rewards, and a harness-fixed opening guess.

## Canonical implementation decisions

| Item | Implementation status | WORDLE-PROTOCOL-002 decision |
|---|---|---|
| LoRA | Adapter factory/configuration | Allowed; the base model stays frozen. |
| SFT | Existing completion/word-weighted implementation | Allowed. |
| DPO | Existing reference-regulated implementation plus registry/config | Allowed, offline training data only. |
| ORPO | Existing reference-free implementation plus registry/config | Allowed, offline training data only. |
| GRPO | Existing on-policy trainer plus stability helpers | Allowed on training splits. |
| Q-SFT | New Bellman-likelihood weighted SFT implementation | Allowed; canonical evaluator does not perform two-policy reranking. |
| SFT to GRPO | New staged plan/orchestrator validation | Allowed with an explicit parent checkpoint and entropy/format gates. |
| ACR | New reward-group diagnostic | Allowed and logged. |
| AVSPO | New virtual-reward advantage helper | Allowed as an ablation; virtual values affect normalization only and never masquerade as environment outcomes. |
| Partial-game curriculum | Already implemented in canonical state generation/curricula | Allowed; provenance remains training-only. |
| Multi-granularity reward | New component-ledger implementation | Allowed for training; evaluation remains reward-free. |
| Structured letter-list prompt | Catalogued but not enabled | It would change the frozen prompt hash; requires a separately named protocol and baseline. |
| Hardcoded opening guess | Explicitly rejected | Harness-selected guesses violate natural generation and the no-cheating study contract. |

## Notebook-reported comparison (not reproduced)

The notebook claims a 100-target Gemma 3 270M comparison with win rates of
4.2% SFT, 6.8% DPO, 7.1% ORPO, 11.3% GRPO, 13.8% Q-SFT, and 15.6% hybrid
SFT+GRPO. Treat these as context and hypotheses only. They are not measurements
from this repository and must never be mixed with local development results as
though they shared data, prompts, seeds, or evaluation code.

## Evidence caveats for the paper

- The current repository's strongest verified development adapter is Qwen2.5-1.5B, so new Qwen experiments are cross-family evidence rather than a clean Gemma-270M replication.
- Q-SFT's paper policy extraction combines a behavior policy with learned Q-like probabilities. Canonical evaluation forbids harness-side combination/reranking, so the planned cell evaluates the trained model's natural generation and must state this difference.
- AVSPO was published for sparse binary verified rewards. Wordle training currently uses shaped continuous components; ACR/AVSPO therefore require a baseline GRPO ablation and bias monitoring rather than an assumed transfer.
- No notebook-reported win rate has been reproduced here.

The NotebookLM rubric is available as an optional GRPO training reward in
`configs/methods/grpo_notebooklm_reward.yaml`. It records format, validity,
completion, repetition, green-position, missing-yellow, and gray-reuse terms.
The default historical `wordle-shaped-v1` reward is unchanged, which lets the
new rubric be tested as a matched ablation rather than quietly changing old runs.
