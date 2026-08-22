# Paper evidence log

This log separates implementation evidence, local empirical evidence, and external claims.

## External primary sources consulted

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685. Supports the adapter/objective distinction and frozen-base low-rank updates.
- Hong, Dragan, and Levine, *Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning*, arXiv:2411.05193 / ICLR 2025. Supports Bellman-likelihood weighted cross entropy and the absence of a separate value head.
- He et al., *Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation*, arXiv:2605.21125. Supports ACR and AVSPO virtual reward normalization. This is recent preprint evidence and should be identified as such.
- Hong, Lee, and Thorne, *ORPO: Monolithic Preference Optimization without Reference Model*, arXiv:2403.07691 / EMNLP 2024. Supports the reference-free odds-ratio objective.

## Reproducibility ledger

- NotebookLM conversation extracted: 2026-08-21.
- Frozen protocol: WORDLE-PROTOCOL-002; protocol modules are not edited by this implementation.
- Locked test: remains closed.
- Implementation validation: pending integration test run.
- Expensive adapter training: not yet executed by this implementation pass.
- Notebook comparison numbers: external/unreproduced and excluded from local score tables.

## Required experiment-card fields

Every executed cell must record method, adapter, base-model identity, parent checkpoint, seed, optimizer/token budget, data manifest/hash, protocol hash, raw game JSONL, aggregate metrics, state diagnostics, wall time, peak VRAM, and selection outcome. Negative or null results remain part of the paper record.
