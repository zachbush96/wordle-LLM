from __future__ import annotations

"""Protocol-safe Q-SFT utilities and an offline LoRA training entry point.

The paper objective represents Q-values as token probabilities and trains them
with a soft-label cross entropy.  This module deliberately does not implement
the paper's two-model policy-extraction rule: WORDLE-PROTOCOL-002 evaluates the
trained model by natural generation, without inference-time reranking.
"""

import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset

from wordle_lab.common import write_json, write_jsonl
from wordle_lab.methods.adapters import attach_adapter
from wordle_lab.methods.sft import Collator, CompletionDataset
from wordle_lab.models import load_base_model, load_tokenizer


_FORBIDDEN_DATA_FIELDS = {
    "allowed_words",
    "answer",
    "candidate_words",
    "locked_test_answer",
    "posterior_candidates",
    "secret",
    "secret_answer",
    "source_state",
    "test_answer",
}


def bellman_likelihood_target(
    reward: float | torch.Tensor,
    discount: float,
    next_q_probabilities: Sequence[float] | torch.Tensor | None = None,
    next_behavior_probabilities: Sequence[float] | torch.Tensor | None = None,
    *,
    terminal: bool,
) -> torch.Tensor:
    """Compute the Q-SFT Bellman-likelihood target from Equation 3.

    For a nonterminal transition this is
    ``r + gamma * max_a p_bar(a|s') / pi_beta(a|s')``.  The probabilities
    must describe the same action support.  Targets outside [0, 1] are not
    clipped here: callers must normalize rewards as required by the paper's
    bounded-return assumption, making invalid experiment data visible.
    """

    if not 0.0 <= float(discount) <= 1.0:
        raise ValueError("discount must be in [0, 1]")
    reward_tensor = torch.as_tensor(reward, dtype=torch.float32)
    if not torch.isfinite(reward_tensor).all():
        raise ValueError("reward must be finite")
    if terminal:
        return reward_tensor
    if next_q_probabilities is None or next_behavior_probabilities is None:
        raise ValueError("nonterminal transitions require both next-state probability vectors")
    q = torch.as_tensor(next_q_probabilities, dtype=torch.float32)
    behavior = torch.as_tensor(next_behavior_probabilities, dtype=torch.float32)
    if q.ndim != 1 or behavior.ndim != 1 or q.numel() == 0 or q.shape != behavior.shape:
        raise ValueError("next-state probability vectors must be nonempty, one-dimensional, and equally sized")
    if not torch.isfinite(q).all() or not torch.isfinite(behavior).all():
        raise ValueError("next-state probability vectors must be finite")
    if torch.any(q < 0) or torch.any(q > 1):
        raise ValueError("next_q_probabilities must be in [0, 1]")
    if torch.any(behavior <= 0) or torch.any(behavior > 1):
        raise ValueError("next_behavior_probabilities must be in (0, 1]")
    return reward_tensor + float(discount) * torch.max(q / behavior)


def q_sft_soft_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    bellman_targets: torch.Tensor,
) -> torch.Tensor:
    """Causal-LM form of Q-SFT's uniformly smoothed WCE objective.

    The observed action token gets mass ``B*p_bar`` and every other vocabulary
    token gets ``(1 - B*p_bar) / (|A| - 1)``. Prompt/padding labels of -100 are
    ignored. A target may be one scalar per sequence or one value per token.
    """

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits must be [batch, length, vocab] and labels [batch, length]")
    if logits.size(-1) < 2:
        raise ValueError("Q-SFT needs an action space of at least two tokens")
    targets = torch.as_tensor(bellman_targets, dtype=torch.float32, device=logits.device)
    if targets.ndim == 1:
        if targets.shape[0] != logits.shape[0]:
            raise ValueError("sequence targets must have one value per batch item")
        targets = targets[:, None].expand_as(labels)
    elif targets.shape != labels.shape:
        raise ValueError("token targets must have the same shape as labels")
    if not torch.isfinite(targets).all() or torch.any(targets < 0) or torch.any(targets > 1):
        raise ValueError("bellman targets must be finite probabilities in [0, 1]")

    shift_log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
    shift_labels = labels[:, 1:]
    shift_targets = targets[:, 1:]
    active = shift_labels.ne(-100)
    if not active.any():
        raise ValueError("batch has no completion tokens")
    safe_labels = shift_labels.masked_fill(~active, 0)
    observed_logp = shift_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    other_mean_logp = (shift_log_probs.sum(-1) - observed_logp) / (logits.size(-1) - 1)
    token_loss = -(shift_targets * observed_logp + (1.0 - shift_targets) * other_mean_logp)
    return token_loss.masked_select(active).mean()


def _resolved_target(row: dict[str, Any], discount: float) -> float:
    if "bellman_target" in row:
        target = float(row["bellman_target"])
    else:
        if "reward" not in row or "terminal" not in row:
            raise ValueError("each row needs bellman_target or reward plus terminal")
        target = float(
            bellman_likelihood_target(
                row["reward"],
                discount,
                row.get("next_q_probabilities"),
                row.get("next_behavior_probabilities"),
                terminal=bool(row["terminal"]),
            )
        )
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError(
            "Bellman target must be in [0, 1]; normalize rewards/returns instead of silently clipping"
        )
    return target


def validate_q_sft_rows(rows: list[dict[str, Any]], discount: float) -> list[float]:
    """Validate static offline transitions and return their Bellman targets."""

    if not rows:
        raise ValueError("Q-SFT requires at least one offline transition")
    targets: list[float] = []
    for index, row in enumerate(rows):
        forbidden = _FORBIDDEN_DATA_FIELDS.intersection(row)
        if forbidden:
            raise ValueError(f"row {index} contains forbidden evaluator/candidate data: {sorted(forbidden)}")
        if not isinstance(row.get("prompt"), list) or not row["prompt"]:
            raise ValueError(f"row {index} prompt must be a nonempty chat-message list")
        if not isinstance(row.get("completion"), list) or not row["completion"]:
            raise ValueError(f"row {index} completion must be a nonempty chat-message list")
        for message in row["prompt"] + row["completion"]:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
                raise ValueError(f"row {index} contains an invalid chat message")
        if any(message["role"] != "assistant" for message in row["completion"]):
            raise ValueError(f"row {index} completion messages must have assistant role")
        targets.append(_resolved_target(row, discount))
    return targets


class QSFTDataset(Dataset):
    """Completion-only tokenization plus one frozen Bellman target per row."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int, discount: float):
        targets = validate_q_sft_rows(rows, discount)
        completion_data = CompletionDataset(rows, tokenizer, max_length, word_token_weight=1.0)
        self.samples = []
        for sample, target in zip(completion_data.samples, targets, strict=True):
            self.samples.append({**sample, "bellman_target": target})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class QSFTCollator:
    def __init__(self, pad_token_id: int):
        self.completion_collator = Collator(pad_token_id)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.completion_collator(rows)
        batch["bellman_targets"] = torch.tensor(
            [row["bellman_target"] for row in rows], dtype=torch.float32
        )
        return batch


def train_q_sft(
    rows: list[dict[str, Any]],
    parent_adapter: Path | None,
    run_dir: Path,
    spec: dict[str, Any],
) -> tuple[object, dict[str, Any]]:
    """Train a Q-SFT LoRA on frozen offline Bellman-likelihood targets.

    ``rows`` must contain either a normalized ``bellman_target`` or sufficient
    values to compute a snapshot target (reward, terminal, and for nonterminal
    rows the aligned next-Q and behavior-policy probability vectors). Updating
    those snapshots is intentionally a separate data-generation stage so this
    function never consults the evaluator, answer list, or candidate set.
    """

    run_dir = Path(run_dir)
    discount = float(spec.get("discount", 0.99))
    tokenizer = load_tokenizer(parent_adapter)
    dataset = QSFTDataset(rows, tokenizer, int(spec["max_length"]), discount)
    generator = torch.Generator().manual_seed(int(spec["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=int(spec["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=QSFTCollator(tokenizer.pad_token_id),
        drop_last=False,
    )

    model = load_base_model(training=True)
    if parent_adapter is not None:
        model = PeftModel.from_pretrained(model, parent_adapter, is_trainable=True)
    else:
        model, normalized_adapter, adapter_validation = attach_adapter(model, spec)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]))
    max_steps = int(spec["max_steps"])
    accumulation = int(spec["gradient_accumulation_steps"])
    warmup = max(1, int(max_steps * float(spec.get("warmup_fraction", 0.05))))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    logs: list[dict[str, Any]] = []
    optimizer_tokens = 0
    checkpoints = sorted({max(1, round(max_steps * fraction)) for fraction in (0.25, 0.5, 0.75, 1.0)})

    for step in range(1, max_steps + 1):
        losses: list[float] = []
        step_tokens = 0
        for _ in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to("cuda") for key, value in batch.items()}
            targets = batch.pop("bellman_targets")
            batch.pop("loss_weights")
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = q_sft_soft_cross_entropy(output.logits, batch["labels"], targets)
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            step_tokens += int(batch["attention_mask"].sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec.get("max_grad_norm", 1.0)))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_tokens += step_tokens
        logs.append(
            {
                "optimizer_step": step,
                "train_loss": sum(losses) / len(losses),
                "learning_rate": scheduler.get_last_lr()[0],
                "optimizer_tokens": optimizer_tokens,
                "wall_time_s": time.perf_counter() - started,
            }
        )
        if step in checkpoints:
            checkpoint = run_dir / "checkpoints" / f"step-{step:06d}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)

    final = run_dir / "checkpoints" / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    accounting = {
        "method": "q_sft",
        "objective": "bellman_likelihood_uniform_wce",
        "bellman_targets": "frozen_offline_snapshots",
        "canonical_evaluation_policy": "trained_model_direct_generation",
        "paper_policy_extraction_enabled": False,
        "discount": discount,
        "train_examples": len(dataset),
        "optimizer_steps": max_steps,
        "effective_batch_size": int(spec["batch_size"]) * accumulation,
        "optimizer_tokens": optimizer_tokens,
        "wall_time_s": time.perf_counter() - started,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "checkpoint_steps": checkpoints,
        "mean_bellman_target": sum(sample["bellman_target"] for sample in dataset.samples) / len(dataset),
    }
    if parent_adapter is None:
        accounting.update({"adapter": normalized_adapter, "adapter_validation": adapter_validation})
    write_jsonl(run_dir / "train_metrics.jsonl", logs)
    write_json(run_dir / "accounting.json", accounting)
    model.config.use_cache = True
    model.eval()
    return model, accounting
