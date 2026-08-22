from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset

from wordle_lab.common import write_json, write_jsonl
from wordle_lab.methods.adapters import attach_adapter, normalize_adapter_config, validate_trainable_targets
from wordle_lab.models import load_base_model, load_tokenizer


class CompletionDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int, word_token_weight: float = 1.0):
        if word_token_weight < 1:
            raise ValueError("word_token_weight must be at least 1")
        self.samples = []
        for row in rows:
            prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            full_messages = row["prompt"] + row["completion"]
            full = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            tokenized = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
            full_ids = tokenized["input_ids"]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError("chat template prompt is not a prefix of the completed conversation")
            full_ids = full_ids[:max_length]
            labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
            if not any(label != -100 for label in labels):
                raise RuntimeError(f"completion fully truncated for {row['example_id']}")
            loss_weights = [0.0 if label == -100 else 1.0 for label in labels]
            target_word = row.get("target_word")
            if target_word is None:
                content = row["completion"][0]["content"]
                target_word = content.rsplit(":", 1)[-1].strip().split()[0]
            if word_token_weight > 1:
                word_start = full.rfind(target_word)
                if word_start < 0:
                    raise RuntimeError(f"target word absent from rendered completion: {row['example_id']}")
                word_end = word_start + len(target_word)
                offsets = tokenized.get("offset_mapping")
                if offsets is None:
                    raise RuntimeError("word-focused loss requires tokenizer offset mappings")
                weighted = 0
                for index, (start, end) in enumerate(offsets[: len(full_ids)]):
                    if labels[index] != -100 and end > word_start and start < word_end:
                        loss_weights[index] = float(word_token_weight)
                        weighted += 1
                if not weighted:
                    raise RuntimeError(f"could not identify target tokens: {row['example_id']}")
            self.samples.append({"input_ids": full_ids, "labels": labels, "loss_weights": loss_weights})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict]):
        width = max(len(row["input_ids"]) for row in rows)
        input_ids, labels, attention, loss_weights = [], [], [], []
        for row in rows:
            pad = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * pad)
            labels.append(row["labels"] + [-100] * pad)
            row_weights = row.get("loss_weights", [0.0 if label == -100 else 1.0 for label in row["labels"]])
            loss_weights.append(row_weights + [0.0] * pad)
            attention.append([1] * len(row["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
        }


def weighted_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Token-normalized causal LM loss with explicit action-token emphasis."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction="none"
    ).view_as(shift_labels)
    active_weights = shift_weights * shift_labels.ne(-100)
    return (token_losses * active_weights).sum() / active_weights.sum().clamp_min(1.0)


def train_sft(rows: list[dict], run_dir: Path, spec: dict) -> tuple[object, dict]:
    tokenizer = load_tokenizer()
    word_token_weight = float(spec.get("word_token_weight", 1.0))
    dataset = CompletionDataset(rows, tokenizer, int(spec["max_length"]), word_token_weight=word_token_weight)
    generator = torch.Generator().manual_seed(int(spec["seed"]))
    loader = DataLoader(
        dataset, batch_size=int(spec["batch_size"]), shuffle=spec["representation"] != "mixed_curriculum", generator=generator,
        collate_fn=Collator(tokenizer.pad_token_id), drop_last=False,
    )
    model = load_base_model(training=True)
    adapter_config = normalize_adapter_config(spec)
    parent = spec.get("parent_checkpoint", "base")
    if parent != "base":
        model = PeftModel.from_pretrained(model, parent, is_trainable=True)
        adapter_validation = validate_trainable_targets(model, adapter_config["target_modules"])
    else:
        model, adapter_config, adapter_validation = attach_adapter(model, adapter_config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]))
    max_steps = int(spec["max_steps"])
    accumulation = int(spec["gradient_accumulation_steps"])
    warmup = max(1, int(max_steps * 0.05))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    log_rows = []
    optimizer_tokens = 0
    weighted_target_tokens = 0.0
    checkpoints = sorted({max(1, round(max_steps * fraction)) for fraction in (0.25, 0.5, 0.75, 1.0)})
    iterator = iter(loader)
    for step in range(1, max_steps + 1):
        losses = []
        step_tokens = 0
        for _ in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to("cuda") for key, value in batch.items()}
            loss_weights = batch.pop("loss_weights")
            if word_token_weight == 1.0:
                output = model(**batch)
                loss = output.loss
            else:
                output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = weighted_causal_lm_loss(output.logits, batch["labels"], loss_weights)
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            step_tokens += int(batch["attention_mask"].sum())
            weighted_target_tokens += float(loss_weights.sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_tokens += step_tokens
        log_rows.append({"optimizer_step": step, "train_loss": sum(losses) / len(losses), "learning_rate": scheduler.get_last_lr()[0], "optimizer_tokens": optimizer_tokens, "wall_time_s": time.perf_counter() - started})
        if step in checkpoints:
            path = run_dir / "checkpoints" / f"step-{step:06d}"
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
    adapter = run_dir / "checkpoints" / "final"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    accounting = {
        "train_examples": len(dataset), "optimizer_steps": max_steps, "effective_batch_size": int(spec["batch_size"]) * accumulation,
        "optimizer_tokens": optimizer_tokens, "wall_time_s": time.perf_counter() - started,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()), "trainable_parameters": trainable,
        "total_parameters": total, "trainable_fraction": trainable / total, "checkpoint_steps": checkpoints,
        "loss_mode": "word_focused" if word_token_weight > 1 else "completion",
        "word_token_weight": word_token_weight,
        "weighted_completion_tokens": weighted_target_tokens,
        "adapter": adapter_config,
        "adapter_validation": adapter_validation,
    }
    write_jsonl(run_dir / "train_metrics.jsonl", log_rows)
    write_json(run_dir / "accounting.json", accounting)
    model.config.use_cache = True
    model.eval()
    return model, accounting
