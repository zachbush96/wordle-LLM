from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset

from wordle_lab.common import write_json, write_jsonl
from wordle_lab.methods.sft import Collator
from wordle_lab.models import load_base_model, load_tokenizer


class PreferenceDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.samples = []
        for row in rows:
            prompt_text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            sample = {}
            for side in ("chosen", "rejected"):
                full = tokenizer.apply_chat_template(row["prompt"] + row[side], tokenize=False, add_generation_prompt=False)
                ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_length]
                labels = [-100] * min(len(prompt_ids), len(ids)) + ids[len(prompt_ids):]
                sample[f"{side}_input_ids"] = ids; sample[f"{side}_labels"] = labels
            self.samples.append(sample)

    def __len__(self): return len(self.samples)
    def __getitem__(self, index): return self.samples[index]


def _side_batch(rows: list[dict], side: str, pad: int) -> dict:
    return Collator(pad)([{"input_ids": row[f"{side}_input_ids"], "labels": row[f"{side}_labels"]} for row in rows])


def _sequence_logp(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(**{key: value for key, value in batch.items() if key != "labels"})
    logits = output.logits[:, :-1].float(); labels = batch["labels"][:, 1:]
    mask = labels != -100; safe = labels.masked_fill(~mask, 0)
    token_logp = torch.log_softmax(logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    summed = (token_logp * mask).sum(-1); mean = summed / mask.sum(-1).clamp_min(1)
    return summed, mean


def train_orpo(rows: list[dict], parent_adapter: Path, run_dir: Path, spec: dict):
    """Reference-free ORPO with chosen NLL plus a stable odds-ratio term."""
    tokenizer = load_tokenizer(parent_adapter)
    model = PeftModel.from_pretrained(load_base_model(training=True), parent_adapter, is_trainable=True)
    dataset = PreferenceDataset(rows, tokenizer, int(spec["max_length"]))
    loader = DataLoader(dataset, batch_size=int(spec["batch_size"]), shuffle=True, collate_fn=lambda batch: batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]))
    max_steps = int(spec["max_steps"]); accumulation = int(spec["gradient_accumulation_steps"]); lam = float(spec.get("lambda_or", 0.1))
    iterator = iter(loader); logs = []; started = time.perf_counter(); model.train()
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True); losses = []
        for _ in range(accumulation):
            try: raw = next(iterator)
            except StopIteration: iterator = iter(loader); raw = next(iterator)
            chosen = {key: value.to("cuda") for key, value in _side_batch(raw, "chosen", tokenizer.pad_token_id).items()}
            rejected = {key: value.to("cuda") for key, value in _side_batch(raw, "rejected", tokenizer.pad_token_id).items()}
            chosen_sum, chosen_mean = _sequence_logp(model, chosen); _, rejected_mean = _sequence_logp(model, rejected)
            nll = -chosen_sum / (chosen["labels"] != -100).sum(-1).clamp_min(1)
            log1m_chosen = torch.log1p(-torch.exp(chosen_mean).clamp(max=1 - 1e-6))
            log1m_rejected = torch.log1p(-torch.exp(rejected_mean).clamp(max=1 - 1e-6))
            log_odds_ratio = (chosen_mean - rejected_mean) - (log1m_chosen - log1m_rejected)
            loss = (nll - lam * torch.nn.functional.logsigmoid(log_odds_ratio)).mean()
            (loss / accumulation).backward(); losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        logs.append({"optimizer_step": step, "loss": sum(losses) / len(losses), "wall_time_s": time.perf_counter() - started})
        if step in {round(max_steps * value) for value in (0.25, 0.5, 0.75, 1.0)}:
            model.save_pretrained(run_dir / "checkpoints" / f"step-{step:06d}")
    final = run_dir / "checkpoints" / "final"; model.save_pretrained(final); tokenizer.save_pretrained(final)
    write_jsonl(run_dir / "train_metrics.jsonl", logs)
    model.config.use_cache = True; model.eval(); return model
