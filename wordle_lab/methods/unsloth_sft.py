from __future__ import annotations

import hashlib
import importlib.metadata
import math
import time
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from wordle_lab.common import MODEL_DIR, write_json, write_jsonl
from wordle_lab.methods.sft import Collator, CompletionDataset
from wordle_lab.models import assert_supported_model


UNSLOTH_BACKEND_ID = "UNSLOTH-GEMMA-SFT-001"


def select_nested_rows(rows: Sequence[dict], limit: int | None) -> list[dict]:
    """Select a deterministic, nested state dose without inspecting labels."""
    if limit is None or limit >= len(rows):
        return list(rows)
    if limit <= 0:
        raise ValueError("train state limit must be positive")
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(str(row["comparison_id"]).encode("utf-8")).hexdigest(),
    )[:limit]


def unsloth_environment() -> dict:
    packages = {}
    for name in ("unsloth", "unsloth_zoo", "torch", "transformers", "peft", "trl", "triton-windows"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "backend_id": UNSLOTH_BACKEND_ID,
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }


def train_unsloth_sft(rows: list[dict], run_dir: Path, spec: dict) -> tuple[object, object, dict]:
    """Train a 16-bit LoRA through Unsloth while preserving the existing SFT envelope."""
    assert_supported_model()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Unsloth model experiments")
    if float(spec.get("word_token_weight", 1.0)) != 1.0:
        raise ValueError("UNSLOTH-GEMMA-SFT-001 preregisters completion-only loss")

    try:
        from unsloth import FastModel
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth is not installed. Use the isolated runtime documented in the experiment report."
        ) from exc

    lora = spec["lora"]
    model, tokenizer = FastModel.from_pretrained(
        model_name=str(MODEL_DIR),
        max_seq_length=int(spec["max_length"]),
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        device_map="sequential",
        use_exact_model_name=True,
        local_files_only=True,
        fullgraph=False,
        random_state=int(spec["seed"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastModel.get_peft_model(
        model,
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(spec["seed"]),
    )

    dataset = CompletionDataset(rows, tokenizer, int(spec["max_length"]))
    generator = torch.Generator().manual_seed(int(spec["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=int(spec["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=Collator(tokenizer.pad_token_id),
        drop_last=False,
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(spec["learning_rate"]))
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
            batch = {key: value.to("cuda") for key, value in batch.items() if key != "loss_weights"}
            loss = model(**batch).loss
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            step_tokens += int(batch["attention_mask"].sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_tokens += step_tokens
        log_rows.append(
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
        "backend": unsloth_environment(),
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
        "loss_mode": "completion",
        "quantization": "none_16bit",
        "adapter": {
            "type": "lora",
            "r": int(lora["r"]),
            "alpha": int(lora["alpha"]),
            "dropout": float(lora["dropout"]),
            "target_modules": list(lora["target_modules"]),
        },
    }
    write_jsonl(run_dir / "train_metrics.jsonl", log_rows)
    write_json(run_dir / "accounting.json", accounting)
    model.config.use_cache = True
    model.eval()
    return model, tokenizer, accounting
