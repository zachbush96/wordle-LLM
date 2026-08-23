from __future__ import annotations

"""Matched full-parameter Gemma 270M fine-tuning for the adapter-capacity test."""

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from wordle_lab.common import MODEL_DIR, write_json, write_jsonl
from wordle_lab.methods.sft import Collator, CompletionDataset, weighted_causal_lm_loss
from wordle_lab.models import assert_supported_model, load_tokenizer


FULL_FINETUNE_BACKEND_ID = "GEMMA-270M-FULL-FINETUNE-001"
FULL_FINETUNE_EXPERIMENT_ID = "GEMMA-270M-LORA-VS-FULL-001"
FULL_FINETUNE_SMOKE_ID = "GEMMA-270M-FULL-MEMORY-SMOKE-001"
PRIMARY_MODE = "matched_primary"
SMOKE_MODE = "nonmatched_memory_smoke"
EXPECTED_MODEL_ID = "google/gemma-3-270m-it"
EXPECTED_MODEL_REVISION = "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
EXPECTED_PROTOCOL_ID = "WORDLE-PROTOCOL-002"
EXPECTED_PROTOCOL_SHA256 = "afb9884a341f51fbf9c902e07bb130c0a4d742f189aadb3dd0f9ce92fa0f681a"
EXPECTED_TRAIN_ROWS_SHA256 = "8a5741e061349243bc9467ba53254fec648b83dafb5944f65c0d61ab65466e7f"
EXPECTED_STATE_MANIFEST_SHA256 = "4ab23b5cd883d8ad9b542befadc23c2aec3a3d631b78f239bb551ca998fd6a3c"
EXPECTED_CHECKPOINT_STEPS = [150, 300, 450, 600]
EXPECTED_CHECKPOINT_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
EXPECTED_PARAMETER_COUNT = 268_098_176
EXPECTED_LORA_WRAPPED_PARAMETER_COUNT = 271_895_168
DEFAULT_VRAM_MARGIN_BYTES = 2 * 1024**3
EXPECTED_DATA_FILE_HASHES = {
    "train.jsonl": EXPECTED_TRAIN_ROWS_SHA256,
    "manifest.json": "091681fd66f3af5b1e329fe457de6ffac0247421e83a04c8d68d95489be26889",
    "state_manifest.jsonl": EXPECTED_STATE_MANIFEST_SHA256,
    "canonical.jsonl": "6bfefc11ce390048ca3cfcb8d44db4f583501d5174dc89d3140caccc49cf958f",
    "universe.json": "1256cd1c1075246251cafb4d01612dae26a73808a4915c3d88006478f3f736ac",
    "train_secrets.json": "e8ace1e06a6f35a1b600702099c029e232b6124fe76265a5cf4da2d981386a4e",
    "dev_secrets.json": "e94dea81d06f464a55ea7463b36837c998d1e405ef3f1e6e0500c78ea627c8a2",
}
EXPECTED_PROTOCOL_COMPONENTS = {
    "env.py": "9764ead89aee0332b385a63a6859b49f2bcd8317941b8d9813e4ae0621dd0871",
    "evaluator.py": "ea5d3205cfb59f81e2fe68593440a6a581e30a14dd777ca33201258b23b7b53c",
    "generation.py": "f231f10442ae8c821beb8d4390661c9a218f3c4a333be5e56cb7439daa47210b",
    "parsing.py": "018757e458b5ad71f9e726a276dadce75e53ea2032b4d9d28c51e19aa7611cd4",
    "prompting.py": "094a3170e4f97c00e07ccf35e25b1b9760b9932c5c5ed40a9a447b11a88bdd16",
    "retention.py": "7a76161392cbfd9499b51cbe611f50d9bc00c51c825016bb9588b9dcce3fe011",
}
EXPECTED_EVALUATION = {
    "split": "balanced_002_dev_32",
    "dev_games": 32,
    "diagnostic_items": 128,
    "prompt_variant": "explicit_feedback",
    "decoder": "greedy",
    "generation": {"do_sample": False, "max_new_tokens": 128, "use_cache": True},
    "allowed_words_sha256": "9df5dad1b44cf1b9e0fa7c3ebff94d69ff7efb59f8691ac88e74e1c7b3da121e",
    "retention_probes_sha256": "f510af87dfb67f3200ec39982e3470a5b08c40ac3e444f35712c603da07975c6",
}
EXPECTED_COMPARATOR_CHECKPOINT_TREES = {
    "step-000150": "6b40afa18ff3d737adafe03d5f8c282d5dbc5ee230aed4b80e8048a6ca2b59cc",
    "step-000300": "c4e6c516ca3d5fa80955da591379121a372e5ea1905c6e90fb5f433b3c04ba08",
    "step-000450": "38d1bb27cf0474d0e38b969338e595fd32ea9b51d249debb6fbe25dcba02450e",
    "step-000600": "7deadbb541d4059d5e90958cf4d83670da490b97b63512509fe45498da6c665a",
    "final": "7deadbb541d4059d5e90958cf4d83670da490b97b63512509fe45498da6c665a",
}

EXPECTED_COMPARATOR_EVIDENCE = {
    "run_id": "sft-common-balanced-word-s2026-0649b4deeb",
    "spec_sha256": "655972a100f33ca26d0e7834f602de9856f88aed89107aed67e6426f9d8c95bc",
    "dataset_manifest_sha256": "091681fd66f3af5b1e329fe457de6ffac0247421e83a04c8d68d95489be26889",
    "accounting_sha256": "4a46e116d8d998ec20c6f05ebd74b2cd8c4525feb52f9dad8cc22327940625b4",
    "summary_sha256": "5501c697996717e9a67be75e90f1ee57dbaefa90a29899b733bbdd8f0d093b9d",
    "final_adapter_tree_sha256": "7deadbb541d4059d5e90958cf4d83670da490b97b63512509fe45498da6c665a",
}


def _mismatches(actual: dict[str, Any], expected: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    return {
        f"{prefix}{key}": {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }


def validate_full_finetune_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Fail closed for either the exact primary cell or explicit non-matched smoke."""
    metadata = assert_supported_model()
    mode = spec.get("experiment_mode")
    common_expected = {
        "backend": FULL_FINETUNE_BACKEND_ID,
        "method": "full_parameter_sft",
        "representation": "common_balanced_curriculum",
        "curriculum_id": "COMMON-WORD-CURRICULUM-002",
        "seed": 2026,
        "learning_rate": 5e-5,
        "max_length": 320,
        "warmup_fraction": 0.05,
        "max_grad_norm": 1.0,
        "word_token_weight": 8.0,
        "gradient_checkpointing": False,
        "precision": "bfloat16",
        "quantization": "none_16bit",
        "optimizer": "torch.optim.AdamW",
        "scheduler": "linear_warmup_5pct_cosine",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
        "harness_selected_guess": False,
    }
    drift = _mismatches(spec, common_expected)
    if mode == PRIMARY_MODE:
        drift.update(
            _mismatches(
                spec,
                {
                    "experiment_id": FULL_FINETUNE_EXPERIMENT_ID,
                    "smoke": False,
                    "matched_comparison": True,
                    "max_steps": 600,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 1,
                    "effective_batch_size": 4,
                    "checkpoint_steps": EXPECTED_CHECKPOINT_STEPS,
                    "checkpoint_fractions": EXPECTED_CHECKPOINT_FRACTIONS,
                },
            )
        )
    elif mode == SMOKE_MODE:
        drift.update(
            _mismatches(
                spec,
                {
                    "experiment_id": FULL_FINETUNE_SMOKE_ID,
                    "smoke": True,
                    "matched_comparison": False,
                    "max_steps": 2,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "effective_batch_size": 1,
                    "checkpoint_steps": [],
                    "checkpoint_fractions": [],
                },
            )
        )
    else:
        drift["experiment_mode"] = {"expected": [PRIMARY_MODE, SMOKE_MODE], "actual": mode}

    declared_model = spec.get("model", {})
    drift.update(
        _mismatches(
            declared_model,
            {"model_id": EXPECTED_MODEL_ID, "revision": EXPECTED_MODEL_REVISION},
            "model.",
        )
    )
    if metadata.get("model_id") != EXPECTED_MODEL_ID or metadata.get("revision") != EXPECTED_MODEL_REVISION:
        drift["local_model"] = {
            "expected": {"model_id": EXPECTED_MODEL_ID, "revision": EXPECTED_MODEL_REVISION},
            "actual": {"model_id": metadata.get("model_id"), "revision": metadata.get("revision")},
        }

    data = spec.get("data", {})
    drift.update(
        _mismatches(
            data,
            {
                "rows": 512,
                "rendered_sha256": EXPECTED_TRAIN_ROWS_SHA256,
                "state_manifest_sha256": EXPECTED_STATE_MANIFEST_SHA256,
                "train_secret_count": 96,
                "dev_secret_count": 32,
                "locked_test_access": False,
            },
            "data.",
        )
    )
    if data.get("files") != EXPECTED_DATA_FILE_HASHES:
        drift["data.files"] = {"expected": EXPECTED_DATA_FILE_HASHES, "actual": data.get("files")}
    protocol = spec.get("protocol", {})
    drift.update(
        _mismatches(
            protocol,
            {"protocol_id": EXPECTED_PROTOCOL_ID, "protocol_sha256": EXPECTED_PROTOCOL_SHA256},
            "protocol.",
        )
    )
    expected_protocol_evidence = {
        "status": "passed",
        "lock_file_sha256": "afb074f6aafa5d30b16595890c1087556c0b8078c92bceacc283a296c3a462e7",
        "component_files": EXPECTED_PROTOCOL_COMPONENTS,
        "allowed_words_sha256": EXPECTED_EVALUATION["allowed_words_sha256"],
        "retention_probes_sha256": EXPECTED_EVALUATION["retention_probes_sha256"],
        "locked_test_access": False,
    }
    drift.update(_mismatches(protocol, expected_protocol_evidence, "protocol."))
    evaluation = spec.get("evaluation", {})
    if evaluation != EXPECTED_EVALUATION:
        drift["evaluation"] = {"expected": EXPECTED_EVALUATION, "actual": evaluation}
    comparator = spec.get("comparator", {})
    drift.update(_mismatches(comparator, EXPECTED_COMPARATOR_EVIDENCE, "comparator."))
    drift.update(
        _mismatches(
            comparator,
            {
                "status": "passed",
                "backend": "native_transformers_peft_lora",
                "checkpoint_adapter_tree_sha256": EXPECTED_COMPARATOR_CHECKPOINT_TREES,
                "locked_test_access": False,
            },
            "comparator.",
        )
    )
    comparison = spec.get("comparison_contract", {})
    expected_matched = mode == PRIMARY_MODE
    drift.update(
        _mismatches(
            comparison,
            {
                "matched": expected_matched,
                "comparator_backend": "native_transformers_peft_lora",
                "intended_difference": "all model parameters trainable instead of rank-16 LoRA parameters",
            },
            "comparison_contract.",
        )
    )
    if drift:
        raise ValueError(f"full fine-tune spec drift: {drift}")
    return metadata


def estimated_adamw_training_bytes(parameter_count: int, *, parameter_bytes: int = 2) -> int:
    """Planning estimate: weights, gradients, and conservatively FP32 Adam moments."""
    if parameter_count <= 0 or parameter_bytes <= 0:
        raise ValueError("parameter counts and widths must be positive")
    return parameter_count * (parameter_bytes + parameter_bytes + 4 + 4)


def full_finetune_vram_preflight(
    *,
    parameter_count: int = EXPECTED_PARAMETER_COUNT,
    margin_bytes: int = DEFAULT_VRAM_MARGIN_BYTES,
) -> dict[str, Any]:
    """Read current CUDA memory without loading the tokenizer or model."""
    if parameter_count <= 0 or margin_bytes < 0:
        raise ValueError("preflight parameter count must be positive and margin non-negative")
    parameter_estimate = estimated_adamw_training_bytes(parameter_count)
    required = parameter_estimate + margin_bytes
    if not torch.cuda.is_available():
        return {
            "status": "blocked_cuda_unavailable",
            "ready": False,
            "read_only": True,
            "model_loaded": False,
            "parameter_count": parameter_count,
            "estimated_parameter_state_bytes": parameter_estimate,
            "activation_allocator_margin_bytes": margin_bytes,
            "required_free_vram_bytes": required,
            "free_vram_bytes": None,
            "total_vram_bytes": None,
            "gpu": None,
            "bf16_supported": False,
        }
    free_bytes, total_bytes = (int(value) for value in torch.cuda.mem_get_info())
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    ready = free_bytes >= required and bf16_supported
    status = (
        "ready_for_full_allocation"
        if ready
        else "blocked_bf16_unsupported"
        if not bf16_supported
        else "blocked_insufficient_free_vram"
    )
    return {
        "status": status,
        "ready": ready,
        "read_only": True,
        "model_loaded": False,
        "parameter_count": parameter_count,
        "estimated_parameter_state_bytes": parameter_estimate,
        "activation_allocator_margin_bytes": margin_bytes,
        "required_free_vram_bytes": required,
        "free_vram_bytes": free_bytes,
        "total_vram_bytes": total_bytes,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "bf16_supported": bf16_supported,
    }


def _checkpoint_steps(max_steps: int, fractions: tuple[float, ...]) -> list[int]:
    if any(not math.isfinite(fraction) or not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("checkpoint fractions must be in (0, 1]")
    return sorted({max(1, round(max_steps * fraction)) for fraction in fractions})


def train_full_finetune(
    rows: list[dict[str, Any]],
    run_dir: Path,
    spec: dict[str, Any],
) -> tuple[object, object, dict[str, Any]]:
    """Fine-tune every Gemma parameter with the same causal objective as LoRA."""
    metadata = validate_full_finetune_spec(spec)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full-fine-tuning experiment")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the matched historical path requires CUDA BF16 support")
    preflight = full_finetune_vram_preflight()
    if not preflight["ready"]:
        raise RuntimeError(f"full fine-tune preflight blocked: {preflight['status']}")
    run_dir = Path(run_dir)
    tokenizer = load_tokenizer()
    weight = float(spec.get("word_token_weight", 1.0))
    dataset = CompletionDataset(rows, tokenizer, int(spec["max_length"]), word_token_weight=weight)
    generator = torch.Generator().manual_seed(int(spec["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=int(spec["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=Collator(tokenizer.pad_token_id),
        drop_last=False,
    )
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    ).to("cuda")
    model.config.use_cache = False
    if bool(spec.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable != total:
        raise RuntimeError("full fine-tuning did not enable every model parameter")
    if total != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"full fine-tuning parameter-count drift: expected {EXPECTED_PARAMETER_COUNT}, got {total}")
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
    fractions = tuple(float(value) for value in spec.get("checkpoint_fractions", (0.25, 0.5, 0.75, 1.0)))
    checkpoints = _checkpoint_steps(max_steps, fractions) if fractions else []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    logs: list[dict[str, Any]] = []
    optimizer_tokens = 0
    weighted_completion_tokens = 0.0

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
            loss_weights = batch.pop("loss_weights")
            if weight == 1.0:
                loss = model(**batch).loss
            else:
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                loss = weighted_causal_lm_loss(output.logits, batch["labels"], loss_weights)
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            step_tokens += int(batch["attention_mask"].sum())
            weighted_completion_tokens += float(loss_weights.sum())
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

    accounting = {
        "backend_id": FULL_FINETUNE_BACKEND_ID,
        "model": metadata,
        "train_examples": len(dataset),
        "optimizer_steps": max_steps,
        "effective_batch_size": int(spec["batch_size"]) * accumulation,
        "optimizer_tokens": optimizer_tokens,
        "wall_time_s": time.perf_counter() - started,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "estimated_weights_gradients_adam_bytes": estimated_adamw_training_bytes(total),
        "checkpoint_steps": checkpoints,
        "final_checkpoint": f"step-{max_steps:06d}" if max_steps in checkpoints else None,
        "loss_mode": "word_focused" if weight > 1 else "completion",
        "word_token_weight": weight,
        "weighted_completion_tokens": weighted_completion_tokens,
        "gradient_checkpointing": bool(spec.get("gradient_checkpointing", False)),
        "quantization": "none_16bit",
    }
    write_jsonl(run_dir / "train_metrics.jsonl", logs)
    write_json(run_dir / "accounting.json", accounting)
    model.config.use_cache = True
    model.eval()
    return model, tokenizer, accounting


def load_full_checkpoint(path: str | Path):
    """Load a saved full-model checkpoint for natural-generation evaluation."""
    assert_supported_model()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("matched full-checkpoint evaluation requires CUDA BF16 support")
    checkpoint = Path(path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    dtype = torch.bfloat16
    tokenizer = load_tokenizer(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    ).to("cuda")
    checkpoint_identity = {
        "model_type": getattr(model.config, "model_type", None),
        "architecture": list(getattr(model.config, "architectures", []) or []),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    expected_identity = {
        "model_type": "gemma3_text",
        "architecture": ["Gemma3ForCausalLM"],
        "parameters": EXPECTED_PARAMETER_COUNT,
    }
    if checkpoint_identity != expected_identity:
        raise RuntimeError(f"full checkpoint identity drift: expected {expected_identity}, got {checkpoint_identity}")
    model.config.use_cache = True
    model.eval()
    return model, tokenizer
