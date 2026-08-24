from __future__ import annotations

"""Double the strongest 600-step Gemma 270M run with an audited continuation phase."""

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from wordle_lab.analysis.state_diagnostics import run_state_diagnostics
from wordle_lab.common import (
    ARTIFACTS,
    DATA,
    ROOT,
    canonical_json,
    read_json,
    read_jsonl,
    set_seed,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from wordle_lab.data.canonical import generate_canonical_states
from wordle_lab.experiments.intervention_sweep import _explicit_feedback_messages
from wordle_lab.methods.sft import Collator, CompletionDataset, weighted_causal_lm_loss
from wordle_lab.models import assert_supported_model, load_tokenizer
from wordle_lab.protocol import generation
from wordle_lab.protocol.evaluator import evaluate
from wordle_lab.protocol.retention import evaluate_retention

from .full_finetune import EXPECTED_PARAMETER_COUNT, full_finetune_vram_preflight, load_full_checkpoint
from .full_finetune_experiment import (
    DEFAULT_DATA,
    EXPECTED_ALLOWED_WORDS_SHA256,
    EXPECTED_RETENTION_SHA256,
    _binding_view,
    _project_path,
    _tree_digest,
    audit_balanced_source,
    audit_protocol,
)


EXPERIMENT_ID = "GEMMA-270M-FULL-FINETUNE-CONTINUATION-002"
BACKEND_ID = "GEMMA-270M-FULL-FINETUNE-CONTINUATION-001"
PARENT_RUN_ID = "full-finetune-balanced-word-primary-s2026-57ba532ae7"
PARENT_RUN = ARTIFACTS / "runs" / PARENT_RUN_ID
PARENT_CHECKPOINT = PARENT_RUN / "checkpoints" / "step-000600"
PARENT_TREE_SHA256 = "a2a1df120cdc370fab91ebc2e1f4b1babcd2affe3aafb39d60cb83d15158c2e5"
PARENT_ARTIFACT_HASHES = {
    "spec.json": "f4327445a8415bf9fb71685d779ce582fcb772f70fe63b3a0588f71053ed3e2a",
    "summary.json": "3c73298b88e00827057e1be4016f743eaa60a9ec0e5373c04898b3ad49ec9368",
    "eval-step-000600-summary.json": "8074f82255e4e523b36165b32a105820bf47291b84f3820877409416e7a1ac0d",
}
CONTINUATION_STEPS = 600
PARENT_STEPS = 600
TOTAL_CHECKPOINT_STEPS = [750, 900, 1050, 1200]
RELATIVE_CHECKPOINT_STEPS = [150, 300, 450, 600]


def audit_parent(directory: Path = PARENT_RUN) -> dict[str, Any]:
    directory = Path(directory)
    observed = {name: sha256_file(directory / name) for name in PARENT_ARTIFACT_HASHES}
    if observed != PARENT_ARTIFACT_HASHES:
        raise AssertionError(f"full-finetune parent artifact drift: {observed}")
    tree_hash, files = _tree_digest(directory / "checkpoints" / "step-000600")
    if tree_hash != PARENT_TREE_SHA256:
        raise AssertionError(f"full-finetune parent checkpoint drift: {tree_hash}")
    parent_spec = read_json(directory / "spec.json")
    parent_eval = read_json(directory / "eval-step-000600-summary.json")
    expected = {
        "seed": 2026,
        "max_steps": 600,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "word_token_weight": 8.0,
        "locked_test_access": False,
    }
    drift = {key: (value, parent_spec.get(key)) for key, value in expected.items() if parent_spec.get(key) != value}
    if drift:
        raise AssertionError(f"full-finetune parent recipe drift: {drift}")
    if parent_eval.get("locked_test_access") is not False or parent_eval.get("split") != "balanced_002_dev_32":
        raise AssertionError("full-finetune parent evaluation boundary drift")
    return {
        "status": "passed",
        "run_id": PARENT_RUN_ID,
        "run_directory": _project_path(directory),
        "checkpoint": "step-000600",
        "checkpoint_directory": _project_path(directory / "checkpoints" / "step-000600"),
        "checkpoint_tree_sha256": tree_hash,
        "checkpoint_files": files,
        "artifact_hashes": observed,
        "optimizer_state_available": False,
        "scheduler_state_available": False,
        "rng_state_available": False,
        "parent_optimizer_steps": PARENT_STEPS,
        "parent_metrics": {
            "wins": parent_eval["gameplay"]["wins"],
            "terminal_marker_compliance": parent_eval["gameplay"]["terminal_marker_compliance"],
            "repeat_guess_rate": parent_eval["gameplay"]["repeat_guess_rate"],
            "posterior_constraint_violation_rate": parent_eval["diagnostics"]["posterior_constraint_violation_rate"],
            "turn_2_posterior_constraint_violation_rate": parent_eval["diagnostics"]["by_turn"]["2"]["posterior_constraint_violation_rate"],
            "singleton_answer_accuracy": parent_eval["diagnostics"]["singleton_answer_accuracy"],
            "action_target_accuracy": parent_eval["diagnostics"]["action_target_accuracy"],
            "retention": parent_eval["retention"]["overall_score"],
        },
        "locked_test_access": False,
    }


def continuation_spec(directory: Path = DEFAULT_DATA) -> dict[str, Any]:
    data = audit_balanced_source(directory)
    protocol = audit_protocol()
    parent = audit_parent()
    spec = {
        "experiment_id": EXPERIMENT_ID,
        "backend": BACKEND_ID,
        "method": "full_parameter_sft_continuation",
        "representation": "common_balanced_curriculum",
        "curriculum_id": "COMMON-WORD-CURRICULUM-002",
        "seed": 2026,
        "parent_optimizer_steps": PARENT_STEPS,
        "continuation_optimizer_steps": CONTINUATION_STEPS,
        "total_optimizer_steps": PARENT_STEPS + CONTINUATION_STEPS,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 4,
        "max_length": 320,
        "warmup_fraction": 0.05,
        "max_grad_norm": 1.0,
        "relative_checkpoint_steps": RELATIVE_CHECKPOINT_STEPS,
        "total_checkpoint_steps": TOTAL_CHECKPOINT_STEPS,
        "word_token_weight": 8.0,
        "precision": "bfloat16",
        "quantization": "none_16bit",
        "optimizer": "torch.optim.AdamW_fresh_continuation_phase",
        "scheduler": "linear_warmup_5pct_cosine_fresh_continuation_phase",
        "optimizer_restart_declared": True,
        "parent": parent,
        "data": data,
        "protocol": protocol,
        "evaluation": {
            "split": "balanced_002_dev_32",
            "dev_games": 32,
            "diagnostic_items": 128,
            "prompt_variant": "explicit_feedback",
            "decoder": "greedy",
            "generation": {"do_sample": False, "max_new_tokens": 128, "use_cache": True},
            "allowed_words_sha256": EXPECTED_ALLOWED_WORDS_SHA256,
            "retention_probes_sha256": EXPECTED_RETENTION_SHA256,
        },
        "intended_change": "add 600 optimizer steps to the exact full-parameter step-600 checkpoint",
        "declared_limitation": (
            "the parent did not save optimizer, scheduler, or RNG state; the continuation therefore restarts "
            "AdamW, the 5%-warmup cosine schedule, and the seed-2026 data order"
        ),
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
        "harness_selected_guess": False,
    }
    validate_spec(spec, directory)
    return spec


def validate_spec(spec: dict[str, Any], directory: Path = DEFAULT_DATA) -> None:
    fixed = {
        "experiment_id": EXPERIMENT_ID,
        "backend": BACKEND_ID,
        "method": "full_parameter_sft_continuation",
        "representation": "common_balanced_curriculum",
        "curriculum_id": "COMMON-WORD-CURRICULUM-002",
        "seed": 2026,
        "parent_optimizer_steps": 600,
        "continuation_optimizer_steps": 600,
        "total_optimizer_steps": 1200,
        "learning_rate": 5e-5,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 4,
        "max_length": 320,
        "warmup_fraction": 0.05,
        "max_grad_norm": 1.0,
        "relative_checkpoint_steps": RELATIVE_CHECKPOINT_STEPS,
        "total_checkpoint_steps": TOTAL_CHECKPOINT_STEPS,
        "word_token_weight": 8.0,
        "precision": "bfloat16",
        "quantization": "none_16bit",
        "optimizer": "torch.optim.AdamW_fresh_continuation_phase",
        "scheduler": "linear_warmup_5pct_cosine_fresh_continuation_phase",
        "optimizer_restart_declared": True,
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
        "harness_selected_guess": False,
    }
    drift = {key: {"expected": value, "actual": spec.get(key)} for key, value in fixed.items() if spec.get(key) != value}
    if drift:
        raise ValueError(f"continuation spec drift: {drift}")
    if _binding_view(spec["data"]) != _binding_view(audit_balanced_source(directory)):
        raise ValueError("continuation data binding drift")
    if spec["protocol"] != audit_protocol():
        raise ValueError("continuation protocol binding drift")
    parent = audit_parent()
    if spec["parent"] != parent:
        raise ValueError("continuation parent binding drift")
    if spec["evaluation"]["allowed_words_sha256"] != EXPECTED_ALLOWED_WORDS_SHA256:
        raise ValueError("continuation allowed-word binding drift")
    if spec["evaluation"]["retention_probes_sha256"] != EXPECTED_RETENTION_SHA256:
        raise ValueError("continuation retention binding drift")


def prepare_run(spec: dict[str, Any], directory: Path = DEFAULT_DATA) -> Path:
    validate_spec(spec, directory)
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:10]
    run_dir = ARTIFACTS / "runs" / f"full-finetune-balanced-word-continuation-s2026-{digest}"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    write_json(run_dir / "spec.json", spec)
    write_json(run_dir / "dataset_manifest.json", spec["data"])
    write_json(run_dir / "parent_manifest.json", spec["parent"])
    return run_dir


def train_continuation(spec: dict[str, Any], directory: Path = DEFAULT_DATA) -> dict[str, Any]:
    validate_spec(spec, directory)
    preflight = full_finetune_vram_preflight(parameter_count=EXPECTED_PARAMETER_COUNT)
    if not preflight["ready"]:
        raise RuntimeError(f"full continuation preflight blocked: {preflight['status']}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required for the matched continuation")
    run_dir = prepare_run(spec, directory)
    write_json(run_dir / "preflight.json", preflight)
    rows = read_jsonl(Path(directory) / "train.jsonl")
    tokenizer = load_tokenizer(PARENT_CHECKPOINT)
    dataset = CompletionDataset(rows, tokenizer, 320, word_token_weight=8.0)
    generator = torch.Generator().manual_seed(2026)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator, collate_fn=Collator(tokenizer.pad_token_id))
    model = AutoModelForCausalLM.from_pretrained(
        PARENT_CHECKPOINT,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"continuation trainable-parameter drift: {trainable}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    warmup = 30

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / (CONTINUATION_STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    iterator = iter(loader)
    logs: list[dict[str, Any]] = []
    optimizer_tokens = 0
    weighted_completion_tokens = 0.0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    set_seed(2026)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        for relative_step in range(1, CONTINUATION_STEPS + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to("cuda") for key, value in batch.items()}
            loss_weights = batch.pop("loss_weights")
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            loss = weighted_causal_lm_loss(output.logits, batch["labels"], loss_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step_tokens = int(batch["attention_mask"].sum())
            optimizer_tokens += step_tokens
            weighted_completion_tokens += float(loss_weights.sum())
            total_step = PARENT_STEPS + relative_step
            logs.append(
                {
                    "continuation_optimizer_step": relative_step,
                    "total_optimizer_step": total_step,
                    "train_loss": float(loss.detach()),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "continuation_optimizer_tokens": optimizer_tokens,
                    "wall_time_s": time.perf_counter() - started,
                }
            )
            if relative_step in RELATIVE_CHECKPOINT_STEPS:
                checkpoint = run_dir / "checkpoints" / f"step-{total_step:06d}"
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
        accounting = {
            "backend_id": BACKEND_ID,
            "train_examples": len(dataset),
            "parent_optimizer_steps": PARENT_STEPS,
            "continuation_optimizer_steps": CONTINUATION_STEPS,
            "total_optimizer_steps": PARENT_STEPS + CONTINUATION_STEPS,
            "effective_batch_size": 4,
            "continuation_optimizer_tokens": optimizer_tokens,
            "wall_time_s": time.perf_counter() - started,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "trainable_parameters": trainable,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_steps": TOTAL_CHECKPOINT_STEPS,
            "loss_mode": "word_focused",
            "word_token_weight": 8.0,
            "weighted_completion_tokens": weighted_completion_tokens,
            "optimizer_restart_declared": True,
            "locked_test_access": False,
        }
        write_jsonl(run_dir / "train_metrics.jsonl", logs)
        write_json(run_dir / "accounting.json", accounting)
        summary = {
            "status": "full_finetune_continuation_completed",
            "experiment_id": EXPERIMENT_ID,
            "run_dir": str(run_dir),
            "initial_continuation_loss": logs[0]["train_loss"],
            "final_continuation_loss": logs[-1]["train_loss"],
            "accounting": accounting,
            "preflight": preflight,
            "parent": spec["parent"],
            "locked_test_access": False,
        }
        write_json(run_dir / "summary.json", summary)
        return summary
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def evaluate_checkpoint(run_dir: Path, checkpoint: str, directory: Path = DEFAULT_DATA) -> dict[str, Any]:
    run_dir, directory = Path(run_dir), Path(directory)
    spec = read_json(run_dir / "spec.json")
    validate_spec(spec, directory)
    expected = {f"step-{step:06d}" for step in TOTAL_CHECKPOINT_STEPS}
    if checkpoint not in expected:
        raise ValueError(f"checkpoint must be one of {sorted(expected)}")
    output_path = run_dir / f"eval-{checkpoint}-summary.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation: {output_path}")
    checkpoint_dir = run_dir / "checkpoints" / checkpoint
    model, tokenizer = load_full_checkpoint(checkpoint_dir)
    allowed = [
        line.strip().upper()
        for line in (ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    universe = read_json(directory / "universe.json")
    dev_answers = read_json(directory / "dev_secrets.json")[:32]
    previous_messages = generation.inference_messages
    previous_generation = dict(generation.GENERATION_CONFIG)
    try:
        set_seed(2026)
        generation.inference_messages = _explicit_feedback_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(spec["evaluation"]["generation"])
        games, gameplay = evaluate(model, tokenizer, dev_answers, allowed, universe)
        dev_records = generate_canonical_states(
            dev_answers,
            "common_dev_diagnostic",
            128,
            seed=2026,
            answer_vocabulary=universe,
        )
        diagnostics_dir, diagnostics = run_state_diagnostics(
            model,
            tokenizer,
            dev_records,
            read_jsonl(directory / "canonical.jsonl"),
            allowed,
            universe,
            run_dir / f"eval-{checkpoint}",
        )
        retention_rows, retention = evaluate_retention(model, tokenizer, read_jsonl(DATA / "retention_probes_v1.jsonl"))
        write_jsonl(run_dir / f"eval-{checkpoint}-games.jsonl", games)
        write_jsonl(run_dir / f"eval-{checkpoint}-retention.jsonl", retention_rows)
        summary = {
            "status": "dev_evaluated",
            "experiment_id": EXPERIMENT_ID,
            "checkpoint": checkpoint,
            "split": "balanced_002_dev_32",
            "decoder": "greedy",
            "prompt_variant": "explicit_feedback",
            "spec_sha256": sha256_text(canonical_json(spec)),
            "evaluation_data": _binding_view(spec["data"]),
            "protocol": spec["protocol"],
            "parent": spec["parent"],
            "locked_test_access": False,
            "gameplay": gameplay,
            "diagnostics": diagnostics,
            "diagnostics_dir": str(diagnostics_dir),
            "retention": retention,
        }
        write_json(output_path, summary)
        return summary
    finally:
        generation.inference_messages = previous_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(previous_generation)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def compact_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    diagnostics = summary["diagnostics"]
    return {
        "wins": summary["gameplay"]["wins"],
        "win_rate": summary["gameplay"]["win_rate"],
        "terminal_marker_compliance": summary["gameplay"]["terminal_marker_compliance"],
        "invalid_guess_rate": summary["gameplay"]["invalid_guess_rate"],
        "repeat_guess_rate": summary["gameplay"]["repeat_guess_rate"],
        "posterior_constraint_violation_rate": diagnostics["posterior_constraint_violation_rate"],
        "turn_2_posterior_constraint_violation_rate": diagnostics["by_turn"]["2"]["posterior_constraint_violation_rate"],
        "singleton_answer_accuracy": diagnostics["singleton_answer_accuracy"],
        "action_target_accuracy": diagnostics["action_target_accuracy"],
        "retention": summary["retention"]["overall_score"],
    }


def summarize(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    spec = read_json(run_dir / "spec.json")
    validate_spec(spec)
    results = {"600": spec["parent"]["parent_metrics"]}
    for step in TOTAL_CHECKPOINT_STEPS:
        path = run_dir / f"eval-step-{step:06d}-summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        results[str(step)] = compact_metrics(read_json(path))
    output = {
        "experiment_id": EXPERIMENT_ID,
        "decision_scope": "development_only_single_seed",
        "parent_steps": 600,
        "final_steps": 1200,
        "optimizer_restart_declared": True,
        "metrics_by_total_step": results,
        "locked_test_access": False,
    }
    write_json(run_dir / "comparison_summary.json", output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Continue the strongest Gemma 270M full tune from 600 to 1,200 steps")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dry-run", action="store_true")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--run-dir", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = {
            "status": "ready",
            "parent": audit_parent(),
            "data": audit_balanced_source(),
            "protocol": audit_protocol(),
            "vram": full_finetune_vram_preflight(),
            "locked_test_access": False,
        }
    elif args.command == "train":
        spec = continuation_spec()
        result = {"status": "dry_run_ready", "spec": spec, "vram": full_finetune_vram_preflight()} if args.dry_run else train_continuation(spec)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(args.run_dir, args.checkpoint)
    else:
        result = summarize(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
