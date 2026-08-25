from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
import types
from pathlib import Path
from typing import Any, Sequence

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from wordle_lab.analysis.state_diagnostics import run_state_diagnostics
from wordle_lab.common import DATA, read_json, read_jsonl, set_seed, sha256_file, utc_now, write_json, write_jsonl
from wordle_lab.methods.sft import Collator, CompletionDataset, train_sft, weighted_causal_lm_loss
from wordle_lab.protocol import generation
from wordle_lab.protocol.evaluator import evaluate
from wordle_lab.protocol.retention import evaluate_retention


ROOT = Path(__file__).resolve().parents[2]
TRACK = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "base" / "Qwen--Qwen3-0.6B"
LOCAL_RUNS = ROOT / "artifacts" / "qwen3-0.6b-2026-08-24"
RESULTS = TRACK / "results"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PROTOCOL_ID = "WORDLE-PROTOCOL-002"
SEED = 2026

DATASETS = {
    "balanced": ROOT / "data" / "common-curriculum-002" / "u128-train96" / "train.jsonl",
    "coverage4096": ROOT / "data" / "common-curriculum-006" / "u128-train96-n4096" / "train.jsonl",
    "constraint": ROOT / "next_steps" / "chatgpt_2026_08_23" / "generated" / "constraint_first" / "train.jsonl",
    "structured": ROOT / "next_steps" / "chatgpt_2026_08_23" / "generated" / "structured_microtasks_v1" / "train" / "mixed.jsonl",
    "tiny-general": ROOT / "next_steps" / "chatgpt_2026_08_23" / "generated" / "tiny_overfit" / "general_32.jsonl",
    "tiny-singleton": ROOT / "next_steps" / "chatgpt_2026_08_23" / "generated" / "tiny_overfit" / "singleton_32.jsonl",
}

DEFAULTS = {
    "balanced": {"steps": 600, "batch": 4, "accumulation": 1, "word_weight": 8.0},
    "coverage4096": {"steps": 256, "batch": 8, "accumulation": 2, "word_weight": 8.0},
    "constraint": {"steps": 600, "batch": 4, "accumulation": 1, "word_weight": 8.0},
    "structured": {
        "steps": 608,
        "batch": 2,
        "accumulation": 1,
        "word_weight": 1.0,
        "max_length": 512,
        "runtime_adjustment": "one_pass_all_1216_rows_after_native_effective_batch_16_failed_to_checkpoint_in_bounded_window",
    },
    "tiny-general": {"steps": 400, "batch": 4, "accumulation": 1, "word_weight": 8.0},
    "tiny-singleton": {"steps": 400, "batch": 4, "accumulation": 1, "word_weight": 8.0},
}

ADAPTERS = {
    "lora": {"use_rslora": False, "use_dora": False},
    "rslora": {"use_rslora": True, "use_dora": False},
    "dora": {"use_rslora": False, "use_dora": True},
    "full": None,
}


def model_manifest() -> dict[str, Any]:
    metadata = read_json(MODEL_DIR / "wordle_lab_model.json")
    config = read_json(MODEL_DIR / "config.json")
    expected = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_type": "qwen3",
        "architecture": ["Qwen3ForCausalLM"],
        "hidden_size": 1024,
        "layers": 28,
    }
    actual = {
        "model_id": metadata.get("model_id"),
        "revision": metadata.get("revision"),
        "model_type": config.get("model_type"),
        "architecture": config.get("architectures"),
        "hidden_size": config.get("hidden_size"),
        "layers": config.get("num_hidden_layers"),
    }
    if actual != expected:
        raise RuntimeError(f"Qwen model identity drift: expected {expected}, got {actual}")
    files = {}
    for path in sorted(MODEL_DIR.iterdir()):
        if path.is_file() and path.name != "wordle_lab_model.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {**actual, "local_path": str(MODEL_DIR), "files": files, "locked_test_access": False}


def _force_nonthinking(tokenizer):
    original = tokenizer.apply_chat_template

    def apply_chat_template(self, *args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return original(*args, **kwargs)

    tokenizer.apply_chat_template = types.MethodType(apply_chat_template, tokenizer)

    tokenizer.wordle_chat_mode = "qwen3_nonthinking"
    return tokenizer


def _qwen_stop_compat(tokenizer):
    original_convert = tokenizer.convert_tokens_to_ids

    def convert_tokens_to_ids(self, token):
        value = original_convert(token)
        # The frozen retention helper names Gemma's turn terminator directly.
        # Qwen has no such token, so use its native EOS only as a stop fallback.
        return self.eos_token_id if token == "<end_of_turn>" and value is None else value

    tokenizer.convert_tokens_to_ids = types.MethodType(convert_tokens_to_ids, tokenizer)
    return tokenizer


def load_tokenizer(path: str | Path | None = None, *, thinking: bool = False):
    model_manifest()
    tokenizer = AutoTokenizer.from_pretrained(path or MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer = _qwen_stop_compat(tokenizer)
    return tokenizer if thinking else _force_nonthinking(tokenizer)


def load_base_model(training: bool = False):
    model_manifest()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        attn_implementation="eager",
    ).to("cuda")
    model.config.use_cache = not training
    return model


def _explicit_feedback_messages(history):
    from wordle_lab.experiments.intervention_sweep import _explicit_feedback_messages as render

    return render(history)


def _evaluation_data(dataset: str):
    directory = DATASETS[dataset].parent
    if not (directory / "dev_secrets.json").exists():
        directory = ROOT / "data" / "common-curriculum-002" / "u128-train96"
    universe = read_json(directory / "universe.json")
    dev_answers = read_json(directory / "dev_secrets.json")[:32]
    probe_path = directory / "dev_diagnostic_states.jsonl"
    if not probe_path.exists():
        probe_path = ROOT / "data" / "common-curriculum-006" / "u128-train96-n4096" / "dev_diagnostic_states.jsonl"
    training_path = DATASETS[dataset]
    training_rows = read_jsonl(training_path)
    training_records = [row.get("source_state", row) for row in training_rows]
    allowed = [
        line.strip().upper()
        for line in (ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return directory, universe, dev_answers, read_jsonl(probe_path), training_records, allowed


def _evaluate_model(
    model,
    tokenizer,
    output_dir: Path,
    dataset: str,
    label: str,
    generation_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory, universe, dev_answers, probes, training_records, allowed = _evaluation_data(dataset)
    previous_messages = generation.inference_messages
    previous_config = dict(generation.GENERATION_CONFIG)
    try:
        set_seed(SEED)
        generation.inference_messages = _explicit_feedback_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update({"do_sample": False, "max_new_tokens": 128, "use_cache": True})
        generation.GENERATION_CONFIG.update(generation_overrides or {})
        games, gameplay = evaluate(model, tokenizer, dev_answers, allowed, universe)
        diagnostic_dir, diagnostics = run_state_diagnostics(
            model, tokenizer, probes, training_records, allowed, universe, output_dir / f"{label}-fixed-states"
        )
        retention_rows, retention = evaluate_retention(model, tokenizer, read_jsonl(DATA / "retention_probes_v1.jsonl"))
        write_jsonl(output_dir / f"{label}-games.jsonl", games)
        write_jsonl(output_dir / f"{label}-retention.jsonl", retention_rows)
        summary = {
            "status": "dev_evaluated",
            "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
            "protocol_id": PROTOCOL_ID,
            "split": "balanced_002_dev_32",
            "dataset_training_context": dataset,
            "decoder": "greedy",
            "generation": dict(generation.GENERATION_CONFIG),
            "chat_mode": getattr(tokenizer, "wordle_chat_mode", "qwen3_native_thinking"),
            "locked_test_access": False,
            "gameplay": gameplay,
            "diagnostics": diagnostics,
            "diagnostics_dir": str(diagnostic_dir),
            "retention": retention,
        }
        write_json(output_dir / f"{label}-summary.json", summary)
        return summary
    finally:
        generation.inference_messages = previous_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(previous_config)


def benchmark(thinking: bool = False) -> dict[str, Any]:
    label = "base-thinking" if thinking else "base-nonthinking"
    output_dir = RESULTS / label
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_tokenizer(thinking=thinking), load_base_model(training=False)
    try:
        return _evaluate_model(model, tokenizer, output_dir, "coverage4096", label)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def train(dataset: str, adapter: str, *, steps: int | None = None) -> dict[str, Any]:
    if dataset not in DATASETS or adapter not in ADAPTERS:
        raise ValueError("unknown dataset or adapter")
    defaults = DEFAULTS[dataset]
    max_steps = int(steps or defaults["steps"])
    run_name = f"{dataset}-{adapter}-s{SEED}-n{max_steps}"
    run_dir = LOCAL_RUNS / run_name
    if (run_dir / "train-summary.json").exists():
        return read_json(run_dir / "train-summary.json")
    rows = read_jsonl(DATASETS[dataset])
    spec = {
        "experiment_id": "QWEN3-0.6B-WORDLE-001",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "method": "sft",
        "representation": dataset,
        "seed": SEED,
        "max_steps": max_steps,
        "batch_size": defaults["batch"],
        "gradient_accumulation_steps": defaults["accumulation"],
        "learning_rate": 5e-5,
        "max_length": int(defaults.get("max_length", 320)),
        "word_token_weight": defaults["word_weight"],
        "adapter": {
            "type": "lora",
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            **(ADAPTERS[adapter] or {}),
        },
        "protocol_id": PROTOCOL_ID,
        "chat_mode": "qwen3_nonthinking",
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
        "harness_selected_guess": False,
        "runtime_adjustment": defaults.get("runtime_adjustment"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "spec.json", {**spec, "created_at": utc_now(), "dataset_path": str(DATASETS[dataset]), "dataset_sha256": sha256_file(DATASETS[dataset]), "rows": len(rows), "model": model_manifest()})
    set_seed(SEED)
    if adapter == "full":
        if dataset != "balanced":
            raise ValueError("the full-parameter ablation is defined only for balanced-002")
        spec.update({"batch_size": 2, "gradient_accumulation_steps": 2, "trainable_scope": "all_parameters"})
        model, accounting = _train_full(rows, run_dir, spec)
    else:
        model, accounting = train_sft(
            rows,
            run_dir,
            spec,
            tokenizer_loader=lambda: load_tokenizer(thinking=False),
            model_loader=load_base_model,
        )
    summary = {"status": "trained", "run_name": run_name, "dataset": dataset, "adapter": adapter, "accounting": accounting, "locked_test_access": False}
    write_json(run_dir / "train-summary.json", summary)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _train_full(rows: list[dict[str, Any]], run_dir: Path, spec: dict[str, Any]):
    """Full BF16 AdamW ablation with the same weighted objective and dose curve."""
    tokenizer = load_tokenizer(thinking=False)
    dataset = CompletionDataset(rows, tokenizer, int(spec["max_length"]), word_token_weight=float(spec["word_token_weight"]))
    generator = torch.Generator().manual_seed(int(spec["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=int(spec["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=Collator(tokenizer.pad_token_id),
        drop_last=False,
    )
    model = load_base_model(training=True)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable != total:
        raise RuntimeError("full tune did not enable every parameter")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]))
    max_steps = int(spec["max_steps"])
    accumulation = int(spec["gradient_accumulation_steps"])
    warmup = max(1, int(max_steps * 0.05))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    checkpoints = sorted({max(1, round(max_steps * fraction)) for fraction in (0.25, 0.5, 0.75, 1.0)})
    iterator = iter(loader)
    logs = []
    optimizer_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
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
            weights = batch.pop("loss_weights")
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            loss = weighted_causal_lm_loss(output.logits, batch["labels"], weights)
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            step_tokens += int(batch["attention_mask"].sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_tokens += step_tokens
        logs.append({"optimizer_step": step, "train_loss": sum(losses) / len(losses), "learning_rate": scheduler.get_last_lr()[0], "optimizer_tokens": optimizer_tokens, "wall_time_s": time.perf_counter() - started})
        if step in checkpoints:
            destination = run_dir / "checkpoints" / f"step-{step:06d}"
            model.save_pretrained(destination)
            tokenizer.save_pretrained(destination)
    accounting = {
        "backend": "native_transformers_full_parameter_bf16",
        "train_examples": len(dataset),
        "optimizer_steps": max_steps,
        "effective_batch_size": int(spec["batch_size"]) * accumulation,
        "optimizer_tokens": optimizer_tokens,
        "wall_time_s": time.perf_counter() - started,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": 1.0,
        "checkpoint_steps": checkpoints,
        "word_token_weight": float(spec["word_token_weight"]),
    }
    write_jsonl(run_dir / "train_metrics.jsonl", logs)
    write_json(run_dir / "accounting.json", accounting)
    model.config.use_cache = True
    model.eval()
    return model, accounting


def evaluate_adapter(
    dataset: str,
    adapter: str,
    *,
    steps: int | None = None,
    checkpoint_name: str = "final",
    repetition_penalty: float | None = None,
) -> dict[str, Any]:
    defaults = DEFAULTS[dataset]
    max_steps = int(steps or defaults["steps"])
    run_name = f"{dataset}-{adapter}-s{SEED}-n{max_steps}"
    run_dir = LOCAL_RUNS / run_name
    if checkpoint_name != "final" and not checkpoint_name.startswith("step-"):
        raise ValueError("checkpoint must be final or step-NNNNNN")
    checkpoint = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    evaluation_name = checkpoint_name + (f"-rep{repetition_penalty:g}" if repetition_penalty else "")
    output_dir = RESULTS / run_name / evaluation_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(checkpoint, thinking=False)
    if adapter == "full":
        base = None
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
        ).to("cuda")
    else:
        base = load_base_model(training=False)
        model = PeftModel.from_pretrained(base, checkpoint).to("cuda")
    model.eval()
    try:
        if dataset.startswith("tiny-"):
            from next_steps.chatgpt_2026_08_23.tiny_overfit import evaluate_memorization

            rows = read_jsonl(DATASETS[dataset])
            universe = read_json(ROOT / "data" / "common-curriculum-002" / "u128-train96" / "universe.json")
            details, summary = evaluate_memorization(model, tokenizer, rows, universe)
            summary.update({"model_id": MODEL_ID, "revision": MODEL_REVISION, "dataset": dataset, "adapter": adapter, "checkpoint": checkpoint_name, "locked_test_access": False})
            write_jsonl(output_dir / "memorization-items.jsonl", details)
            write_json(output_dir / "memorization-summary.json", summary)
            return summary
        if dataset == "structured":
            from next_steps.chatgpt_2026_08_23.structured_microtasks_experiment import (
                _generate_natural_outputs,
                _record_from_dict,
                evaluate_raw_outputs,
            )

            bundle = DATASETS[dataset].parent.parent
            rendered = read_jsonl(bundle / "dev" / "mixed.jsonl")
            logical = [_record_from_dict(row) for row in read_jsonl(bundle / "dev" / "records.jsonl")]
            allowed = [
                line.strip().upper()
                for line in (ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            raw = _generate_natural_outputs(model, tokenizer, rendered, batch_size=16, max_new_tokens=128)
            raw_rows = [{"record_id": key, "raw_output": raw[key]} for key in sorted(raw)]
            parsed, metrics = evaluate_raw_outputs(logical, raw, allowed)
            summary = {
                "status": "dev_evaluated",
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "dataset": dataset,
                "adapter": adapter,
                "checkpoint": checkpoint_name,
                "metrics": metrics,
                "locked_test_access": False,
            }
            write_jsonl(output_dir / "raw-outputs.jsonl", raw_rows)
            write_jsonl(output_dir / "parsed-outputs.jsonl", parsed)
            write_json(output_dir / "structured-summary.json", summary)
            return summary
        overrides = {"repetition_penalty": repetition_penalty} if repetition_penalty else None
        summary = _evaluate_model(model, tokenizer, output_dir, dataset, f"{run_name}-{evaluation_name}", overrides)
        summary["checkpoint"] = checkpoint_name
        write_json(output_dir / f"{run_name}-{checkpoint_name}-summary.json", summary)
        return summary
    finally:
        del model
        if base is not None:
            del base
        gc.collect()
        torch.cuda.empty_cache()


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    if "gameplay" not in summary:
        return summary
    gameplay, diagnostics = summary["gameplay"], summary["diagnostics"]
    return {
        "wins": gameplay["wins"],
        "win_rate": gameplay["win_rate"],
        "terminal_compliance": gameplay["terminal_marker_compliance"],
        "invalid_guess_rate": gameplay["invalid_guess_rate"],
        "repeat_guess_rate": gameplay["repeat_guess_rate"],
        "posterior_violation_rate": diagnostics["posterior_constraint_violation_rate"],
        "turn_2_violation_rate": diagnostics["by_turn"]["2"]["posterior_constraint_violation_rate"],
        "singleton_accuracy": diagnostics["singleton_answer_accuracy"],
        "action_target_accuracy": diagnostics["action_target_accuracy"],
        "retention": summary["retention"]["overall_score"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3-0.6B audited Wordle experiment track")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-model")
    base = sub.add_parser("benchmark"); base.add_argument("--thinking", action="store_true")
    for command in ("train", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
        p.add_argument("--adapter", choices=sorted(ADAPTERS), default="lora")
        p.add_argument("--steps", type=int)
        if command == "evaluate":
            p.add_argument("--checkpoint", default="final")
            p.add_argument("--repetition-penalty", type=float)
    args = parser.parse_args(argv)
    if args.command == "verify-model":
        result = model_manifest()
        write_json(TRACK / "model_manifest.json", result)
    elif args.command == "benchmark":
        result = benchmark(args.thinking)
    elif args.command == "train":
        result = train(args.dataset, args.adapter, steps=args.steps)
    else:
        result = evaluate_adapter(
            args.dataset,
            args.adapter,
            steps=args.steps,
            checkpoint_name=args.checkpoint,
            repetition_penalty=args.repetition_penalty,
        )
    print(json.dumps(compact(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
