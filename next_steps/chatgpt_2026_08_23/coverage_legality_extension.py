from __future__ import annotations

"""Add a second disjoint coverage pass to the 4K Gemma 270M leader."""

import argparse
import gc
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from wordle_lab.analysis.state_diagnostics import run_state_diagnostics
from wordle_lab.common import ARTIFACTS, DATA, ROOT, canonical_json, read_json, read_jsonl, set_seed, sha256_file, sha256_text, write_json, write_jsonl
from wordle_lab.experiments.common_curriculum import _targeted_state_pools, ranked_common_words
from wordle_lab.experiments.intervention_sweep import _explicit_feedback_messages
from wordle_lab.methods.sft import Collator, CompletionDataset, weighted_causal_lm_loss
from wordle_lab.models import load_tokenizer
from wordle_lab.protocol import generation
from wordle_lab.protocol.env import score_wordle
from wordle_lab.protocol.evaluator import evaluate
from wordle_lab.protocol.retention import evaluate_retention

from . import coverage_max_experiment as phase1
from .full_finetune import EXPECTED_PARAMETER_COUNT, full_finetune_vram_preflight, load_full_checkpoint
from .full_finetune_experiment import _tree_digest, audit_protocol


EXPERIMENT_ID = "GEMMA-270M-COVERAGE-STACK-002"
CURRICULUM_ID = "COMMON-WORD-CURRICULUM-007"
BACKEND_ID = "GEMMA-270M-FULL-COVERAGE-STACK-001"
PARENT_RUN = ARTIFACTS / "runs" / "gemma-270m-coverage-max-s2026-b452e4f6a5"
PARENT_CHECKPOINT = PARENT_RUN / "checkpoints" / "step-001024"
PARENT_TREE_SHA256 = "b6c913be0d423819c5107fb6c3355c3f691c439dfe275f66c4cafcb58f2a1cf6"
PARENT_HASHES = {
    "spec.json": "971fd10aac71cc7caf0804e904e4cd22798dc493322455c7bf36b46cbe50c108",
    "summary.json": "b49ef943cd0e39dee3fa8f260c40b83c9bcbfca9bc68437b407f7a8286c7692e",
    "eval-step-001024-summary.json": "e149633c368a5e613d94c89d4ecf0b6024ff4f257a8ac5f5b1655cb41d4f6462",
}
ROWS = 4096
BATCH_SIZE = 4
STEPS = 1024
CHECKPOINT_STEPS = [256, 512, 768, 1024]
CUMULATIVE_EXAMPLES = [5120, 6144, 7168, 8192]
QUOTAS = {"turn_2": 1024, "low_posterior": 1792, "true_singleton": 768, "later_broad": 512}
TARGET_CAP = 96
LEARNING_RATE = 1e-5
DEFAULT_OUTPUT = ROOT / "data" / "common-curriculum-007" / "u128-train96-n4096-disjoint"


def _rank(label: str, value: str) -> str:
    return sha256_text(canonical_json({"seed": phase1.SEED, "label": label, "value": value}))


def audit_parent() -> dict[str, Any]:
    observed = {name: sha256_file(PARENT_RUN / name) for name in PARENT_HASHES}
    if observed != PARENT_HASHES:
        raise AssertionError("coverage-stack parent artifact drift")
    tree, files = _tree_digest(PARENT_CHECKPOINT)
    if tree != PARENT_TREE_SHA256:
        raise AssertionError("coverage-stack parent checkpoint drift")
    summary = read_json(PARENT_RUN / "eval-step-001024-summary.json")
    return {
        "status": "passed",
        "run_directory": PARENT_RUN.relative_to(ROOT).as_posix(),
        "checkpoint": "step-001024",
        "checkpoint_directory": PARENT_CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_tree_sha256": tree,
        "checkpoint_files": files,
        "artifact_hashes": observed,
        "metrics": phase1.compact_metrics(summary),
        "optimizer_state_available": False,
        "locked_test_access": False,
    }


def build_bundle(output: Path = DEFAULT_OUTPUT, *, force: bool = False) -> tuple[Path, dict[str, Any]]:
    output = Path(output)
    if (output / "manifest.json").is_file() and not force:
        return output, audit_bundle(output)
    phase1_audit = phase1.audit_bundle()
    universe = read_json(phase1.DEFAULT_OUTPUT / "universe.json")
    train_secrets = read_json(phase1.DEFAULT_OUTPUT / "train_secrets.json")
    dev_secrets = read_json(phase1.DEFAULT_OUTPUT / "dev_secrets.json")
    dev_records = read_jsonl(phase1.DEFAULT_OUTPUT / "dev_diagnostic_states.jsonl")
    used_phase1 = {row["state_id"] for row in read_jsonl(phase1.DEFAULT_OUTPUT / "train.jsonl") if row["state_type"] != "format_root"}
    dev_keys = {phase1._history_key(record) for record in dev_records if record["history"]}
    pools = _targeted_state_pools(train_secrets, universe, ROWS, phase1.SEED)
    selected: list[tuple[dict[str, Any], str]] = []
    target_counts: Counter[str] = Counter()
    used: set[str] = set()
    for kind, quota in QUOTAS.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in pools[kind]:
            state_id = str(record["state_id"])
            if state_id in used_phase1 or phase1._history_key(record) in dev_keys:
                continue
            grouped[str(record["facts"]["oracle_action"])].append(record)
        for target, records in grouped.items():
            records.sort(key=lambda row: _rank(f"{kind}:{target}", str(row["state_id"])), reverse=True)
        targets = sorted(grouped, key=lambda target: _rank(f"{kind}:target", target))
        chosen = 0
        while chosen < quota:
            progressed = False
            targets.sort(key=lambda target: (target_counts[target], _rank(f"{kind}:round", target)))
            for target in targets:
                if chosen >= quota:
                    break
                if target_counts[target] >= TARGET_CAP:
                    grouped[target].clear()
                    continue
                while grouped[target]:
                    record = grouped[target].pop()
                    if record["state_id"] in used:
                        continue
                    selected.append((record, kind))
                    used.add(record["state_id"])
                    target_counts[target] += 1
                    chosen += 1
                    progressed = True
                    break
            if not progressed:
                raise RuntimeError(f"coverage-stack cannot fill {kind}: {chosen}/{quota}")
    selected.sort(key=lambda item: _rank("final", f"{item[1]}:{item[0]['state_id']}"))
    rows = []
    for index, (record, kind) in enumerate(selected):
        history = [(item["guess"], item["feedback"]) for item in record["history"]]
        target = record["facts"]["oracle_action"]
        rows.append(
            {
                "example_id": f"coverage-stack-{index:06d}-{record['state_id']}",
                "state_id": record["state_id"],
                "source_state": record,
                "state_type": kind,
                "turn": record["turn"],
                "posterior_size": record["facts"]["posterior_count"],
                "target_word": target,
                "target_frequency": target_counts[target],
                "prompt": _explicit_feedback_messages(history),
                "completion": [{"role": "assistant", "content": f"Final answer: {target}"}],
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    train_path = write_jsonl(output / "train.jsonl", rows)
    write_json(output / "universe.json", universe)
    write_json(output / "train_secrets.json", train_secrets)
    write_json(output / "dev_secrets.json", dev_secrets)
    write_jsonl(output / "dev_diagnostic_states.jsonl", dev_records)
    manifest = {
        "curriculum_id": CURRICULUM_ID,
        "experiment_id": EXPERIMENT_ID,
        "parent_curriculum_id": phase1.CURRICULUM_ID,
        "rows": len(rows),
        "unique_states": len(used),
        "disjoint_from_parent_states": True,
        "quotas": QUOTAS,
        "composition": dict(sorted(Counter(row["state_type"] for row in rows).items())),
        "target_cap": max(target_counts.values()),
        "prompt_version": "explicit-constraints-v2-compact",
        "learning_rate_rationale": "conservative continuation rate after high-rate repeated-data continuation degraded compliance",
        "phase1_audit": phase1_audit,
        "train_sha256": sha256_file(train_path),
        "locked_test_access": False,
    }
    write_json(output / "manifest.json", manifest)
    return output, audit_bundle(output)


def audit_bundle(directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    directory = Path(directory)
    rows = read_jsonl(directory / "train.jsonl")
    manifest = read_json(directory / "manifest.json")
    train_secrets = set(read_json(directory / "train_secrets.json"))
    dev_secrets = set(read_json(directory / "dev_secrets.json"))
    phase1_ids = {row["state_id"] for row in read_jsonl(phase1.DEFAULT_OUTPUT / "train.jsonl") if row["state_type"] != "format_root"}
    dev_keys = {phase1._history_key(record) for record in read_jsonl(directory / "dev_diagnostic_states.jsonl") if record["history"]}
    seen: set[str] = set()
    targets: Counter[str] = Counter()
    for row in rows:
        source = row["source_state"]
        if source["secret_answer"] not in train_secrets or source["secret_answer"] in dev_secrets:
            raise AssertionError("coverage-stack secret leakage")
        if row["state_id"] in phase1_ids or row["state_id"] in seen:
            raise AssertionError("coverage-stack state overlap")
        seen.add(row["state_id"])
        if phase1._history_key(source) in dev_keys:
            raise AssertionError("coverage-stack development history collision")
        history = [(item["guess"], item["feedback"]) for item in source["history"]]
        if any(score_wordle(source["secret_answer"], guess) != feedback for guess, feedback in history):
            raise AssertionError("coverage-stack feedback mismatch")
        if row["prompt"] != _explicit_feedback_messages(history):
            raise AssertionError("coverage-stack prompt drift")
        if row["completion"] != [{"role": "assistant", "content": f"Final answer: {row['target_word']}"}]:
            raise AssertionError("coverage-stack completion drift")
        targets[row["target_word"]] += 1
    if len(rows) != ROWS or len(seen) != ROWS or Counter(row["state_type"] for row in rows) != Counter(QUOTAS):
        raise AssertionError("coverage-stack composition drift")
    if max(targets.values()) > TARGET_CAP:
        raise AssertionError("coverage-stack target cap drift")
    if manifest["train_sha256"] != sha256_file(directory / "train.jsonl"):
        raise AssertionError("coverage-stack hash drift")
    return {
        "status": "passed",
        "directory": directory.as_posix(),
        "rows": len(rows),
        "unique_states": len(seen),
        "disjoint_parent_states": len(phase1_ids),
        "target_cap": max(targets.values()),
        "checks": ["training_only_secrets", "feedback_recomputed", "parent_state_disjoint", "dev_history_disjoint", "prompt_exact", "completion_exact", "target_cap", "locked_test_unread"],
        "train_sha256": sha256_file(directory / "train.jsonl"),
        "locked_test_access": False,
    }


def build_spec(directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    spec = {
        "experiment_id": EXPERIMENT_ID,
        "curriculum_id": CURRICULUM_ID,
        "backend": BACKEND_ID,
        "seed": phase1.SEED,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "examples_seen": ROWS,
        "cumulative_unique_coverage": 8192,
        "learning_rate": LEARNING_RATE,
        "word_token_weight": 8.0,
        "max_length": 320,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "checkpoint_cumulative_examples": dict(zip((str(step) for step in CHECKPOINT_STEPS), CUMULATIVE_EXAMPLES)),
        "optimizer": "fresh_AdamW_declared",
        "scheduler": "fresh_linear_warmup_5pct_cosine_declared",
        "parent": audit_parent(),
        "data": audit_bundle(directory),
        "protocol": audit_protocol(),
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
    }
    validate_spec(spec, directory)
    return spec


def validate_spec(spec: dict[str, Any], directory: Path = DEFAULT_OUTPUT) -> None:
    fixed = {"experiment_id": EXPERIMENT_ID, "curriculum_id": CURRICULUM_ID, "backend": BACKEND_ID, "seed": phase1.SEED, "steps": STEPS, "batch_size": BATCH_SIZE, "examples_seen": ROWS, "cumulative_unique_coverage": 8192, "learning_rate": LEARNING_RATE, "word_token_weight": 8.0, "max_length": 320, "checkpoint_steps": CHECKPOINT_STEPS, "locked_test_access": False, "candidate_injection": False, "vocabulary_masking": False, "reranking": False, "repeat_ban": False, "output_repair": False}
    drift = {key: (value, spec.get(key)) for key, value in fixed.items() if spec.get(key) != value}
    if drift:
        raise ValueError(f"coverage-stack spec drift: {drift}")
    if spec["parent"] != audit_parent() or spec["data"] != audit_bundle(directory) or spec["protocol"] != audit_protocol():
        raise ValueError("coverage-stack provenance binding drift")


def prepare_run(spec: dict[str, Any], directory: Path = DEFAULT_OUTPUT) -> Path:
    validate_spec(spec, directory)
    digest = hashlib.sha256(canonical_json(spec).encode()).hexdigest()[:10]
    run_dir = ARTIFACTS / "runs" / f"gemma-270m-coverage-stack-s{phase1.SEED}-{digest}"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    write_json(run_dir / "spec.json", spec)
    return run_dir


def train(spec: dict[str, Any], directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    validate_spec(spec, directory)
    preflight = full_finetune_vram_preflight(parameter_count=EXPECTED_PARAMETER_COUNT)
    if not preflight["ready"]:
        raise RuntimeError("coverage-stack VRAM preflight failed")
    run_dir = prepare_run(spec, directory)
    rows = read_jsonl(Path(directory) / "train.jsonl")
    tokenizer = load_tokenizer(PARENT_CHECKPOINT)
    dataset = CompletionDataset(rows, tokenizer, 320, word_token_weight=8.0)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(phase1.SEED), collate_fn=Collator(tokenizer.pad_token_id))
    model = AutoModelForCausalLM.from_pretrained(PARENT_CHECKPOINT, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager").to("cuda")
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    warmup = max(1, int(STEPS * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / (STEPS - warmup))))
    iterator = iter(loader)
    logs = []
    tokens = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    set_seed(phase1.SEED)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        for step in range(1, STEPS + 1):
            batch = next(iterator)
            batch = {key: value.to("cuda") for key, value in batch.items()}
            weights = batch.pop("loss_weights")
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            loss = weighted_causal_lm_loss(output.logits, batch["labels"], weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            tokens += int(batch["attention_mask"].sum())
            logs.append({"optimizer_step": step, "phase2_examples_seen": step * BATCH_SIZE, "cumulative_examples_seen": 4096 + step * BATCH_SIZE, "train_loss": float(loss.detach()), "learning_rate": scheduler.get_last_lr()[0], "optimizer_tokens": tokens, "wall_time_s": time.perf_counter() - started})
            if step in CHECKPOINT_STEPS:
                label = spec["checkpoint_cumulative_examples"][str(step)]
                checkpoint = run_dir / "checkpoints" / f"coverage-{label:06d}"
                model.save_pretrained(checkpoint); tokenizer.save_pretrained(checkpoint)
        accounting = {"phase2_examples_seen": ROWS, "cumulative_unique_coverage": 8192, "optimizer_steps": STEPS, "optimizer_tokens": tokens, "wall_time_s": time.perf_counter() - started, "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()), "learning_rate": LEARNING_RATE, "trainable_parameters": EXPECTED_PARAMETER_COUNT, "locked_test_access": False}
        write_jsonl(run_dir / "train_metrics.jsonl", logs); write_json(run_dir / "accounting.json", accounting)
        summary = {"status": "coverage_stack_training_completed", "run_dir": str(run_dir), "initial_loss": logs[0]["train_loss"], "final_loss": logs[-1]["train_loss"], "accounting": accounting, "locked_test_access": False}
        write_json(run_dir / "summary.json", summary)
        return summary
    finally:
        del model; gc.collect(); torch.cuda.empty_cache()


def evaluate_checkpoint(run_dir: Path, checkpoint: str, directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    run_dir = Path(run_dir)
    spec = read_json(run_dir / "spec.json"); validate_spec(spec, directory)
    expected = {f"coverage-{value:06d}" for value in CUMULATIVE_EXAMPLES}
    if checkpoint not in expected:
        raise ValueError(f"checkpoint must be one of {sorted(expected)}")
    output_path = run_dir / f"eval-{checkpoint}-summary.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    model, tokenizer = load_full_checkpoint(run_dir / "checkpoints" / checkpoint)
    allowed = [line.strip().upper() for line in (ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    universe = read_json(directory / "universe.json"); dev_answers = read_json(directory / "dev_secrets.json")[:32]; dev_records = read_jsonl(directory / "dev_diagnostic_states.jsonl")
    previous_messages = generation.inference_messages; previous_generation = dict(generation.GENERATION_CONFIG)
    try:
        set_seed(phase1.SEED); generation.inference_messages = _explicit_feedback_messages; generation.GENERATION_CONFIG.clear(); generation.GENERATION_CONFIG.update({"do_sample": False, "max_new_tokens": 128, "use_cache": True})
        games, gameplay = evaluate(model, tokenizer, dev_answers, allowed, universe)
        training_records = [row["source_state"] for row in read_jsonl(directory / "train.jsonl")]
        diagnostics_dir, diagnostics = run_state_diagnostics(model, tokenizer, dev_records, training_records, allowed, universe, run_dir / f"eval-{checkpoint}")
        retention_rows, retention = evaluate_retention(model, tokenizer, read_jsonl(DATA / "retention_probes_v1.jsonl"))
        write_jsonl(run_dir / f"eval-{checkpoint}-games.jsonl", games); write_jsonl(run_dir / f"eval-{checkpoint}-retention.jsonl", retention_rows)
        summary = {"status": "dev_evaluated", "experiment_id": EXPERIMENT_ID, "checkpoint": checkpoint, "cumulative_unique_coverage": int(checkpoint.split("-")[-1]), "locked_test_access": False, "gameplay": gameplay, "diagnostics": diagnostics, "diagnostics_dir": str(diagnostics_dir), "retention": retention}
        write_json(output_path, summary); return summary
    finally:
        generation.inference_messages = previous_messages; generation.GENERATION_CONFIG.clear(); generation.GENERATION_CONFIG.update(previous_generation); del model; gc.collect(); torch.cuda.empty_cache()


def summarize(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir); spec = read_json(run_dir / "spec.json"); validate_spec(spec)
    metrics = {"4096_parent": spec["parent"]["metrics"]}
    for coverage in CUMULATIVE_EXAMPLES:
        summary = read_json(run_dir / f"eval-coverage-{coverage:06d}-summary.json")
        metrics[str(coverage)] = phase1.compact_metrics({**summary, "examples_seen": coverage})
    result = {"experiment_id": EXPERIMENT_ID, "metrics": metrics, "locked_test_access": False}
    write_json(run_dir / "comparison_summary.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Second disjoint Gemma 270M coverage phase")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build"); build.add_argument("--force", action="store_true")
    sub.add_parser("preflight")
    train_parser = sub.add_parser("train"); train_parser.add_argument("--dry-run", action="store_true")
    evaluate_parser = sub.add_parser("evaluate"); evaluate_parser.add_argument("--run-dir", type=Path, required=True); evaluate_parser.add_argument("--checkpoint", required=True)
    summarize_parser = sub.add_parser("summarize"); summarize_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        directory, audit = build_bundle(force=args.force); result = {"directory": str(directory), "audit": audit}
    elif args.command == "preflight":
        result = {"status": "ready", "parent": audit_parent(), "data": audit_bundle(), "protocol": audit_protocol(), "vram": full_finetune_vram_preflight(), "locked_test_access": False}
    elif args.command == "train":
        spec = build_spec(); result = {"status": "dry_run_ready", "spec": spec} if args.dry_run else train(spec)
    elif args.command == "evaluate":
        result = evaluate_checkpoint(args.run_dir, args.checkpoint)
    else:
        result = summarize(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
