from __future__ import annotations

"""Conditionally extend Gemma 3 270M unique state coverage from 7,168 to 20,480."""

import argparse
import gc
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
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
from wordle_lab.experiments.common_curriculum import _targeted_state_pools
from wordle_lab.experiments.intervention_sweep import _explicit_feedback_messages
from wordle_lab.methods.sft import Collator, CompletionDataset, weighted_causal_lm_loss
from wordle_lab.models import load_tokenizer
from wordle_lab.protocol import generation
from wordle_lab.protocol.env import score_wordle
from wordle_lab.protocol.evaluator import evaluate
from wordle_lab.protocol.retention import evaluate_retention

from . import coverage_max_experiment as phase1
from .full_finetune import EXPECTED_PARAMETER_COUNT, full_finetune_vram_preflight
from .full_finetune_experiment import _tree_digest, audit_protocol


EXPERIMENT_ID = "GEMMA-270M-COVERAGE-GROWTH-003"
CURRICULUM_ID = "COMMON-WORD-CURRICULUM-008"
BACKEND_ID = "GEMMA-270M-FULL-COVERAGE-GROWTH-001"
PARENT_COVERAGE = 7168
PARENT_RUN = ARTIFACTS / "runs" / "gemma-270m-coverage-stack-s2026-99814e6413"
PARENT_CHECKPOINT = PARENT_RUN / "checkpoints" / "coverage-007168"
PARENT_TREE_SHA256 = "07f339db3d9f5599029e5d1afd3c60bf7fa0db267da9a78f83f5f0aa2507f4b1"
PARENT_HASHES = {
    "spec.json": "670c0db0a29a441643a66fad5c933d2d200df3568ae7f51c3b66996b4093ee66",
    "summary.json": "21f8e33eace8f0448981004e7b29c4aeb7873beb705b5d71106aaf44c6a69e96",
    "eval-coverage-007168-summary.json": "9dafb8e624742219a8eb9ebdcc6acc578dfcbb586445c492ba33cb9c7d83cd25",
}
MILESTONES = [10240, 12288, 15360, 20480]
CHECKPOINT_STEPS = [(coverage - PARENT_COVERAGE) // 4 for coverage in MILESTONES]
ROWS = MILESTONES[-1] - PARENT_COVERAGE
BATCH_SIZE = 4
MAX_STEPS = ROWS // BATCH_SIZE
LEARNING_RATE = 5e-6
POOL_TOTAL = 20000
TARGET_CAP = 256
QUOTAS = {"turn_2": 48, "low_posterior": 7274, "true_singleton": 3994, "later_broad": 1996}
DEFAULT_OUTPUT = ROOT / "data" / "common-curriculum-008" / "u128-train96-growth-07168-to-20480"
FORCE_PARENT_COVERAGE = 10240
FORCE_TARGET_COVERAGE = 15360
FORCE_PARENT_RUN = ARTIFACTS / "runs" / "gemma-270m-coverage-growth-s2026-10126fb1ed"
FORCE_PARENT_CHECKPOINT = FORCE_PARENT_RUN / "checkpoints" / "coverage-010240"
FORCE_PARENT_TREE_SHA256 = "32240ec3403dc8639692d063ad7ee38b21507ce6bd0b18631f77b5bd20a90fa3"
FORCE_PARENT_HASHES = {
    "spec.json": "46be837011a8f4a67c0d1cce46ede5847b94f0cbad1eafb28ab40ee6ea5237df",
    "summary.json": "7373ae8ff2d2dc52688b3532d756df223a07d5dc0a4e79b9067aa826fffa0a46",
    "eval-coverage-010240-summary.json": "28145ff58b9ce3fa54856e9c9528ccea693a7439b3a3a56a9a979639e1d260ca",
}
FORCE_MILESTONES = [12288, 15360]
FORCE_ROWS = FORCE_TARGET_COVERAGE - FORCE_PARENT_COVERAGE
FORCE_STEPS = FORCE_ROWS // BATCH_SIZE
EXCLUDED_DATASETS = [
    phase1.DEFAULT_OUTPUT,
    ROOT / "data" / "common-curriculum-007" / "u128-train96-n4096-disjoint",
]


def _rank(label: str, value: str) -> str:
    return sha256_text(canonical_json({"seed": phase1.SEED, "label": label, "value": value}))


def _history_key(record: dict[str, Any]) -> str:
    history = [(item["guess"], item["feedback"]) for item in record["history"]]
    return sha256_text(canonical_json(history))


def compact_metrics(summary: dict[str, Any], examples_seen: int) -> dict[str, Any]:
    diagnostics = summary["diagnostics"]
    return {
        "examples_seen": examples_seen,
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


def audit_parent() -> dict[str, Any]:
    observed = {name: sha256_file(PARENT_RUN / name) for name in PARENT_HASHES}
    if observed != PARENT_HASHES:
        raise AssertionError("growth-ladder parent artifact drift")
    tree, files = _tree_digest(PARENT_CHECKPOINT)
    if tree != PARENT_TREE_SHA256:
        raise AssertionError("growth-ladder parent checkpoint drift")
    summary = read_json(PARENT_RUN / "eval-coverage-007168-summary.json")
    if summary.get("locked_test_access") is not False or summary.get("cumulative_unique_coverage") != PARENT_COVERAGE:
        raise AssertionError("growth-ladder parent evaluation drift")
    return {
        "status": "passed",
        "run_directory": PARENT_RUN.relative_to(ROOT).as_posix(),
        "checkpoint_directory": PARENT_CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_tree_sha256": tree,
        "checkpoint_files": files,
        "artifact_hashes": observed,
        "metrics": compact_metrics(summary, PARENT_COVERAGE),
        "optimizer_state_available": False,
        "locked_test_access": False,
    }


def audit_force_parent() -> dict[str, Any]:
    observed = {name: sha256_file(FORCE_PARENT_RUN / name) for name in FORCE_PARENT_HASHES}
    if observed != FORCE_PARENT_HASHES:
        raise AssertionError("forced-15k parent artifact drift")
    tree, files = _tree_digest(FORCE_PARENT_CHECKPOINT)
    if tree != FORCE_PARENT_TREE_SHA256:
        raise AssertionError("forced-15k parent checkpoint drift")
    summary = read_json(FORCE_PARENT_RUN / "eval-coverage-010240-summary.json")
    if summary.get("locked_test_access") is not False or summary.get("cumulative_unique_coverage") != FORCE_PARENT_COVERAGE:
        raise AssertionError("forced-15k parent evaluation drift")
    return {
        "status": "passed",
        "run_directory": FORCE_PARENT_RUN.relative_to(ROOT).as_posix(),
        "checkpoint_directory": FORCE_PARENT_CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_tree_sha256": tree,
        "checkpoint_files": files,
        "artifact_hashes": observed,
        "metrics": compact_metrics(summary, FORCE_PARENT_COVERAGE),
        "optimizer_state_available": False,
        "locked_test_access": False,
    }


def _used_state_ids() -> set[str]:
    used: set[str] = set()
    for directory in EXCLUDED_DATASETS:
        for row in read_jsonl(directory / "train.jsonl"):
            if row["state_type"] != "format_root":
                used.add(str(row["state_id"]))
    return used


def build_bundle(output: Path = DEFAULT_OUTPUT, *, force: bool = False) -> tuple[Path, dict[str, Any]]:
    output = Path(output)
    if (output / "manifest.json").is_file() and not force:
        return output, audit_bundle(output)
    universe = read_json(phase1.DEFAULT_OUTPUT / "universe.json")
    train_secrets = read_json(phase1.DEFAULT_OUTPUT / "train_secrets.json")
    dev_secrets = read_json(phase1.DEFAULT_OUTPUT / "dev_secrets.json")
    dev_records = read_jsonl(phase1.DEFAULT_OUTPUT / "dev_diagnostic_states.jsonl")
    used_before = _used_state_ids()
    dev_keys = {_history_key(record) for record in dev_records if record["history"]}
    pools = _targeted_state_pools(train_secrets, universe, POOL_TOTAL, phase1.SEED)
    selected: list[tuple[dict[str, Any], str]] = []
    selected_ids: set[str] = set()
    target_counts: Counter[str] = Counter()
    available_counts: dict[str, int] = {}
    for kind, quota in QUOTAS.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in pools[kind]:
            state_id = str(record["state_id"])
            if state_id in used_before or _history_key(record) in dev_keys:
                continue
            grouped[str(record["facts"]["oracle_action"])].append(record)
        available_counts[kind] = sum(len(records) for records in grouped.values())
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
                    state_id = str(record["state_id"])
                    if state_id in selected_ids:
                        continue
                    selected.append((record, kind))
                    selected_ids.add(state_id)
                    target_counts[target] += 1
                    chosen += 1
                    progressed = True
                    break
            if not progressed:
                raise RuntimeError(f"growth ladder cannot fill {kind}: {chosen}/{quota}")
    # Use a declared deterministic order so each intermediate checkpoint has exact,
    # auditable unique-state accounting rather than relying on DataLoader internals.
    selected.sort(key=lambda item: _rank("training-order", f"{item[1]}:{item[0]['state_id']}"))
    rows: list[dict[str, Any]] = []
    for index, (record, kind) in enumerate(selected):
        history = [(item["guess"], item["feedback"]) for item in record["history"]]
        target = str(record["facts"]["oracle_action"])
        rows.append(
            {
                "example_id": f"coverage-growth-{index:06d}-{record['state_id']}",
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
    prefix_composition = {}
    for coverage in MILESTONES:
        count = coverage - PARENT_COVERAGE
        prefix_composition[str(coverage)] = dict(sorted(Counter(row["state_type"] for row in rows[:count]).items()))
    output.mkdir(parents=True, exist_ok=True)
    train_path = write_jsonl(output / "train.jsonl", rows)
    write_json(output / "universe.json", universe)
    write_json(output / "train_secrets.json", train_secrets)
    write_json(output / "dev_secrets.json", dev_secrets)
    write_jsonl(output / "dev_diagnostic_states.jsonl", dev_records)
    manifest = {
        "curriculum_id": CURRICULUM_ID,
        "experiment_id": EXPERIMENT_ID,
        "rows": len(rows),
        "unique_states": len(selected_ids),
        "multi_turn_only": True,
        "parent_coverage": PARENT_COVERAGE,
        "maximum_cumulative_coverage": MILESTONES[-1],
        "milestones": MILESTONES,
        "quotas": QUOTAS,
        "composition": dict(sorted(Counter(row["state_type"] for row in rows).items())),
        "prefix_composition": prefix_composition,
        "available_after_exclusions": available_counts,
        "excluded_dataset_hashes": {
            directory.relative_to(ROOT).as_posix(): sha256_file(directory / "train.jsonl") for directory in EXCLUDED_DATASETS
        },
        "excluded_state_ids": len(used_before),
        "target_cap": max(target_counts.values()),
        "prompt_version": "explicit-constraints-v2-compact",
        "ordering": "sha256_seeded_declared_training_order",
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
    used_before = _used_state_ids()
    dev_keys = {_history_key(record) for record in read_jsonl(directory / "dev_diagnostic_states.jsonl") if record["history"]}
    seen: set[str] = set()
    targets: Counter[str] = Counter()
    for row in rows:
        source = row["source_state"]
        state_id = str(row["state_id"])
        if source["secret_answer"] not in train_secrets or source["secret_answer"] in dev_secrets:
            raise AssertionError("growth-ladder secret leakage")
        if state_id in used_before or state_id in seen:
            raise AssertionError("growth-ladder state overlap")
        if not source["history"]:
            raise AssertionError("growth-ladder root example")
        seen.add(state_id)
        if _history_key(source) in dev_keys:
            raise AssertionError("growth-ladder development history collision")
        history = [(item["guess"], item["feedback"]) for item in source["history"]]
        if any(score_wordle(source["secret_answer"], guess) != feedback for guess, feedback in history):
            raise AssertionError("growth-ladder feedback mismatch")
        if row["prompt"] != _explicit_feedback_messages(history):
            raise AssertionError("growth-ladder prompt drift")
        if row["completion"] != [{"role": "assistant", "content": f"Final answer: {row['target_word']}"}]:
            raise AssertionError("growth-ladder completion drift")
        targets[row["target_word"]] += 1
    if len(rows) != ROWS or len(seen) != ROWS or Counter(row["state_type"] for row in rows) != Counter(QUOTAS):
        raise AssertionError("growth-ladder composition drift")
    if max(targets.values()) > TARGET_CAP:
        raise AssertionError("growth-ladder target cap drift")
    if manifest["train_sha256"] != sha256_file(directory / "train.jsonl"):
        raise AssertionError("growth-ladder data hash drift")
    return {
        "status": "passed",
        "directory": directory.as_posix(),
        "rows": len(rows),
        "unique_states": len(seen),
        "multi_turn_only": True,
        "excluded_state_ids": len(used_before),
        "target_cap": max(targets.values()),
        "prefix_composition": manifest["prefix_composition"],
        "checks": [
            "training_only_secrets",
            "feedback_recomputed",
            "prior_state_disjoint",
            "development_history_disjoint",
            "non_root_only",
            "prompt_exact",
            "completion_exact",
            "target_cap",
            "declared_prefix_order",
            "locked_test_unread",
        ],
        "train_sha256": sha256_file(directory / "train.jsonl"),
        "locked_test_access": False,
    }


def build_spec(directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    spec = {
        "experiment_id": EXPERIMENT_ID,
        "curriculum_id": CURRICULUM_ID,
        "backend": BACKEND_ID,
        "seed": phase1.SEED,
        "parent_coverage": PARENT_COVERAGE,
        "milestones": MILESTONES,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "maximum_steps": MAX_STEPS,
        "maximum_new_examples": ROWS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "word_token_weight": 8.0,
        "max_length": 320,
        "optimizer": "fresh_AdamW_parent_optimizer_unavailable",
        "scheduler": "single_5pct_warmup_cosine_planned_to_20480",
        "conditional_stop_policy": {
            "reliability": "compliance >= 0.99 and invalid <= 0.01",
            "wins": "candidate wins >= last accepted wins",
            "meaningful_growth": "more wins, or equal wins plus >=2 singleton answers, >=5pp posterior improvement, >=5pp turn-2 improvement, >=3pp repeat improvement, or >=3pp target-accuracy improvement",
            "regression_guards": "singleton no worse by >1/74; posterior, turn-2, and repeats no worse by >5pp",
        },
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
    fixed = {
        "experiment_id": EXPERIMENT_ID,
        "curriculum_id": CURRICULUM_ID,
        "backend": BACKEND_ID,
        "seed": phase1.SEED,
        "parent_coverage": PARENT_COVERAGE,
        "milestones": MILESTONES,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "maximum_steps": MAX_STEPS,
        "maximum_new_examples": ROWS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "word_token_weight": 8.0,
        "max_length": 320,
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
    }
    drift = {key: (value, spec.get(key)) for key, value in fixed.items() if spec.get(key) != value}
    if drift:
        raise ValueError(f"growth-ladder spec drift: {drift}")
    if spec["parent"] != audit_parent() or spec["data"] != audit_bundle(directory) or spec["protocol"] != audit_protocol():
        raise ValueError("growth-ladder provenance binding drift")


def build_force_spec(directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    start = FORCE_PARENT_COVERAGE - PARENT_COVERAGE
    end = FORCE_TARGET_COVERAGE - PARENT_COVERAGE
    spec = {
        "experiment_id": "GEMMA-270M-COVERAGE-FORCED-15K-004",
        "curriculum_id": CURRICULUM_ID,
        "backend": "GEMMA-270M-FULL-COVERAGE-FORCED-15K-001",
        "seed": phase1.SEED,
        "parent_coverage": FORCE_PARENT_COVERAGE,
        "target_coverage": FORCE_TARGET_COVERAGE,
        "milestones": FORCE_MILESTONES,
        "dataset_slice": {"start_inclusive": start, "end_exclusive": end, "rows": FORCE_ROWS},
        "steps": FORCE_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "word_token_weight": 8.0,
        "max_length": 320,
        "optimizer": "fresh_AdamW_parent_optimizer_unavailable",
        "scheduler": "fresh_5pct_warmup_cosine_to_15360",
        "forced_override": "user_requested_15k_after_10k_stop_rule_failed",
        "parent": audit_force_parent(),
        "data": audit_bundle(directory),
        "protocol": audit_protocol(),
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
    }
    validate_force_spec(spec, directory)
    return spec


def validate_force_spec(spec: dict[str, Any], directory: Path = DEFAULT_OUTPUT) -> None:
    fixed = {
        "experiment_id": "GEMMA-270M-COVERAGE-FORCED-15K-004",
        "curriculum_id": CURRICULUM_ID,
        "backend": "GEMMA-270M-FULL-COVERAGE-FORCED-15K-001",
        "seed": phase1.SEED,
        "parent_coverage": FORCE_PARENT_COVERAGE,
        "target_coverage": FORCE_TARGET_COVERAGE,
        "milestones": FORCE_MILESTONES,
        "dataset_slice": {"start_inclusive": 3072, "end_exclusive": 8192, "rows": FORCE_ROWS},
        "steps": FORCE_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "word_token_weight": 8.0,
        "max_length": 320,
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
    }
    drift = {key: (value, spec.get(key)) for key, value in fixed.items() if spec.get(key) != value}
    if drift:
        raise ValueError(f"forced-15k spec drift: {drift}")
    if spec["parent"] != audit_force_parent() or spec["data"] != audit_bundle(directory) or spec["protocol"] != audit_protocol():
        raise ValueError("forced-15k provenance binding drift")


def growth_decision(previous: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    singleton_unit = 1.0 / 74.0
    improvements = []
    if candidate["wins"] > previous["wins"]:
        improvements.append("wins")
    if candidate["singleton_answer_accuracy"] - previous["singleton_answer_accuracy"] >= 2 * singleton_unit - 1e-12:
        improvements.append("singleton")
    if previous["posterior_constraint_violation_rate"] - candidate["posterior_constraint_violation_rate"] >= 0.05 - 1e-12:
        improvements.append("posterior_legality")
    if previous["turn_2_posterior_constraint_violation_rate"] - candidate["turn_2_posterior_constraint_violation_rate"] >= 0.05 - 1e-12:
        improvements.append("turn_2_legality")
    if previous["repeat_guess_rate"] - candidate["repeat_guess_rate"] >= 0.03 - 1e-12:
        improvements.append("repeat_rate")
    if candidate["action_target_accuracy"] - previous["action_target_accuracy"] >= 0.03 - 1e-12:
        improvements.append("target_accuracy")
    reliability = candidate["terminal_marker_compliance"] >= 0.99 and candidate["invalid_guess_rate"] <= 0.01
    non_regression = (
        candidate["wins"] >= previous["wins"]
        and candidate["singleton_answer_accuracy"] >= previous["singleton_answer_accuracy"] - singleton_unit - 1e-12
        and candidate["posterior_constraint_violation_rate"] <= previous["posterior_constraint_violation_rate"] + 0.05 + 1e-12
        and candidate["turn_2_posterior_constraint_violation_rate"] <= previous["turn_2_posterior_constraint_violation_rate"] + 0.05 + 1e-12
        and candidate["repeat_guess_rate"] <= previous["repeat_guess_rate"] + 0.05 + 1e-12
    )
    continue_training = reliability and non_regression and bool(improvements)
    return {
        "continue": continue_training,
        "reliability_passed": reliability,
        "non_regression_passed": non_regression,
        "meaningful_improvements": improvements,
        "previous_examples": previous["examples_seen"],
        "candidate_examples": candidate["examples_seen"],
    }


def _evaluate_model(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    examples_seen: int,
    run_dir: Path,
    directory: Path,
) -> dict[str, Any]:
    allowed = [
        line.strip().upper()
        for line in (ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    universe = read_json(directory / "universe.json")
    dev_answers = read_json(directory / "dev_secrets.json")[:32]
    dev_records = read_jsonl(directory / "dev_diagnostic_states.jsonl")
    new_examples_seen = examples_seen - PARENT_COVERAGE
    training_records = [row["source_state"] for row in rows[:new_examples_seen]]
    previous_messages = generation.inference_messages
    previous_generation = dict(generation.GENERATION_CONFIG)
    try:
        model.eval()
        model.config.use_cache = True
        set_seed(phase1.SEED)
        generation.inference_messages = _explicit_feedback_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update({"do_sample": False, "max_new_tokens": 128, "use_cache": True})
        games, gameplay = evaluate(model, tokenizer, dev_answers, allowed, universe)
        diagnostics_dir, diagnostics = run_state_diagnostics(
            model,
            tokenizer,
            dev_records,
            training_records,
            allowed,
            universe,
            run_dir / f"eval-coverage-{examples_seen:06d}",
        )
        retention_rows, retention = evaluate_retention(model, tokenizer, read_jsonl(DATA / "retention_probes_v1.jsonl"))
        write_jsonl(run_dir / f"eval-coverage-{examples_seen:06d}-games.jsonl", games)
        write_jsonl(run_dir / f"eval-coverage-{examples_seen:06d}-retention.jsonl", retention_rows)
        summary = {
            "status": "dev_evaluated",
            "experiment_id": EXPERIMENT_ID,
            "checkpoint": f"coverage-{examples_seen:06d}",
            "cumulative_unique_coverage": examples_seen,
            "new_unique_examples_seen": new_examples_seen,
            "locked_test_access": False,
            "gameplay": gameplay,
            "diagnostics": diagnostics,
            "diagnostics_dir": str(diagnostics_dir),
            "retention": retention,
        }
        write_json(run_dir / f"eval-coverage-{examples_seen:06d}-summary.json", summary)
        return summary
    finally:
        generation.inference_messages = previous_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(previous_generation)
        model.config.use_cache = False
        model.train()


def train(spec: dict[str, Any], directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    validate_spec(spec, directory)
    preflight = full_finetune_vram_preflight(parameter_count=EXPECTED_PARAMETER_COUNT)
    if not preflight["ready"]:
        raise RuntimeError("growth-ladder VRAM preflight failed")
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:10]
    run_dir = ARTIFACTS / "runs" / f"gemma-270m-coverage-growth-s{phase1.SEED}-{digest}"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    write_json(run_dir / "spec.json", spec)
    write_json(run_dir / "preflight.json", preflight)
    rows = read_jsonl(Path(directory) / "train.jsonl")
    tokenizer = load_tokenizer(PARENT_CHECKPOINT)
    dataset = CompletionDataset(rows, tokenizer, 320, word_token_weight=8.0)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=Collator(tokenizer.pad_token_id))
    model = AutoModelForCausalLM.from_pretrained(
        PARENT_CHECKPOINT,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    warmup = max(1, int(MAX_STEPS * 0.05))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / (MAX_STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    logs: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    evaluated: dict[str, dict[str, Any]] = {}
    tokens = 0
    started = time.perf_counter()
    previous_metrics = spec["parent"]["metrics"]
    stop_reason = None
    torch.cuda.reset_peak_memory_stats()
    set_seed(phase1.SEED)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to("cuda") for key, value in batch.items()}
            weights = batch.pop("loss_weights")
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            loss = weighted_causal_lm_loss(output.logits, batch["labels"], weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            tokens += int(batch["attention_mask"].sum())
            coverage = PARENT_COVERAGE + step * BATCH_SIZE
            logs.append(
                {
                    "optimizer_step": step,
                    "new_examples_seen": step * BATCH_SIZE,
                    "cumulative_unique_coverage": coverage,
                    "train_loss": float(loss.detach()),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "optimizer_tokens": tokens,
                    "wall_time_s": time.perf_counter() - started,
                }
            )
            if step not in CHECKPOINT_STEPS:
                continue
            checkpoint = run_dir / "checkpoints" / f"coverage-{coverage:06d}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            evaluation = _evaluate_model(model, tokenizer, rows, coverage, run_dir, Path(directory))
            metrics = compact_metrics(evaluation, coverage)
            evaluated[str(coverage)] = metrics
            decision = growth_decision(previous_metrics, metrics)
            decisions.append(decision)
            write_json(run_dir / f"decision-coverage-{coverage:06d}.json", decision)
            write_jsonl(run_dir / "train_metrics.jsonl", logs)
            write_json(run_dir / "interim_results.json", {"metrics": evaluated, "decisions": decisions, "locked_test_access": False})
            if coverage != MILESTONES[-1] and not decision["continue"]:
                stop_reason = f"no_meaningful_growth_at_{coverage}"
                break
            previous_metrics = metrics
        accounting = {
            "parent_coverage": PARENT_COVERAGE,
            "maximum_planned_coverage": MILESTONES[-1],
            "final_coverage": max(int(value) for value in evaluated),
            "new_examples_seen": max(int(value) for value in evaluated) - PARENT_COVERAGE,
            "optimizer_steps": logs[-1]["optimizer_step"],
            "optimizer_tokens": tokens,
            "wall_time_s": time.perf_counter() - started,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "learning_rate": LEARNING_RATE,
            "trainable_parameters": EXPECTED_PARAMETER_COUNT,
            "stopped_early": stop_reason is not None,
            "stop_reason": stop_reason,
            "locked_test_access": False,
        }
        write_jsonl(run_dir / "train_metrics.jsonl", logs)
        write_json(run_dir / "accounting.json", accounting)
        result = {
            "status": "stopped_no_meaningful_growth" if stop_reason else "maximum_coverage_completed",
            "experiment_id": EXPERIMENT_ID,
            "run_dir": str(run_dir),
            "initial_loss": logs[0]["train_loss"],
            "final_loss": logs[-1]["train_loss"],
            "metrics": {str(PARENT_COVERAGE): spec["parent"]["metrics"], **evaluated},
            "decisions": decisions,
            "accounting": accounting,
            "locked_test_access": False,
        }
        write_json(run_dir / "comparison_summary.json", result)
        write_json(run_dir / "summary.json", result)
        return result
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def force_train_15k(spec: dict[str, Any], directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    validate_force_spec(spec, directory)
    preflight = full_finetune_vram_preflight(parameter_count=EXPECTED_PARAMETER_COUNT)
    if not preflight["ready"]:
        raise RuntimeError("forced-15k VRAM preflight failed")
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:10]
    run_dir = ARTIFACTS / "runs" / f"gemma-270m-coverage-forced-15k-s{phase1.SEED}-{digest}"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    write_json(run_dir / "spec.json", spec)
    write_json(run_dir / "preflight.json", preflight)
    all_rows = read_jsonl(Path(directory) / "train.jsonl")
    start = spec["dataset_slice"]["start_inclusive"]
    end = spec["dataset_slice"]["end_exclusive"]
    rows = all_rows[start:end]
    if len(rows) != FORCE_ROWS or len({row["state_id"] for row in rows}) != FORCE_ROWS:
        raise AssertionError("forced-15k slice drift")
    tokenizer = load_tokenizer(FORCE_PARENT_CHECKPOINT)
    dataset = CompletionDataset(rows, tokenizer, 320, word_token_weight=8.0)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=Collator(tokenizer.pad_token_id))
    model = AutoModelForCausalLM.from_pretrained(
        FORCE_PARENT_CHECKPOINT,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    warmup = max(1, int(FORCE_STEPS * 0.05))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / (FORCE_STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    checkpoint_steps = {(coverage - FORCE_PARENT_COVERAGE) // BATCH_SIZE: coverage for coverage in FORCE_MILESTONES}
    logs: list[dict[str, Any]] = []
    evaluated: dict[str, dict[str, Any]] = {}
    tokens = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    set_seed(phase1.SEED)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to("cuda") for key, value in batch.items()}
            weights = batch.pop("loss_weights")
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            loss = weighted_causal_lm_loss(output.logits, batch["labels"], weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            tokens += int(batch["attention_mask"].sum())
            coverage = FORCE_PARENT_COVERAGE + step * BATCH_SIZE
            logs.append(
                {
                    "optimizer_step": step,
                    "new_examples_seen": step * BATCH_SIZE,
                    "cumulative_unique_coverage": coverage,
                    "train_loss": float(loss.detach()),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "optimizer_tokens": tokens,
                    "wall_time_s": time.perf_counter() - started,
                }
            )
            if step not in checkpoint_steps:
                continue
            checkpoint_coverage = checkpoint_steps[step]
            checkpoint = run_dir / "checkpoints" / f"coverage-{checkpoint_coverage:06d}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            evaluation = _evaluate_model(model, tokenizer, all_rows, checkpoint_coverage, run_dir, Path(directory))
            evaluated[str(checkpoint_coverage)] = compact_metrics(evaluation, checkpoint_coverage)
            write_jsonl(run_dir / "train_metrics.jsonl", logs)
            write_json(run_dir / "interim_results.json", {"metrics": evaluated, "locked_test_access": False})
        accounting = {
            "parent_coverage": FORCE_PARENT_COVERAGE,
            "final_coverage": FORCE_TARGET_COVERAGE,
            "new_examples_seen": FORCE_ROWS,
            "optimizer_steps": FORCE_STEPS,
            "optimizer_tokens": tokens,
            "wall_time_s": time.perf_counter() - started,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "learning_rate": LEARNING_RATE,
            "trainable_parameters": EXPECTED_PARAMETER_COUNT,
            "forced_override": True,
            "locked_test_access": False,
        }
        write_jsonl(run_dir / "train_metrics.jsonl", logs)
        write_json(run_dir / "accounting.json", accounting)
        result = {
            "status": "forced_15k_completed",
            "experiment_id": spec["experiment_id"],
            "run_dir": str(run_dir),
            "initial_loss": logs[0]["train_loss"],
            "final_loss": logs[-1]["train_loss"],
            "metrics": {str(FORCE_PARENT_COVERAGE): spec["parent"]["metrics"], **evaluated},
            "accounting": accounting,
            "locked_test_access": False,
        }
        write_json(run_dir / "comparison_summary.json", result)
        write_json(run_dir / "summary.json", result)
        return result
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conditional Gemma 270M coverage growth ladder")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--force", action="store_true")
    sub.add_parser("preflight")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--dry-run", action="store_true")
    force_parser = sub.add_parser("force-15k")
    force_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "build":
        directory, audit = build_bundle(force=args.force)
        result = {"directory": str(directory), "audit": audit}
    elif args.command == "preflight":
        result = {
            "status": "ready",
            "parent": audit_parent(),
            "data": audit_bundle(),
            "protocol": audit_protocol(),
            "vram": full_finetune_vram_preflight(),
            "locked_test_access": False,
        }
    elif args.command == "train":
        spec = build_spec()
        result = {"status": "dry_run_ready", "spec": spec} if args.dry_run else train(spec)
    else:
        spec = build_force_spec()
        result = {"status": "forced_15k_dry_run_ready", "spec": spec} if args.dry_run else force_train_15k(spec)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
