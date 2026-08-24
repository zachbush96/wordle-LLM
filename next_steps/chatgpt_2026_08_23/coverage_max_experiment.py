from __future__ import annotations

"""Maximum-coverage full-parameter Gemma 270M Wordle experiment."""

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
from wordle_lab.common import (
    ARTIFACTS,
    DATA,
    MODEL_DIR,
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
from wordle_lab.data.canonical import _facts, generate_canonical_states
from wordle_lab.experiments.common_curriculum import _targeted_state_pools, ranked_common_words
from wordle_lab.experiments.intervention_sweep import _explicit_feedback_messages
from wordle_lab.methods.sft import Collator, CompletionDataset, weighted_causal_lm_loss
from wordle_lab.models import assert_supported_model, load_tokenizer
from wordle_lab.protocol import generation
from wordle_lab.protocol.env import score_wordle
from wordle_lab.protocol.evaluator import evaluate
from wordle_lab.protocol.oracle import GreedyPartitionOracle
from wordle_lab.protocol.retention import evaluate_retention

from .full_finetune import EXPECTED_PARAMETER_COUNT, full_finetune_vram_preflight, load_full_checkpoint
from .full_finetune_experiment import EXPECTED_ALLOWED_WORDS_SHA256, EXPECTED_RETENTION_SHA256, audit_protocol


EXPERIMENT_ID = "GEMMA-270M-COVERAGE-MAX-001"
CURRICULUM_ID = "COMMON-WORD-CURRICULUM-006"
BACKEND_ID = "GEMMA-270M-FULL-COVERAGE-001"
SEED = 2026
UNIVERSE_SIZE = 128
TRAIN_SECRET_COUNT = 96
TRAIN_ROWS = 4096
BATCH_SIZE = 4
MAX_STEPS = TRAIN_ROWS // BATCH_SIZE
CHECKPOINT_STEPS = [256, 512, 768, 1024]
EXAMPLES_SEEN = [1024, 2048, 3072, 4096]
TARGET_CAP = 48
QUOTAS = {
    "format_root": 32,
    "turn_2": 1280,
    "low_posterior": 1024,
    "true_singleton": 1536,
    "later_broad": 224,
}
DEFAULT_OUTPUT = ROOT / "data" / "common-curriculum-006" / "u128-train96-n4096"


def _history_key(source: Mapping[str, Any]) -> str:
    visible = [{"guess": item["guess"], "feedback": item["feedback"]} for item in source["history"]]
    return sha256_text(canonical_json(visible))


def _rank(label: str, value: str) -> str:
    return sha256_text(canonical_json({"seed": SEED, "label": label, "value": value}))


def _select_coverage_rows(
    pools: Mapping[str, Sequence[dict[str, Any]]],
    train_secrets: Sequence[str],
    dev_history_keys: set[str],
) -> list[tuple[dict[str, Any], str]]:
    selected: list[tuple[dict[str, Any], str]] = []
    target_counts: Counter[str] = Counter()
    used_states: set[str] = set()

    root = list(pools["format_root"])
    if not root:
        raise RuntimeError("coverage-max pool has no root anchor")
    selected.extend((root[0], "format_root") for _ in range(QUOTAS["format_root"]))

    def add(record: dict[str, Any], kind: str) -> bool:
        state_id = str(record["state_id"])
        target = str(record["facts"]["oracle_action"])
        if state_id in used_states or _history_key(record) in dev_history_keys:
            return False
        if target_counts[target] >= TARGET_CAP:
            return False
        selected.append((record, kind))
        used_states.add(state_id)
        target_counts[target] += 1
        return True

    singleton_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pools["true_singleton"]:
        singleton_by_target[str(record["facts"]["oracle_action"])].append(record)
    for target in singleton_by_target:
        singleton_by_target[target].sort(key=lambda row: _rank("singleton", str(row["state_id"])))
    missing = [target for target in train_secrets if target not in singleton_by_target]
    if missing:
        raise RuntimeError(f"coverage-max singleton pool misses training targets: {missing}")
    # Guarantee at least one distinct singleton history for every training answer.
    for target in sorted(train_secrets):
        while singleton_by_target[target] and not add(singleton_by_target[target].pop(), "true_singleton"):
            pass
    singleton_seed_count = sum(kind == "true_singleton" for _, kind in selected)
    if singleton_seed_count != len(train_secrets):
        raise RuntimeError("coverage-max could not seed every singleton target")

    for kind in ("turn_2", "low_posterior", "true_singleton", "later_broad"):
        quota = QUOTAS[kind]
        already = sum(selected_kind == kind for _, selected_kind in selected)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in pools[kind]:
            if str(record["state_id"]) not in used_states and _history_key(record) not in dev_history_keys:
                grouped[str(record["facts"]["oracle_action"])].append(record)
        for target, records in grouped.items():
            records.sort(key=lambda row: _rank(f"{kind}:{target}", str(row["state_id"])), reverse=True)
        target_order = sorted(grouped, key=lambda target: _rank(f"{kind}:target", target))
        while already < quota:
            progressed = False
            target_order.sort(key=lambda target: (target_counts[target], _rank(f"{kind}:round", target)))
            for target in target_order:
                if already >= quota:
                    break
                while grouped[target]:
                    record = grouped[target].pop()
                    if add(record, kind):
                        already += 1
                        progressed = True
                        break
                if target_counts[target] >= TARGET_CAP:
                    grouped[target].clear()
            if not progressed:
                available = sum(len(records) for records in grouped.values())
                raise RuntimeError(
                    f"coverage-max cannot fill {kind}: selected={already}, quota={quota}, "
                    f"remaining={available}, target_cap={TARGET_CAP}"
                )
    if len(selected) != TRAIN_ROWS:
        raise AssertionError(f"coverage-max selected {len(selected)} rows, expected {TRAIN_ROWS}")
    # A deterministic hash order plus DataLoader shuffle makes every dose a
    # nested no-replacement coverage sample while avoiding kind blocks.
    return sorted(selected, key=lambda item: _rank("final-order", f"{item[1]}:{item[0]['state_id']}"))


def build_bundle(output: Path = DEFAULT_OUTPUT, *, force: bool = False) -> tuple[Path, dict[str, Any]]:
    output = Path(output)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file() and not force:
        return output, audit_bundle(output)
    universe = ranked_common_words(UNIVERSE_SIZE)
    shuffled = list(universe)
    random.Random(SEED).shuffle(shuffled)
    train_secrets = sorted(shuffled[:TRAIN_SECRET_COUNT])
    dev_secrets = sorted(shuffled[TRAIN_SECRET_COUNT:])
    dev_records = generate_canonical_states(
        dev_secrets,
        "common_dev_diagnostic",
        128,
        seed=SEED,
        answer_vocabulary=universe,
    )
    dev_history_keys = {_history_key(record) for record in dev_records if record["history"]}
    pools = _targeted_state_pools(train_secrets, universe, TRAIN_ROWS, SEED)
    selected = _select_coverage_rows(pools, train_secrets, dev_history_keys)
    rows: list[dict[str, Any]] = []
    state_manifest: list[dict[str, Any]] = []
    target_frequency = Counter(str(record["facts"]["oracle_action"]) for record, _ in selected)
    for index, (record, kind) in enumerate(selected):
        history = [(item["guess"], item["feedback"]) for item in record["history"]]
        target = str(record["facts"]["oracle_action"])
        row = {
            "example_id": f"coverage-max-{index:06d}-{record['state_id']}",
            "state_id": record["state_id"],
            "source_state": record,
            "state_type": kind,
            "turn": record["turn"],
            "posterior_size": record["facts"]["posterior_count"],
            "target_word": target,
            "target_frequency": target_frequency[target],
            "prompt": _explicit_feedback_messages(history),
            "completion": [{"role": "assistant", "content": f"Final answer: {target}"}],
        }
        rows.append(row)
        state_manifest.append(
            {key: row[key] for key in (
                "example_id",
                "state_id",
                "state_type",
                "turn",
                "posterior_size",
                "target_word",
                "target_frequency",
            )}
        )
    output.mkdir(parents=True, exist_ok=True)
    rows_path = write_jsonl(output / "train.jsonl", rows)
    states_path = write_jsonl(output / "state_manifest.jsonl", state_manifest)
    dev_path = write_jsonl(output / "dev_diagnostic_states.jsonl", dev_records)
    write_json(output / "universe.json", universe)
    write_json(output / "train_secrets.json", train_secrets)
    write_json(output / "dev_secrets.json", dev_secrets)
    composition = Counter(row["state_type"] for row in rows)
    singleton_targets = {row["target_word"] for row in rows if row["state_type"] == "true_singleton"}
    manifest = {
        "curriculum_id": CURRICULUM_ID,
        "experiment_id": EXPERIMENT_ID,
        "seed": SEED,
        "universe_size": len(universe),
        "train_secret_count": len(train_secrets),
        "dev_secret_count": len(dev_secrets),
        "rendered_examples": len(rows),
        "unique_non_root_states": len({row["state_id"] for row in rows if row["state_type"] != "format_root"}),
        "examples_per_optimizer_epoch": TRAIN_ROWS,
        "state_copy_cap_non_root": 1,
        "target_cap_non_root": TARGET_CAP,
        "requested_quotas": QUOTAS,
        "achieved_composition": dict(sorted(composition.items())),
        "singleton_target_coverage": len(singleton_targets),
        "singleton_target_coverage_required": len(train_secrets),
        "posterior_distribution": dict(sorted(Counter(str(row["posterior_size"]) for row in rows).items())),
        "turn_distribution": dict(sorted(Counter(str(row["turn"]) for row in rows).items())),
        "target_frequency_distribution": dict(sorted(target_frequency.items())),
        "prompt_version": "explicit-constraints-v2-compact",
        "prompt_renderer": "wordle_lab.experiments.intervention_sweep._explicit_feedback_messages",
        "dev_non_root_history_collisions": 0,
        "locked_test_access": False,
        "hashes": {
            "train.jsonl": sha256_file(rows_path),
            "state_manifest.jsonl": sha256_file(states_path),
            "dev_diagnostic_states.jsonl": sha256_file(dev_path),
            "universe.json": sha256_file(output / "universe.json"),
            "train_secrets.json": sha256_file(output / "train_secrets.json"),
            "dev_secrets.json": sha256_file(output / "dev_secrets.json"),
        },
    }
    write_json(manifest_path, manifest)
    return output, audit_bundle(output)


def audit_bundle(directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    rows = read_jsonl(directory / "train.jsonl")
    state_manifest = read_jsonl(directory / "state_manifest.jsonl")
    dev_records = read_jsonl(directory / "dev_diagnostic_states.jsonl")
    universe = read_json(directory / "universe.json")
    train_secrets = set(read_json(directory / "train_secrets.json"))
    dev_secrets = set(read_json(directory / "dev_secrets.json"))
    if len(rows) != TRAIN_ROWS or len(state_manifest) != TRAIN_ROWS:
        raise AssertionError("coverage-max row count drift")
    if train_secrets & dev_secrets or len(train_secrets) != TRAIN_SECRET_COUNT or len(dev_secrets) != 32:
        raise AssertionError("coverage-max secret split drift")
    expected_hashes = {
        name: sha256_file(directory / name)
        for name in (
            "train.jsonl",
            "state_manifest.jsonl",
            "dev_diagnostic_states.jsonl",
            "universe.json",
            "train_secrets.json",
            "dev_secrets.json",
        )
    }
    if manifest.get("hashes") != expected_hashes:
        raise AssertionError("coverage-max artifact hash drift")
    dev_history_keys = {_history_key(record) for record in dev_records if record["history"]}
    oracle = GreedyPartitionOracle(universe)
    seen_non_root: set[str] = set()
    target_counts: Counter[str] = Counter()
    singleton_targets: set[str] = set()
    for row in rows:
        source = row["source_state"]
        secret = source["secret_answer"]
        if secret not in train_secrets or secret in dev_secrets:
            raise AssertionError("coverage-max label leakage")
        history = [(item["guess"], item["feedback"]) for item in source["history"]]
        if any(score_wordle(secret, guess) != feedback for guess, feedback in history):
            raise AssertionError("coverage-max feedback mismatch")
        if any(guess == secret for guess, _ in history):
            raise AssertionError("coverage-max contains post-solve state")
        facts = _facts(oracle, history, secret)
        if facts["posterior_count"] != row["posterior_size"] or facts["oracle_action"] != row["target_word"]:
            raise AssertionError("coverage-max oracle fact mismatch")
        if row["prompt"] != _explicit_feedback_messages(history):
            raise AssertionError("coverage-max prompt drift")
        if row["completion"] != [{"role": "assistant", "content": f"Final answer: {row['target_word']}"}]:
            raise AssertionError("coverage-max completion drift")
        if row["state_type"] != "format_root":
            if row["state_id"] in seen_non_root:
                raise AssertionError("coverage-max repeated non-root state")
            seen_non_root.add(row["state_id"])
            if _history_key(source) in dev_history_keys:
                raise AssertionError("coverage-max non-root development history collision")
            target_counts[row["target_word"]] += 1
        if row["state_type"] == "true_singleton":
            if row["posterior_size"] != 1 or row["target_word"] != secret:
                raise AssertionError("coverage-max singleton label mismatch")
            singleton_targets.add(row["target_word"])
    if max(target_counts.values()) > TARGET_CAP:
        raise AssertionError("coverage-max target cap exceeded")
    if singleton_targets != train_secrets:
        raise AssertionError("coverage-max singleton target coverage incomplete")
    return {
        "status": "passed",
        "directory": directory.as_posix(),
        "rows": len(rows),
        "unique_non_root_states": len(seen_non_root),
        "singleton_target_coverage": len(singleton_targets),
        "train_secret_count": len(train_secrets),
        "dev_secret_count": len(dev_secrets),
        "target_cap_non_root": max(target_counts.values()),
        "hashes": expected_hashes,
        "checks": [
            "held_out_secret_split",
            "feedback_recomputed",
            "no_post_solve_states",
            "oracle_facts_recomputed",
            "non_root_state_unique",
            "target_frequency_capped",
            "all_training_singleton_targets_covered",
            "prompt_renderer_exact",
            "completion_target_exact",
            "dev_non_root_history_separation",
            "locked_test_unread",
        ],
        "locked_test_access": False,
    }


def build_spec(directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    audit = audit_bundle(directory)
    protocol = audit_protocol()
    metadata = assert_supported_model()
    spec = {
        "experiment_id": EXPERIMENT_ID,
        "curriculum_id": CURRICULUM_ID,
        "backend": BACKEND_ID,
        "method": "full_parameter_sft",
        "representation": "coverage_max_explicit_feedback",
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "learning_rate": 5e-5,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": BATCH_SIZE,
        "max_length": 320,
        "warmup_fraction": 0.05,
        "max_grad_norm": 1.0,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "checkpoint_examples_seen": dict(zip((str(step) for step in CHECKPOINT_STEPS), EXAMPLES_SEEN)),
        "word_token_weight": 8.0,
        "precision": "bfloat16",
        "quantization": "none_16bit",
        "optimizer": "torch.optim.AdamW",
        "scheduler": "linear_warmup_5pct_cosine",
        "training_epochs": 1.0,
        "shuffle_without_replacement": True,
        "model": metadata,
        "data": audit,
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
        "intended_change": "replace repeated 512-row exposure with one pass over 4096 coverage-max rows",
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


def validate_spec(spec: dict[str, Any], directory: Path = DEFAULT_OUTPUT) -> None:
    fixed = {
        "experiment_id": EXPERIMENT_ID,
        "curriculum_id": CURRICULUM_ID,
        "backend": BACKEND_ID,
        "method": "full_parameter_sft",
        "representation": "coverage_max_explicit_feedback",
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "learning_rate": 5e-5,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": BATCH_SIZE,
        "max_length": 320,
        "warmup_fraction": 0.05,
        "max_grad_norm": 1.0,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "word_token_weight": 8.0,
        "precision": "bfloat16",
        "quantization": "none_16bit",
        "training_epochs": 1.0,
        "shuffle_without_replacement": True,
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
        "harness_selected_guess": False,
    }
    drift = {key: (value, spec.get(key)) for key, value in fixed.items() if spec.get(key) != value}
    if drift:
        raise ValueError(f"coverage-max spec drift: {drift}")
    if spec["data"] != audit_bundle(directory):
        raise ValueError("coverage-max data binding drift")
    if spec["protocol"] != audit_protocol():
        raise ValueError("coverage-max protocol binding drift")
    if spec["evaluation"]["allowed_words_sha256"] != EXPECTED_ALLOWED_WORDS_SHA256:
        raise ValueError("coverage-max allowed-word drift")
    if spec["evaluation"]["retention_probes_sha256"] != EXPECTED_RETENTION_SHA256:
        raise ValueError("coverage-max retention drift")


def prepare_run(spec: dict[str, Any], directory: Path = DEFAULT_OUTPUT) -> Path:
    validate_spec(spec, directory)
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:10]
    run_dir = ARTIFACTS / "runs" / f"gemma-270m-coverage-max-s{SEED}-{digest}"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    write_json(run_dir / "spec.json", spec)
    write_json(run_dir / "dataset_manifest.json", spec["data"])
    return run_dir


def train(spec: dict[str, Any], directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    validate_spec(spec, directory)
    preflight = full_finetune_vram_preflight(parameter_count=EXPECTED_PARAMETER_COUNT)
    if not preflight["ready"]:
        raise RuntimeError(f"coverage-max preflight blocked: {preflight['status']}")
    rows = read_jsonl(Path(directory) / "train.jsonl")
    run_dir = prepare_run(spec, directory)
    write_json(run_dir / "preflight.json", preflight)
    tokenizer = load_tokenizer()
    dataset = CompletionDataset(rows, tokenizer, 320, word_token_weight=8.0)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, collate_fn=Collator(tokenizer.pad_token_id))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"coverage-max trainable parameter drift: {trainable}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    warmup = max(1, int(MAX_STEPS * 0.05))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / (MAX_STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    iterator = iter(loader)
    logs: list[dict[str, Any]] = []
    optimizer_tokens = 0
    weighted_completion_tokens = 0.0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    set_seed(SEED)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        for step in range(1, MAX_STEPS + 1):
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError("coverage-max unexpectedly exhausted before one complete epoch") from exc
            batch = {key: value.to("cuda") for key, value in batch.items()}
            loss_weights = batch.pop("loss_weights")
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            loss = weighted_causal_lm_loss(output.logits, batch["labels"], loss_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_tokens += int(batch["attention_mask"].sum())
            weighted_completion_tokens += float(loss_weights.sum())
            logs.append(
                {
                    "optimizer_step": step,
                    "examples_seen": step * BATCH_SIZE,
                    "train_loss": float(loss.detach()),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "optimizer_tokens": optimizer_tokens,
                    "wall_time_s": time.perf_counter() - started,
                }
            )
            if step in CHECKPOINT_STEPS:
                checkpoint = run_dir / "checkpoints" / f"step-{step:06d}"
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
        accounting = {
            "backend_id": BACKEND_ID,
            "train_examples": len(dataset),
            "unique_non_root_states": spec["data"]["unique_non_root_states"],
            "optimizer_steps": MAX_STEPS,
            "examples_seen": MAX_STEPS * BATCH_SIZE,
            "training_epochs": 1.0,
            "effective_batch_size": BATCH_SIZE,
            "optimizer_tokens": optimizer_tokens,
            "wall_time_s": time.perf_counter() - started,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "trainable_parameters": trainable,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_steps": CHECKPOINT_STEPS,
            "checkpoint_examples_seen": spec["checkpoint_examples_seen"],
            "loss_mode": "word_focused",
            "word_token_weight": 8.0,
            "weighted_completion_tokens": weighted_completion_tokens,
            "locked_test_access": False,
        }
        write_jsonl(run_dir / "train_metrics.jsonl", logs)
        write_json(run_dir / "accounting.json", accounting)
        summary = {
            "status": "coverage_max_training_completed",
            "experiment_id": EXPERIMENT_ID,
            "run_dir": str(run_dir),
            "initial_loss": logs[0]["train_loss"],
            "final_loss": logs[-1]["train_loss"],
            "accounting": accounting,
            "preflight": preflight,
            "locked_test_access": False,
        }
        write_json(run_dir / "summary.json", summary)
        return summary
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def evaluate_checkpoint(run_dir: Path, checkpoint: str, directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    run_dir, directory = Path(run_dir), Path(directory)
    spec = read_json(run_dir / "spec.json")
    validate_spec(spec, directory)
    expected = {f"step-{step:06d}" for step in CHECKPOINT_STEPS}
    if checkpoint not in expected:
        raise ValueError(f"checkpoint must be one of {sorted(expected)}")
    summary_path = run_dir / f"eval-{checkpoint}-summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation: {summary_path}")
    model, tokenizer = load_full_checkpoint(run_dir / "checkpoints" / checkpoint)
    allowed = [
        line.strip().upper()
        for line in (ROOT / "data" / "wordlists" / "tabatkins_wordle_list_pinned.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    universe = read_json(directory / "universe.json")
    dev_answers = read_json(directory / "dev_secrets.json")[:32]
    dev_records = read_jsonl(directory / "dev_diagnostic_states.jsonl")
    previous_messages = generation.inference_messages
    previous_generation = dict(generation.GENERATION_CONFIG)
    try:
        set_seed(SEED)
        generation.inference_messages = _explicit_feedback_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(spec["evaluation"]["generation"])
        games, gameplay = evaluate(model, tokenizer, dev_answers, allowed, universe)
        training_records = [row["source_state"] for row in read_jsonl(directory / "train.jsonl")]
        diagnostics_dir, diagnostics = run_state_diagnostics(
            model,
            tokenizer,
            dev_records,
            training_records,
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
            "examples_seen": spec["checkpoint_examples_seen"][str(int(checkpoint.split("-")[-1]))],
            "split": "balanced_002_dev_32",
            "decoder": "greedy",
            "prompt_variant": "explicit_feedback",
            "spec_sha256": sha256_text(canonical_json(spec)),
            "data": spec["data"],
            "protocol": spec["protocol"],
            "locked_test_access": False,
            "gameplay": gameplay,
            "diagnostics": diagnostics,
            "diagnostics_dir": str(diagnostics_dir),
            "retention": retention,
        }
        write_json(summary_path, summary)
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
        "examples_seen": summary["examples_seen"],
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
    baseline = read_json(ARTIFACTS / "runs" / "full-finetune-balanced-word-primary-s2026-57ba532ae7" / "eval-step-000600-summary.json")
    metrics = {
        "baseline_512_rows_repeated_600_steps": {
            "examples_seen": 2400,
            "wins": baseline["gameplay"]["wins"],
            "win_rate": baseline["gameplay"]["win_rate"],
            "terminal_marker_compliance": baseline["gameplay"]["terminal_marker_compliance"],
            "invalid_guess_rate": baseline["gameplay"]["invalid_guess_rate"],
            "repeat_guess_rate": baseline["gameplay"]["repeat_guess_rate"],
            "posterior_constraint_violation_rate": baseline["diagnostics"]["posterior_constraint_violation_rate"],
            "turn_2_posterior_constraint_violation_rate": baseline["diagnostics"]["by_turn"]["2"]["posterior_constraint_violation_rate"],
            "singleton_answer_accuracy": baseline["diagnostics"]["singleton_answer_accuracy"],
            "action_target_accuracy": baseline["diagnostics"]["action_target_accuracy"],
            "retention": baseline["retention"]["overall_score"],
        }
    }
    for step in CHECKPOINT_STEPS:
        metrics[f"coverage_step_{step}"] = compact_metrics(read_json(run_dir / f"eval-step-{step:06d}-summary.json"))
    result = {
        "experiment_id": EXPERIMENT_ID,
        "decision_scope": "development_only_single_seed",
        "metrics": metrics,
        "locked_test_access": False,
    }
    write_json(run_dir / "comparison_summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gemma 270M maximum state-coverage experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("preflight")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dry-run", action="store_true")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--run-dir", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        directory, result = build_bundle(force=args.force)
        output = {"directory": str(directory), "audit": result}
    elif args.command == "preflight":
        output = {
            "status": "ready",
            "data": audit_bundle(),
            "protocol": audit_protocol(),
            "vram": full_finetune_vram_preflight(),
            "locked_test_access": False,
        }
    elif args.command == "train":
        spec = build_spec()
        output = {"status": "dry_run_ready", "spec": spec, "vram": full_finetune_vram_preflight()} if args.dry_run else train(spec)
    elif args.command == "evaluate":
        output = evaluate_checkpoint(args.run_dir, args.checkpoint)
    else:
        output = summarize(args.run_dir)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
