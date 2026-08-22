from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import torch

from wordle_lab.common import ARTIFACTS, DATA, git_commit, read_json, read_jsonl, sha256_file, source_tree_sha256, utc_now, write_json, write_jsonl
from wordle_lab.data.manifests import file_manifest
from wordle_lab.methods.sft import train_sft
from wordle_lab.models import load_adapter, load_base_model, load_tokenizer, model_metadata
from wordle_lab.protocol.evaluator import evaluate
from wordle_lab.protocol.lock import PROTOCOL_ID
from wordle_lab.protocol.retention import evaluate_retention
from .accounting import environment
from .registry import run_id
from .schema import ALLOWED_STATES, validate_spec


def _status(run_dir: Path, state: str, **extra) -> None:
    if state not in ALLOWED_STATES:
        raise ValueError(state)
    write_json(run_dir / "status.json", {"state": state, "updated_at": utc_now(), **extra})


def run_sft(spec: dict, train_limit: int | None = None, dev_games: int = 25) -> tuple[str, dict]:
    protocol_lock = read_json(DATA / "protocol_lock.json")
    spec = validate_spec({**spec, "train_limit": train_limit, "dev_games": dev_games, "state_sampling": "shared-state-hash-turn-floor-v3", "minimum_examples_per_turn": 32, "protocol_sha256": protocol_lock["protocol_sha256"]})
    rid = run_id(spec)
    run_dir = ARTIFACTS / "runs" / rid
    if (run_dir / "summary.json").exists():
        return rid, read_json(run_dir / "summary.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    _status(run_dir, "PLANNED")
    dataset_path = DATA / "rendered" / f"train_{spec['representation']}.jsonl"
    rows = read_jsonl(dataset_path)
    if train_limit:
        root_rows = [row for row in rows if row.get("turn") == 1]
        remaining = [row for row in rows if row.get("turn") != 1]
        selected_ids = {
            row["state_id"]
            for row in root_rows + sorted(remaining, key=lambda item: hashlib.sha256(item["state_id"].encode("utf-8")).hexdigest())[: max(0, train_limit - len(root_rows))]
        }
        rows = [row for row in rows if row["state_id"] in selected_ids]
    # Rare root and late-turn states are otherwise seen too few times to be
    # learnable. The same declared exposure floor is used for every
    # representation; unique-state accounting remains unchanged.
    for turn in range(1, 7):
        bucket = [row for row in rows if row.get("turn") == turn]
        if bucket:
            rows.extend(bucket[index % len(bucket)] for index in range(max(0, 32 - len(bucket))))
    if not rows:
        raise RuntimeError(f"dataset missing: {dataset_path}; run prepare-data")
    tokenizer = load_tokenizer()
    complete_spec = {
        **spec, "run_id": rid, "protocol_id": PROTOCOL_ID, "parent_checkpoint": spec.get("parent_checkpoint", "base"),
        "git_commit": git_commit(), "source_tree_sha256": source_tree_sha256(), "created_at": utc_now(), "model": model_metadata(), "environment": environment(),
        "dataset_sha256": sha256_file(dataset_path),
    }
    write_json(run_dir / "spec.json", complete_spec)
    dataset_manifest = file_manifest(dataset_path, rows, tokenizer)
    dataset_manifest["selected_rows_sha256"] = hashlib.sha256(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows).encode("utf-8")
    ).hexdigest()
    write_json(run_dir / "dataset_manifest.json", dataset_manifest)
    _status(run_dir, "DATA_READY")
    _status(run_dir, "TRAINING")
    model, accounting = train_sft(rows, run_dir, spec)
    _status(run_dir, "TRAINED", final_checkpoint="checkpoints/final")
    try:
        dev_answers = read_json(DATA / "splits" / "dev_answers.json")[:dev_games]
        allowed = [line.strip().upper() for line in (DATA / "wordlists" / "allowed_words.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        answer_vocab = read_json(DATA / "splits" / "all_answer_words.json")
        games, summary = evaluate(model, tokenizer, dev_answers, allowed, answer_vocab)
        summary.update({"run_id": rid, "split": "dev", "dev_games": dev_games, "protocol_id": PROTOCOL_ID, "accounting": accounting})
        write_jsonl(run_dir / "games.jsonl", games)
        write_json(run_dir / "summary.json", summary)
        _status(run_dir, "DEV_EVALUATED")
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rid, summary


def evaluate_base(split: str = "dev", games: int | None = None) -> tuple[str, dict]:
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    answers = read_json(DATA / "splits" / f"{split}_answers.json")
    if games is not None:
        answers = answers[:games]
    count = len(answers)
    protocol_sha = read_json(DATA / "protocol_lock.json")["protocol_sha256"]
    rid = f"base-{PROTOCOL_ID.lower()}-{protocol_sha[:8]}-{split}-{count}"
    run_dir = ARTIFACTS / "runs" / rid
    if (run_dir / "summary.json").exists():
        return rid, read_json(run_dir / "summary.json")
    run_dir.mkdir(parents=True, exist_ok=True)
    _status(run_dir, "PLANNED")
    allowed = [line.strip().upper() for line in (DATA / "wordlists" / "allowed_words.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    answer_vocab = read_json(DATA / "splits" / "all_answer_words.json")
    tokenizer = load_tokenizer()
    model = load_base_model(training=False)
    try:
        records, summary = evaluate(model, tokenizer, answers, allowed, answer_vocab)
        summary.update({"run_id": rid, "split": split, "protocol_id": PROTOCOL_ID, "protocol_sha256": protocol_sha, "model": model_metadata()})
        write_jsonl(run_dir / "games.jsonl", records)
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "spec.json", {"run_id": rid, "kind": "frozen_base_evaluation", "split": split, "games": count, "protocol_id": PROTOCOL_ID, "created_at": utc_now(), "environment": environment()})
        _status(run_dir, "TEST_EVALUATED" if split == "test" else "DEV_EVALUATED")
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rid, summary


def select_run(rid: str, score: float, rationale: str) -> None:
    run_dir = ARTIFACTS / "runs" / rid
    status = read_json(run_dir / "status.json")
    if status["state"] != "DEV_EVALUATED":
        raise RuntimeError("only a DEV_EVALUATED run may be selected")
    _status(run_dir, "SELECTED", selection_score=score, selection_rationale=rationale)


def evaluate_selected_test(rid: str) -> dict:
    run_dir = ARTIFACTS / "runs" / rid
    if read_json(run_dir / "status.json")["state"] != "SELECTED":
        raise RuntimeError("locked test refused: run has not passed the study-level SELECTED transition")
    answers = read_json(DATA / "splits" / "test_answers.json")
    allowed = [line.strip().upper() for line in (DATA / "wordlists" / "allowed_words.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    answer_vocab = read_json(DATA / "splits" / "all_answer_words.json")
    tokenizer = load_tokenizer(run_dir / "checkpoints" / "final")
    model = load_adapter(run_dir / "checkpoints" / "final")
    try:
        records, summary = evaluate(model, tokenizer, answers, allowed, answer_vocab)
        summary.update({"run_id": rid, "split": "test", "protocol_id": PROTOCOL_ID})
        write_jsonl(run_dir / "test_games.jsonl", records)
        write_json(run_dir / "test_summary.json", summary)
        _status(run_dir, "TEST_EVALUATED")
        return summary
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run_retention(rid: str = "base") -> dict:
    if rid == "base":
        output_dir = ARTIFACTS / "runs" / f"retention-base-{read_json(DATA / 'protocol_lock.json')['protocol_sha256'][:8]}"
        tokenizer = load_tokenizer(); model = load_base_model(training=False)
    else:
        run_dir = ARTIFACTS / "runs" / rid
        output_dir = run_dir
        tokenizer = load_tokenizer(run_dir / "checkpoints" / "final")
        model = load_adapter(run_dir / "checkpoints" / "final")
    try:
        probes = read_jsonl(DATA / "retention_probes_v1.jsonl")
        records, summary = evaluate_retention(model, tokenizer, probes)
        summary.update({"run_id": rid, "probe_sha256": sha256_file(DATA / "retention_probes_v1.jsonl")})
        write_jsonl(output_dir / "retention.jsonl", records)
        write_json(output_dir / "retention_summary.json", summary)
        return summary
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
