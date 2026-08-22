from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from wordle_lab.common import DATA, ROOT, canonical_json, read_json, read_jsonl, sha256_file, sha256_text, utc_now, write_json, write_jsonl
from wordle_lab.data.builders import REPRESENTATIONS, render_representation
from wordle_lab.data.canonical import generate_canonical_states
from wordle_lab.data.manifests import assert_no_test_leakage, file_manifest
from wordle_lab.data.preferences import build_preferences
from wordle_lab.experiments.runner import evaluate_base, evaluate_selected_test, run_retention, run_sft, select_run
from wordle_lab.experiments.technique_catalog import technique_manifest
from wordle_lab.models import load_tokenizer
from wordle_lab.protocol.env import score_wordle
from wordle_lab.protocol.lock import build_lock
from wordle_lab.protocol.parsing import parse_terminal_answer
from wordle_lab.protocol.references import evaluate_reference
from wordle_lab.protocol.retention import build_retention_probes


def prepare_data(train_states: int, dev_states: int, force: bool = False) -> dict:
    source_data = ROOT / "data"
    wordlist_source = source_data / "wordlists" / "tabatkins_wordle_list_pinned.txt"
    split_manifest_source = source_data / "manifests" / "split_manifest_v1.json"
    for directory in (DATA / "wordlists", DATA / "splits", DATA / "canonical", DATA / "rendered", DATA / "preferences", DATA / "manifests"):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wordlist_source, DATA / "wordlists" / "allowed_words.txt")
    for split in ("train", "dev", "test"):
        shutil.copyfile(source_data / f"{split}_answers.json", DATA / "splits" / f"{split}_answers.json")
    all_answers = read_json(source_data / "train_answers.json") + read_json(source_data / "dev_answers.json") + read_json(source_data / "test_answers.json")
    write_json(DATA / "splits" / "all_answer_words.json", all_answers)
    shutil.copyfile(split_manifest_source, DATA / "splits" / "source_split_manifest.json")
    retention_path = DATA / "retention_probes_v1.jsonl"
    write_jsonl(retention_path, build_retention_probes())
    lock = build_lock(DATA / "wordlists" / "allowed_words.txt", DATA / "splits" / "source_split_manifest.json", retention_path)
    tokenizer = load_tokenizer()
    manifests = {}
    canonical_by_split = {}
    for split, count in (("train", train_states), ("dev", dev_states)):
        path = DATA / "canonical" / f"{split}_states.jsonl"
        if path.exists() and not force:
            rows = read_jsonl(path)
            if len(rows) != count:
                raise RuntimeError(f"{path} has {len(rows)} rows, requested {count}; use --force or the original count")
        else:
            rows = generate_canonical_states(read_json(DATA / "splits" / f"{split}_answers.json"), split, count)
            write_jsonl(path, rows)
        canonical_by_split[split] = rows
        manifests[f"canonical_{split}"] = {"path": str(path), "sha256": sha256_file(path), "records": len(rows), "unique_states": len({r['state_id'] for r in rows})}
        for representation in REPRESENTATIONS:
            rendered = render_representation(rows, representation)
            rendered_path = DATA / "rendered" / f"{split}_{representation}.jsonl"
            write_jsonl(rendered_path, rendered)
            if split == "train":
                assert_no_test_leakage(rendered, read_json(DATA / "splits" / "test_answers.json"))
            manifests[f"{split}_{representation}"] = file_manifest(rendered_path, rendered, tokenizer)
    for rationale in (False, True):
        label = "rationale" if rationale else "direct"
        for split in ("train", "dev"):
            pairs = build_preferences(canonical_by_split[split], rationale=rationale)
            path = DATA / "preferences" / f"{split}_{label}.jsonl"
            write_jsonl(path, pairs)
            if split == "train":
                assert_no_test_leakage(pairs, read_json(DATA / "splits" / "test_answers.json"))
            manifest = file_manifest(path, pairs)
            counts = {kind: sum(row["negative_type"] == kind for row in pairs) for kind in ("hard_strategic", "behavioral", "malformed")}
            manifests[f"preference_{split}_{label}"] = {**manifest, "negative_counts": counts}
    manifest = {"dataset_id": "wordle-canonical-v2", "created_at": utc_now(), "protocol_sha256": lock["protocol_sha256"], "train_states": train_states, "dev_states": dev_states, "files": manifests, "leakage": {"answer_splits_disjoint": True, "test_answers_absent_from_rendered_training_views": True}}
    write_json(DATA / "manifests" / "dataset_manifest.json", manifest)
    return manifest


def validate() -> dict:
    assert score_wordle("APPLE", "ALLEY") == "GYBYB"
    allowed = ["CRANE", "SLATE"]
    assert parse_terminal_answer("thinking\nFinal answer: CRANE", allowed)["status"] == "ok"
    assert parse_terminal_answer("Final answer: CRANE\nextra", allowed)["status"] == "prose_after_terminal"
    assert parse_terminal_answer("maybe CRANE", allowed)["parsed_guess"] is None
    manifest = read_json(DATA / "manifests" / "dataset_manifest.json")
    for item in manifest["files"].values():
        assert sha256_file(item["path"]) == item["sha256"]
    return {"status": "passed", "protocol": read_json(DATA / "protocol_lock.json")["protocol_id"], "dataset_files_verified": len(manifest["files"])}


def default_spec(representation: str, seed: int, max_steps: int) -> dict:
    return {"method": "sft", "representation": representation, "seed": seed, "max_steps": max_steps, "learning_rate": 2e-4, "batch_size": 4, "gradient_accumulation_steps": 4, "max_length": 512, "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="WORDLE-PROTOCOL-002 experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare-data"); data.add_argument("--train-states", type=int, default=2048); data.add_argument("--dev-states", type=int, default=512); data.add_argument("--force", action="store_true")
    sub.add_parser("validate")
    sub.add_parser("techniques")
    baseline = sub.add_parser("baseline"); baseline.add_argument("--split", choices=("dev", "test"), default="dev"); baseline.add_argument("--games", type=int)
    reference = sub.add_parser("reference"); reference.add_argument("--policy", choices=("random_allowed", "random_posterior", "oracle"), required=True); reference.add_argument("--split", choices=("dev", "test"), default="dev"); reference.add_argument("--games", type=int)
    train = sub.add_parser("train-sft"); train.add_argument("--representation", choices=REPRESENTATIONS, required=True); train.add_argument("--seed", type=int, default=1337); train.add_argument("--max-steps", type=int, default=40); train.add_argument("--train-limit", type=int); train.add_argument("--dev-games", type=int, default=25)
    retention = sub.add_parser("retention"); retention.add_argument("--run-id", default="base")
    select = sub.add_parser("select"); select.add_argument("run_id"); select.add_argument("--score", type=float, required=True); select.add_argument("--rationale", required=True)
    test = sub.add_parser("test-selected"); test.add_argument("run_id")
    args = parser.parse_args(argv)
    if args.command == "prepare-data": result = prepare_data(args.train_states, args.dev_states, args.force)
    elif args.command == "validate": result = validate()
    elif args.command == "techniques": result = {"protocol_id": "WORDLE-PROTOCOL-002", "techniques": technique_manifest()}
    elif args.command == "baseline": rid, summary = evaluate_base(args.split, args.games); result = {"run_id": rid, **summary}
    elif args.command == "reference":
        answers = read_json(DATA / "splits" / f"{args.split}_answers.json"); answers = answers[:args.games] if args.games else answers
        allowed = [line.strip().upper() for line in (DATA / "wordlists" / "allowed_words.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        records, summary = evaluate_reference(args.policy, answers, allowed, read_json(DATA / "splits" / "all_answer_words.json"))
        protocol_sha = read_json(DATA / "protocol_lock.json")["protocol_sha256"]
        rid = f"reference-{args.policy}-{protocol_sha[:8]}-{args.split}-{len(answers)}"; run_dir = ROOT / "artifacts" / "runs" / rid; write_jsonl(run_dir / "games.jsonl", records); write_json(run_dir / "summary.json", {**summary, "run_id": rid, "split": args.split, "protocol_sha256": protocol_sha}); result = {"run_id": rid, **summary}
    elif args.command == "train-sft": rid, summary = run_sft(default_spec(args.representation, args.seed, args.max_steps), args.train_limit, args.dev_games); result = {"run_id": rid, **summary}
    elif args.command == "retention": result = run_retention(args.run_id)
    elif args.command == "select": select_run(args.run_id, args.score, args.rationale); result = {"selected": args.run_id}
    else: result = evaluate_selected_test(args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
