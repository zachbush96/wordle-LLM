from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from wordle_lab.analysis.state_diagnostics import run_state_diagnostics
from wordle_lab.common import ARTIFACTS, DATA, read_json, read_jsonl, write_json, write_jsonl
from wordle_lab.data.comparison import audit_comparison_bundle, default_directory
from wordle_lab.models import load_base_model, load_tokenizer, model_metadata
from wordle_lab.protocol.evaluator import evaluate
from wordle_lab.protocol.retention import evaluate_retention


def main() -> int:
    parser = argparse.ArgumentParser(description="Matched base Gemma evaluation for the audited u128 comparison bundle")
    parser.add_argument("--data-dir", type=Path, default=default_directory())
    parser.add_argument("--dev-games", type=int, default=32)
    parser.add_argument("--diagnostic-items", type=int, default=512)
    args = parser.parse_args()

    audit = audit_comparison_bundle(args.data_dir)
    output_dir = ARTIFACTS / "runs" / "base-gemma-270m-u128-comparison-v1-dev"
    tokenizer = load_tokenizer()
    model = load_base_model(training=False)
    allowed = [line.strip().upper() for line in (DATA / "wordlists" / "allowed_words.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    universe = read_json(args.data_dir / "universe.json")
    dev_answers = read_json(args.data_dir / "dev_secrets.json")[: args.dev_games]
    dev_probes = read_jsonl(args.data_dir / "dev_probe_states.jsonl")[: args.diagnostic_items]
    training_records = read_jsonl(args.data_dir / "source_states.jsonl")
    try:
        games, gameplay = evaluate(model, tokenizer, dev_answers, allowed, universe)
        write_jsonl(output_dir / "games.jsonl", games)
        diagnostics_dir, diagnostics = run_state_diagnostics(
            model, tokenizer, dev_probes, training_records, allowed, universe, output_dir
        )
        retention_rows, retention = evaluate_retention(model, tokenizer, read_jsonl(DATA / "retention_probes_v1.jsonl"))
        write_jsonl(output_dir / "retention.jsonl", retention_rows)
        summary = {
            "status": "dev_evaluated",
            "condition": "frozen_base",
            "split": "dev",
            "locked_test_access": False,
            "model": model_metadata(),
            "data_audit": audit,
            "gameplay": gameplay,
            "diagnostics": diagnostics,
            "diagnostics_dir": str(diagnostics_dir),
            "retention": retention,
        }
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
