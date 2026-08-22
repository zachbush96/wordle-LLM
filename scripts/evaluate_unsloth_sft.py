from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from wordle_lab.analysis.state_diagnostics import run_state_diagnostics
from wordle_lab.common import DATA, read_json, read_jsonl, write_json, write_jsonl
from wordle_lab.data.comparison import audit_comparison_bundle, default_directory
from wordle_lab.methods.unsloth_sft import UNSLOTH_BACKEND_ID
from wordle_lab.models import load_adapter, load_tokenizer
from wordle_lab.protocol.evaluator import evaluate
from wordle_lab.protocol.retention import evaluate_retention


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an Unsloth-trained adapter under WORDLE-PROTOCOL-002")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--data-dir", type=Path, default=default_directory())
    parser.add_argument("--dev-games", type=int, default=32)
    parser.add_argument("--diagnostic-items", type=int, default=512)
    args = parser.parse_args()

    spec = read_json(args.run_dir / "spec.json")
    if spec.get("backend") != UNSLOTH_BACKEND_ID or spec.get("locked_test_access") is not False:
        raise RuntimeError("run is not a locked-test-free Unsloth Gemma SFT artifact")
    audit = audit_comparison_bundle(args.data_dir)
    checkpoint = args.run_dir / "checkpoints" / args.checkpoint
    tokenizer = load_tokenizer(checkpoint)
    model = load_adapter(checkpoint)
    allowed = [line.strip().upper() for line in (DATA / "wordlists" / "allowed_words.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    universe = read_json(args.data_dir / "universe.json")
    dev_answers = read_json(args.data_dir / "dev_secrets.json")[: args.dev_games]
    dev_probes = read_jsonl(args.data_dir / "dev_probe_states.jsonl")[: args.diagnostic_items]
    training_records = read_jsonl(args.data_dir / "source_states.jsonl")
    suffix = args.checkpoint.replace("/", "-")
    try:
        games, gameplay = evaluate(model, tokenizer, dev_answers, allowed, universe)
        write_jsonl(args.run_dir / f"eval-{suffix}-games.jsonl", games)
        diagnostics_dir, diagnostics = run_state_diagnostics(
            model, tokenizer, dev_probes, training_records, allowed, universe, args.run_dir / f"eval-{suffix}"
        )
        retention_rows, retention = evaluate_retention(model, tokenizer, read_jsonl(DATA / "retention_probes_v1.jsonl"))
        write_jsonl(args.run_dir / f"eval-{suffix}-retention.jsonl", retention_rows)
        summary = {
            "status": "dev_evaluated",
            "split": "dev",
            "locked_test_access": False,
            "checkpoint": args.checkpoint,
            "data_audit": audit,
            "gameplay": gameplay,
            "diagnostics": diagnostics,
            "diagnostics_dir": str(diagnostics_dir),
            "retention": retention,
        }
        write_json(args.run_dir / f"eval-{suffix}-summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
