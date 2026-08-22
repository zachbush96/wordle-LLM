from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from wordle_lab.common import ARTIFACTS, canonical_json, read_json, read_jsonl, write_json, write_jsonl
from wordle_lab.data.comparison import PARTITIONS, audit_comparison_bundle, default_directory
from wordle_lab.models import model_metadata


def comparison_context(directory: str | Path | None, partition: str) -> tuple[Path, list[dict], list[dict], dict]:
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {sorted(PARTITIONS)}")
    root = Path(directory) if directory else default_directory()
    audit = audit_comparison_bundle(root)
    rows = read_jsonl(root / f"{partition}.jsonl")
    sources = read_jsonl(root / "source_states.jsonl")
    if not rows or len(rows) != len(sources):
        raise RuntimeError(f"missing or incomplete comparison data at {root}")
    return root, rows, sources, audit


def base_spec(method: str, partition: str, seed: int, steps: int, learning_rate: float) -> dict[str, Any]:
    return {
        "method": method,
        "representation": partition,
        "seed": seed,
        "max_steps": steps,
        "learning_rate": learning_rate,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_length": 320,
        "lora": {
            "r": 16, "alpha": 32, "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "model": model_metadata(),
        "locked_test_access": False,
        "candidate_injection": False,
        "reranking": False,
        "output_repair": False,
    }


def prepare_run(method: str, partition: str, spec: dict, data_dir: Path) -> Path:
    digest = hashlib.sha256(canonical_json(spec).encode()).hexdigest()[:10]
    run_dir = ARTIFACTS / "runs" / f"{method}-{partition}-s{spec['seed']}-{digest}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "spec.json", {**spec, "data_directory": str(data_dir)})
    return run_dir


def preference_rows(rendered: list[dict], sources: list[dict], partition: str) -> list[dict]:
    by_id = {row["comparison_id"]: row for row in sources}
    output = []
    for row in rendered:
        source = by_id[row["comparison_id"]]
        rejected_word = source["facts"]["hard_negative"]
        chosen_word = source["facts"]["oracle_action"]
        if rejected_word == chosen_word:
            history = source["history"]
            if not history:
                continue
            rejected_word = history[-1]["guess"]
            negative_type = "prior_repeat"
        else:
            negative_type = "oracle_ranked_hard_negative"
        if partition == "reasoning_single_step":
            facts = source["facts"]
            fixed = ", ".join(f"{p}={v}" for p, v in facts["fixed_positions"].items()) or "none"
            required = ", ".join(facts["letters_in_every_candidate"]) or "none"
            excluded = ", ".join(facts["excluded_seen_letters"]) or "none"
            def neutral(action: str) -> str:
                return (
                    f"Constraints: {facts['posterior_count']} candidates remain. Fixed positions: {fixed}. "
                    f"Required letters: {required}. Excluded seen letters: {excluded}.\n"
                    f"Action assessment: {action} is the action under consideration.\n"
                    "Choice rationale: This is the proposed next action for comparison.\n"
                    f"Final answer: {action}"
                )
            chosen = [{"role": "assistant", "content": neutral(chosen_word)}]
            rejected_text = neutral(rejected_word)
        else:
            chosen = row["completion"]
            rejected_text = f"Final answer: {rejected_word}"
        output.append({
            "pair_id": f"{row['comparison_id']}-{partition}",
            "comparison_id": row["comparison_id"],
            "state_id": source["state_id"],
            "negative_type": negative_type,
            "prompt": row["prompt"],
            "chosen": chosen,
            "rejected": [{"role": "assistant", "content": rejected_text}],
            "chosen_word": chosen_word,
            "rejected_word": rejected_word,
        })
    if not output:
        raise RuntimeError("no non-identical preference pairs were generated")
    return output


def dry_run_summary(method: str, partition: str, data_dir: Path, rows: list[dict], spec: dict, **extra: Any) -> dict:
    return {
        "status": "dry_run_passed",
        "method": method,
        "partition": partition,
        "training_rows": len(rows),
        "data_directory": str(data_dir),
        "model": spec["model"],
        "spec": spec,
        **extra,
    }


def load_jsonl_required(path: str | Path) -> list[dict]:
    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError(f"required non-empty JSONL not found: {path}")
    return rows


def assert_gemma_parent_adapter(path: str | Path) -> Path:
    adapter = Path(path)
    config_path = adapter / "adapter_config.json"
    if not adapter.is_dir() or not config_path.exists():
        raise RuntimeError(f"parent must be a trained PEFT adapter directory: {adapter}")
    config = read_json(config_path)
    base = str(config.get("base_model_name_or_path", "")).replace("\\", "/").lower()
    if "qwen" in base or (base and "gemma-3-270m-it" not in base and "google--gemma-3-270m-it" not in base):
        raise RuntimeError(f"parent adapter is not based on google/gemma-3-270m-it: {base}")
    return adapter
