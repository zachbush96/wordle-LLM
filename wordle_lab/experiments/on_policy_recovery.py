from __future__ import annotations

import hashlib
import gc
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

from wordle_lab.common import canonical_json, sha256_file, write_json, write_jsonl
from wordle_lab.common import ARTIFACTS, read_jsonl, set_seed
from wordle_lab.experiments.intervention_sweep import _explicit_feedback_messages
from wordle_lab.protocol.env import WordleEnv, normalize_word, posterior_candidates
from wordle_lab.protocol.oracle import GreedyPartitionOracle
from wordle_lab.protocol.parsing import parse_terminal_answer


DAGGER_ID = "COMMON-DAGGER-001"


def _error_type(parsed: dict, guess: str | None, history: Sequence[tuple[str, str]], posterior: Sequence[str]) -> str | None:
    if not parsed["format_valid"]:
        return "malformed_output"
    if parsed["status"] != "ok":
        return "invalid_word"
    if guess in {word for word, _ in history}:
        return "repeat"
    if len(posterior) == 1 and guess != posterior[0]:
        return "singleton_miss"
    if guess not in posterior:
        return "constraint_violation"
    return None


def collect_recovery_rows(
    model,
    tokenizer,
    train_secrets: Sequence[str],
    allowed_words: Sequence[str],
    answer_vocabulary: Sequence[str],
    source_parent: str,
    rollout_seed: int,
    generate_fn: Callable | None = None,
    prompt_builder: Callable = _explicit_feedback_messages,
    max_calls: int = 12,
) -> list[dict]:
    """Collect only mistakes from real model histories on training secrets."""
    if not set(map(normalize_word, train_secrets)) <= set(map(normalize_word, answer_vocabulary)):
        raise ValueError("all rollout secrets must be in the public answer vocabulary")
    if generate_fn is None:
        from wordle_lab.protocol import generation

        def generate_fn(model, tokenizer, histories, batch_size=1):
            previous = generation.inference_messages
            try:
                generation.inference_messages = prompt_builder
                return generation.generate(model, tokenizer, histories, batch_size=batch_size)
            finally:
                generation.inference_messages = previous
    oracle = GreedyPartitionOracle(answer_vocabulary)
    secrets = list(map(normalize_word, train_secrets))
    random.Random(rollout_seed).shuffle(secrets)
    rows = []
    for game_index, secret in enumerate(secrets):
        env = WordleEnv(secret, allowed_words)
        calls = 0
        while not env.done and calls < max_calls:
            history = list(env.history)
            posterior = posterior_candidates(history, answer_vocabulary)
            output = generate_fn(model, tokenizer, [history], batch_size=1)[0]
            parsed = parse_terminal_answer(output.get("raw_output", ""), allowed_words)
            guess = parsed["parsed_guess"] if parsed["status"] == "ok" else None
            error = _error_type(parsed, guess, history, posterior)
            if error:
                target = oracle.best(history)["guess"]
                state_hash = hashlib.sha256(canonical_json(history).encode("utf-8")).hexdigest()[:16]
                rows.append(
                    {
                        "example_id": f"dagger-{rollout_seed}-{game_index:04d}-{calls:02d}-{state_hash}",
                        "state_id": state_hash,
                        "source_parent": source_parent,
                        "rollout_seed": rollout_seed,
                        "error_type": error,
                        "history": [{"guess": word, "feedback": feedback} for word, feedback in history],
                        "model_output": output.get("raw_output", ""),
                        "model_guess": parsed["parsed_guess"],
                        "turn": len(history) + 1,
                        "posterior_size": len(posterior),
                        "target_word": target,
                        "prompt": prompt_builder(history),
                        "completion": [{"role": "assistant", "content": f"Final answer: {target}"}],
                        "secret_split": "train",
                    }
                )
            step = env.step(guess)
            calls += 1
            if not step["valid"] and calls >= max_calls:
                break
    # Stable priority makes singleton recovery survive a later row budget.
    priority = {"singleton_miss": 0, "constraint_violation": 1, "repeat": 2, "malformed_output": 3, "invalid_word": 4}
    ordered = sorted(rows, key=lambda row: (priority[row["error_type"]], row["example_id"]))
    # Invalid outputs do not advance the environment, so a deterministic model
    # can revisit the exact same history until max_calls. Training copies of the
    # same labelled state would recreate the exposure skew DAgger is meant to
    # fix; keep only the highest-priority error observed for each state.
    unique = {}
    for row in ordered:
        unique.setdefault(row["state_id"], row)
    return list(unique.values())


def mix_static_and_recovery(
    static_rows: Sequence[dict], recovery_rows: Sequence[dict], total: int, seed: int
) -> list[dict]:
    if not static_rows or not recovery_rows:
        raise ValueError("both static and recovery rows are required")
    static_count = total // 2
    recovery_count = total - static_count
    rng = random.Random(seed)

    def sample(rows: Sequence[dict], count: int, source: str) -> list[dict]:
        order = list(rows)
        rng.shuffle(order)
        return [{**order[index % len(order)], "dagger_mix_source": source} for index in range(count)]

    mixed = sample(static_rows, static_count, "static") + sample(recovery_rows, recovery_count, "on_policy_recovery")
    rng.shuffle(mixed)
    return mixed


def save_dagger_round(
    output_parent: Path,
    static_rows: Sequence[dict],
    recovery_rows: Sequence[dict],
    source_parent: str,
    rollout_seed: int,
    round_number: int,
    total: int,
) -> tuple[Path, dict]:
    mixed = mix_static_and_recovery(static_rows, recovery_rows, total, rollout_seed + round_number)
    spec = {
        "dagger_id": DAGGER_ID,
        "source_parent": source_parent,
        "rollout_seed": rollout_seed,
        "round": round_number,
        "total": total,
        "recovery_state_ids": [row["state_id"] for row in recovery_rows],
    }
    artifact_id = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:12]
    directory = Path(output_parent) / f"dagger-round-{round_number}-{artifact_id}"
    recovery_path = write_jsonl(directory / "recovery.jsonl", recovery_rows)
    mixed_path = write_jsonl(directory / "train.jsonl", mixed)
    summary = {
        **spec,
        "artifact_id": artifact_id,
        "static_examples": sum(row["dagger_mix_source"] == "static" for row in mixed),
        "recovery_examples": sum(row["dagger_mix_source"] == "on_policy_recovery" for row in mixed),
        "unique_recovery_states": len({row["state_id"] for row in recovery_rows}),
        "state_coverage_gained": len({row["state_id"] for row in recovery_rows}),
        "error_type_counts": dict(sorted(Counter(row["error_type"] for row in recovery_rows).items())),
        "recovery_sha256": sha256_file(recovery_path),
        "train_sha256": sha256_file(mixed_path),
    }
    write_json(directory / "manifest.json", summary)
    return directory, summary


def train_dagger_round(
    parent_run_id: str,
    round_directory: Path,
    max_steps: int = 600,
    learning_rate: float = 2.5e-5,
    seed: int = 2026,
    dev_games: int = 32,
    word_token_weight: float = 8.0,
) -> tuple[str, dict]:
    """Continue a parent on an already-audited 50/50 DAgger round."""
    import torch

    from wordle_lab.experiments.common_curriculum import evaluate_saved_checkpoint
    from wordle_lab.methods.sft import train_sft
    parent_dir = ARTIFACTS / "runs" / parent_run_id
    parent_spec = json.loads((parent_dir / "spec.json").read_text(encoding="utf-8"))
    round_directory = Path(round_directory)
    rows_path = round_directory / "train.jsonl"
    round_manifest = json.loads((round_directory / "manifest.json").read_text(encoding="utf-8"))
    rows = read_jsonl(rows_path)
    spec = {
        "method": "continued_sft_dagger",
        "representation": "balanced_static_50_on_policy_recovery_50",
        "parent_run_id": parent_run_id,
        "parent_checkpoint": str(parent_dir / "checkpoints" / "final"),
        "seed": seed,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_length": int(parent_spec["max_length"]),
        "word_token_weight": word_token_weight,
        "lora": parent_spec["lora"],
        "curriculum": parent_spec["curriculum"],
        "dagger_round": round_manifest,
        "dataset_sha256": sha256_file(rows_path),
    }
    run_hash = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:10]
    run_id = f"dagger-common-s{seed}-{run_hash}"
    run_dir = ARTIFACTS / "runs" / run_id
    if (run_dir / "summary.json").exists():
        return run_id, json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "spec.json", spec)
    write_json(run_dir / "dataset_manifest.json", round_manifest)
    write_jsonl(run_dir / "train.jsonl", rows)
    set_seed(seed)
    model, accounting = train_sft(rows, run_dir, spec)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    summary = evaluate_saved_checkpoint(run_id, "final", dev_games, "greedy_rep105")
    summary.update({"accounting": accounting, "dagger_round": round_manifest["round"]})
    write_json(run_dir / "summary.json", summary)
    return run_id, summary
