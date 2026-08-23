from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, TypeVar


MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parents[1]
DEFAULT_CONFIG = MODULE_DIR / "structured_microtasks_config.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Preserve Unsloth's required import order only for the explicit training
# command. Builds, audits, tests, and ordinary PEFT evaluation do not import it.
if __name__ == "__main__" and "train-unsloth" in sys.argv[1:]:
    import unsloth  # noqa: F401

from next_steps.chatgpt_2026_08_23.microtasks import (  # noqa: E402
    BUILDER_VERSION,
    INVALID_REASONS,
    CurriculumRecord,
    TrainingContext,
    TrainingState,
    audit_curriculum_records,
    build_balanced_candidate_validity_records,
    build_constraint_merge_record,
    build_feedback_decode_record,
    build_full_policy_record,
    build_singleton_record,
    evaluate_curriculum_predictions,
)
from wordle_lab.common import (  # noqa: E402
    ARTIFACTS,
    canonical_json,
    read_json,
    read_jsonl,
    set_seed,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from wordle_lab.data.canonical import generate_canonical_states  # noqa: E402
from wordle_lab.experiments.intervention_sweep import _explicit_feedback_messages  # noqa: E402
from wordle_lab.methods.unsloth_sft import (  # noqa: E402
    UNSLOTH_BACKEND_ID,
    train_unsloth_sft,
    unsloth_environment,
)
from wordle_lab.methods.sft import train_sft  # noqa: E402
from wordle_lab.models import SUPPORTED_MODEL_ID, SUPPORTED_REVISION, load_tokenizer  # noqa: E402
from wordle_lab.protocol.env import is_five_ascii_letters, normalize_word, posterior_candidates  # noqa: E402
from wordle_lab.protocol.parsing import parse_terminal_answer  # noqa: E402


EXPERIMENT_ID = "STRUCTURED-MICROTASKS-SFT-001"
TRANSFORMERS_BACKEND_ID = "TRANSFORMERS-PEFT-COMPLETION-SFT-001"
TASKS = (
    "feedback_decode",
    "constraint_merge",
    "candidate_validity",
    "singleton_solve",
    "full_policy",
)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
JSON_SYSTEM_PROMPT = (
    "Solve the structured Wordle microtask using only the visible input. "
    "Return exactly one JSON object and no markdown or prose. Positions are one-based."
)
T = TypeVar("T")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _rank(seed: int, label: str, identity: Any) -> str:
    return _hash({"seed": seed, "label": label, "identity": identity})


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_json(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("builder_version") != BUILDER_VERSION:
        raise ValueError("structured microtask experiment/builder id drift")
    if config.get("protocol_id") != "WORDLE-PROTOCOL-002":
        raise ValueError("structured microtasks require WORDLE-PROTOCOL-002")
    if config.get("model") != {"model_id": SUPPORTED_MODEL_ID, "revision": SUPPORTED_REVISION}:
        raise ValueError("structured microtasks require the pinned Gemma 3 270M model")
    if config.get("seed") != 2026:
        raise ValueError("the preregistered builder seed must remain 2026")
    if not 0 < float(config["selection"]["duplicate_heavy_min_fraction"]) <= 1:
        raise ValueError("duplicate-heavy fraction must be in (0, 1]")
    for split in ("train", "dev"):
        selection = config["selection"][split]
        if any(int(value) <= 0 for value in selection.values()):
            raise ValueError(f"all {split} selection counts must be positive")
    training = config["training"]
    if float(training.get("word_token_weight", 0)) != 1.0:
        raise ValueError("structured microtasks preregister completion-only SFT")
    if training.get("lora", {}).get("target_modules") != TARGET_MODULES:
        raise ValueError("structured microtasks require all seven pinned LoRA targets")
    for key in (
        "locked_test_access",
        "candidate_injection",
        "vocabulary_masking",
        "reranking",
        "repeat_ban",
        "output_repair",
        "harness_selected_guess",
    ):
        if config.get(key) is not False:
            raise ValueError(f"{key} must remain false")


def audit_source(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    directory = _resolve(source["directory"])
    expected = {
        "manifest.json": source["manifest_sha256"],
        "train.jsonl": source["train_rows_sha256"],
        "universe.json": source["universe_sha256"],
        "train_secrets.json": source["train_secrets_sha256"],
        "dev_secrets.json": source["dev_secrets_sha256"],
    }
    actual = {name: sha256_file(directory / name) for name in expected}
    if actual != expected:
        raise AssertionError("COMMON-WORD-CURRICULUM-002 source hash drift")
    manifest = read_json(directory / "manifest.json")
    if manifest.get("curriculum_id") != source["dataset_id"]:
        raise AssertionError("source curriculum id mismatch")
    universe = set(read_json(directory / "universe.json"))
    train = set(read_json(directory / "train_secrets.json"))
    dev = set(read_json(directory / "dev_secrets.json"))
    if train & dev or train | dev != universe:
        raise AssertionError("source train/dev split is not a disjoint partition")
    rows = read_jsonl(directory / "train.jsonl")
    if any(row["source_state"]["secret_answer"] not in train for row in rows):
        raise AssertionError("balanced training row has non-training provenance")
    return {
        "status": "passed",
        "dataset_id": source["dataset_id"],
        "hashes": actual,
        "training_rows": len(rows),
        "universe_size": len(universe),
        "train_secret_count": len(train),
        "dev_secret_count": len(dev),
        "train_dev_overlap": 0,
        "locked_test_access": False,
    }


def audit_evaluation_source(config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    allowed_path = _resolve(evaluation["allowed_words"])
    actual = sha256_file(allowed_path)
    expected = str(evaluation["allowed_words_sha256"])
    if actual != expected:
        raise AssertionError("evaluation allowed-word hash drift")
    words = [
        line.strip().upper()
        for line in allowed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(words) != len(set(words)) or any(not is_five_ascii_letters(word) for word in words):
        raise AssertionError("evaluation allowed-word vocabulary is not canonical")
    return {
        "status": "passed",
        "allowed_words": str(evaluation["allowed_words"]),
        "allowed_words_sha256": actual,
        "allowed_word_count": len(words),
        "locked_test_access": False,
    }


def _training_state(source: Mapping[str, Any]) -> TrainingState:
    return TrainingState(
        state_id=str(source["state_id"]),
        secret_answer=str(source["secret_answer"]),
        history=tuple((str(row["guess"]), str(row["feedback"])) for row in source["history"]),
    )


def _state_payload(state: TrainingState) -> dict[str, Any]:
    return state.private_payload()


def _history_key(state: TrainingState) -> str:
    return _hash([{"guess": guess, "feedback": feedback} for guess, feedback in state.history])


def _has_duplicate_word(word: str) -> bool:
    normalized = normalize_word(word)
    return len(set(normalized)) < len(normalized)


def _state_is_duplicate_heavy(state: TrainingState) -> bool:
    return any(_has_duplicate_word(guess) for guess, _ in state.history)


def _record_is_duplicate_heavy(record: CurriculumRecord, state: TrainingState) -> bool:
    if record.task_type == "feedback_decode":
        return _has_duplicate_word(str(record.input["guess"]))
    if record.task_type == "candidate_validity":
        return _state_is_duplicate_heavy(state) or _has_duplicate_word(str(record.input["candidate"]))
    return _state_is_duplicate_heavy(state)


def _select_with_duplicate_floor(
    items: Sequence[T],
    count: int,
    *,
    seed: int,
    label: str,
    identity: Callable[[T], Any],
    duplicate_heavy: Callable[[T], bool],
    fraction: float,
) -> list[T]:
    if count > len(items):
        raise ValueError(f"insufficient {label} examples: need {count}, found {len(items)}")
    duplicate_needed = math.ceil(count * fraction)
    duplicate_rows = sorted(
        (item for item in items if duplicate_heavy(item)),
        key=lambda item: _rank(seed, f"{label}:duplicate", identity(item)),
    )
    if len(duplicate_rows) < duplicate_needed:
        raise ValueError(
            f"insufficient duplicate-heavy {label} examples: need {duplicate_needed}, found {len(duplicate_rows)}"
        )
    selected = duplicate_rows[:duplicate_needed]
    selected_ids = {_hash(identity(item)) for item in selected}
    remaining = sorted(
        (item for item in items if _hash(identity(item)) not in selected_ids),
        key=lambda item: _rank(seed, f"{label}:fill", identity(item)),
    )
    selected.extend(remaining[: count - len(selected)])
    return sorted(selected, key=lambda item: _rank(seed, f"{label}:order", identity(item)))


def _contexts_and_sources(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_dir = _resolve(config["source"]["directory"])
    universe = tuple(read_json(source_dir / "universe.json"))
    train_secrets = tuple(read_json(source_dir / "train_secrets.json"))
    dev_secrets = tuple(read_json(source_dir / "dev_secrets.json"))
    manifest_sha = config["source"]["manifest_sha256"]

    balanced_rows = read_jsonl(source_dir / "train.jsonl")
    train_sources: dict[str, Mapping[str, Any]] = {}
    for row in balanced_rows:
        train_sources.setdefault(row["source_state"]["state_id"], row["source_state"])
    train_states = {state_id: _training_state(source) for state_id, source in train_sources.items()}
    train_history_hashes = {_history_key(state) for state in train_states.values()}

    dev_generated = generate_canonical_states(
        dev_secrets,
        "structured_microtasks_dev",
        int(config["source"]["dev_generated_states"]),
        seed=int(config["seed"]),
        answer_vocabulary=universe,
    )
    dev_states: dict[str, TrainingState] = {}
    for source in dev_generated:
        state = _training_state(source)
        if not state.history:
            continue
        if config["source"]["exclude_train_history_from_dev"] and _history_key(state) in train_history_hashes:
            continue
        dev_states.setdefault(state.state_id, state)
    if set(train_states) & set(dev_states):
        raise AssertionError("train/dev source state ids overlap")
    if train_history_hashes & {_history_key(state) for state in dev_states.values()}:
        raise AssertionError("train/dev visible histories overlap")

    return {
        "train": {
            "context": TrainingContext(
                source_dataset_id=config["source"]["dataset_id"],
                source_manifest_sha256=manifest_sha,
                training_secret_source="common-curriculum-002/train_secrets.json",
                training_secrets=train_secrets,
                answer_universe=universe,
            ),
            "states": train_states,
            "policy_sources": balanced_rows,
        },
        "dev": {
            "context": TrainingContext(
                source_dataset_id=f"{config['source']['dataset_id']}-DEV-EVALUATION-ONLY",
                source_manifest_sha256=manifest_sha,
                training_secret_source="common-curriculum-002/dev_secrets.json:evaluation_only",
                training_secrets=dev_secrets,
                answer_universe=universe,
            ),
            "states": dev_states,
            "policy_sources": dev_generated,
        },
    }


def _feedback_records(
    states: Sequence[TrainingState], context: TrainingContext, count: int, split: str, config: dict[str, Any]
) -> list[CurriculumRecord]:
    pool = [(state, index) for state in states for index in range(len(state.history))]
    chosen = _select_with_duplicate_floor(
        pool,
        count,
        seed=config["seed"],
        label=f"{split}:feedback_decode",
        identity=lambda item: [item[0].state_id, item[1]],
        duplicate_heavy=lambda item: _has_duplicate_word(item[0].history[item[1]][0]),
        fraction=config["selection"]["duplicate_heavy_min_fraction"],
    )
    return [build_feedback_decode_record(state, context, index) for state, index in chosen]


def _state_task_records(
    states: Sequence[TrainingState],
    context: TrainingContext,
    count: int,
    split: str,
    task: str,
    config: dict[str, Any],
) -> list[CurriculumRecord]:
    if task == "singleton_solve":
        eligible = [
            state
            for state in states
            if len(posterior_candidates(state.history, context.answer_universe)) == 1
        ]
        builder = build_singleton_record
    elif task == "constraint_merge":
        eligible = [state for state in states if state.history]
        builder = build_constraint_merge_record
    else:
        raise ValueError(task)
    chosen = _select_with_duplicate_floor(
        eligible,
        count,
        seed=config["seed"],
        label=f"{split}:{task}",
        identity=lambda state: state.state_id,
        duplicate_heavy=_state_is_duplicate_heavy,
        fraction=config["selection"]["duplicate_heavy_min_fraction"],
    )
    return [builder(state, context) for state in chosen]


def _policy_record_pool(
    split: str,
    policy_sources: Sequence[Mapping[str, Any]],
    states: Mapping[str, TrainingState],
    context: TrainingContext,
) -> list[tuple[TrainingState, str, str]]:
    pool: list[tuple[TrainingState, str, str]] = []
    for index, source_row in enumerate(policy_sources):
        source = source_row.get("source_state", source_row)
        state = states.get(str(source["state_id"]))
        if state is None:
            continue
        target = str(source_row.get("target_word", source.get("facts", {}).get("oracle_action", ""))).upper()
        if target not in context.training_secrets:
            continue
        source_id = str(source_row.get("example_id", f"generated-{index:06d}-{state.state_id}"))
        pool.append((state, target, f"{split}:natural-oracle:{source_id}"))
    return pool


def _full_policy_records(
    split: str,
    policy_sources: Sequence[Mapping[str, Any]],
    states: Mapping[str, TrainingState],
    context: TrainingContext,
    count: int,
    config: dict[str, Any],
) -> list[CurriculumRecord]:
    pool = _policy_record_pool(split, policy_sources, states, context)
    chosen = _select_with_duplicate_floor(
        pool,
        count,
        seed=config["seed"],
        label=f"{split}:full_policy",
        identity=lambda item: [item[0].state_id, item[1], item[2]],
        duplicate_heavy=lambda item: _state_is_duplicate_heavy(item[0]),
        fraction=config["selection"]["duplicate_heavy_min_fraction"],
    )
    return [build_full_policy_record(state, context, target, policy_id=policy_id) for state, target, policy_id in chosen]


def build_split_records(
    split: str,
    context: TrainingContext,
    states: Mapping[str, TrainingState],
    policy_sources: Sequence[Mapping[str, Any]],
    config: dict[str, Any],
) -> tuple[list[CurriculumRecord], dict[str, Any]]:
    counts = config["selection"][split]
    state_rows = list(states.values())
    feedback = _feedback_records(state_rows, context, counts["feedback_decode"], split, config)
    merged = _state_task_records(
        state_rows, context, counts["constraint_merge"], split, "constraint_merge", config
    )
    candidates = build_balanced_candidate_validity_records(
        state_rows,
        context,
        context.training_secrets,
        per_invalid_reason=counts["candidate_validity_per_invalid_reason"],
        seed=config["seed"],
    )
    singletons = _state_task_records(
        state_rows, context, counts["singleton_solve"], split, "singleton_solve", config
    )
    policies = _full_policy_records(
        split, policy_sources, states, context, counts["full_policy"], config
    )
    records = feedback + merged + candidates + singletons + policies
    source_ids = {record.source_state_id for record in records}
    used_states = [states[state_id] for state_id in sorted(source_ids)]
    primitive_audit = {
        **audit_curriculum_records(records, used_states, context),
        # The reusable primitive names its secret-set field ``training_secrets``.
        # Scope it explicitly here because the development invocation supplies
        # the evaluation-only dev partition to the same label-recomputation API.
        "split": split,
        "secret_role": "training" if split == "train" else "evaluation_only",
    }
    by_id = {state.state_id: state for state in used_states}
    duplicate_counts = Counter(
        record.task_type
        for record in records
        if _record_is_duplicate_heavy(record, by_id[record.source_state_id])
    )
    distribution = Counter(record.task_type for record in records)
    fraction = config["selection"]["duplicate_heavy_min_fraction"]
    for task in ("feedback_decode", "constraint_merge", "singleton_solve", "full_policy"):
        if duplicate_counts[task] < math.ceil(distribution[task] * fraction):
            raise AssertionError(f"duplicate-heavy floor not retained for {split} {task}")
    return records, {
        "split": split,
        "records": len(records),
        "source_states": len(used_states),
        "task_distribution": dict(sorted(distribution.items())),
        "duplicate_heavy_distribution": {
            task: duplicate_counts[task] for task in sorted(distribution)
        },
        "candidate_balance": primitive_audit["candidate_balance"],
        "primitive_audit": primitive_audit,
        "locked_test_access": False,
    }


def _json_prompt(record: CurriculumRecord) -> list[dict[str, str]]:
    # The primitive keeps decoded row constraints for label auditing. The model
    # sees only the compact visible history for the merge task; this both makes
    # the task substantive and keeps duplicate-heavy late turns under the
    # preregistered sequence ceiling.
    rendered_input = (
        {"history": record.input["history"]}
        if record.task_type == "constraint_merge"
        else record.input
    )
    request = {
        "task": record.task_type,
        "input": rendered_input,
        "output_contract": (
            {"valid": "boolean", "reason": f"null or one of {list(INVALID_REASONS)}", "violations": "array"}
            if record.task_type == "candidate_validity"
            else "match the demonstrated JSON schema exactly"
        ),
    }
    return [
        {"role": "system", "content": JSON_SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(request)},
    ]


def render_record(record: CurriculumRecord, split: str, state: TrainingState) -> dict[str, Any]:
    if record.task_type == "full_policy":
        prompt = _explicit_feedback_messages(state.history)
        completion_text = f"Final answer: {record.target['word']}"
        output_format = "natural_terminal_answer"
    else:
        prompt = _json_prompt(record)
        completion_text = canonical_json(record.target)
        output_format = "strict_json_object"
    completion = [{"role": "assistant", "content": completion_text}]
    return {
        "example_id": record.record_id,
        "record_id": record.record_id,
        "schema_version": "wordle-structured-sft-render-v1",
        "split": split,
        "task_type": record.task_type,
        "source_state_id": record.source_state_id,
        "prompt": prompt,
        "completion": completion,
        "messages": prompt + completion,
        "target": record.target,
        "metadata": record.metadata,
        "provenance": record.provenance,
        "output_format": output_format,
        "duplicate_heavy": _record_is_duplicate_heavy(record, state),
        "locked_test_access": False,
    }


def _record_from_dict(row: Mapping[str, Any]) -> CurriculumRecord:
    return CurriculumRecord(
        record_id=str(row["record_id"]),
        task_type=str(row["task_type"]),
        source_state_id=str(row["source_state_id"]),
        input=dict(row["input"]),
        target=dict(row["target"]),
        metadata=dict(row["metadata"]),
        provenance=dict(row["provenance"]),
        schema_version=str(row["schema_version"]),
    )


def _mixed_order(rows: Sequence[dict[str, Any]], split: str, seed: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _rank(seed, f"{split}:mixed", row["record_id"]))


def build_bundle(
    config: dict[str, Any], output_dir: str | Path | None = None, *, force: bool = False
) -> tuple[Path, dict[str, Any]]:
    source_audit = audit_source(config)
    evaluation_source_audit = audit_evaluation_source(config)
    destination = Path(output_dir).resolve() if output_dir else _resolve(config["bundle_directory"])
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and not force:
        return destination, audit_bundle(config, destination)

    material = _contexts_and_sources(config)
    split_manifests: dict[str, Any] = {}
    all_record_ids: dict[str, set[str]] = {}
    train_histories: set[str] = set()
    for split in ("train", "dev"):
        context: TrainingContext = material[split]["context"]
        states: dict[str, TrainingState] = material[split]["states"]
        records, record_audit = build_split_records(
            split, context, states, material[split]["policy_sources"], config
        )
        source_ids = sorted({record.source_state_id for record in records})
        used_states = {state_id: states[state_id] for state_id in source_ids}
        rendered = [render_record(record, split, used_states[record.source_state_id]) for record in records]
        by_task = {
            task: sorted(
                (row for row in rendered if row["task_type"] == task),
                key=lambda row: row["record_id"],
            )
            for task in TASKS
        }
        split_dir = destination / split
        files: dict[str, Path] = {
            "records.jsonl": write_jsonl(
                split_dir / "records.jsonl",
                [record.to_dict() for record in sorted(records, key=lambda row: row.record_id)],
            ),
            "source_states_private.jsonl": write_jsonl(
                split_dir / "source_states_private.jsonl",
                [_state_payload(used_states[state_id]) for state_id in source_ids],
            ),
        }
        for task, rows in by_task.items():
            files[f"{task}.jsonl"] = write_jsonl(split_dir / f"{task}.jsonl", rows)
        mixed = _mixed_order(rendered, split, config["seed"])
        files["mixed.jsonl"] = write_jsonl(split_dir / "mixed.jsonl", mixed)
        file_hashes = {name: sha256_file(path) for name, path in sorted(files.items())}
        all_record_ids[split] = {record.record_id for record in records}
        histories = {_history_key(state) for state in used_states.values()}
        if split == "train":
            train_histories = histories
        elif histories & train_histories:
            raise AssertionError("rendered development source histories overlap training")
        split_manifests[split] = {
            **record_audit,
            "context": {
                "source_dataset_id": context.source_dataset_id,
                "source_manifest_sha256": context.source_manifest_sha256,
                "secret_source": context.training_secret_source,
                "declared_secret_set_sha256": context.training_secret_set_sha256,
                "answer_universe_sha256": context.answer_universe_sha256,
            },
            "files": file_hashes,
            "mixed_rows": len(mixed),
            "rendered_prompt_secret_fields": False,
        }
    if all_record_ids["train"] & all_record_ids["dev"]:
        raise AssertionError("train/dev curriculum record ids overlap")

    manifest = {
        "experiment_id": config["experiment_id"],
        "builder_version": config["builder_version"],
        "protocol_id": config["protocol_id"],
        "seed": config["seed"],
        "config_sha256": _hash(config),
        "source": source_audit,
        "evaluation_source": evaluation_source_audit,
        "splits": split_manifests,
        "train_dev_record_overlap": 0,
        "train_dev_history_overlap": 0,
        "training_file": "train/mixed.jsonl",
        "development_file": "dev/mixed.jsonl",
        "natural_generation_evaluation": True,
        "locked_test_access": False,
        "checks": [
            "source_hashes",
            "evaluation_allowed_words_hash",
            "train_dev_secret_separation",
            "train_dev_history_separation",
            "primitive_label_recomputation",
            "candidate_50_50_balance",
            "six_invalid_reasons_balanced",
            "duplicate_heavy_examples_preserved",
            "machine_readable_auxiliary_outputs",
            "natural_full_policy_outputs",
            "locked_test_unread",
        ],
    }
    write_json(manifest_path, manifest)
    return destination, manifest


def audit_bundle(config: dict[str, Any], bundle_dir: str | Path | None = None) -> dict[str, Any]:
    source_audit = audit_source(config)
    evaluation_source_audit = audit_evaluation_source(config)
    directory = Path(bundle_dir).resolve() if bundle_dir else _resolve(config["bundle_directory"])
    manifest = read_json(directory / "manifest.json")
    if manifest.get("experiment_id") != config["experiment_id"]:
        raise AssertionError("generated bundle experiment id mismatch")
    if manifest.get("config_sha256") != _hash(config):
        raise AssertionError("generated bundle config hash mismatch")
    if manifest.get("locked_test_access") is not False:
        raise AssertionError("generated bundle does not keep the locked test closed")
    if manifest.get("evaluation_source") != evaluation_source_audit:
        raise AssertionError("generated bundle evaluation-source audit drift")
    record_ids: dict[str, set[str]] = {}
    history_keys: dict[str, set[str]] = {}
    split_results: dict[str, Any] = {}
    for split in ("train", "dev"):
        split_dir = directory / split
        expected_hashes = manifest["splits"][split]["files"]
        actual_hashes = {name: sha256_file(split_dir / name) for name in expected_hashes}
        if actual_hashes != expected_hashes:
            raise AssertionError(f"generated {split} bundle hash mismatch")
        records = [_record_from_dict(row) for row in read_jsonl(split_dir / "records.jsonl")]
        states = [_training_state(row) for row in read_jsonl(split_dir / "source_states_private.jsonl")]
        context_data = manifest["splits"][split]["context"]
        source_dir = _resolve(config["source"]["directory"])
        secrets_name = "train_secrets.json" if split == "train" else "dev_secrets.json"
        context = TrainingContext(
            source_dataset_id=context_data["source_dataset_id"],
            source_manifest_sha256=context_data["source_manifest_sha256"],
            training_secret_source=context_data["secret_source"],
            training_secrets=tuple(read_json(source_dir / secrets_name)),
            answer_universe=tuple(read_json(source_dir / "universe.json")),
        )
        primitive_audit = audit_curriculum_records(records, states, context)
        by_state = {state.state_id: state for state in states}
        rendered_by_task: dict[str, list[dict[str, Any]]] = {}
        for task in TASKS:
            rows = read_jsonl(split_dir / f"{task}.jsonl")
            expected = [
                render_record(record, split, by_state[record.source_state_id])
                for record in records
                if record.task_type == task
            ]
            expected.sort(key=lambda row: row["record_id"])
            if rows != expected:
                raise AssertionError(f"generated {split} {task} rendering drift")
            rendered_by_task[task] = rows
        rendered = [row for task in TASKS for row in rendered_by_task[task]]
        if read_jsonl(split_dir / "mixed.jsonl") != _mixed_order(rendered, split, config["seed"]):
            raise AssertionError(f"generated {split} mixed ordering drift")
        record_ids[split] = {record.record_id for record in records}
        history_keys[split] = {_history_key(state) for state in states}
        split_results[split] = {
            "status": "passed",
            "records": len(records),
            "source_states": len(states),
            "task_distribution": primitive_audit["task_distribution"],
            "candidate_balance": primitive_audit["candidate_balance"],
            "hashes": actual_hashes,
        }
    if record_ids["train"] & record_ids["dev"] or history_keys["train"] & history_keys["dev"]:
        raise AssertionError("generated train/dev overlap")
    return {
        "status": "passed",
        "experiment_id": config["experiment_id"],
        "source": source_audit,
        "evaluation_source": evaluation_source_audit,
        "splits": split_results,
        "train_dev_record_overlap": 0,
        "train_dev_history_overlap": 0,
        "locked_test_access": False,
    }


def _is_upper_letter(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 1 and "A" <= value <= "Z"


def _is_position_map(value: Any) -> bool:
    if not isinstance(value, dict) or any(not _is_upper_letter(key) for key in value):
        return False
    for positions in value.values():
        if (
            not isinstance(positions, list)
            or any(isinstance(position, bool) or not isinstance(position, int) for position in positions)
            or positions != sorted(set(positions))
            or any(position not in range(1, 6) for position in positions)
        ):
            return False
    return True


def _is_count_map(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and all(_is_upper_letter(key) for key in value)
        and all(
            not isinstance(count, bool) and isinstance(count, int) and count in range(0, 6)
            for count in value.values()
        )
    )


def _is_constraint_object(value: Any) -> bool:
    expected_keys = {
        "fixed",
        "forbidden",
        "min_counts",
        "max_counts",
        "excluded",
        "position_evidence",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        return False
    fixed = value["fixed"]
    excluded = value["excluded"]
    evidence = value["position_evidence"]
    return (
        isinstance(fixed, list)
        and len(fixed) == 5
        and all(item is None or _is_upper_letter(item) for item in fixed)
        and _is_position_map(value["forbidden"])
        and _is_count_map(value["min_counts"])
        and _is_count_map(value["max_counts"])
        and isinstance(excluded, list)
        and all(_is_upper_letter(letter) for letter in excluded)
        and excluded == sorted(set(excluded))
        and isinstance(evidence, dict)
        and set(evidence) == {"yellow", "gray"}
        and _is_position_map(evidence["yellow"])
        and _is_position_map(evidence["gray"])
    )


def parse_generated_output(
    record: CurriculumRecord, raw_output: str, allowed_words: Sequence[str]
) -> tuple[Any | None, dict[str, Any]]:
    raw = str(raw_output)
    if record.task_type == "full_policy":
        parsed = parse_terminal_answer(raw, allowed_words)
        word = parsed.get("parsed_guess") if parsed.get("status") == "ok" else None
        return (
            {"word": word} if word else None,
            {
                "format_valid": bool(parsed.get("format_valid")),
                "parse_status": parsed.get("status"),
                "parsed_prediction": {"word": word} if word else None,
            },
        )
    try:
        value = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        return None, {"format_valid": False, "parse_status": "invalid_json", "parsed_prediction": None}
    if not isinstance(value, dict):
        return None, {"format_valid": False, "parse_status": "not_object", "parsed_prediction": None}
    if record.task_type in {"feedback_decode", "constraint_merge"}:
        if not _is_constraint_object(value):
            return None, {
                "format_valid": False,
                "parse_status": "constraint_schema_error",
                "parsed_prediction": None,
            }
    elif record.task_type == "candidate_validity":
        valid = value.get("valid")
        reason = value.get("reason")
        violations = value.get("violations")
        schema_valid = (
            set(value) == {"valid", "reason", "violations"}
            and isinstance(valid, bool)
            and isinstance(violations, list)
            and all(isinstance(item, str) and item in INVALID_REASONS for item in violations)
            and len(violations) == len(set(violations))
            and (
                (valid and reason is None and not violations)
                or (not valid and reason in INVALID_REASONS and reason in violations)
            )
        )
        if not schema_valid:
            return None, {
                "format_valid": False,
                "parse_status": "candidate_schema_error",
                "parsed_prediction": None,
            }
    elif record.task_type == "singleton_solve":
        if set(value) != {"word"}:
            return None, {"format_valid": False, "parse_status": "word_schema_error", "parsed_prediction": None}
        word = normalize_word(str(value.get("word", "")))
        if not is_five_ascii_letters(word):
            return None, {"format_valid": False, "parse_status": "word_schema_error", "parsed_prediction": None}
        value = {"word": word}
    return value, {"format_valid": True, "parse_status": "ok", "parsed_prediction": value}


def evaluate_raw_outputs(
    records: Sequence[CurriculumRecord],
    raw_outputs: Mapping[str, str],
    allowed_words: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    format_counts: dict[str, Counter[str]] = {task: Counter() for task in TASKS}
    for record in records:
        raw = raw_outputs.get(record.record_id)
        if raw is None:
            parsed, parse = None, {"format_valid": False, "parse_status": "missing", "parsed_prediction": None}
        else:
            parsed, parse = parse_generated_output(record, raw, allowed_words)
        if parsed is not None:
            predictions[record.record_id] = parsed
        format_counts[record.task_type]["total"] += 1
        format_counts[record.task_type]["valid"] += int(parse["format_valid"])
        rows.append({
            "record_id": record.record_id,
            "task_type": record.task_type,
            "raw_output": raw,
            **parse,
            "expected_target": record.target,
        })
    metrics = evaluate_curriculum_predictions(records, predictions)
    metrics["format_by_task"] = {
        task: {
            "valid": format_counts[task]["valid"],
            "total": format_counts[task]["total"],
            "compliance": (
                format_counts[task]["valid"] / format_counts[task]["total"]
                if format_counts[task]["total"]
                else None
            ),
        }
        for task in TASKS
    }
    metrics["natural_generation"] = True
    metrics["output_repair"] = False
    metrics["locked_test_access"] = False
    return rows, metrics


def evaluate_gates(metrics: Mapping[str, Any], gates: Mapping[str, float]) -> dict[str, Any]:
    by_task = metrics["by_task"]
    reason_metrics = metrics["candidate_invalid_reason_accuracy"]
    checks = {
        "coverage": metrics["coverage"] >= gates["coverage_min"],
        "feedback_decode_accuracy": by_task["feedback_decode"]["accuracy"] >= gates["feedback_decode_accuracy_min"],
        "constraint_merge_accuracy": by_task["constraint_merge"]["accuracy"] >= gates["constraint_merge_accuracy_min"],
        "candidate_validity_accuracy": by_task["candidate_validity"]["accuracy"] >= gates["candidate_validity_accuracy_min"],
        "candidate_invalid_reason_accuracy": all(
            item["accuracy"] is not None and item["accuracy"] >= gates["candidate_invalid_reason_accuracy_min"]
            for item in reason_metrics.values()
        ),
        "singleton_solve_accuracy": by_task["singleton_solve"]["accuracy"] >= gates["singleton_solve_accuracy_min"],
        "full_policy_accuracy": by_task["full_policy"]["accuracy"] >= gates["full_policy_accuracy_min"],
        "full_policy_format_compliance": (
            metrics["format_by_task"]["full_policy"]["compliance"]
            >= gates["full_policy_format_compliance_min"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": dict(gates)}


def _generate_natural_outputs(
    model,
    tokenizer,
    rendered_rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, str]:
    import torch

    from wordle_lab.protocol.generation import stop_token_ids

    device = next(model.parameters()).device
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs: dict[str, str] = {}
    try:
        for start in range(0, len(rendered_rows), batch_size):
            batch = rendered_rows[start : start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
                for row in batch
            ]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    eos_token_id=stop_token_ids(tokenizer),
                    pad_token_id=tokenizer.pad_token_id,
                )
            width = inputs["input_ids"].shape[1]
            for row, output in zip(batch, generated):
                outputs[str(row["record_id"])] = tokenizer.decode(
                    output[width:], skip_special_tokens=True
                ).strip()
    finally:
        tokenizer.padding_side = old_padding
    return outputs


def training_spec(config: dict[str, Any], backend: str, bundle_manifest: Mapping[str, Any]) -> dict[str, Any]:
    if backend not in {"transformers", "unsloth"}:
        raise ValueError("backend must be transformers or unsloth")
    training = config["training"]
    backend_id = TRANSFORMERS_BACKEND_ID if backend == "transformers" else UNSLOTH_BACKEND_ID
    return {
        "experiment_id": config["experiment_id"],
        "method": f"structured-microtasks-{backend}-completion-sft",
        "backend": backend_id,
        "representation": "structured_microtasks_mixed",
        "protocol_id": config["protocol_id"],
        "seed": config["seed"],
        "max_steps": training["max_steps"],
        "learning_rate": training["learning_rate"],
        "batch_size": training["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "max_length": training["max_length"],
        "precision": training["precision"],
        "quantization": training["quantization"],
        "word_token_weight": 1.0,
        "lora": training["lora"],
        "model": config["model"],
        "data": {
            "bundle_directory": config["bundle_directory"],
            "training_file": bundle_manifest["training_file"],
            "training_rows": bundle_manifest["splits"]["train"]["mixed_rows"],
            "training_sha256": bundle_manifest["splits"]["train"]["files"]["mixed.jsonl"],
            "development_file": bundle_manifest["development_file"],
            "development_sha256": bundle_manifest["splits"]["dev"]["files"]["mixed.jsonl"],
            "config_sha256": bundle_manifest["config_sha256"],
            "dev_role": "evaluation_only_never_training",
        },
        "locked_test_access": False,
        "candidate_injection": False,
        "vocabulary_masking": False,
        "reranking": False,
        "repeat_ban": False,
        "output_repair": False,
        "harness_selected_guess": False,
    }


def run_id_for_spec(spec: Mapping[str, Any]) -> str:
    digest = _hash(spec)[:10]
    backend = "unsloth" if spec["backend"] == UNSLOTH_BACKEND_ID else "transformers"
    return f"structured-microtasks-{backend}-s{spec['seed']}-{digest}"


def dry_run(config: dict[str, Any], backend: str) -> dict[str, Any]:
    bundle_dir = _resolve(config["bundle_directory"])
    bundle_audit = audit_bundle(config, bundle_dir)
    manifest = read_json(bundle_dir / "manifest.json")
    spec = training_spec(config, backend, manifest)
    run_id = run_id_for_spec(spec)
    environment = unsloth_environment() if backend == "unsloth" else {"backend_id": TRANSFORMERS_BACKEND_ID}
    if backend == "unsloth":
        environment = {**environment, "backend_id": UNSLOTH_BACKEND_ID}
    return {
        "status": "dry_run_passed",
        "backend": backend,
        "run_id": run_id,
        "planned_run_dir": str(_resolve(config["run_output_root"]) / run_id),
        "spec": spec,
        "bundle_audit": bundle_audit,
        "environment": environment,
        "training_started": False,
        "locked_test_access": False,
    }


def _prepare_training_run(
    config: dict[str, Any], backend: str, run_dir: str | Path | None
) -> tuple[Path, dict[str, Any]]:
    bundle_dir = _resolve(config["bundle_directory"])
    audit_bundle(config, bundle_dir)
    manifest = read_json(bundle_dir / "manifest.json")
    spec = training_spec(config, backend, manifest)
    destination = Path(run_dir).resolve() if run_dir else _resolve(config["run_output_root"]) / run_id_for_spec(spec)
    spec_path = destination / "spec.json"
    if spec_path.exists() and read_json(spec_path) != spec:
        raise RuntimeError(f"refusing to overwrite a different run at {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(spec_path, spec)
    write_json(destination / "dataset_manifest.json", manifest)
    return destination, spec


def train(config: dict[str, Any], backend: str, run_dir: str | Path | None = None) -> dict[str, Any]:
    destination, spec = _prepare_training_run(config, backend, run_dir)
    if (destination / "summary.json").exists() or (destination / "checkpoints").exists():
        raise RuntimeError(f"refusing to overwrite existing training artifacts at {destination}")
    rows = read_jsonl(_resolve(config["bundle_directory"]) / "train" / "mixed.jsonl")
    set_seed(config["seed"])
    if backend == "unsloth":
        model, _, accounting = train_unsloth_sft(rows, destination, spec)
    else:
        model, accounting = train_sft(rows, destination, spec)
    summary = {
        "status": "trained",
        "experiment_id": config["experiment_id"],
        "backend": backend,
        "run_id": run_id_for_spec(spec),
        "run_dir": str(destination),
        "accounting": accounting,
        "locked_test_access": False,
    }
    write_json(destination / "summary.json", summary)
    del model
    gc.collect()
    return summary


def evaluate_checkpoint(
    config: dict[str, Any], run_dir: str | Path, checkpoint: str = "final"
) -> dict[str, Any]:
    import torch

    from wordle_lab.models import load_adapter

    bundle_dir = _resolve(config["bundle_directory"])
    audit_bundle(config, bundle_dir)
    destination = Path(run_dir).resolve()
    spec = read_json(destination / "spec.json")
    manifest = read_json(bundle_dir / "manifest.json")
    expected = training_spec(
        config,
        "unsloth" if spec.get("backend") == UNSLOTH_BACKEND_ID else "transformers",
        manifest,
    )
    if spec != expected or spec.get("locked_test_access") is not False:
        raise RuntimeError("run spec is not an audited structured-microtask SFT run")
    checkpoint_dir = destination / "checkpoints" / checkpoint
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    output_dir = destination / "microtask-evaluation" / checkpoint
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        return read_json(summary_path)

    logical = [_record_from_dict(row) for row in read_jsonl(bundle_dir / "dev" / "records.jsonl")]
    rendered = read_jsonl(bundle_dir / "dev" / "mixed.jsonl")
    allowed = [
        line.strip().upper()
        for line in _resolve(config["evaluation"]["allowed_words"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer = load_tokenizer(checkpoint_dir)
    model = load_adapter(checkpoint_dir)
    try:
        set_seed(config["seed"])
        raw_outputs = _generate_natural_outputs(
            model,
            tokenizer,
            rendered,
            batch_size=config["evaluation"]["batch_size"],
            max_new_tokens=config["evaluation"]["max_new_tokens"],
        )
        parsed_rows, metrics = evaluate_raw_outputs(logical, raw_outputs, allowed)
        write_jsonl(output_dir / "raw_outputs.jsonl", [
            {"record_id": record_id, "raw_output": raw_outputs[record_id]}
            for record_id in sorted(raw_outputs)
        ])
        write_jsonl(output_dir / "parsed_outputs.jsonl", parsed_rows)
        result = {
            "status": "dev_evaluated",
            "experiment_id": config["experiment_id"],
            "checkpoint": checkpoint,
            "split": "development_only",
            "metrics": metrics,
            "gates": evaluate_gates(metrics, config["evaluation"]["gates"]),
            "natural_generation": True,
            "locked_test_access": False,
        }
        result["decision"] = (
            "development_gates_passed_replication_may_proceed_locked_test_closed"
            if result["gates"]["passed"]
            else "development_gates_failed_stop_locked_test_closed"
        )
        write_json(summary_path, result)
        return result
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def token_length_audit(config: dict[str, Any]) -> dict[str, Any]:
    bundle_dir = _resolve(config["bundle_directory"])
    audit_bundle(config, bundle_dir)
    tokenizer = load_tokenizer()
    max_length = int(config["training"]["max_length"])
    result: dict[str, Any] = {"max_length": max_length, "splits": {}}
    for split in ("train", "dev"):
        rows = read_jsonl(bundle_dir / split / "mixed.jsonl")
        lengths = []
        for row in rows:
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
            lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        result["splits"][split] = {
            "rows": len(rows),
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "over_limit": sum(length > max_length for length in lengths),
        }
        if result["splits"][split]["over_limit"]:
            raise AssertionError(f"{split} contains rows above the preregistered token limit")
    result.update({"status": "passed", "locked_test_access": False})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Structured Wordle microtask curriculum experiment")
    parser.add_argument(
        "command",
        choices=("build", "audit", "token-audit", "dry-run", "train-sft", "train-unsloth", "evaluate"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.bundle_dir:
        config = {**config, "bundle_directory": str(args.bundle_dir)}
    if args.command == "build":
        directory, manifest = build_bundle(config, args.bundle_dir, force=args.force)
        result = {"status": "built", "bundle_dir": str(directory), "manifest": manifest}
    elif args.command == "audit":
        result = audit_bundle(config, args.bundle_dir)
    elif args.command == "token-audit":
        result = token_length_audit(config)
    elif args.command == "dry-run":
        backend = "unsloth"
        result = dry_run(config, backend)
    elif args.command in {"train-sft", "train-unsloth"}:
        backend = "transformers" if args.command == "train-sft" else "unsloth"
        result = train(config, backend, args.run_dir)
    else:
        if args.run_dir is None:
            parser.error("evaluate requires --run-dir")
        result = evaluate_checkpoint(config, args.run_dir, args.checkpoint)
    if args.compact:
        result = {
            key: result.get(key)
            for key in (
                "status",
                "experiment_id",
                "bundle_dir",
                "run_id",
                "run_dir",
                "decision",
                "locked_test_access",
            )
            if key in result
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
