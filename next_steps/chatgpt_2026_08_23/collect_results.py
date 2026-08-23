from __future__ import annotations

"""Collect compact, content-addressed development evidence from ignored runs.

The collector is driven by an explicit JSON declaration. It never discovers
runs or artifacts by walking ``artifacts/`` and it rejects checkpoint paths,
locked-test paths, locked-test payloads, binary model files, and path escapes.
Missing or blocked experiments are recorded as unavailable; they are never
represented by synthetic zero metrics.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from wordle_lab.common import ROOT, canonical_json


SCHEMA_VERSION = "wordle-results-collection-v1"
OUTPUT_SCHEMA_VERSION = "wordle-results-evidence-v1"
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = MODULE_DIR / "results"
DEFAULT_MAX_BYTES = 2 * 1024 * 1024
HARD_MAX_BYTES = 5 * 1024 * 1024
ALLOWED_ROLES = {
    "spec": {".json"},
    "accounting": {".json"},
    "train_metrics": {".jsonl"},
    "eval_summary": {".json"},
    "comparison_summary": {".json"},
    "gate_metrics": {".json"},
    "raw_jsonl": {".jsonl"},
}
UNAVAILABLE_STATUSES = {"blocked", "unavailable"}
ACTIVE_STATUSES = {"complete", "expected"}
FORBIDDEN_PATH_TOKENS = {"test", "tests", "lockedtest"}
FORBIDDEN_PATH_PARTS = {
    "checkpoint",
    "checkpoints",
    "adapter_model.bin",
    "adapter_model.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
}
FORBIDDEN_PAYLOAD_KEYS = {
    "locked_test_answer",
    "locked_test_answers",
    "locked_test_secret",
    "locked_test_secrets",
    "test_answer",
    "test_answers",
    "test_secret",
    "test_secrets",
}


class CollectionViolation(RuntimeError):
    """Raised when a declaration or artifact crosses an evidence boundary."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _path_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}


def _assert_declarative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CollectionViolation(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise CollectionViolation(f"{label} must stay inside its declared root")
    normalized_parts = [part.lower() for part in path.parts]
    tokens = _path_tokens(value)
    if tokens & FORBIDDEN_PATH_TOKENS or any(
        part in FORBIDDEN_PATH_PARTS or part.startswith("checkpoint-")
        for part in normalized_parts
    ):
        raise CollectionViolation(f"{label} names a locked-test or checkpoint path")
    if path.suffix.lower() in {".bin", ".pt", ".pth", ".safetensors", ".ckpt"}:
        raise CollectionViolation(f"{label} names a binary model artifact")
    return path.as_posix()


def _assert_payload_safe(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).lower()
            child = f"{path}.{raw_key}"
            if key in FORBIDDEN_PAYLOAD_KEYS:
                raise CollectionViolation(f"artifact contains locked-test payload at {child}")
            if key == "locked_test_access" and nested is not False:
                raise CollectionViolation(f"artifact declares locked-test access at {child}")
            if key == "test_access" and nested not in (False, "closed", "forbidden", None):
                raise CollectionViolation(f"artifact declares test access at {child}")
            if key in {"split", "evaluation_split", "secret_split"} and str(nested).lower() in {
                "test",
                "locked_test",
                "locked-test",
            }:
                raise CollectionViolation(f"artifact contains a locked-test split at {child}")
            _assert_payload_safe(nested, child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _assert_payload_safe(nested, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"artifact contains a non-finite number at {path}")


def template_manifest() -> dict[str, Any]:
    """Return a valid empty declaration plus a non-operative schema example."""

    return {
        "schema_version": SCHEMA_VERSION,
        "collection_id": "chatgpt-2026-08-23-next-steps",
        "locked_test_access": False,
        "experiments": [],
        "artifact_example": {
            "experiment_id": "replace-with-experiment-id",
            "run_dir": "artifacts/runs/replace-with-run-id",
            "status": "complete",
            "artifacts": [
                {
                    "source": "spec.json",
                    "destination": "spec.json",
                    "role": "spec",
                    "scope": "metadata",
                    "required": True,
                },
                {
                    "source": "train_metrics.jsonl",
                    "destination": "train_metrics.jsonl",
                    "role": "train_metrics",
                    "scope": "train",
                    "required": True,
                },
                {
                    "source": "games.jsonl",
                    "destination": "games.jsonl",
                    "role": "raw_jsonl",
                    "scope": "dev",
                    "required": False,
                    "max_bytes": DEFAULT_MAX_BYTES,
                },
            ],
        },
    }


def validate_collection_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and validate a collection declaration without reading runs."""

    if not isinstance(manifest, Mapping):
        raise CollectionViolation("collection manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CollectionViolation(f"collection manifest schema_version must be {SCHEMA_VERSION}")
    collection_id = manifest.get("collection_id")
    if not isinstance(collection_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", collection_id):
        raise CollectionViolation("collection_id must be a filesystem-safe identifier")
    if manifest.get("locked_test_access") is not False:
        raise CollectionViolation("collection manifest must explicitly keep locked_test_access false")
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list):
        raise CollectionViolation("collection manifest experiments must be a list")

    normalized_experiments: list[dict[str, Any]] = []
    seen_experiments: set[str] = set()
    seen_outputs: set[str] = set()
    for index, raw_experiment in enumerate(experiments):
        if not isinstance(raw_experiment, Mapping):
            raise CollectionViolation(f"experiment {index} must be a JSON object")
        experiment_id = raw_experiment.get("experiment_id")
        if not isinstance(experiment_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", experiment_id):
            raise CollectionViolation(f"experiment {index} has an invalid experiment_id")
        if experiment_id in seen_experiments:
            raise CollectionViolation(f"duplicate experiment_id: {experiment_id}")
        seen_experiments.add(experiment_id)
        output_name = raw_experiment.get("output_name", experiment_id)
        if not isinstance(output_name, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", output_name):
            raise CollectionViolation(f"experiment {experiment_id} has an invalid output_name")
        if output_name in seen_outputs:
            raise CollectionViolation(f"duplicate experiment output_name: {output_name}")
        seen_outputs.add(output_name)
        status = raw_experiment.get("status", "expected")
        if status not in ACTIVE_STATUSES | UNAVAILABLE_STATUSES:
            raise CollectionViolation(f"experiment {experiment_id} has an invalid status")
        reason = raw_experiment.get("reason")
        if status in UNAVAILABLE_STATUSES and (not isinstance(reason, str) or not reason.strip()):
            raise CollectionViolation(f"experiment {experiment_id} requires a non-empty unavailable reason")
        run_dir = raw_experiment.get("run_dir")
        if status in ACTIVE_STATUSES:
            run_dir = _assert_declarative_path(run_dir, f"experiment {experiment_id} run_dir")
        elif run_dir is not None:
            run_dir = _assert_declarative_path(run_dir, f"experiment {experiment_id} run_dir")
        artifacts = raw_experiment.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise CollectionViolation(f"experiment {experiment_id} artifacts must be a list")
        if status in ACTIVE_STATUSES and not artifacts:
            raise CollectionViolation(f"active experiment {experiment_id} must declare evidence artifacts")
        normalized_artifacts: list[dict[str, Any]] = []
        destinations: set[str] = set()
        for artifact_index, raw_artifact in enumerate(artifacts):
            if not isinstance(raw_artifact, Mapping):
                raise CollectionViolation(
                    f"experiment {experiment_id} artifact {artifact_index} must be a JSON object"
                )
            source = _assert_declarative_path(
                raw_artifact.get("source"),
                f"experiment {experiment_id} artifact {artifact_index} source",
            )
            destination = _assert_declarative_path(
                raw_artifact.get("destination", Path(source).name),
                f"experiment {experiment_id} artifact {artifact_index} destination",
            )
            if destination in destinations:
                raise CollectionViolation(
                    f"experiment {experiment_id} has duplicate artifact destination {destination}"
                )
            destinations.add(destination)
            role = raw_artifact.get("role")
            if role not in ALLOWED_ROLES:
                raise CollectionViolation(f"experiment {experiment_id} artifact {source} has invalid role")
            scope = raw_artifact.get("scope")
            if scope not in {"metadata", "train", "dev"}:
                raise CollectionViolation(
                    f"experiment {experiment_id} artifact {source} scope must be metadata, train, or dev"
                )
            if Path(source).suffix.lower() not in ALLOWED_ROLES[role] or Path(destination).suffix.lower() not in ALLOWED_ROLES[role]:
                raise CollectionViolation(
                    f"experiment {experiment_id} artifact {source} extension does not match role {role}"
                )
            required = raw_artifact.get("required", True)
            if not isinstance(required, bool):
                raise CollectionViolation(f"experiment {experiment_id} artifact {source} required must be boolean")
            max_bytes = raw_artifact.get("max_bytes", DEFAULT_MAX_BYTES)
            if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= HARD_MAX_BYTES:
                raise CollectionViolation(
                    f"experiment {experiment_id} artifact {source} max_bytes must be 1..{HARD_MAX_BYTES}"
                )
            normalized_artifacts.append(
                {
                    "source": source,
                    "destination": destination,
                    "role": role,
                    "scope": scope,
                    "required": required,
                    "max_bytes": max_bytes,
                }
            )
        normalized_experiments.append(
            {
                "experiment_id": experiment_id,
                "output_name": output_name,
                "status": status,
                "reason": reason,
                "run_dir": run_dir,
                "artifacts": normalized_artifacts,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "collection_id": collection_id,
        "locked_test_access": False,
        "experiments": normalized_experiments,
    }


def load_collection_manifest(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(path).resolve()
    raw = manifest_path.read_bytes()
    if len(raw) > DEFAULT_MAX_BYTES:
        raise CollectionViolation("collection declaration is unexpectedly large")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectionViolation("collection declaration is not valid UTF-8 JSON") from exc
    normalized = validate_collection_manifest(payload)
    provenance = {
        "path": str(manifest_path),
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }
    return normalized, provenance


def _resolve_inside(root: Path, relative: str, label: str) -> Path:
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CollectionViolation(f"{label} escapes its declared root") from exc
    return candidate


def _normalize_artifact(path: Path, raw: bytes | None = None) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes() if raw is None else raw
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("artifact is not UTF-8") from exc
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(
                text,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("artifact is not valid JSON") from exc
        _assert_payload_safe(payload)
        normalized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        metadata = {"format": "json", "jsonl_rows": None}
    elif suffix == ".jsonl":
        rows: list[Any] = []
        try:
            for line in text.splitlines():
                if line.strip():
                    rows.append(
                        json.loads(
                            line,
                            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
                        )
                    )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("artifact is not valid JSONL") from exc
        if not rows:
            raise ValueError("artifact JSONL contains no rows")
        for row in rows:
            _assert_payload_safe(row)
        normalized = ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")
        metadata = {"format": "jsonl", "jsonl_rows": len(rows)}
    else:  # The manifest validator prevents this branch.
        raise CollectionViolation(f"unsupported evidence extension: {suffix}")
    return normalized, {
        **metadata,
        "source_bytes": len(raw),
        "source_sha256": _sha256_bytes(raw),
        "collected_bytes": len(normalized),
        "collected_sha256": _sha256_bytes(normalized),
        "normalization": "sorted-json-v1" if suffix == ".json" else "canonical-jsonl-v1",
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def collect_results(
    manifest: Mapping[str, Any] | str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    workspace_root: str | Path = ROOT,
    runs_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Collect declared evidence, preserving explicit unavailable states."""

    workspace = Path(workspace_root).resolve()
    allowed_runs = Path(runs_root).resolve() if runs_root is not None else (workspace / "artifacts" / "runs").resolve()
    if isinstance(manifest, Mapping):
        declaration = validate_collection_manifest(manifest)
        declaration_provenance = {
            "path": None,
            "bytes": len(canonical_json(declaration).encode("utf-8")),
            "sha256": _sha256_bytes(canonical_json(declaration).encode("utf-8")),
        }
    else:
        declaration, declaration_provenance = load_collection_manifest(manifest)
        manifest_path = Path(manifest).resolve()
        try:
            declaration_provenance["path"] = manifest_path.relative_to(workspace).as_posix()
        except ValueError:
            declaration_provenance["path"] = str(manifest_path)
    output = Path(output_dir).resolve()
    planned_files: list[tuple[Path, bytes]] = []
    experiment_reports: list[dict[str, Any]] = []

    for experiment in declaration["experiments"]:
        report: dict[str, Any] = {
            "experiment_id": experiment["experiment_id"],
            "declared_status": experiment["status"],
            "availability": None,
            "reason": None,
            "run_directory": experiment["run_dir"],
            "artifacts": [],
            "metrics": None,
            "locked_test_access": False,
        }
        if experiment["status"] in UNAVAILABLE_STATUSES:
            report["availability"] = "unavailable"
            report["reason"] = experiment["reason"]
            experiment_reports.append(report)
            continue

        run_dir = _resolve_inside(workspace, experiment["run_dir"], "run_dir")
        try:
            run_dir.relative_to(allowed_runs)
        except ValueError as exc:
            raise CollectionViolation(
                f"experiment {experiment['experiment_id']} run_dir is outside artifacts/runs"
            ) from exc
        if not run_dir.is_dir():
            report["availability"] = "unavailable"
            report["reason"] = "run_directory_missing"
            experiment_reports.append(report)
            continue

        required_failures: list[str] = []
        for artifact in experiment["artifacts"]:
            source_path = _resolve_inside(run_dir, artifact["source"], "artifact source")
            destination_relative = f"{experiment['output_name']}/{artifact['destination']}"
            destination_path = _resolve_inside(output, destination_relative, "artifact destination")
            artifact_report: dict[str, Any] = {
                "role": artifact["role"],
                "scope": artifact["scope"],
                "required": artifact["required"],
                "source": str(source_path.relative_to(workspace).as_posix()),
                "destination": str(destination_path.relative_to(output).as_posix()),
                "availability": None,
                "reason": None,
            }
            if not source_path.is_file():
                artifact_report.update({"availability": "unavailable", "reason": "source_artifact_missing"})
            elif source_path.stat().st_size > artifact["max_bytes"]:
                artifact_report.update(
                    {
                        "availability": "unavailable",
                        "reason": "source_artifact_exceeds_size_limit",
                        "source_bytes": source_path.stat().st_size,
                        "max_bytes": artifact["max_bytes"],
                    }
                )
            else:
                try:
                    before = source_path.stat()
                    raw = source_path.read_bytes()
                    after = source_path.stat()
                    if (
                        len(raw) > artifact["max_bytes"]
                        or before.st_size != after.st_size
                        or before.st_mtime_ns != after.st_mtime_ns
                    ):
                        raise ValueError("source_artifact_changed_or_exceeded_limit_during_collection")
                    normalized, provenance = _normalize_artifact(source_path, raw)
                except CollectionViolation:
                    raise
                except ValueError as exc:
                    artifact_report.update({"availability": "unavailable", "reason": str(exc)})
                else:
                    artifact_report.update({"availability": "available", **provenance})
                    planned_files.append((destination_path, normalized))
            if artifact_report["availability"] != "available" and artifact["required"]:
                required_failures.append(artifact["source"])
            report["artifacts"].append(artifact_report)
        if required_failures:
            report["availability"] = "unavailable"
            report["reason"] = "required_artifacts_unavailable"
            report["required_artifacts_unavailable"] = required_failures
        else:
            report["availability"] = "available"
        experiment_reports.append(report)

    available = sum(report["availability"] == "available" for report in experiment_reports)
    unavailable = len(experiment_reports) - available
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "collection_id": declaration["collection_id"],
        "declaration": declaration_provenance,
        "status": "complete" if unavailable == 0 else "complete_with_unavailable_experiments",
        "dry_run": bool(dry_run),
        "output_directory": (
            output.relative_to(workspace).as_posix()
            if output == workspace or workspace in output.parents
            else str(output)
        ),
        "experiments_declared": len(experiment_reports),
        "experiments_available": available,
        "experiments_unavailable": unavailable,
        "files_available": sum(
            artifact["availability"] == "available"
            for report in experiment_reports
            for artifact in report["artifacts"]
        ),
        "experiments": experiment_reports,
        "locked_test_access": False,
    }
    result["manifest_content_sha256"] = _sha256_bytes(canonical_json(result).encode("utf-8"))
    if not dry_run:
        for destination, payload in planned_files:
            _atomic_write(destination, payload)
        manifest_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _atomic_write(output / "collection_manifest.json", manifest_bytes)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect compact, Git-trackable Wordle experiment evidence")
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="print an empty declarative collection manifest")
    template.set_defaults(command="template")
    validate = commands.add_parser("validate", help="validate a declaration without reading any run")
    validate.add_argument("--manifest", type=Path, required=True)
    collect = commands.add_parser("collect", help="collect declared non-checkpoint evidence")
    collect.add_argument("--manifest", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    collect.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            result = template_manifest()
        elif args.command == "validate":
            declaration, provenance = load_collection_manifest(args.manifest)
            result = {
                "status": "valid",
                "collection_id": declaration["collection_id"],
                "experiments_declared": len(declaration["experiments"]),
                "declaration": provenance,
                "runs_read": False,
                "locked_test_access": False,
            }
        else:
            result = collect_results(args.manifest, args.output_dir, dry_run=args.dry_run)
    except (CollectionViolation, OSError) as exc:
        print(f"collection error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
