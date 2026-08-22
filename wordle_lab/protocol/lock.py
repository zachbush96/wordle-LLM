from __future__ import annotations

from pathlib import Path

from wordle_lab.common import DATA, canonical_json, sha256_file, sha256_text, write_json
from .generation import GENERATION_CONFIG
from .parsing import TERMINAL_PATTERN
from .prompting import PROMPT_VERSION, SYSTEM_PROMPT

PROTOCOL_ID = "WORDLE-PROTOCOL-002"


def build_lock(wordlist: Path, split_manifest: Path, retention_probes: Path | None = None) -> dict:
    protocol_dir = Path(__file__).resolve().parent
    components = {
        "protocol_id": PROTOCOL_ID,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "terminal_pattern": TERMINAL_PATTERN,
        "generation": GENERATION_CONFIG,
        "wordlist_sha256": sha256_file(wordlist),
        "split_manifest_sha256": sha256_file(split_manifest),
        "retention_probe_sha256": sha256_file(retention_probes) if retention_probes else None,
        "component_files": {
            name: sha256_file(protocol_dir / name)
            for name in ("env.py", "prompting.py", "parsing.py", "generation.py", "evaluator.py", "retention.py")
        },
    }
    lock = {**components, "protocol_sha256": sha256_text(canonical_json(components))}
    write_json(DATA / "protocol_lock.json", lock)
    return lock
