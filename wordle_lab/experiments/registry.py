from __future__ import annotations

from wordle_lab.common import canonical_json, sha256_text
from .schema import validate_spec


def run_id(spec: dict) -> str:
    validate_spec(spec)
    digest = sha256_text(canonical_json(spec))[:10]
    return f"{spec['method']}-{spec['representation']}-s{spec['seed']}-{digest}"
