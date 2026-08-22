from __future__ import annotations

import pandas as pd
from wordle_lab.common import ARTIFACTS, read_json


def collect_runs() -> pd.DataFrame:
    rows = []
    for summary_path in sorted((ARTIFACTS / "runs").glob("*/summary.json")):
        summary = read_json(summary_path)
        spec_path = summary_path.parent / "spec.json"
        spec = read_json(spec_path) if spec_path.exists() else {}
        rows.append({"run_id": summary_path.parent.name, "method": spec.get("method", "base"), "representation": spec.get("representation", "protocol"), "seed": spec.get("seed"), **{key: summary.get(key) for key in ("split", "n_games", "win_rate", "format_failure_rate", "invalid_guess_rate", "repeat_guess_rate", "constraint_violation_rate", "mean_generated_tokens", "mean_latency_s")}})
    return pd.DataFrame(rows)
