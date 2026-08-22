from __future__ import annotations

from pathlib import Path
from .collect import collect_runs


def write_pilot_report(path: Path) -> Path:
    table = collect_runs()
    lines = ["# WORDLE-PROTOCOL-002 pilot report", "", "These are development results; the locked test was not used for model selection.", ""]
    lines.append("No completed runs." if table.empty else table.to_markdown(index=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
