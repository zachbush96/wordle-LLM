from __future__ import annotations

from collections.abc import Sequence

from .builders import direct_completion, rationale_completion, state_messages


def build_preferences(records: Sequence[dict], rationale: bool = False) -> list[dict]:
    """Deterministic 80/10/10 hard/behavioral/malformed pair mix."""
    rows = []
    for index, record in enumerate(records):
        bucket = index % 10
        chosen = rationale_completion(record) if rationale else direct_completion(record)
        if bucket < 8:
            negative_type = "hard_strategic"
            rejected = rationale_completion(record, "hard_negative") if rationale else f"Final answer: {record['facts']['hard_negative']}"
        elif bucket == 8:
            negative_type = "behavioral"
            if record["history"]:
                rejected = f"Final answer: {record['history'][-1]['guess']}"
            else:
                rejected = "Final answer: XXXXX"
        else:
            negative_type = "malformed"
            rejected = f"I would choose {record['facts']['hard_negative']} because it seems useful."
        rows.append({
            "schema_version": "wordle-preference-pair-v2", "pair_id": f"pair-{record['state_id']}",
            "state_id": record["state_id"], "prompt": state_messages(record),
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}], "negative_type": negative_type,
        })
    return rows
