from __future__ import annotations

from collections.abc import Sequence

from wordle_lab.protocol.prompting import SYSTEM_PROMPT, render_user_prompt

REPRESENTATIONS = ("state_direct", "episode_multiturn", "state_rationale", "mixed_curriculum")


def history_tuples(record: dict) -> list[tuple[str, str]]:
    return [(item["guess"], item["feedback"]) for item in record["history"]]


def direct_completion(record: dict) -> str:
    return f"Final answer: {record['facts']['oracle_action']}"


def rationale_completion(record: dict, action_key: str = "oracle_action") -> str:
    facts = record["facts"]
    action = facts[action_key]
    expected_key = "oracle_expected_remaining" if action_key == "oracle_action" else "hard_negative_expected_remaining"
    fixed = ", ".join(f"{position}={letter}" for position, letter in facts["fixed_positions"].items()) or "none"
    excluded = ", ".join(facts["excluded_seen_letters"]) or "none"
    required = ", ".join(facts["letters_in_every_candidate"]) or "none"
    if action_key == "oracle_action":
        rationale = "It has the lowest expected remaining candidate count under the deterministic oracle."
    else:
        rationale = f"It is worse than the oracle action by {facts['hard_negative_regret']:.3f} expected candidates."
    return (
        f"Constraints: {facts['posterior_count']} candidates remain. Fixed positions: {fixed}. "
        f"Required letters: {required}. Excluded seen letters: {excluded}.\n"
        f"Action assessment: {action} has expected remaining candidates {facts[expected_key]:.3f}.\n"
        f"Choice rationale: {rationale}\n"
        f"Final answer: {action}"
    )


def state_messages(record: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(history_tuples(record))},
    ]


def episode_messages(record: dict) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    history = history_tuples(record)
    messages.append({"role": "user", "content": "WORDLE\nNo previous guesses. Choose the first guess."})
    for index, (guess, feedback) in enumerate(history):
        messages.append({"role": "assistant", "content": f"Final answer: {guess}"})
        messages.append({"role": "user", "content": f"Feedback: {guess} -> {feedback}. Choose guess {index + 2}."})
    return messages


def render_representation(records: Sequence[dict], representation: str) -> list[dict]:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    output = []
    for index, record in enumerate(records):
        kind = representation
        messages = state_messages(record)
        completion = direct_completion(record)
        if representation == "episode_multiturn":
            messages = episode_messages(record)
        elif representation == "state_rationale":
            completion = rationale_completion(record)
        elif representation == "mixed_curriculum":
            bucket = index % 20
            if bucket < 9:
                kind = "direct"
            elif bucket < 15:
                kind = "later_turn_direct"
            elif bucket < 18:
                kind = "failure_correction"
                messages = state_messages(record) + [
                    {"role": "assistant", "content": "I choose XXXXX."},
                    {"role": "user", "content": "That response was malformed and ignored. Use the required terminal-answer envelope."},
                ]
            else:
                kind = "rationale"
                completion = rationale_completion(record)
        output.append({
            "schema_version": "wordle-rendered-example-v2",
            "example_id": f"{record['state_id']}-{representation}",
            "state_id": record["state_id"],
            "split": record["split"],
            "representation": representation,
            "curriculum_kind": kind,
            "turn": record["turn"],
            "prompt": messages,
            "completion": [{"role": "assistant", "content": completion}],
        })
    if representation == "mixed_curriculum":
        output.sort(key=lambda row: ({"direct": 0, "later_turn_direct": 1, "failure_correction": 2, "rationale": 3}.get(row["curriculum_kind"], 0), row["turn"], row["state_id"]))
    return output
