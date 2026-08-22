from __future__ import annotations

import re

from wordle_lab.protocol.env import posterior_candidates

_COUNT = re.compile(r"Constraints:\s*(\d+) candidates remain", re.IGNORECASE)
_FIXED = re.compile(r"Fixed positions:\s*([^.]*)", re.IGNORECASE)
_EXCLUDED = re.compile(r"Excluded seen letters:\s*([^.]*)", re.IGNORECASE)


def score_trace_facts(games: list[dict], answer_vocabulary: list[str]) -> dict:
    facts = []
    for game in games:
        history = []
        for turn_number, turn in enumerate(game["turns"], 1):
            trace = turn.get("reasoning_text", "")
            posterior = posterior_candidates(history, answer_vocabulary)
            count_match = _COUNT.search(trace)
            if count_match:
                facts.append({"game_id": game["game_id"], "turn": turn_number, "fact_type": "posterior_count", "correct": int(count_match.group(1)) == len(posterior)})
            fixed_match = _FIXED.search(trace)
            if fixed_match:
                stated_text = fixed_match.group(1).strip()
                stated = {} if stated_text.lower() == "none" else {int(pos): letter for pos, letter in re.findall(r"([1-5])=([A-Z])", stated_text.upper())}
                actual = {i + 1: letters[0] for i in range(5) if len(letters := sorted({word[i] for word in posterior})) == 1}
                facts.append({"game_id": game["game_id"], "turn": turn_number, "fact_type": "fixed_positions", "correct": stated == actual})
            excluded_match = _EXCLUDED.search(trace)
            if excluded_match:
                stated_text = excluded_match.group(1).strip()
                stated = set() if stated_text.lower() == "none" else set(re.findall(r"[A-Z]", stated_text.upper()))
                seen = set("".join(guess for guess, _ in history)); possible = set("".join(posterior))
                facts.append({"game_id": game["game_id"], "turn": turn_number, "fact_type": "excluded_letters", "correct": stated == seen - possible})
            if turn.get("valid"):
                history.append((turn["guess"], turn["feedback"]))
    by_type = {}
    for fact_type in sorted({row["fact_type"] for row in facts}):
        group = [row for row in facts if row["fact_type"] == fact_type]
        by_type[fact_type] = {"facts": len(group), "accuracy": sum(row["correct"] for row in group) / len(group)}
    return {"facts": facts, "summary": by_type, "overall_accuracy": sum(row["correct"] for row in facts) / len(facts) if facts else None}
