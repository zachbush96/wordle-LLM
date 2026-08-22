from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence

from .env import WordleEnv, posterior_candidates
from .generation import generate
from .parsing import parse_terminal_answer


def evaluate(model, tokenizer, answers: Sequence[str], allowed_words: Sequence[str], answer_vocabulary: Sequence[str], max_calls: int = 12, batch_size: int = 16) -> tuple[list[dict], dict]:
    games = [{"game_id": i, "answer": answer, "env": WordleEnv(answer, allowed_words), "turns": [], "calls": 0} for i, answer in enumerate(answers)]
    while True:
        active = [game for game in games if not game["env"].done and game["calls"] < max_calls]
        if not active:
            break
        generated = generate(model, tokenizer, [game["env"].history for game in active], batch_size=batch_size)
        for game, raw in zip(active, generated):
            env = game["env"]
            before = list(env.history)
            posterior_before = posterior_candidates(before, answer_vocabulary)
            parsed = parse_terminal_answer(raw["raw_output"], allowed_words)
            parsed["reasoning_tokens"] = len(tokenizer(parsed["reasoning_text"], add_special_tokens=False)["input_ids"])
            step = env.step(parsed["parsed_guess"] if parsed["status"] == "ok" else None)
            posterior_after = posterior_candidates(env.history, answer_vocabulary) if step["valid"] else posterior_before
            game["calls"] += 1
            game["turns"].append({**raw, **parsed, **step, "posterior_before": len(posterior_before), "posterior_after": len(posterior_after), "information_gain": len(posterior_before) - len(posterior_after), "constraint_violation": bool(step["valid"] and step["guess"] not in posterior_before)})
    records = []
    for game in games:
        env = game.pop("env")
        records.append({**game, "won": env.won, "guesses": len(env.history), "invalid_guesses": env.invalid_guesses})
    turns = [turn for game in records for turn in game["turns"]]
    valid = [turn for turn in turns if turn["valid"]]
    wins_by_guess = Counter(str(game["guesses"]) for game in records if game["won"])
    summary = {
        "n_games": len(records), "wins": sum(game["won"] for game in records),
        "win_rate": sum(game["won"] for game in records) / len(records),
        "wins_by_guess": {str(i): wins_by_guess[str(i)] for i in range(1, 7)},
        "mean_guesses_on_wins": statistics.mean([game["guesses"] for game in records if game["won"]]) if any(game["won"] for game in records) else None,
        "model_calls": len(turns), "format_failure_rate": sum(not turn["format_valid"] for turn in turns) / max(1, len(turns)),
        "invalid_guess_rate": sum(not turn["valid"] for turn in turns) / max(1, len(turns)),
        "repeat_guess_rate": sum(turn["repeat"] for turn in valid) / max(1, len(valid)),
        "constraint_violation_rate": sum(turn["constraint_violation"] for turn in valid) / max(1, len(valid)),
        "terminal_marker_compliance": sum(turn["format_valid"] for turn in turns) / max(1, len(turns)),
        "reasoning_presence_rate": sum(turn["reasoning_tokens"] > 0 for turn in turns) / max(1, len(turns)),
        "mean_reasoning_tokens": statistics.mean([turn["reasoning_tokens"] for turn in turns]) if turns else 0,
        "mean_generated_tokens": statistics.mean([turn["generated_tokens"] for turn in turns]) if turns else 0,
        "mean_latency_s": statistics.mean([turn["latency_s"] for turn in turns]) if turns else 0,
    }
    return records, summary
