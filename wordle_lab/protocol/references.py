from __future__ import annotations

import random
from collections import Counter

from .env import WordleEnv, posterior_candidates
from .oracle import GreedyPartitionOracle


def evaluate_reference(policy: str, answers: list[str], allowed_words: list[str], answer_vocabulary: list[str], seed: int = 1337) -> tuple[list[dict], dict]:
    if policy not in {"random_allowed", "random_posterior", "oracle"}:
        raise ValueError(policy)
    rng = random.Random(seed)
    oracle = GreedyPartitionOracle(answer_vocabulary) if policy == "oracle" else None
    records = []
    for game_id, answer in enumerate(answers):
        env = WordleEnv(answer, allowed_words)
        turns = []
        while not env.done:
            posterior = posterior_candidates(env.history, answer_vocabulary)
            if oracle:
                guess = oracle.best(env.history)["guess"]
            elif policy == "random_posterior":
                guess = rng.choice(posterior)
            else:
                used = {guess for guess, _ in env.history}
                guess = rng.choice([word for word in allowed_words if word not in used])
            step = env.step(guess)
            turns.append({**step, "posterior_before": len(posterior)})
        records.append({"game_id": game_id, "answer": answer, "won": env.won, "guesses": len(env.history), "turns": turns})
    wins = sum(row["won"] for row in records)
    by_turn = Counter(str(row["guesses"]) for row in records if row["won"])
    return records, {"policy": policy, "n_games": len(records), "wins": wins, "win_rate": wins / len(records), "wins_by_guess": {str(i): by_turn[str(i)] for i in range(1, 7)}}
