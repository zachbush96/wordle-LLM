from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import torch

from wordle_lab.common import ARTIFACTS, DATA, canonical_json, read_json, set_seed, write_json, write_jsonl
from wordle_lab.models import load_adapter, load_tokenizer
from wordle_lab.protocol import generation
from wordle_lab.protocol.evaluator import evaluate
from wordle_lab.protocol.prompting import SYSTEM_PROMPT, inference_messages


PromptBuilder = Callable[[Sequence[tuple[str, str]]], list[dict[str, str]]]


def _episode_messages(history: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": "WORDLE\nNo previous guesses. Choose the first guess."})
    for index, (guess, feedback) in enumerate(history):
        messages.append({"role": "assistant", "content": f"Final answer: {guess}"})
        messages.append(
            {
                "role": "user",
                "content": f"Feedback: {guess} -> {feedback}. Choose guess {index + 2}. Do not repeat {guess}.",
            }
        )
    return messages


def _explicit_feedback_messages(history: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    system = (
        "Play Wordle. G means fixed in that position, Y means present in another position, and B means absent after "
        "duplicate-letter accounting. Use every feedback row. Never repeat a prior guess. End with `Final answer:` "
        "and one five-letter English guess as the final non-empty line."
    )
    if not history:
        body = "Turn 1 of 6. There are no prior guesses. Choose a strong opening guess."
    else:
        lines = [f"{number}. {guess} -> {feedback}" for number, (guess, feedback) in enumerate(history, 1)]
        fixed: dict[int, str] = {}
        required_counts: dict[str, int] = {}
        forbidden_positions: dict[str, set[int]] = {}
        maximum_counts: dict[str, int] = {}
        for guess, feedback in history:
            positive = Counter(letter for letter, code in zip(guess, feedback) if code in "GY")
            for letter, count in positive.items():
                required_counts[letter] = max(required_counts.get(letter, 0), count)
            for position, (letter, code) in enumerate(zip(guess, feedback), 1):
                if code == "G":
                    fixed[position] = letter
                elif code in "YB":
                    forbidden_positions.setdefault(letter, set()).add(position)
            for letter in set(guess):
                if any(code == "B" for tile, code in zip(guess, feedback) if tile == letter):
                    maximum_counts[letter] = min(maximum_counts.get(letter, 5), positive[letter])
        fixed_text = ", ".join(f"{position}={letter}" for position, letter in sorted(fixed.items())) or "none"
        required_text = ", ".join(
            f"{letter}x{count}" +
            (f" not@{','.join(map(str, sorted(forbidden_positions.get(letter, set()))))}" if forbidden_positions.get(letter) else "")
            for letter, count in sorted(required_counts.items())
        ) or "none"
        absent_text = ", ".join(sorted(letter for letter, count in maximum_counts.items() if count == 0)) or "none"
        capped_text = ", ".join(
            f"{letter}<={count}" for letter, count in sorted(maximum_counts.items()) if count > 0
        ) or "none"
        banned = ", ".join(guess for guess, _ in history)
        body = (
            f"Turn {len(history) + 1} of 6.\nHistory:\n" + "\n".join(lines) +
            f"\nDerived constraints: fixed [{fixed_text}]; required [{required_text}]; absent [{absent_text}]; "
            f"duplicate caps [{capped_text}].\nForbidden repeats: {banned}. Choose a different consistent word."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": body}]


def _strict_explicit_feedback_messages(history: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    """Inference-shaped constraints with an unambiguous five-letter envelope."""
    messages = _explicit_feedback_messages(history)
    messages[0] = {
        "role": "system",
        "content": (
            "Play Wordle. G means fixed in that position, Y means present in another position, and B means absent "
            "after duplicate-letter accounting. Use every feedback row and never repeat a prior guess. Reply with "
            "exactly one line in the form `Final answer: WORD`, where WORD is exactly five uppercase ASCII letters "
            "and a valid English word. Do not add reasoning or any other text."
        ),
    }
    return messages


def _repeat_recovery_messages(history: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    messages = inference_messages(history)
    if history:
        repeated = history[-1][0]
        messages.extend(
            [
                {"role": "assistant", "content": f"Final answer: {repeated}"},
                {
                    "role": "user",
                    "content": (
                        f"That repeats the already-used word {repeated}, so it was ignored. Re-read every feedback "
                        "row and choose a different, consistent Wordle guess in the required final-answer format."
                    ),
                },
            ]
        )
    return messages


PROMPT_VARIANTS: dict[str, PromptBuilder] = {
    "legacy": inference_messages,
    "native_episode": _episode_messages,
    "explicit_feedback": _explicit_feedback_messages,
    "strict_explicit_feedback": _strict_explicit_feedback_messages,
    "repeat_recovery": _repeat_recovery_messages,
}

DECODING_VARIANTS: dict[str, dict] = {
    "greedy": {"do_sample": False, "max_new_tokens": 128, "use_cache": True},
    "greedy_rep105": {
        "do_sample": False,
        "max_new_tokens": 128,
        "use_cache": True,
        "repetition_penalty": 1.05,
    },
    "greedy_rep102": {
        "do_sample": False,
        "max_new_tokens": 128,
        "use_cache": True,
        "repetition_penalty": 1.02,
    },
    "greedy_rep110": {
        "do_sample": False,
        "max_new_tokens": 128,
        "use_cache": True,
        "repetition_penalty": 1.10,
    },
    "sample_t03": {
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.9,
        "max_new_tokens": 128,
        "use_cache": True,
    },
    "sample_t07": {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 128,
        "use_cache": True,
    },
}


def run_sweep(
    run_id: str,
    games: int = 5,
    prompt_variants: Sequence[str] = tuple(PROMPT_VARIANTS),
    decoding_variants: Sequence[str] = tuple(DECODING_VARIANTS),
    seed: int = 1337,
) -> dict:
    checkpoint = ARTIFACTS / "runs" / run_id / "checkpoints" / "final"
    if not checkpoint.exists():
        raise FileNotFoundError(f"adapter checkpoint not found: {checkpoint}")
    unknown_prompts = set(prompt_variants) - PROMPT_VARIANTS.keys()
    unknown_decoders = set(decoding_variants) - DECODING_VARIANTS.keys()
    if unknown_prompts or unknown_decoders:
        raise ValueError(f"unknown variants: prompts={sorted(unknown_prompts)}, decoders={sorted(unknown_decoders)}")

    answers = read_json(DATA / "splits" / "dev_answers.json")[:games]
    allowed = [
        line.strip().upper()
        for line in (DATA / "wordlists" / "allowed_words.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    answer_vocabulary = read_json(DATA / "splits" / "all_answer_words.json")
    tokenizer = load_tokenizer(checkpoint)
    model = load_adapter(checkpoint)
    original_messages = generation.inference_messages
    original_config = dict(generation.GENERATION_CONFIG)
    summaries = []
    spec = {
        "run_id": run_id,
        "games": games,
        "prompt_variants": list(prompt_variants),
        "decoding_variants": list(decoding_variants),
        "seed": seed,
    }
    sweep_id = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()[:10]
    output_dir = ARTIFACTS / "intervention_sweeps" / f"{run_id}-{sweep_id}"
    try:
        for prompt_name in prompt_variants:
            for decoding_name in decoding_variants:
                set_seed(seed)
                generation.inference_messages = PROMPT_VARIANTS[prompt_name]
                generation.GENERATION_CONFIG.clear()
                generation.GENERATION_CONFIG.update(DECODING_VARIANTS[decoding_name])
                records, summary = evaluate(model, tokenizer, answers, allowed, answer_vocabulary)
                turns = [turn for game in records for turn in game["turns"]]
                valid_guesses = [turn["guess"] for turn in turns if turn["valid"]]
                summary.update(
                    {
                        "prompt_variant": prompt_name,
                        "decoding_variant": decoding_name,
                        "unique_guesses": len(set(valid_guesses)),
                        "unique_noninitial_guesses": len(
                            {turn["guess"] for game in records for turn in game["turns"][1:] if turn["valid"]}
                        ),
                    }
                )
                summaries.append(summary)
                write_jsonl(output_dir / f"games-{prompt_name}-{decoding_name}.jsonl", records)
    finally:
        generation.inference_messages = original_messages
        generation.GENERATION_CONFIG.clear()
        generation.GENERATION_CONFIG.update(original_config)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    result = {"sweep_id": sweep_id, "spec": spec, "conditions": summaries}
    write_json(output_dir / "summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Development-only prompt and decoding intervention sweep")
    parser.add_argument("run_id")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--prompts", nargs="+", choices=tuple(PROMPT_VARIANTS), default=list(PROMPT_VARIANTS))
    parser.add_argument("--decoders", nargs="+", choices=tuple(DECODING_VARIANTS), default=list(DECODING_VARIANTS))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    print(json.dumps(run_sweep(args.run_id, args.games, args.prompts, args.decoders, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
