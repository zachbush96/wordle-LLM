from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

from wordle_lab.data.builders import state_messages
from wordle_lab.methods.rewards import DEFAULT_WEIGHTS, shaped_reward
from wordle_lab.methods.reward_rubrics import (
    NOTEBOOKLM_REWARD_VERSION,
    RewardSignals,
    multigranularity_reward,
    wordle_constraint_violations,
)
from wordle_lab.methods.avspo_trainer import AVSPOGRPOTrainer
from wordle_lab.models import load_base_model, load_tokenizer
from wordle_lab.protocol.env import posterior_candidates, score_wordle
from wordle_lab.protocol.oracle import GreedyPartitionOracle
from wordle_lab.protocol.parsing import parse_terminal_answer


def train_grpo(canonical_rows: list[dict], parent_adapter: Path, run_dir: Path, spec: dict, allowed_words: list[str], train_answers: list[str]):
    """Online grouped rollout training; reward components remain observation-only at evaluation."""
    tokenizer = load_tokenizer(parent_adapter)
    model = PeftModel.from_pretrained(load_base_model(training=True), parent_adapter, is_trainable=True)
    oracle = GreedyPartitionOracle(train_answers)
    dataset = Dataset.from_list([{"prompt": state_messages(row), "history": row["history"], "secret_answer": row["secret_answer"]} for row in canonical_rows])

    def reward_func(completions, history, secret_answer, **_kwargs):
        values = []
        for completion, serialized_history, secret in zip(completions, history, secret_answer):
            raw = completion[0]["content"] if isinstance(completion, list) else str(completion)
            parsed = parse_terminal_answer(raw, allowed_words)
            hist = [(item["guess"], item["feedback"]) for item in serialized_history]
            before = posterior_candidates(hist, train_answers)
            guess = parsed["parsed_guess"]
            valid = parsed["status"] == "ok" and guess in allowed_words
            after = posterior_candidates(hist + ([(guess, score_wordle(secret, guess))] if valid else []), train_answers)
            ranked = oracle.ranked(oracle.remaining(hist))
            score_by_guess = {row["guess"]: row for row in ranked}
            regret = score_by_guess.get(guess, {"regret": max(row["regret"] for row in ranked) + 1})["regret"]
            rubric = spec.get("reward_rubric", {})
            if rubric.get("version") == NOTEBOOKLM_REWARD_VERSION:
                violations = wordle_constraint_violations(hist, guess) if valid else {
                    "green_violations": 0,
                    "missing_yellow_violations": 0,
                    "gray_reuse_violations": 0,
                }
                value = multigranularity_reward(
                    RewardSignals(
                        format_valid=parsed["format_valid"],
                        valid_word=valid,
                        solved=bool(valid and guess == secret),
                        repeated=bool(valid and guess in {old for old, _ in hist}),
                        **violations,
                    ),
                    weights=rubric.get("weights"),
                )
            else:
                value = shaped_reward(solved=guess == secret, information_gain=len(before) - len(after), oracle_regret=regret, repeated=guess in {old for old, _ in hist}, format_valid=parsed["format_valid"], weights=rubric.get("weights", spec.get("reward_weights", DEFAULT_WEIGHTS)))
            values.append(value["total"])
        return values

    args = GRPOConfig(output_dir=str(run_dir / "checkpoints"), max_steps=int(spec["max_steps"]), per_device_train_batch_size=int(spec["batch_size"]), gradient_accumulation_steps=int(spec["gradient_accumulation_steps"]), learning_rate=float(spec["learning_rate"]), num_generations=int(spec.get("group_size", 8)), max_completion_length=128, temperature=float(spec.get("temperature", 1.0)), bf16=torch.cuda.is_bf16_supported(), fp16=False, report_to="none", logging_steps=1, save_strategy="steps", save_steps=max(1, int(spec["max_steps"]) // 4), seed=int(spec["seed"]))
    virtual_support = spec.get("virtual_support", {})
    trainer_class = AVSPOGRPOTrainer if virtual_support.get("enabled") else GRPOTrainer
    trainer_kwargs = dict(model=model, reward_funcs=reward_func, args=args, train_dataset=dataset, processing_class=tokenizer)
    if trainer_class is AVSPOGRPOTrainer:
        trainer_kwargs["avspo_spec"] = virtual_support
    trainer = trainer_class(**trainer_kwargs)
    trainer.train(); final = run_dir / "checkpoints" / "final"; trainer.save_model(final); tokenizer.save_pretrained(final)
    return trainer.model
