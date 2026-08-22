import json
from pathlib import Path

import torch

from wordle_lab.analysis.state_diagnostics import build_probe_items, score_probe_outputs
from wordle_lab.common import write_json, write_jsonl
from wordle_lab.data.canonical import generate_canonical_states
from wordle_lab.experiments.common_curriculum import _balanced_select, _targeted_select, _targeted_state_type
from wordle_lab.experiments.common_preference import build_mixed_preferences
from wordle_lab.experiments.intervention_sweep import _explicit_feedback_messages, _strict_explicit_feedback_messages
from wordle_lab.experiments.on_policy_recovery import collect_recovery_rows, mix_static_and_recovery
from wordle_lab.methods.sft import CompletionDataset, weighted_causal_lm_loss
from wordle_lab.protocol.env import score_wordle


class TinyTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def test_word_focused_dataset_weights_only_target_word():
    row = {
        "example_id": "one",
        "target_word": "SHARE",
        "prompt": [{"role": "user", "content": "Choose"}],
        "completion": [{"role": "assistant", "content": "Final answer: SHARE"}],
    }
    sample = CompletionDataset([row], TinyTokenizer(), 500, word_token_weight=8)[0]
    assert sample["loss_weights"].count(8.0) == 5
    assert all(weight == 0 for label, weight in zip(sample["labels"], sample["loss_weights"]) if label == -100)


def test_weighted_causal_loss_emphasizes_selected_token():
    logits = torch.zeros((1, 3, 3))
    labels = torch.tensor([[-100, 1, 2]])
    weights = torch.tensor([[0.0, 1.0, 8.0]])
    loss = weighted_causal_lm_loss(logits, labels, weights)
    assert torch.isclose(loss, torch.tensor(3.0).log())


def test_balanced_selector_enforces_state_and_target_caps():
    def record(state, target):
        return {"state_id": state, "facts": {"oracle_action": target}}

    pools = {
        "root": [record("root", "SHARE")],
        "turn_2": [record(f"t2-{i}", word) for i, word in enumerate(("CRANE", "SLATE", "STORE", "CRANE"))],
        "later_on_policy": [record(f"later-{i}", word) for i, word in enumerate(("SLATE", "STORE", "CRANE"))],
        "recovery_singleton": [record(f"recovery-{i}", word) for i, word in enumerate(("STORE", "CRANE", "SLATE"))],
    }
    selected = _balanced_select(pools, total=10, seed=1, state_cap=2, target_cap=3)
    state_counts, target_counts = {}, {}
    for row, _ in selected:
        state_counts[row["state_id"]] = state_counts.get(row["state_id"], 0) + 1
        target = row["facts"]["oracle_action"]
        target_counts[target] = target_counts.get(target, 0) + 1
    assert max(state_counts.values()) <= 2
    assert max(target_counts.values()) <= 3


def test_targeted_buckets_use_actual_posterior_size():
    assert _targeted_state_type([], 128) == "format_root"
    assert _targeted_state_type([("SHARE", "BBBBB")], 1) == "true_singleton"
    assert _targeted_state_type([("SHARE", "BBBBB")], 3) == "turn_2"
    assert _targeted_state_type([("SHARE", "BBBBB"), ("POINT", "BBBBB")], 2) == "low_posterior"
    assert _targeted_state_type([("SHARE", "BBBBB"), ("POINT", "BBBBB")], 8) == "later_broad"


def test_targeted_selector_fills_exact_mix_and_caps_non_root_targets():
    def records(kind, count):
        return [
            {"state_id": f"{kind}-{i}", "facts": {"oracle_action": f"W{i % 20:04d}"}}
            for i in range(count)
        ]

    pools = {kind: records(kind, 80) for kind in (
        "format_root", "turn_2", "low_posterior", "true_singleton", "later_broad"
    )}
    selected = _targeted_select(pools, total=100, seed=2, target_cap=10)
    assert len(selected) == 100
    counts = {}
    for row, kind in selected:
        if kind != "format_root":
            target = row["facts"]["oracle_action"]
            counts[target] = counts.get(target, 0) + 1
    assert max(counts.values()) <= 10


def test_strict_prompt_requires_one_five_ascii_letter_line():
    messages = _strict_explicit_feedback_messages([("SHARE", "BBBBB")])
    assert "exactly one line" in messages[0]["content"]
    assert "exactly five uppercase ASCII letters" in messages[0]["content"]
    assert messages[-1]["content"] == _explicit_feedback_messages([("SHARE", "BBBBB")])[-1]["content"]


def test_state_diagnostics_separates_valid_words_from_policy_violations():
    universe = ["SHARE", "SHORE", "CRANE"]
    history = [("SHORE", score_wordle("SHARE", "SHORE"))]
    record = {
        "state_id": "dev-1",
        "secret_answer": "SHARE",
        "history": [{"guess": word, "feedback": feedback} for word, feedback in history],
        "facts": {"posterior_count": 1, "oracle_action": "SHARE"},
    }
    items = build_probe_items([record])
    rows, summary = score_probe_outputs(items, ["Final answer: CRANE"], universe, universe)
    assert rows[0]["valid_word"] is True
    assert rows[0]["posterior_consistent"] is False
    assert summary["singleton_answer_accuracy"] == 0
    _, solved = score_probe_outputs(items, ["Final answer: SHARE"], universe, universe)
    assert solved["singleton_answer_accuracy"] == 1


def test_dagger_keeps_actual_history_and_uses_identical_prompt_renderer():
    universe = ["SHARE", "SHORE", "CRANE"]

    def fake_generate(model, tokenizer, histories, batch_size=1):
        guess = "CRANE" if not histories[0] else "CRANE" if len(histories[0]) == 1 else "SHARE"
        return [{"raw_output": f"Final answer: {guess}"}]

    rows = collect_recovery_rows(
        None, None, ["SHARE"], universe, universe, "parent", 7, generate_fn=fake_generate, max_calls=4
    )
    assert any(row["error_type"] == "repeat" for row in rows)
    assert len(rows) == len({row["state_id"] for row in rows})
    for row in rows:
        history = [(item["guess"], item["feedback"]) for item in row["history"]]
        assert row["prompt"] == _explicit_feedback_messages(history)
        assert row["secret_split"] == "train"


def test_dagger_mix_is_exactly_half_recovery():
    static = [{"example_id": f"s{i}"} for i in range(3)]
    recovery = [{"example_id": f"r{i}"} for i in range(2)]
    mixed = mix_static_and_recovery(static, recovery, 10, 4)
    assert sum(row["dagger_mix_source"] == "static" for row in mixed) == 5
    assert sum(row["dagger_mix_source"] == "on_policy_recovery" for row in mixed) == 5


def test_mixed_preferences_have_declared_negative_mix_and_same_envelope(tmp_path: Path):
    universe = ["CRANE", "SHARE", "SHORE", "SCORE", "STORE", "SLATE"]
    canonical = generate_canonical_states(universe[:4], "train", 8, seed=3, answer_vocabulary=universe)
    write_jsonl(tmp_path / "canonical.jsonl", canonical)
    write_json(tmp_path / "universe.json", universe)
    source = next(row for row in canonical if row["history"])
    history = [(item["guess"], item["feedback"]) for item in source["history"]]
    posterior = {source["facts"]["oracle_action"]}
    bad = next(word for word in universe if word not in posterior and word != source["facts"]["oracle_action"])
    recovery = [{
        "state_id": source["state_id"],
        "error_type": "constraint_violation",
        "model_guess": bad,
        "target_word": source["facts"]["oracle_action"],
        "prompt": _explicit_feedback_messages(history),
    }]
    rows = build_mixed_preferences(tmp_path, recovery, total=8, seed=1)
    counts = {kind: sum(row["negative_type"] == kind for row in rows) for kind in {
        "model_constraint_violation", "prior_repeat", "strategically_inferior_consistent"
    }}
    assert counts == {"model_constraint_violation": 4, "prior_repeat": 2, "strategically_inferior_consistent": 2}
    assert all(row["chosen"][0]["content"].startswith("Final answer: ") for row in rows)
    assert all(row["rejected"][0]["content"].startswith("Final answer: ") for row in rows)
