import math

import pytest
import torch

from wordle_lab.methods.q_sft import (
    bellman_likelihood_target,
    q_sft_soft_cross_entropy,
    validate_q_sft_rows,
)


def _row(**updates):
    row = {
        "prompt": [{"role": "user", "content": "Choose a legal Wordle guess."}],
        "completion": [{"role": "assistant", "content": "Final answer: CRANE"}],
        "bellman_target": 0.75,
        "example_id": "train-only-1",
    }
    row.update(updates)
    return row


def test_bellman_likelihood_target_uses_behavior_corrected_maximum():
    target = bellman_likelihood_target(
        0.1,
        0.5,
        [0.2, 0.3],
        [0.5, 0.25],
        terminal=False,
    )
    assert float(target) == pytest.approx(0.7)
    assert float(bellman_likelihood_target(0.8, 0.99, terminal=True)) == pytest.approx(0.8)


def test_bellman_likelihood_target_rejects_zero_behavior_support():
    with pytest.raises(ValueError, match="behavior"):
        bellman_likelihood_target(0.0, 0.9, [0.2], [0.0], terminal=False)


def test_q_sft_loss_matches_uniform_soft_label_cross_entropy():
    # Position zero predicts label 1 at position one. Target distribution is
    # [0.125, 0.75, 0.125] for a Bellman target of 0.75.
    logits = torch.tensor([[[0.0, 2.0, -1.0], [9.0, 9.0, 9.0]]])
    labels = torch.tensor([[-100, 1]])
    target = torch.tensor([0.75])
    log_probs = torch.log_softmax(logits[0, 0], dim=-1)
    expected = -(0.125 * log_probs[0] + 0.75 * log_probs[1] + 0.125 * log_probs[2])
    actual = q_sft_soft_cross_entropy(logits, labels, target)
    assert float(actual) == pytest.approx(float(expected))


def test_q_sft_loss_has_finite_gradient():
    logits = torch.randn(2, 4, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, 2], [-100, 1, 4, 5]])
    loss = q_sft_soft_cross_entropy(logits, labels, torch.tensor([0.2, 0.9]))
    loss.backward()
    assert math.isfinite(float(loss.detach()))
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_row_validation_computes_targets_without_evaluator_data():
    row = _row(
        bellman_target=None,
        reward=0.2,
        terminal=False,
        next_q_probabilities=[0.1, 0.2],
        next_behavior_probabilities=[0.5, 0.5],
    )
    row.pop("bellman_target")
    assert validate_q_sft_rows([row], discount=0.5) == pytest.approx([0.4])


@pytest.mark.parametrize("field", ["answer", "candidate_words", "posterior_candidates", "secret"])
def test_row_validation_rejects_candidate_or_secret_injection(field):
    with pytest.raises(ValueError, match="forbidden"):
        validate_q_sft_rows([_row(**{field: "CRANE"})], discount=0.99)


def test_row_validation_rejects_out_of_range_target_instead_of_clipping():
    with pytest.raises(ValueError, match="normalize"):
        validate_q_sft_rows([_row(bellman_target=1.01)], discount=0.99)
