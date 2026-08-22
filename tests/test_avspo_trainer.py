import pytest

from wordle_lab.methods.avspo_trainer import avspo_group_advantages


def test_avspo_group_advantages_preserve_only_real_sample_count():
    values, audit = avspo_group_advantages(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        {
            "enabled": True,
            "collapse_std_threshold": 1e-6,
            "alpha": 0.5,
            "zero_reward_anchor": 0.1,
            "adaptive_threshold_initial": 0.4,
            "adaptive_eta": 0.01,
            "normalization_epsilon": 1e-6,
        },
        adaptive_threshold=0.4,
    )
    assert len(values) == 8
    assert audit["real_sample_count"] == 8
    assert audit["virtual_sample_count"] > 0
    assert values[:4] == pytest.approx([values[0]] * 4)
    assert values[0] < 0
