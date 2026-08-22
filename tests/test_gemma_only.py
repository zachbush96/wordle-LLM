import json
from pathlib import Path

import pytest

from wordle_lab.models import SUPPORTED_MODEL_ID, SUPPORTED_REVISION, assert_supported_model


def _model_dir(tmp_path: Path, *, model_id=SUPPORTED_MODEL_ID, revision=SUPPORTED_REVISION, model_type="gemma3_text"):
    (tmp_path / "wordle_lab_model.json").write_text(
        json.dumps({"model_id": model_id, "revision": revision}), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_type": model_type,
            "architectures": ["Gemma3ForCausalLM"],
            "hidden_size": 640,
            "num_hidden_layers": 18,
        }),
        encoding="utf-8",
    )
    return tmp_path


def test_exact_small_gemma_is_accepted(tmp_path):
    result = assert_supported_model(_model_dir(tmp_path))
    assert result["model_id"] == SUPPORTED_MODEL_ID
    assert result["gemma_270m_only"] is True


@pytest.mark.parametrize(
    "updates",
    [
        {"model_id": "Qwen/Qwen2.5-1.5B-Instruct"},
        {"model_id": "google/gemma-3-1b-it"},
        {"revision": "wrong"},
        {"model_type": "qwen2"},
    ],
)
def test_other_models_and_revisions_are_rejected(tmp_path, updates):
    with pytest.raises(RuntimeError, match="only google/gemma-3-270m-it"):
        assert_supported_model(_model_dir(tmp_path, **updates))
