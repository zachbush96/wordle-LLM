from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import MODEL_DIR


SUPPORTED_MODEL_ID = "google/gemma-3-270m-it"
SUPPORTED_MODEL_TYPE = "gemma3_text"
SUPPORTED_REVISION = "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"


def assert_supported_model(path: str | Path = MODEL_DIR) -> dict:
    """Fail closed unless ``path`` is the study's exact small Gemma model."""
    model_path = Path(path).resolve()
    metadata_path = model_path / "wordle_lab_model.json"
    config_path = model_path / "config.json"
    if not metadata_path.exists() or not config_path.exists():
        raise RuntimeError(f"Gemma 3 270M model metadata/config missing at {model_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checks = {
        "model_id": metadata.get("model_id") == SUPPORTED_MODEL_ID,
        "revision": metadata.get("revision") == SUPPORTED_REVISION,
        "model_type": config.get("model_type") == SUPPORTED_MODEL_TYPE,
        "architecture": config.get("architectures") == ["Gemma3ForCausalLM"],
        "hidden_size": config.get("hidden_size") == 640,
        "layers": config.get("num_hidden_layers") == 18,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            f"unsupported model at {model_path}; only {SUPPORTED_MODEL_ID}@{SUPPORTED_REVISION} is allowed; "
            f"failed checks: {failed}"
        )
    return {
        **metadata,
        "model_type": config["model_type"],
        "architecture": config["architectures"][0],
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "local_path": str(model_path),
        "gemma_270m_only": True,
    }


def model_metadata() -> dict:
    verified = assert_supported_model()
    metadata_path = MODEL_DIR / "wordle_lab_model.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
        metadata = {
            "model_id": config.get("_name_or_path", MODEL_DIR.name),
            "model_type": config.get("model_type"),
        }
    return {**metadata, **verified, "local_path": str(MODEL_DIR)}


def load_tokenizer(path: str | Path | None = None):
    assert_supported_model()
    tokenizer = AutoTokenizer.from_pretrained(path or MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(training: bool = False):
    assert_supported_model()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model experiments")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=dtype, attn_implementation="eager"
    ).to("cuda")
    model.config.use_cache = not training
    return model


def load_adapter(adapter_path: str | Path):
    from peft import PeftModel

    model = load_base_model(training=False)
    return PeftModel.from_pretrained(model, adapter_path).to("cuda")
