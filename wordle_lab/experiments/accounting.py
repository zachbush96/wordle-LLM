from __future__ import annotations

import importlib.metadata
import platform

import torch


def environment() -> dict:
    packages = {}
    for name in ("torch", "transformers", "peft", "trl", "datasets", "numpy", "pandas"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
