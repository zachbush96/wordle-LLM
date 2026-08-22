# -*- coding: utf-8 -*-
"""Local Tiny LLM Wordle Lab.

Refactored from the original Colab notebook to run as a normal local CLI.

Original file is located at
    https://colab.research.google.com/drive/1n4J98vmqrG76QLHZ6No4hGDIciCda1pl

# Tiny LLM Wordle Lab

**A Drive-backed, reproducible first experiment for parameter-efficient Wordle specialization.**

This notebook establishes one defensible loop:

`Gemma 3 270M IT baseline → frozen 1,000-answer test → greedy partition teacher → LoRA SFT → same test → small fixed retention probe → saved artifacts`

It deliberately excludes GRPO, full fine-tuning, model sweeps, and broad public benchmarks. The initial claim is modest: whether a frozen-base LoRA adapter improves a strict Wordle-style five-letter task while a small fixed retention probe stays near the base model.

**Before running:**

1. In Colab, choose **Runtime → Change runtime type → T4 GPU** (or another CUDA GPU).
2. At [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it), accept the Gemma license with the same Hugging Face account that will supply the token.
3. In Colab's Secrets pane, add `HF_TOKEN`, enable notebook access, and never paste it into a code cell.

The local version keeps its model cache and experiment artifacts beside the script.

## 00 - Configuration
"""

# Run this first. It mounts Drive and defines the canonical experiment layout.
from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import importlib.metadata as importlib_metadata
import inspect
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.request import urlretrieve

SEED = 1337
MODEL_ID = "google/gemma-3-270m-it"
BASELINE_ID = "BASELINE-003"
DEFAULT_EXPERIMENT_ID = "SFT-001"
DATA_VERSION = "wordle-oracle-v1"
SPLIT_VERSION = "v1"

# All persistent state lives beside this script unless WORDLE_LAB_DIR overrides it.
# In particular, the base model is stored under models/base/ and reused on later runs.
PROJECT_DIR = Path(os.environ.get("WORDLE_LAB_DIR", Path(__file__).resolve().parent)).resolve()
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
PLOTS_DIR = PROJECT_DIR / "plots"
LOCAL_CACHE_DIR = PROJECT_DIR / ".cache"
BASE_MODEL_DIR = MODELS_DIR / "base" / MODEL_ID.replace("/", "--")

WORDLIST_URL = (
    "https://raw.githubusercontent.com/tabatkins/wordle-list/"
    "255b9469c4dad99a3b95cc4ddbe139b3d3747868/words"
)
WORDLIST_SHA256 = "9df5dad1b44cf1b9e0fa7c3ebff94d69ff7efb59f8691ac88e74e1c7b3da121e"
WORDLIST_LICENSE = "MIT (tabatkins/wordle-list; pinned commit 255b9469)"

# The experiment is a pinned Wordle-style environment, not a claim about the
# current NYT answer schedule. Keeping this smaller universe makes the exact
# greedy teacher feasible on a single T4 while preserving a 1,000-answer test.
ANSWER_UNIVERSE_SIZE = 2_500
TRAIN_ANSWER_COUNT = 1_250
DEV_ANSWER_COUNT = 250
TEST_ANSWER_COUNT = 1_000
assert TRAIN_ANSWER_COUNT + DEV_ANSWER_COUNT + TEST_ANSWER_COUNT == ANSWER_UNIVERSE_SIZE

TRAIN_EXAMPLES = 10_000
DEV_EXAMPLES = 1_000
MAX_WORDLE_GUESSES = 6
MAX_MODEL_CALLS_PER_GAME = 12  # protects evaluation from endless malformed outputs
EVAL_BATCH_SIZE = 32
GENERATION_MAX_NEW_TOKENS = 12
# Gemma 3 270M IT sometimes assigns the highest first-token logit to
# <end_of_turn>, especially for terse structured-output prompts.  Requiring one
# generated token prevents a vacuous answer without selecting a word for the
# model or constraining generation to the Wordle vocabulary.
GENERATION_MIN_NEW_TOKENS = 1
RETENTION_BATCH_SIZE = 32
RETENTION_MAX_NEW_TOKENS = 16

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}
TRAINING_CONFIG = {
    "num_train_epochs": 2,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "max_length": 256,
    "fp16": True,
    "bf16": False,
    "optim": "adamw_torch",
}

# Never silently overwrite an experiment. Use a new ID for an ablation, or set
# this to True only after intentionally inspecting the old artifacts.
FORCE_OVERWRITE = False

for directory in [
    DATA_DIR,
    DATA_DIR / "wordlists",
    DATA_DIR / "splits" / SPLIT_VERSION,
    DATA_DIR / "oracle" / DATA_VERSION,
    DATA_DIR / "manifests",
    MODELS_DIR,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    EXPERIMENTS_DIR,
    PLOTS_DIR,
    LOCAL_CACHE_DIR,
    BASE_MODEL_DIR.parent,
]:
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(LOCAL_CACHE_DIR / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(payload: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    output = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError:
                # A single interrupted final append is recoverable. Any earlier
                # malformed line is a data-integrity error.
                if line_number != len(lines):
                    raise
    return output


def guarded_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        if not FORCE_OVERWRITE:
            raise FileExistsError(
                f"{path} already contains artifacts. Preserve it, choose a new experiment ID, "
                "or explicitly set FORCE_OVERWRITE=True."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def package_versions(names: Sequence[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


set_global_seed()
print(f"Local project directory: {PROJECT_DIR}")
print(f"Seed: {SEED} | model: {MODEL_ID} | test answers: {TEST_ANSWER_COUNT}")

"""## 01 - Environment Setup"""

# Install dependencies once with: python -m pip install -r requirements.txt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from huggingface_hub import HfApi, get_token, snapshot_download
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU is available to PyTorch. Install a CUDA-enabled PyTorch build "
        "and verify that your NVIDIA driver is working."
    )

DEVICE = torch.device("cuda")
# The checkpoint is natively BF16. On GPUs without BF16 support (for example a
# Colab T4), FP16 inference can produce non-finite logits with some Torch /
# Transformers combinations. This model is small enough for FP32 inference on
# those devices; Trainer mixed precision remains configured separately below.
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
GPU_NAME = torch.cuda.get_device_name(0)
print(f"GPU preflight passed: {GPU_NAME}; inference dtype={DTYPE}.")
TRAINING_CONFIG["bf16"] = DTYPE == torch.bfloat16
TRAINING_CONFIG["fp16"] = DTYPE != torch.bfloat16

HF_TOKEN = os.environ.get("HF_TOKEN") or get_token()


def ensure_local_model() -> tuple[Path, str]:
    """Download the gated Gemma snapshot once, then always load the local copy."""
    config_path = BASE_MODEL_DIR / "config.json"
    metadata_path = BASE_MODEL_DIR / "wordle_lab_model.json"
    if config_path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        revision = metadata["revision"]
        print(f"Using local model: {BASE_MODEL_DIR}")
        return BASE_MODEL_DIR, revision
    if not HF_TOKEN:
        raise RuntimeError(
            "The Gemma model is not downloaded yet. Accept its Hugging Face license, then "
            "set HF_TOKEN or run `hf auth login` before retrying."
        )
    print(f"Downloading {MODEL_ID} once to {BASE_MODEL_DIR} ...")
    info = HfApi().model_info(MODEL_ID, token=HF_TOKEN)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=info.sha,
        local_dir=BASE_MODEL_DIR,
        token=HF_TOKEN,
    )
    atomic_json_dump(
        {"model_id": MODEL_ID, "revision": info.sha, "downloaded_at": utc_now()},
        metadata_path,
    )
    return BASE_MODEL_DIR, info.sha


MODEL_PATH, MODEL_REVISION = ensure_local_model()

ENVIRONMENT = {
    "captured_at": utc_now(),
    "python": sys.version,
    "platform": platform.platform(),
    "gpu": GPU_NAME,
    "cuda": torch.version.cuda,
    "torch": torch.__version__,
    "packages": package_versions(
        ["transformers", "trl", "peft", "datasets", "accelerate", "evaluate", "huggingface_hub", "pandas", "matplotlib"]
    ),
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
}
atomic_json_dump(ENVIRONMENT, EXPERIMENTS_DIR / "environment.json")
print(json.dumps(ENVIRONMENT, indent=2))


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # training default; generation temporarily uses left padding.
    return tokenizer


def load_base_model():
    '''Load a fresh, unmodified base model for a controlled comparison.'''
    set_global_seed()
    load_kwargs = {
        "local_files_only": True,
        "attn_implementation": "eager",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE, **load_kwargs)
    except TypeError:
        # Compatibility with older Transformers releases.
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=DTYPE, **load_kwargs)
    model = model.to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    model.eval()
    return model


tokenizer = load_tokenizer()
print(f"Tokenizer pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}, model revision={MODEL_REVISION}")

"""## 02 - Wordle Environment"""

FEEDBACK_ALPHABET = {"B", "Y", "G"}
ALL_GREEN = "G" * 5


def normalize_word(word: str) -> str:
    return word.strip().upper()


def is_five_ascii_letters(word: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]{5}", word.strip()))


def score_wordle(answer: str, guess: str) -> str:
    '''Duplicate-correct two-pass Wordle score: greens consume letters first.'''
    answer = normalize_word(answer)
    guess = normalize_word(guess)
    if not (is_five_ascii_letters(answer) and is_five_ascii_letters(guess)):
        raise ValueError("answer and guess must each be exactly five ASCII letters")

    feedback = ["B"] * 5
    unmatched_answer = Counter()
    for index, (answer_char, guess_char) in enumerate(zip(answer, guess)):
        if answer_char == guess_char:
            feedback[index] = "G"
        else:
            unmatched_answer[answer_char] += 1

    for index, guess_char in enumerate(guess):
        if feedback[index] == "G":
            continue
        if unmatched_answer[guess_char] > 0:
            feedback[index] = "Y"
            unmatched_answer[guess_char] -= 1
    return "".join(feedback)


class WordleEnv:
    '''Deterministic six-valid-guess Wordle environment.

    Invalid words are recorded but do not consume a Wordle turn. The evaluator
    separately caps model calls so malformed models cannot loop forever.
    '''

    def __init__(self, answer: str, allowed_words: Sequence[str], max_guesses: int = MAX_WORDLE_GUESSES):
        self.allowed_words = {normalize_word(word) for word in allowed_words}
        self.max_guesses = max_guesses
        self.reset(answer)

    def reset(self, answer: str | None = None) -> dict[str, Any]:
        if answer is None:
            answer = self.answer
        answer = normalize_word(answer)
        if not is_five_ascii_letters(answer):
            raise ValueError("answer must be five ASCII letters")
        if answer not in self.allowed_words:
            raise ValueError("answer must belong to allowed_words")
        self.answer = answer
        self.history: list[tuple[str, str]] = []
        self.guessed_words: set[str] = set()
        self.invalid_guesses = 0
        self.done = False
        self.won = False
        return self.state()

    def score_guess(self, guess: str) -> str:
        return score_wordle(self.answer, guess)

    def state(self) -> dict[str, Any]:
        return {
            "history": list(self.history),
            "turns_used": len(self.history),
            "guesses_left": self.max_guesses - len(self.history),
            "done": self.done,
            "won": self.won,
            "invalid_guesses": self.invalid_guesses,
        }

    def step(self, guess: str) -> dict[str, Any]:
        if self.done:
            raise RuntimeError("Game is finished; call reset before stepping again.")
        guess = normalize_word(guess)
        if not is_five_ascii_letters(guess) or guess not in self.allowed_words:
            self.invalid_guesses += 1
            return {
                "valid": False,
                "guess": guess,
                "feedback": None,
                "repeat": False,
                "won": False,
                "done": False,
                "reason": "invalid_word",
                **self.state(),
            }

        repeated = guess in self.guessed_words
        feedback = self.score_guess(guess)
        self.history.append((guess, feedback))
        self.guessed_words.add(guess)
        self.won = feedback == ALL_GREEN
        self.done = self.won or len(self.history) >= self.max_guesses
        return {
            "valid": True,
            "guess": guess,
            "feedback": feedback,
            "repeat": repeated,
            "won": self.won,
            "done": self.done,
            "reason": "win" if self.won else ("loss" if self.done else "continue"),
            **self.state(),
        }


def make_wordle_prompt(history: Sequence[tuple[str, str]], constraints: str | None = None) -> str:
    previous = "\n".join(f"{guess} -> {feedback}" for guess, feedback in history) or "(none)"
    prompt = "WORDLE\nPrevious guesses:\n" + previous
    if constraints:
        prompt += "\n\n" + constraints
    prompt += (
        "\n\nWhat valid five-letter English word would you intelligently choose as your "
        "next Wordle guess, consistent with all feedback? Never repeat a word shown "
        "under Previous guesses. Reply with only that word."
    )
    return prompt


def posterior_candidates(
    history: Sequence[tuple[str, str]], answer_vocabulary: Sequence[str]
) -> list[str]:
    candidates = [normalize_word(answer) for answer in answer_vocabulary]
    for guess, feedback in history:
        candidates = [answer for answer in candidates if score_wordle(answer, guess) == feedback]
    return candidates


def is_consistent_with_history(
    guess: str, history: Sequence[tuple[str, str]], answer_vocabulary: Sequence[str]
) -> bool:
    # In this strict experiment contract, a guess must itself remain a possible
    # secret answer. This is a precise duplicate-safe "constraint violation".
    return normalize_word(guess) in set(posterior_candidates(history, answer_vocabulary))


# Deterministic tests. These are deliberately run before any model work.
TEST_ALLOWED = ["CRANE", "REACT", "SLATE", "APPLE", "ALLEY", "BANAL", "LLAMA", "GEESE", "EERIE", "AABCD", "AAAEE"]
assert score_wordle("CRANE", "CRANE") == "GGGGG"       # exact match
assert score_wordle("CRANE", "REACT") == "YYGYB"       # yellow letters
assert score_wordle("CRANE", "SLATE") == "BBGBG"       # absent letters
assert score_wordle("APPLE", "ALLEY") == "GYBYB"       # duplicate handling
assert score_wordle("BANAL", "LLAMA") == "YBYBY"       # duplicate handling
assert score_wordle("GEESE", "EERIE") == "YGBBG"       # duplicate handling
assert score_wordle("AABCD", "AAAEE") == "GGBBB"       # greens consume duplicates first

env_test = WordleEnv("CRANE", TEST_ALLOWED)
invalid = env_test.step("NOPE")
assert not invalid["valid"] and invalid["turns_used"] == 0
assert env_test.step("CRANE")["won"]

loss_env = WordleEnv("CRANE", TEST_ALLOWED)
for candidate in ["REACT", "SLATE", "APPLE", "ALLEY", "BANAL", "LLAMA"]:
    result = loss_env.step(candidate)
assert result["done"] and not result["won"] and result["turns_used"] == 6
print("Wordle environment tests passed.")

"""## 03 - Oracle Solver"""

FEEDBACK_TO_INT = {"B": 0, "Y": 1, "G": 2}
INT_TO_FEEDBACK = {value: key for key, value in FEEDBACK_TO_INT.items()}


def encode_feedback(feedback: str) -> int:
    code = 0
    for char in feedback:
        code = code * 3 + FEEDBACK_TO_INT[char]
    return code


def decode_feedback(code: int) -> str:
    output = ["B"] * 5
    for index in range(4, -1, -1):
        output[index] = INT_TO_FEEDBACK[code % 3]
        code //= 3
    return "".join(output)


class GreedyPartitionOracle:
    '''Exact greedy partition teacher over a declared candidate-only action space.

    It is not described as globally optimal. At each state it scores every
    remaining candidate word as a next guess and minimizes expected remaining
    candidates, sum_r |partition_r|^2 / N. Ties break alphabetically.
    '''

    def __init__(self, answer_vocabulary: Sequence[str]):
        self.answers = sorted({normalize_word(word) for word in answer_vocabulary})
        if not self.answers:
            raise ValueError("oracle requires at least one answer")
        self.index = {word: index for index, word in enumerate(self.answers)}
        self.matrix = np.empty((len(self.answers), len(self.answers)), dtype=np.uint8)
        for guess_index, guess in enumerate(self.answers):
            for answer_index, answer in enumerate(self.answers):
                self.matrix[guess_index, answer_index] = encode_feedback(score_wordle(answer, guess))
        self._best_cache: dict[tuple[int, ...], dict[str, Any]] = {}

    def remaining_indices(self, history: Sequence[tuple[str, str]]) -> np.ndarray:
        remaining = np.arange(len(self.answers), dtype=np.int32)
        for guess, feedback in history:
            guess_index = self.index.get(normalize_word(guess))
            if guess_index is None:
                # Training trajectories use candidate guesses, but this fallback
                # keeps the oracle correct if an external state is supplied.
                remaining = np.array(
                    [
                        index
                        for index in remaining
                        if score_wordle(self.answers[index], guess) == feedback
                    ],
                    dtype=np.int32,
                )
            else:
                remaining = remaining[self.matrix[guess_index, remaining] == encode_feedback(feedback)]
        return remaining

    def best_guess(self, remaining: np.ndarray) -> dict[str, Any]:
        if len(remaining) == 0:
            raise ValueError("No remaining candidates: history is inconsistent with this oracle vocabulary.")
        key = tuple(int(value) for value in remaining)
        if key in self._best_cache:
            return self._best_cache[key]

        objectives: list[tuple[float, float, int]] = []
        denominator = float(len(remaining))
        for guess_index in remaining:
            feedback_codes = self.matrix[guess_index, remaining]
            counts = np.bincount(feedback_codes, minlength=3 ** 5)
            nonzero = counts[counts > 0]
            expected_remaining = float(np.square(nonzero).sum() / denominator)
            probabilities = nonzero / denominator
            entropy = float(-(probabilities * np.log2(probabilities)).sum())
            objectives.append((expected_remaining, -entropy, int(guess_index)))

        # answers is alphabetically ordered, so the final index is a stable
        # alphabetical tie breaker after objective and entropy.
        objectives.sort()
        best_expected, negative_entropy, best_index = objectives[0]
        tie_count = sum(
            math.isclose(expected, best_expected, rel_tol=0.0, abs_tol=1e-12)
            for expected, _, _ in objectives
        )
        result = {
            "guess": self.answers[best_index],
            "expected_remaining": best_expected,
            "entropy": -negative_entropy,
            "remaining_candidates": len(remaining),
            "action_space": "all remaining candidate answers",
            "tie_count": tie_count,
            "oracle_rank": 1,
        }
        self._best_cache[key] = result
        return result


_ORACLES: dict[tuple[str, ...], GreedyPartitionOracle] = {}


def get_oracle(answer_vocabulary: Sequence[str]) -> GreedyPartitionOracle:
    key = tuple(sorted({normalize_word(word) for word in answer_vocabulary}))
    if key not in _ORACLES:
        print(f"Precomputing {len(key)}×{len(key)} duplicate-correct feedback matrix...")
        _ORACLES[key] = GreedyPartitionOracle(key)
    return _ORACLES[key]


assert decode_feedback(encode_feedback("GYBYB")) == "GYBYB"
toy_oracle = GreedyPartitionOracle(["CRANE", "SLATE", "PLANT", "APPLE"])
toy_choice = toy_oracle.best_guess(toy_oracle.remaining_indices([]))
assert toy_choice["guess"] in toy_oracle.answers
print("Greedy partition oracle smoke test passed.")

"""## 04 - Model Adapter"""

WORD_TOKEN_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]{5}(?![A-Za-z])")


def parse_wordle_output(raw_output: str, allowed_words: Sequence[str]) -> dict[str, Any]:
    '''Recover the last valid standalone word while keeping formatting metrics honest.

    For example, ``I think the optimal answer is PLANT.`` yields `PLANT`, not
    `THINK`. `format_valid` remains false, so formatting and reasoning are
    separately measurable.
    '''
    allowed = {normalize_word(word) for word in allowed_words}
    raw_output = raw_output or ""
    exact = normalize_word(raw_output)
    if is_five_ascii_letters(raw_output) and exact in allowed:
        return {"parsed_guess": exact, "format_valid": True, "parse_status": "exact"}
    tokens = [normalize_word(token) for token in WORD_TOKEN_RE.findall(raw_output)]
    for token in reversed(tokens):
        if token in allowed:
            return {"parsed_guess": token, "format_valid": False, "parse_status": "recovered_last_allowed"}
    return {"parsed_guess": None, "format_valid": False, "parse_status": "no_allowed_word"}


def render_generation_prompts(user_prompts: Sequence[str]) -> list[str]:
    messages = [[{"role": "user", "content": prompt}] for prompt in user_prompts]
    return [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def generation_stop_token_ids() -> list[int]:
    '''Return every Gemma terminal token, including the chat end-of-turn token.'''
    token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    return list(dict.fromkeys(token_id for token_id in token_ids if token_id is not None))


def generate_raw_outputs(
    model,
    user_prompts: Sequence[str],
    max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
    batch_size: int = EVAL_BATCH_SIZE,
    decoder_config: dict[str, Any] | None = None,
) -> list[str]:
    '''Shared batched generation path for baseline, adapter, and probes.'''
    generation_kwargs: dict[str, Any] = {
        "do_sample": False,
        "min_new_tokens": GENERATION_MIN_NEW_TOKENS,
    }
    generation_kwargs.update(decoder_config or {})
    outputs: list[str] = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(user_prompts), batch_size):
            batch_prompts = list(user_prompts[start : start + batch_size])
            rendered = render_generation_prompts(batch_prompts)
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=TRAINING_CONFIG["max_length"],
                add_special_tokens=False,
            )
            inputs = {name: tensor.to(model_device(model)) for name, tensor in inputs.items()}
            with torch.inference_mode():
                if not getattr(model, "_wordle_logits_checked", False):
                    next_token_logits = model(**inputs).logits[:, -1, :]
                    if not torch.isfinite(next_token_logits).all():
                        raise RuntimeError(
                            "Model produced non-finite next-token logits. Do not interpret "
                            "the resulting empty output as model behavior; retry with BF16 "
                            "or FP32 inference."
                        )
                    model._wordle_logits_checked = True
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=generation_stop_token_ids(),
                    use_cache=True,
                    **generation_kwargs,
                )
            prompt_width = inputs["input_ids"].shape[1]
            for row in generated:
                outputs.append(tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip())
    finally:
        tokenizer.padding_side = original_padding_side
    return outputs


assert parse_wordle_output("I think the optimal answer is PLANT.", ["THINK", "PLANT"])["parsed_guess"] == "PLANT"
assert parse_wordle_output("PLANT", ["PLANT"])["format_valid"]
assert tokenizer.convert_tokens_to_ids("<end_of_turn>") in generation_stop_token_ids()
print("Model I/O contract and parser tests passed.")

"""## 05 - Baseline Evaluation"""

WORDLIST_PATH = DATA_DIR / "wordlists" / "tabatkins_wordle_list_pinned.txt"
SPLIT_DIR = DATA_DIR / "splits" / SPLIT_VERSION
TRAIN_ANSWERS_PATH = DATA_DIR / "train_answers.json"
DEV_ANSWERS_PATH = DATA_DIR / "dev_answers.json"
TEST_ANSWERS_PATH = DATA_DIR / "test_answers.json"
SPLIT_MANIFEST_PATH = DATA_DIR / "manifests" / f"split_manifest_{SPLIT_VERSION}.json"


def ensure_wordlist_and_splits() -> tuple[list[str], list[str], list[str], list[str]]:
    if not WORDLIST_PATH.exists():
        print("Downloading pinned MIT-licensed Wordle list...")
        urlretrieve(WORDLIST_URL, WORDLIST_PATH)
    observed_hash = sha256_file(WORDLIST_PATH)
    if observed_hash != WORDLIST_SHA256:
        raise RuntimeError(
            f"Pinned wordlist hash mismatch: expected {WORDLIST_SHA256}, got {observed_hash}. "
            "Stop rather than silently changing the benchmark."
        )

    allowed_words = sorted(
        {
            normalize_word(line)
            for line in WORDLIST_PATH.read_text(encoding="utf-8").splitlines()
            if is_five_ascii_letters(line)
        }
    )
    if len(allowed_words) < ANSWER_UNIVERSE_SIZE:
        raise RuntimeError("The pinned word list is too small for the requested answer split.")

    expected_paths = [TRAIN_ANSWERS_PATH, DEV_ANSWERS_PATH, TEST_ANSWERS_PATH, SPLIT_MANIFEST_PATH]
    if all(path.exists() for path in expected_paths):
        train_answers = read_json(TRAIN_ANSWERS_PATH)
        dev_answers = read_json(DEV_ANSWERS_PATH)
        test_answers = read_json(TEST_ANSWERS_PATH)
    else:
        shuffled = list(allowed_words)
        random.Random(SEED).shuffle(shuffled)
        answer_universe = shuffled[:ANSWER_UNIVERSE_SIZE]
        train_answers = answer_universe[:TRAIN_ANSWER_COUNT]
        dev_answers = answer_universe[TRAIN_ANSWER_COUNT : TRAIN_ANSWER_COUNT + DEV_ANSWER_COUNT]
        test_answers = answer_universe[TRAIN_ANSWER_COUNT + DEV_ANSWER_COUNT :]
        atomic_json_dump(train_answers, TRAIN_ANSWERS_PATH)
        atomic_json_dump(dev_answers, DEV_ANSWERS_PATH)
        atomic_json_dump(test_answers, TEST_ANSWERS_PATH)

    train_set, dev_set, test_set = map(set, [train_answers, dev_answers, test_answers])
    assert not (train_set & dev_set or train_set & test_set or dev_set & test_set), "Split leakage detected"
    assert len(test_answers) == TEST_ANSWER_COUNT
    assert set(train_answers + dev_answers + test_answers).issubset(set(allowed_words))

    manifest = {
        "created_or_verified_at": utc_now(),
        "split_version": SPLIT_VERSION,
        "seed": SEED,
        "wordlist": {
            "url": WORDLIST_URL,
            "sha256": observed_hash,
            "license": WORDLIST_LICENSE,
            "allowed_word_count": len(allowed_words),
        },
        "answer_universe_size": ANSWER_UNIVERSE_SIZE,
        "counts": {"train": len(train_answers), "dev": len(dev_answers), "test": len(test_answers)},
        "hashes": {
            "train_answers": sha256_text(json.dumps(train_answers)),
            "dev_answers": sha256_text(json.dumps(dev_answers)),
            "test_answers": sha256_text(json.dumps(test_answers)),
        },
        "leakage_assertions": {
            "train_dev_disjoint": not bool(train_set & dev_set),
            "train_test_disjoint": not bool(train_set & test_set),
            "dev_test_disjoint": not bool(dev_set & test_set),
        },
    }
    atomic_json_dump(manifest, SPLIT_MANIFEST_PATH)
    return allowed_words, train_answers, dev_answers, test_answers


ALLOWED_WORDS, TRAIN_ANSWERS, DEV_ANSWERS, TEST_ANSWERS = ensure_wordlist_and_splits()
ALL_ANSWER_WORDS = TRAIN_ANSWERS + DEV_ANSWERS + TEST_ANSWERS
print(f"Frozen splits: train={len(TRAIN_ANSWERS)}, dev={len(DEV_ANSWERS)}, test={len(TEST_ANSWERS)}")
print(f"Saved held-out answers: {TEST_ANSWERS_PATH}")


def summarize_games(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No game records to summarize")
    wins = [record for record in records if record["won"]]
    valid_guesses = sum(record["guesses"] for record in records)
    model_calls = sum(record["model_calls"] for record in records)
    summary = {
        "n_games": len(records),
        "wins": len(wins),
        "win_rate": len(wins) / len(records),
        "mean_guesses_on_wins": (float(np.mean([record["guesses"] for record in wins])) if wins else None),
        "invalid_guess_rate": sum(record["invalid_guesses"] for record in records) / max(model_calls, 1),
        "repeat_guess_rate": sum(record["repeated_guesses"] for record in records) / max(valid_guesses, 1),
        "constraint_violation_rate": sum(record["constraint_violations"] for record in records) / max(valid_guesses, 1),
        "format_failure_rate": sum(record["format_failures"] for record in records) / max(model_calls, 1),
        "exhausted_output_budget_rate": sum(record["exhausted_output_budget"] for record in records) / len(records),
        "wins_by_guess": {
            str(turn): sum(record["won"] and record["guesses"] == turn for record in records)
            for turn in range(1, MAX_WORDLE_GUESSES + 1)
        },
        "metric_definitions": {
            "invalid_guess_rate": "invalid parsed guesses divided by model calls",
            "repeat_guess_rate": "valid repeated guesses divided by valid guesses",
            "constraint_violation_rate": "valid guesses outside the duplicate-correct posterior divided by valid guesses",
            "format_failure_rate": "responses not exactly one valid five-letter word divided by model calls",
        },
    }
    return summary


def evaluate_wordle(
    model,
    experiment_id: str,
    mode: str,
    answers: Sequence[str] = TEST_ANSWERS,
    resume: bool = True,
    decoder_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    '''Run the frozen evaluator, checkpointing raw game records in Drive.'''
    normalized_answers = [normalize_word(answer) for answer in answers]
    if len(normalized_answers) != len(set(normalized_answers)):
        raise ValueError("Evaluation answers must be unique; duplicate hidden answers detected.")
    result_dir = RESULTS_DIR / experiment_id
    result_dir.mkdir(parents=True, exist_ok=True)
    games_path = result_dir / "games.jsonl"
    summary_path = result_dir / "summary.json"
    expected_answer_hash = sha256_text(json.dumps(list(answers)))
    if summary_path.exists():
        completed_summary = read_json(summary_path)
        if completed_summary.get("test_answers_sha256") != expected_answer_hash:
            raise RuntimeError("Existing summary uses a different held-out-answer protocol.")
    existing = read_jsonl(games_path) if resume else []
    if existing and any(record.get("mode") != mode for record in existing):
        raise RuntimeError("Existing game records use a different evaluation mode.")
    completed = {record["game_id"] for record in existing}
    records = list(existing)
    pending = [(game_id, answer) for game_id, answer in enumerate(answers) if game_id not in completed]
    if existing:
        print(f"Resuming {experiment_id}: {len(existing)}/{len(answers)} games already saved.")

    # Active games advance in lockstep; each decoder call sees only its own
    # feedback history. Batching does not change the model contract.
    for outer_start in range(0, len(pending), EVAL_BATCH_SIZE):
        batch = pending[outer_start : outer_start + EVAL_BATCH_SIZE]
        states = []
        for game_id, answer in batch:
            states.append(
                {
                    "game_id": game_id,
                    "answer": normalize_word(answer),
                    "env": WordleEnv(answer, ALLOWED_WORDS),
                    "turns": [],
                    "model_calls": 0,
                    "repeated_guesses": 0,
                    "constraint_violations": 0,
                    "format_failures": 0,
                    "exhausted_output_budget": False,
                }
            )

        while any(not state["env"].done for state in states):
            active = [
                state
                for state in states
                if not state["env"].done and state["model_calls"] < MAX_MODEL_CALLS_PER_GAME
            ]
            if not active:
                for state in states:
                    if not state["env"].done:
                        state["exhausted_output_budget"] = True
                break
            prompts = [make_wordle_prompt(state["env"].history) for state in active]
            for state, prompt in zip(active, prompts):
                history = list(state["env"].history)
                expected_history = "\n".join(
                    f"{guess} -> {feedback}" for guess, feedback in history
                ) or "(none)"
                if f"Previous guesses:\n{expected_history}" not in prompt:
                    raise RuntimeError("Generated prompt does not contain the full Wordle history.")
            started = time.perf_counter()
            raws = generate_raw_outputs(
                model,
                prompts,
                batch_size=EVAL_BATCH_SIZE,
                decoder_config=decoder_config,
            )
            elapsed = time.perf_counter() - started
            for state, prompt, raw in zip(active, prompts, raws):
                env = state["env"]
                parsed = parse_wordle_output(raw, ALLOWED_WORDS)
                state["model_calls"] += 1
                if not parsed["format_valid"]:
                    state["format_failures"] += 1
                guess = parsed["parsed_guess"]
                previous_history = list(env.history)
                constraint_violation = bool(
                    guess and not is_consistent_with_history(guess, previous_history, ALL_ANSWER_WORDS)
                )
                if constraint_violation:
                    state["constraint_violations"] += 1
                step = env.step(guess or "")
                if step["valid"] and step["repeat"]:
                    state["repeated_guesses"] += 1
                state["turns"].append(
                    {
                        "call": state["model_calls"],
                        "wordle_turn_before_call": len(previous_history) + 1,
                        "history_before_call": [
                            {"guess": old_guess, "feedback": old_feedback}
                            for old_guess, old_feedback in previous_history
                        ],
                        "prompt_text": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "raw_output": raw,
                        "parsed_guess": guess,
                        "parse_status": parsed["parse_status"],
                        "format_valid": parsed["format_valid"],
                        "valid_guess": step["valid"],
                        "feedback": step["feedback"],
                        "repeat": step["repeat"],
                        "constraint_violation": constraint_violation if step["valid"] else False,
                        "latency_s": elapsed / max(len(active), 1),
                    }
                )

        completed_batch = []
        for state in states:
            env = state["env"]
            completed_batch.append(
                {
                    "schema_version": "wordle-game-v2",
                    "experiment": experiment_id,
                    "mode": mode,
                    "model": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "seed": SEED,
                    "game_id": state["game_id"],
                    "answer": state["answer"],
                    "won": env.won,
                    "guesses": len(env.history),
                    "model_calls": state["model_calls"],
                    "invalid_guesses": env.invalid_guesses,
                    "repeated_guesses": state["repeated_guesses"],
                    "constraint_violations": state["constraint_violations"],
                    "format_failures": state["format_failures"],
                    "exhausted_output_budget": state["exhausted_output_budget"],
                    "turns": state["turns"],
                }
            )
        append_jsonl(completed_batch, games_path)
        records.extend(completed_batch)
        print(f"{experiment_id}: saved {len(records)}/{len(answers)} games")

    records = sorted(records, key=lambda record: record["game_id"])
    if len(records) != len(answers):
        raise RuntimeError("Evaluation did not produce exactly one record per frozen test answer.")
    summary = summarize_games(records)
    summary.update(
        {
            "experiment": experiment_id,
            "mode": mode,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "seed": SEED,
            "test_answers_sha256": expected_answer_hash,
            "answer_protocol": {
                "source": str(TEST_ANSWERS_PATH),
                "unique_hidden_answers": True,
                "answer_is_not_in_model_prompt": True,
            },
            "decoder": {
                **({"do_sample": False, "min_new_tokens": GENERATION_MIN_NEW_TOKENS} | (decoder_config or {})),
                "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
                "eos_token_ids": generation_stop_token_ids(),
                "batch_size": EVAL_BATCH_SIZE,
                "max_model_calls_per_game": MAX_MODEL_CALLS_PER_GAME,
            },
        }
    )
    atomic_json_dump(summary, result_dir / "summary.json")
    return records, summary


def run_baseline(smoke_only: bool = False, game_count: int | None = None) -> dict[str, Any]:
    if game_count is not None and not 1 <= game_count <= len(TEST_ANSWERS):
        raise ValueError(f"game_count must be between 1 and {len(TEST_ANSWERS)}")
    count = game_count if game_count is not None else (25 if smoke_only else len(TEST_ANSWERS))
    answers = TEST_ANSWERS[:count]
    experiment_id = BASELINE_ID if count == len(TEST_ANSWERS) else f"{BASELINE_ID}-SMOKE-{count}"
    baseline_dir = RESULTS_DIR / experiment_id
    summary_path = baseline_dir / "summary.json"
    if summary_path.exists() and read_json(summary_path).get("n_games") == len(answers):
        print(f"Using frozen {experiment_id}; no baseline artifacts were overwritten.")
        return read_json(summary_path)
    config = {
        "experiment": experiment_id,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "kind": "base_model_evaluation",
        "test_answers_sha256": sha256_text(json.dumps(answers)),
        "created_at": utc_now(),
    }
    atomic_json_dump(config, baseline_dir / "config.json")
    model = load_base_model()
    try:
        _, summary = evaluate_wordle(model, experiment_id=experiment_id, mode="base", answers=answers)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps(summary, indent=2))
    return summary


DECODING_SWEEP_ID = "DECODING-SWEEP-001"
DECODING_STRATEGIES: dict[str, dict[str, Any]] = {
    "greedy": {
        "do_sample": False,
        "min_new_tokens": GENERATION_MIN_NEW_TOKENS,
    },
    "beam4": {
        "do_sample": False,
        "min_new_tokens": GENERATION_MIN_NEW_TOKENS,
        "num_beams": 4,
        "early_stopping": True,
    },
    "greedy_repeat_penalty": {
        "do_sample": False,
        "min_new_tokens": GENERATION_MIN_NEW_TOKENS,
        "repetition_penalty": 1.15,
    },
    "sample_t07": {
        "do_sample": True,
        "min_new_tokens": GENERATION_MIN_NEW_TOKENS,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
    },
    "sample_t10": {
        "do_sample": True,
        "min_new_tokens": GENERATION_MIN_NEW_TOKENS,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    },
}


def run_decoding_sweep(game_count: int = 25) -> dict[str, Any]:
    '''Compare decoding strategies on the same ordered hidden-answer slice.'''
    if not 1 <= game_count <= len(TEST_ANSWERS):
        raise ValueError(f"game_count must be between 1 and {len(TEST_ANSWERS)}")
    answers = TEST_ANSWERS[:game_count]
    sweep_dir = RESULTS_DIR / f"{DECODING_SWEEP_ID}-{game_count}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    model = load_base_model()
    try:
        for strategy_name, strategy_config in DECODING_STRATEGIES.items():
            set_global_seed(SEED)
            experiment_id = f"{DECODING_SWEEP_ID}-{strategy_name.upper()}-{game_count}"
            _, summary = evaluate_wordle(
                model,
                experiment_id=experiment_id,
                mode=f"base_decode_{strategy_name}",
                answers=answers,
                decoder_config=strategy_config,
            )
            summaries[strategy_name] = summary
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    comparison_rows = []
    for strategy_name, summary in summaries.items():
        game_records = read_jsonl(RESULTS_DIR / summary["experiment"] / "games.jsonl")
        parsed_guesses = [
            turn["parsed_guess"]
            for game in game_records
            for turn in game["turns"]
            if turn["parsed_guess"]
        ]
        first_guesses = [
            game["turns"][0]["parsed_guess"]
            for game in game_records
            if game["turns"] and game["turns"][0]["parsed_guess"]
        ]
        context_audit_passed = all(
            (
                "Previous guesses:\n"
                + (
                    "\n".join(
                        f"{item['guess']} -> {item['feedback']}"
                        for item in turn["history_before_call"]
                    )
                    or "(none)"
                )
            )
            in turn["prompt_text"]
            for game in game_records
            for turn in game["turns"]
        )
        comparison_rows.append(
            {
                "strategy": strategy_name,
                "games": summary["n_games"],
                "wins": summary["wins"],
                "win_rate": summary["win_rate"],
                "unique_guesses": len(set(parsed_guesses)),
                "unique_first_guesses": len(set(first_guesses)),
                "invalid_guess_rate": summary["invalid_guess_rate"],
                "repeat_guess_rate": summary["repeat_guess_rate"],
                "constraint_violation_rate": summary["constraint_violation_rate"],
                "format_failure_rate": summary["format_failure_rate"],
                "exhausted_output_budget_rate": summary["exhausted_output_budget_rate"],
                "full_context_audit_passed": context_audit_passed,
                "result_dir": str(RESULTS_DIR / summary["experiment"]),
            }
        )
    comparison_path = sweep_dir / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    aggregate = {
        "sweep": DECODING_SWEEP_ID,
        "seed": SEED,
        "games_per_strategy": game_count,
        "test_answers_sha256": sha256_text(json.dumps(answers)),
        "strategies": DECODING_STRATEGIES,
        "results": comparison_rows,
        "prompt_and_history_logged_per_call": True,
    }
    atomic_json_dump(aggregate, sweep_dir / "summary.json")
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    return aggregate


# First validate the full pipeline cheaply. Then run the frozen 1,000-answer
# baseline and leave it untouched for all later comparisons.
# SMOKE_SUMMARY = run_baseline(smoke_only=True)
# Baselines are started explicitly by the CLI at the bottom of this file.

"""## 06 - Dataset Generation"""

ORACLE_DIR = DATA_DIR / "oracle" / DATA_VERSION
TRAIN_DIRECT_PATH = ORACLE_DIR / "train_direct.jsonl"
DEV_DIRECT_PATH = ORACLE_DIR / "dev_direct.jsonl"
TRAIN_CONSTRAINTS_PATH = ORACLE_DIR / "train_constraints.jsonl"
DEV_CONSTRAINTS_PATH = ORACLE_DIR / "dev_constraints.jsonl"
TRAIN_METADATA_PATH = ORACLE_DIR / "train_metadata.jsonl"
DEV_METADATA_PATH = ORACLE_DIR / "dev_metadata.jsonl"
DATASET_MANIFEST_PATH = ORACLE_DIR / "manifest.json"


def derive_explicit_constraints(history: Sequence[tuple[str, str]], answer_vocabulary: Sequence[str]) -> str:
    '''A deterministic, posterior-derived representation for SFT-002.

    It is deliberately appended to the *input* while the completion stays one
    word, preserving the same evaluator/output contract as SFT-001.
    '''
    candidates = posterior_candidates(history, answer_vocabulary)
    if not candidates:
        raise ValueError("Cannot derive constraints from an inconsistent history")
    position_sets = [sorted({candidate[index] for candidate in candidates}) for index in range(5)]
    fixed = [f"{index + 1}={letters[0]}" for index, letters in enumerate(position_sets) if len(letters) == 1]
    required = sorted(set.intersection(*(set(word) for word in candidates))) if candidates else []
    position_lines = [f"P{index + 1}:[{''.join(letters)}]" for index, letters in enumerate(position_sets)]
    return "\n".join(
        [
            "Computed candidate constraints:",
            f"Remaining possible answers: {len(candidates)}",
            "Fixed positions: " + (", ".join(fixed) if fixed else "none"),
            "Letters in every candidate: " + (", ".join(required) if required else "none"),
            "Allowed letters by position: " + " | ".join(position_lines),
        ]
    )


def deterministic_openers(answer_vocabulary: Sequence[str], count: int = 16) -> list[str]:
    pool = sorted({normalize_word(word) for word in answer_vocabulary})
    rng = random.Random(SEED + 17)
    return rng.sample(pool, k=min(count, len(pool)))


def generate_oracle_states(
    answer_vocabulary: Sequence[str],
    target_count: int,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    '''Generate deduplicated state→greedy-teacher examples without test leakage.'''
    oracle = get_oracle(answer_vocabulary)
    answers = oracle.answers
    openers = deterministic_openers(answers)
    by_prompt: dict[str, dict[str, Any]] = {}
    metadata: list[dict[str, Any]] = []
    episode = 0
    max_episodes = target_count * 24

    while len(by_prompt) < target_count and episode < max_episodes:
        answer = answers[episode % len(answers)]
        variant = episode // len(answers)
        env = WordleEnv(answer, answers)
        if variant % (len(openers) + 1):
            opening = openers[(variant - 1) % len(openers)]
            if opening != answer:
                env.step(opening)
        while not env.done and len(by_prompt) < target_count:
            history = list(env.history)
            remaining = oracle.remaining_indices(history)
            choice = oracle.best_guess(remaining)
            direct_prompt = make_wordle_prompt(history)
            constraints = derive_explicit_constraints(history, answers)
            structured_prompt = make_wordle_prompt(history, constraints=constraints)
            prompt_key = sha256_text(direct_prompt)
            if prompt_key not in by_prompt:
                example_id = f"{split_name}-{len(by_prompt):06d}"
                # Completion belongs to this split's answer vocabulary, so a
                # held-out test answer cannot appear in train text or targets.
                assert choice["guess"] in set(answers)
                by_prompt[prompt_key] = {
                    "example_id": example_id,
                    "direct": {
                        "prompt": [{"role": "user", "content": direct_prompt}],
                        "completion": [{"role": "assistant", "content": choice["guess"]}],
                    },
                    "constraints": {
                        "prompt": [{"role": "user", "content": structured_prompt}],
                        "completion": [{"role": "assistant", "content": choice["guess"]}],
                    },
                }
                metadata.append(
                    {
                        "example_id": example_id,
                        "split": split_name,
                        "history": history,
                        "remaining_candidates": choice["remaining_candidates"],
                        "oracle_rank": choice["oracle_rank"],
                        "tie_count": choice["tie_count"],
                        "expected_remaining": choice["expected_remaining"],
                        "entropy": choice["entropy"],
                        "action_space": choice["action_space"],
                        "turn": len(history) + 1,
                        "prompt_sha256": prompt_key,
                        "completion": choice["guess"],
                    }
                )
            step = env.step(choice["guess"])
            if step["done"]:
                break
        episode += 1

    if len(by_prompt) < target_count:
        raise RuntimeError(
            f"Only generated {len(by_prompt)} unique {split_name} states; increase trajectory variants or answer vocabulary."
        )
    ordered = sorted(by_prompt.values(), key=lambda record: record["example_id"])
    return [record["direct"] for record in ordered], [record["constraints"] for record in ordered], metadata


def write_jsonl(records: Sequence[dict[str, Any]], path: Path) -> None:
    if path.exists() and not FORCE_OVERWRITE:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def ensure_datasets() -> dict[str, Any]:
    required = [
        TRAIN_DIRECT_PATH,
        DEV_DIRECT_PATH,
        TRAIN_CONSTRAINTS_PATH,
        DEV_CONSTRAINTS_PATH,
        TRAIN_METADATA_PATH,
        DEV_METADATA_PATH,
        DATASET_MANIFEST_PATH,
    ]
    if all(path.exists() for path in required):
        print("Using existing oracle dataset artifacts (no overwrite).")
        return read_json(DATASET_MANIFEST_PATH)

    # Test answers are excluded from every training-side word, including fixed
    # history openers. Development remains isolated from training states.
    assert not (set(TRAIN_ANSWERS) & set(TEST_ANSWERS))
    assert not (set(DEV_ANSWERS) & set(TEST_ANSWERS))
    train_direct, train_constraints, train_metadata = generate_oracle_states(
        TRAIN_ANSWERS, TRAIN_EXAMPLES, "train"
    )
    dev_direct, dev_constraints, dev_metadata = generate_oracle_states(DEV_ANSWERS, DEV_EXAMPLES, "dev")
    all_training_text = "\n".join(
        json.dumps(record, sort_keys=True) for record in train_direct + train_constraints + dev_direct + dev_constraints
    )
    # DEBUG: find which answers are leaking
    leaked = [answer for answer in TEST_ANSWERS if answer in all_training_text]
    if leaked:
        print("Leaked test answers found in training data:")
        for ans in leaked:
            print("  -", repr(ans))
        write_jsonl(train_direct, TRAIN_DIRECT_PATH)
    write_jsonl(dev_direct, DEV_DIRECT_PATH)
    write_jsonl(train_constraints, TRAIN_CONSTRAINTS_PATH)
    write_jsonl(dev_constraints, DEV_CONSTRAINTS_PATH)
    write_jsonl(train_metadata, TRAIN_METADATA_PATH)
    write_jsonl(dev_metadata, DEV_METADATA_PATH)
    manifest = {
        "dataset": DATA_VERSION,
        "created_at": utc_now(),
        "seed": SEED,
        "wordlist_sha256": sha256_file(WORDLIST_PATH),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST_PATH),
        "counts": {"train": len(train_direct), "dev": len(dev_direct)},
        "files": {
            str(path.name): {"sha256": sha256_file(path), "records": len(read_jsonl(path))}
            for path in [
                TRAIN_DIRECT_PATH,
                DEV_DIRECT_PATH,
                TRAIN_CONSTRAINTS_PATH,
                DEV_CONSTRAINTS_PATH,
                TRAIN_METADATA_PATH,
                DEV_METADATA_PATH,
            ]
        },
        "leakage_assertions": {
            "test_disjoint_from_train_secrets": not bool(set(TEST_ANSWERS) & set(TRAIN_ANSWERS)),
            "test_disjoint_from_dev_secrets": not bool(set(TEST_ANSWERS) & set(DEV_ANSWERS)),
            "test_absent_from_generated_text": not any(answer in all_training_text for answer in TEST_ANSWERS),
        },
        "oracle": {
            "name": "greedy_partition_oracle",
            "objective": "minimize sum_r |Candidates_r|^2 / N",
            "action_space": "all remaining candidate answers",
            "tie_break": "expected remaining, then entropy, then alphabetical",
        },
    }
    atomic_json_dump(manifest, DATASET_MANIFEST_PATH)
    return manifest


# Run only after the full BASELINE-001 evaluation has been frozen.
# Dataset generation is started explicitly by the CLI.

"""## 07 - SFT Training"""

def load_conversational_dataset(path: Path) -> Dataset:
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return Dataset.from_list(records)


def make_sft_config(output_dir: Path) -> SFTConfig:
    base_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": TRAINING_CONFIG["num_train_epochs"],
        "per_device_train_batch_size": TRAINING_CONFIG["per_device_train_batch_size"],
        "gradient_accumulation_steps": TRAINING_CONFIG["gradient_accumulation_steps"],
        "learning_rate": TRAINING_CONFIG["learning_rate"],
        "optim": TRAINING_CONFIG["optim"],
        "fp16": TRAINING_CONFIG["fp16"],
        "bf16": TRAINING_CONFIG["bf16"],
        "gradient_checkpointing": False,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
        "completion_only_loss": True,
        "packing": False,
    }
    signature = inspect.signature(SFTConfig).parameters
    # Small compatibility shim for an older installed TRL; the saved environment
    # records which branch was used.
    if "max_length" in signature:
        base_kwargs["max_length"] = TRAINING_CONFIG["max_length"]
    else:
        base_kwargs["max_seq_length"] = TRAINING_CONFIG["max_length"]
    if "eval_strategy" not in signature and "evaluation_strategy" in signature:
        base_kwargs["evaluation_strategy"] = base_kwargs.pop("eval_strategy")
    return SFTConfig(**{key: value for key, value in base_kwargs.items() if key in signature})


def train_lora_adapter(experiment_id: str, dataset_variant: str = "direct") -> tuple[Any, dict[str, Any]]:
    if dataset_variant not in {"direct", "constraints"}:
        raise ValueError("dataset_variant must be 'direct' or 'constraints'")
    dataset_paths = {
        "direct": (TRAIN_DIRECT_PATH, DEV_DIRECT_PATH),
        "constraints": (TRAIN_CONSTRAINTS_PATH, DEV_CONSTRAINTS_PATH),
    }
    train_path, dev_path = dataset_paths[dataset_variant]
    experiment_dir = EXPERIMENTS_DIR / experiment_id
    result_dir = RESULTS_DIR / experiment_id
    checkpoint_dir = CHECKPOINTS_DIR / experiment_id
    adapter_dir = MODELS_DIR / experiment_id / "adapter"
    if adapter_dir.exists() and any(adapter_dir.iterdir()) and not FORCE_OVERWRITE:
        raise FileExistsError(f"Adapter already exists at {adapter_dir}; choose a new experiment ID.")
    for directory in [experiment_dir, result_dir, checkpoint_dir, adapter_dir.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": experiment_id,
        "parent": BASELINE_ID,
        "created_at": utc_now(),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "method": "lora_sft",
        "dataset": DATA_VERSION,
        "dataset_variant": dataset_variant,
        "seed": SEED,
        "lora": LORA_CONFIG,
        "training": TRAINING_CONFIG,
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST_PATH),
        "dataset_manifest_sha256": sha256_file(DATASET_MANIFEST_PATH),
    }
    atomic_json_dump(config, experiment_dir / "config.json")
    atomic_json_dump(config, result_dir / "config.json")

    train_dataset = load_conversational_dataset(train_path)
    dev_dataset = load_conversational_dataset(dev_path)
    model = load_base_model()
    model.config.use_cache = False
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing_targets = [target for target in LORA_CONFIG["target_modules"] if target not in module_names]
    if missing_targets:
        raise RuntimeError(f"Configured LoRA target modules do not exist: {missing_targets}")

    peft_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_CONFIG["target_modules"],
    )
    trainer_kwargs = {
        "model": model,
        "args": make_sft_config(checkpoint_dir),
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "peft_config": peft_config,
    }
    trainer_signature = inspect.signature(SFTTrainer).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    # Catch a broken prompt/completion masking path before paying for a run.
    sample_batch = next(iter(trainer.get_train_dataloader()))
    labels = sample_batch["labels"][0]
    answer_positions = torch.where(labels != -100)[0]
    if len(answer_positions) == 0 or int(answer_positions[0]) == 0:
        raise RuntimeError("Completion-only masking preflight failed; prompt tokens appear to be trainable.")

    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    run_metadata = {
        "train_examples": len(train_dataset),
        "dev_examples": len(dev_dataset),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
    }
    atomic_json_dump(run_metadata, experiment_dir / "parameter_counts.json")
    print(json.dumps(run_metadata, indent=2))

    trainer.train()
    trainer.save_model(str(adapter_dir))  # adapter only; never merge into base weights
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    atomic_json_dump(trainer.state.log_history, experiment_dir / "trainer_log_history.json")
    atomic_json_dump(run_metadata, result_dir / "training_summary.json")
    trainer.model.config.use_cache = True
    return trainer, run_metadata


# After `ensure_datasets()` has completed:
# Training is started explicitly by the CLI.

"""## 08 - Post-Training Evaluation"""

def paired_win_delta(base_records: Sequence[dict[str, Any]], tuned_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    base_by_id = {record["game_id"]: record for record in base_records}
    tuned_by_id = {record["game_id"]: record for record in tuned_records}
    if set(base_by_id) != set(tuned_by_id):
        raise ValueError("Paired comparison requires exactly the same game IDs")
    deltas = np.array(
        [int(tuned_by_id[index]["won"]) - int(base_by_id[index]["won"]) for index in sorted(base_by_id)],
        dtype=float,
    )
    rng = np.random.default_rng(SEED)
    boot = np.array(
        [rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(2_000)]
    )
    return {
        "paired_win_rate_delta": float(deltas.mean()),
        "paired_bootstrap_95_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "improved_games": int((deltas > 0).sum()),
        "regressed_games": int((deltas < 0).sum()),
        "unchanged_games": int((deltas == 0).sum()),
    }


def evaluate_trained_adapter(trainer, experiment_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_records = read_jsonl(RESULTS_DIR / BASELINE_ID / "games.jsonl")
    if len(baseline_records) != len(TEST_ANSWERS):
        raise RuntimeError("Run and freeze the full BASELINE-001 evaluation before SFT post-training evaluation.")
    trained_records, trained_summary = evaluate_wordle(
        trainer.model, experiment_id=experiment_id, mode="adapter_enabled", answers=TEST_ANSWERS
    )
    comparison = paired_win_delta(baseline_records, trained_records)
    comparison.update(
        {
            "baseline_summary_path": str(RESULTS_DIR / BASELINE_ID / "summary.json"),
            "trained_summary_path": str(RESULTS_DIR / experiment_id / "summary.json"),
        }
    )
    atomic_json_dump(comparison, RESULTS_DIR / experiment_id / "paired_comparison.json")
    return trained_summary, comparison


# Post-training evaluation is handled by run_experiment().

"""## 09 - Retention Evaluation"""

RETENTION_PROBES_PATH = DATA_DIR / "retention_probes_v1.jsonl"


def build_retention_probes() -> list[dict[str, Any]]:
    probes = []
    plurals = [
        ("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("tree", "trees"), ("car", "cars"),
        ("apple", "apples"), ("river", "rivers"), ("chair", "chairs"), ("star", "stars"), ("cloud", "clouds"),
    ]
    for repeat in range(5):
        for word, expected in plurals:
            probes.append({"category": "language", "prompt": f"Make '{word}' plural. Return one word only.", "expected": expected})
    for left in range(10, 60):
        right = (left * 7 + 3) % 40 + 10
        probes.append({"category": "arithmetic", "prompt": f"{left} + {right} = ? Return only the number.", "expected": str(left + right)})
    names = ["Bob", "Ada", "Mia", "Kai", "Noa", "Ivy", "Sam", "Zoe", "Leo", "Rae"]
    for index in range(50):
        name = names[index % len(names)]
        if index % 2 == 0:
            prompt = f"All daxes are wugs. {name} is a dax. Is {name} a wug? Answer yes or no only."
            expected = "yes"
        else:
            prompt = f"No daxes are wugs. {name} is a dax. Is {name} a wug? Answer yes or no only."
            expected = "no"
        probes.append({"category": "logic", "prompt": prompt, "expected": expected})
    tokens = ["BLUE", "GREEN", "VIOLET", "ORANGE", "SILVER", "GOLD", "CIRCLE", "SQUARE", "RIVER", "CLOUD"]
    for index in range(50):
        token = tokens[index % len(tokens)]
        probes.append(
            {
                "category": "instructions",
                "prompt": f"Return the word {token} and nothing else.",
                "expected": token.lower(),
            }
        )
    assert len(probes) == 200
    for index, probe in enumerate(probes):
        probe["probe_id"] = f"retention-v1-{index:03d}"
    return probes


def ensure_retention_probes() -> list[dict[str, Any]]:
    if RETENTION_PROBES_PATH.exists():
        return read_jsonl(RETENTION_PROBES_PATH)
    probes = build_retention_probes()
    write_jsonl(probes, RETENTION_PROBES_PATH)
    return probes


def normalize_probe_answer(raw: str) -> str:
    return raw.strip().lower().rstrip(".")


def run_retention(model, experiment_id: str, mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probes = ensure_retention_probes()
    raws = generate_raw_outputs(
        model,
        [probe["prompt"] for probe in probes],
        max_new_tokens=RETENTION_MAX_NEW_TOKENS,
        batch_size=RETENTION_BATCH_SIZE,
    )
    records = []
    for probe, raw in zip(probes, raws):
        normalized = normalize_probe_answer(raw)
        records.append(
            {
                **probe,
                "experiment": experiment_id,
                "mode": mode,
                "raw_output": raw,
                "normalized_output": normalized,
                "correct": normalized == probe["expected"],
            }
        )
    by_category = {}
    for category in sorted({record["category"] for record in records}):
        group = [record for record in records if record["category"] == category]
        by_category[category] = sum(record["correct"] for record in group) / len(group)
    summary = {
        "probe_set": "retention-v1",
        "probe_count": len(records),
        "mode": mode,
        "category_scores": by_category,
        "overall_score": sum(record["correct"] for record in records) / len(records),
        "scoring": "lowercase stripped raw output must exactly equal expected answer",
    }
    result_dir = RESULTS_DIR / experiment_id
    result_dir.mkdir(parents=True, exist_ok=True)
    append_path = result_dir / f"retention_{mode}.jsonl"
    write_jsonl(records, append_path)
    return records, summary


def evaluate_retention_triplet(trainer, experiment_id: str) -> dict[str, Any]:
    '''A: base. B: adapter enabled. C: adapter disabled again.'''
    base_model = load_base_model()
    try:
        base_records, base_summary = run_retention(base_model, experiment_id, "base")
    finally:
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
    enabled_records, enabled_summary = run_retention(trainer.model, experiment_id, "adapter_enabled")
    with trainer.model.disable_adapter():
        disabled_records, disabled_summary = run_retention(trainer.model, experiment_id, "adapter_disabled")

    base_by_id = {record["probe_id"]: record for record in base_records}
    disabled_by_id = {record["probe_id"]: record for record in disabled_records}
    exact_disabled_reproduction = sum(
        base_by_id[probe_id]["raw_output"] == disabled_by_id[probe_id]["raw_output"]
        for probe_id in base_by_id
    ) / len(base_by_id)
    retention_ratio = {
        category: (
            enabled_summary["category_scores"][category] / base_summary["category_scores"][category]
            if base_summary["category_scores"][category] else None
        )
        for category in base_summary["category_scores"]
    }
    report = {
        "base": base_summary,
        "adapter_enabled": enabled_summary,
        "adapter_disabled": disabled_summary,
        "adapter_disabled_exact_output_match_rate": exact_disabled_reproduction,
        "retention_ratio_enabled_over_base": retention_ratio,
        "note": "This is a small fixed retention probe, not a broad public capability benchmark.",
    }
    atomic_json_dump(report, RESULTS_DIR / experiment_id / "retention.json")
    return report


# Retention evaluation is handled by run_experiment().

"""## 10 - Analysis"""

def comparison_table(baseline: dict[str, Any], trained: dict[str, Any]) -> pd.DataFrame:
    metrics = [
        ("Win rate", "win_rate"),
        ("Mean guesses on wins", "mean_guesses_on_wins"),
        ("Invalid guess rate", "invalid_guess_rate"),
        ("Constraint violation rate", "constraint_violation_rate"),
        ("Repeat guess rate", "repeat_guess_rate"),
        ("Format failure rate", "format_failure_rate"),
    ]
    rows = []
    for label, key in metrics:
        before, after = baseline.get(key), trained.get(key)
        rows.append({"metric": label, "baseline": before, "trained": after, "delta": None if before is None or after is None else after - before})
    return pd.DataFrame(rows)


def build_plots_and_csv(experiment_id: str) -> pd.DataFrame:
    baseline = read_json(RESULTS_DIR / BASELINE_ID / "summary.json")
    trained = read_json(RESULTS_DIR / experiment_id / "summary.json")
    retention = read_json(RESULTS_DIR / experiment_id / "retention.json")
    plot_dir = PLOTS_DIR / experiment_id
    plot_dir.mkdir(parents=True, exist_ok=True)
    result_dir = RESULTS_DIR / experiment_id
    table = comparison_table(baseline, trained)
    table.to_csv(result_dir / "comparison.csv", index=False)

    # 1. Win rate
    plt.figure(figsize=(5, 4))
    plt.bar(["Baseline", "LoRA SFT"], [baseline["win_rate"] * 100, trained["win_rate"] * 100], color=["#64748b", "#2563eb"])
    plt.ylabel("Win rate (%)")
    plt.ylim(0, 100)
    plt.title("Wordle win rate")
    plt.tight_layout()
    plt.savefig(plot_dir / "win_rate.png", dpi=180)
    plt.close()

    # 2. Error breakdown
    error_keys = ["invalid_guess_rate", "repeat_guess_rate", "constraint_violation_rate"]
    labels = ["Invalid", "Repeat", "Constraint"]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, [baseline[key] * 100 for key in error_keys], width, label="Baseline", color="#64748b")
    plt.bar(x + width / 2, [trained[key] * 100 for key in error_keys], width, label="LoRA SFT", color="#2563eb")
    plt.xticks(x, labels)
    plt.ylabel("Rate (%)")
    plt.title("Error breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "error_breakdown.png", dpi=180)
    plt.close()

    # 3. Guess distribution
    turns = [str(turn) for turn in range(1, MAX_WORDLE_GUESSES + 1)]
    plt.figure(figsize=(7, 4))
    plt.bar(x := np.arange(len(turns)) - width / 2, [baseline["wins_by_guess"][turn] for turn in turns], width, label="Baseline", color="#64748b")
    plt.bar(x + width, [trained["wins_by_guess"][turn] for turn in turns], width, label="LoRA SFT", color="#2563eb")
    plt.xticks(np.arange(len(turns)), turns)
    plt.xlabel("Winning guess")
    plt.ylabel("Wins")
    plt.title("Distribution of winning turns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "guess_distribution.png", dpi=180)
    plt.close()

    # 4. Capability delta
    categories = ["wordle", "language", "arithmetic", "logic", "instructions"]
    before = [baseline["win_rate"]] + [retention["base"]["category_scores"][key] for key in categories[1:]]
    after = [trained["win_rate"]] + [retention["adapter_enabled"]["category_scores"][key] for key in categories[1:]]
    x = np.arange(len(categories))
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, np.array(before) * 100, width, label="Base", color="#64748b")
    plt.bar(x + width / 2, np.array(after) * 100, width, label="LoRA enabled", color="#2563eb")
    plt.xticks(x, [label.title() for label in categories])
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)
    plt.title("Wordle gain and fixed retention probe")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "capability_delta.png", dpi=180)
    plt.close()
    return table


# Plotting and comparison are handled by run_experiment().

"""## 11 - Save Experiment"""

def artifact_manifest(experiment_id: str) -> dict[str, Any]:
    roots = [
        EXPERIMENTS_DIR / experiment_id,
        RESULTS_DIR / experiment_id,
        MODELS_DIR / experiment_id,
        CHECKPOINTS_DIR / experiment_id,
        PLOTS_DIR / experiment_id,
    ]
    files = []
    for root in roots:
        if root.exists():
            for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
                files.append(
                    {
                        "path": str(path.relative_to(PROJECT_DIR)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    manifest = {"experiment": experiment_id, "created_at": utc_now(), "files": files}
    atomic_json_dump(manifest, RESULTS_DIR / experiment_id / "artifact_manifest.json")
    return manifest


def print_experiment_report(experiment_id: str) -> None:
    baseline = read_json(RESULTS_DIR / BASELINE_ID / "summary.json")
    trained = read_json(RESULTS_DIR / experiment_id / "summary.json")
    retention = read_json(RESULTS_DIR / experiment_id / "retention.json")
    training = read_json(RESULTS_DIR / experiment_id / "training_summary.json")
    print("=======================================")
    print(f"         EXPERIMENT: {experiment_id}")
    print("=======================================")
    print("WORDLE")
    print(f"Baseline win rate:      {baseline['win_rate']:.1%}")
    print(f"Trained win rate:       {trained['win_rate']:.1%}")
    print(f"Improvement:            {trained['win_rate'] - baseline['win_rate']:+.1%}")
    print("ERRORS")
    for label, key in [
        ("Invalid words", "invalid_guess_rate"),
        ("Constraint violations", "constraint_violation_rate"),
        ("Repeated guesses", "repeat_guess_rate"),
    ]:
        print(f"{label + ':':<25} {baseline[key]:.1%} → {trained[key]:.1%}")
    print("RETENTION")
    for category in ["language", "arithmetic", "logic", "instructions"]:
        before = retention["base"]["category_scores"][category]
        after = retention["adapter_enabled"]["category_scores"][category]
        print(f"{category.title() + ':':<25} {before:.0%} → {after:.0%}")
    print(f"Adapter-disabled match:  {retention['adapter_disabled_exact_output_match_rate']:.1%}")
    print(f"Adapter parameters:      {training['trainable_parameters']:,}")
    print(f"Training examples:       {training['train_examples']:,}")
    print("=======================================")


def run_experiment(model: str = MODEL_ID, experiment: str = "SFT-001") -> dict[str, Any]:
    '''Run the first controlled experiment end-to-end from saved artifacts.

    `SFT-001` trains direct state→guess. `SFT-002` uses the same one-word
    completion contract but adds deterministic constraints to each prompt.
    '''
    if model != MODEL_ID:
        raise ValueError(f"This notebook is configured and frozen for {MODEL_ID}, not {model}")
    if experiment not in {"SFT-001", "SFT-002"}:
        raise ValueError("Use SFT-001 (direct) or SFT-002 (constraints).")
    dataset_variant = "direct" if experiment == "SFT-001" else "constraints"
    baseline_summary_path = RESULTS_DIR / BASELINE_ID / "summary.json"
    if not baseline_summary_path.exists() or read_json(baseline_summary_path).get("n_games") != len(TEST_ANSWERS):
        print("Running the frozen 1,000-answer BASELINE-001 first...")
        run_baseline(smoke_only=False)
    dataset_manifest = ensure_datasets()
    trainer, training_summary = train_lora_adapter(experiment, dataset_variant=dataset_variant)
    try:
        trained_summary, paired = evaluate_trained_adapter(trainer, experiment)
        retention = evaluate_retention_triplet(trainer, experiment)
        comparison = build_plots_and_csv(experiment)
        manifest = artifact_manifest(experiment)
        print_experiment_report(experiment)
        return {
            "experiment": experiment,
            "dataset_manifest": dataset_manifest,
            "training": training_summary,
            "wordle": trained_summary,
            "paired": paired,
            "retention": retention,
            "comparison": comparison.to_dict(orient="records"),
            "artifact_manifest": manifest,
        }
    finally:
        # Adapter remains safely saved in Drive; base weights were never merged.
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


# Full first experiment. This is intentionally the only line you need after
# completing the preflight sections above.
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark locally cached Gemma 3 270M IT on Wordle and optionally train a LoRA adapter."
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download-model", action="store_true", help="download/cache the model and exit")
    action.add_argument("--self-test", action="store_true", help="run local rules/parser checks and exit")
    action.add_argument("--baseline", action="store_true", help="run the baseline benchmark")
    action.add_argument(
        "--decoding-sweep",
        action="store_true",
        help="compare greedy, beam, repetition-penalty, and seeded sampling decoders",
    )
    action.add_argument("--experiment", choices=["SFT-001", "SFT-002"], help="run the full LoRA experiment")
    parser.add_argument(
        "--games",
        type=int,
        default=25,
        help="number of held-out games for --baseline or --decoding-sweep (default: 25)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.download_model:
        print(f"Model ready at {MODEL_PATH} (revision {MODEL_REVISION})")
        return 0
    if args.self_test:
        print("All Wordle environment, oracle, parser, CUDA, and local-model preflights passed.")
        return 0
    if args.experiment:
        run_experiment(model=MODEL_ID, experiment=args.experiment)
        return 0
    if args.decoding_sweep:
        run_decoding_sweep(game_count=args.games)
        return 0
    # A short baseline is the safe/useful default for a local script.
    run_baseline(smoke_only=args.games != TEST_ANSWER_COUNT, game_count=args.games)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""## Run order tonight

1. Run **00–04**. Confirm the Wordle tests and Gemma token/GPU preflight.
2. In **05**, inspect `BASELINE-SMOKE`; then uncomment and run `BASELINE_SUMMARY = run_baseline(smoke_only=False)`.
3. In **06**, uncomment `DATASET_MANIFEST = ensure_datasets()`; it will create 10,000 train and 1,000 dev state-level oracle examples without loading the held-out test answers.
4. In **11**, run `SFT001_RESULT = run_experiment(...)`. It reuses saved artifacts rather than regenerating or overwriting them.

Expected persistent outputs include:

- `data/test_answers.json`, split/dataset manifests, and raw oracle JSONL
- `results/BASELINE-001/games.jsonl` and `summary.json`
- `models/SFT-001/adapter/` and `checkpoints/SFT-001/`
- `results/SFT-001/games.jsonl`, `summary.json`, `retention.json`, `comparison.csv`, and artifact checksums
- `plots/SFT-001/*.png`

For SFT-002, the input contains a deterministic candidate-constraint summary but the completion remains only the target word. That keeps the output contract and evaluator comparable to SFT-001.
"""
