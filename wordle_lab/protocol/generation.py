from __future__ import annotations

import time
from collections.abc import Sequence

import torch

from .prompting import inference_messages

GENERATION_CONFIG = {"do_sample": False, "max_new_tokens": 128, "use_cache": True}


def stop_token_ids(tokenizer) -> list[int]:
    values = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    return list(dict.fromkeys(value for value in values if value is not None and value >= 0))


def generate(model, tokenizer, histories: Sequence[Sequence[tuple[str, str]]], batch_size: int = 16) -> list[dict]:
    device = next(model.parameters()).device
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    records: list[dict] = []
    try:
        for start in range(0, len(histories), batch_size):
            batch = histories[start : start + batch_size]
            rendered = [tokenizer.apply_chat_template(inference_messages(history), tokenize=False, add_generation_prompt=True) for history in batch]
            inputs = tokenizer(rendered, padding=True, return_tensors="pt", add_special_tokens=False).to(device)
            began = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    **GENERATION_CONFIG,
                    eos_token_id=stop_token_ids(tokenizer),
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = time.perf_counter() - began
            width = inputs["input_ids"].shape[1]
            for row in output:
                token_ids = row[width:]
                terminal_ids = set(stop_token_ids(tokenizer))
                if tokenizer.pad_token_id is not None:
                    terminal_ids.add(tokenizer.pad_token_id)
                count = len(token_ids)
                for index, token_id in enumerate(token_ids.tolist()):
                    if token_id in terminal_ids:
                        count = index + 1
                        break
                token_ids = token_ids[:count]
                records.append({
                    "raw_output": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                    "generated_tokens": int(count),
                    "latency_s": elapsed / len(batch),
                })
    finally:
        tokenizer.padding_side = old_padding
    return records
