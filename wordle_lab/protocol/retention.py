from __future__ import annotations

from collections import Counter

import torch


def build_retention_probes() -> list[dict]:
    probes = []
    plurals = [("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("tree", "trees"), ("car", "cars"), ("apple", "apples"), ("river", "rivers"), ("chair", "chairs"), ("star", "stars"), ("cloud", "clouds")]
    for repeat in range(5):
        for word, expected in plurals:
            probes.append({"category": "language", "prompt": f"Make '{word}' plural. Return one word only.", "expected": expected})
    for left in range(10, 60):
        right = (left * 7 + 3) % 40 + 10
        probes.append({"category": "arithmetic", "prompt": f"{left} + {right} = ? Return only the number.", "expected": str(left + right)})
    names = ["Bob", "Ada", "Mia", "Kai", "Noa", "Ivy", "Sam", "Zoe", "Leo", "Rae"]
    for index in range(50):
        yes = index % 2 == 0; name = names[index % len(names)]
        probes.append({"category": "logic", "prompt": f"{'All' if yes else 'No'} daxes are wugs. {name} is a dax. Is {name} a wug? Answer yes or no only.", "expected": "yes" if yes else "no"})
    tokens = ["BLUE", "GREEN", "VIOLET", "ORANGE", "SILVER", "GOLD", "CIRCLE", "SQUARE", "RIVER", "CLOUD"]
    for index in range(50):
        token = tokens[index % len(tokens)]
        probes.append({"category": "instructions", "prompt": f"Return the word {token} and nothing else.", "expected": token.lower()})
    for index, probe in enumerate(probes):
        probe["probe_id"] = f"retention-v1-{index:03d}"
    return probes


def evaluate_retention(model, tokenizer, probes: list[dict], batch_size: int = 32) -> tuple[list[dict], dict]:
    device = next(model.parameters()).device
    records = []
    old_side = tokenizer.padding_side; tokenizer.padding_side = "left"
    try:
        for start in range(0, len(probes), batch_size):
            batch = probes[start:start + batch_size]
            rendered = [tokenizer.apply_chat_template([{"role": "user", "content": probe["prompt"]}], tokenize=False, add_generation_prompt=True) for probe in batch]
            inputs = tokenizer(rendered, padding=True, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, do_sample=False, max_new_tokens=16, eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")], pad_token_id=tokenizer.pad_token_id)
            width = inputs["input_ids"].shape[1]
            for probe, output in zip(batch, outputs):
                raw = tokenizer.decode(output[width:], skip_special_tokens=True).strip()
                normalized = raw.lower().rstrip(".").strip()
                records.append({**probe, "raw_output": raw, "normalized_output": normalized, "correct": normalized == probe["expected"]})
    finally:
        tokenizer.padding_side = old_side
    by_category = {category: sum(row["correct"] for row in records if row["category"] == category) / sum(row["category"] == category for row in records) for category in sorted({row["category"] for row in records})}
    return records, {"probe_count": len(records), "overall_score": sum(row["correct"] for row in records) / len(records), "category_scores": by_category}
