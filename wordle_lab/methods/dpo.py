from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

from wordle_lab.common import write_json
from wordle_lab.models import load_base_model, load_tokenizer


def train_dpo(rows: list[dict], parent_adapter: Path, run_dir: Path, spec: dict):
    """Reference-policy DPO from a seed-matched SFT LoRA warm start."""
    tokenizer = load_tokenizer(parent_adapter)
    policy = PeftModel.from_pretrained(load_base_model(training=True), parent_adapter, is_trainable=True)
    reference = PeftModel.from_pretrained(load_base_model(training=False), parent_adapter, is_trainable=False)
    dataset = Dataset.from_list([{"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"], "negative_type": row["negative_type"]} for row in rows])
    args = DPOConfig(
        output_dir=str(run_dir / "checkpoints"), max_steps=int(spec["max_steps"]),
        per_device_train_batch_size=int(spec["batch_size"]), gradient_accumulation_steps=int(spec["gradient_accumulation_steps"]),
        learning_rate=float(spec["learning_rate"]), beta=float(spec.get("beta", 0.1)), max_length=int(spec["max_length"]),
        bf16=torch.cuda.is_bf16_supported(), fp16=False, gradient_checkpointing=False, report_to="none",
        logging_steps=1, save_strategy="steps", save_steps=max(1, int(spec["max_steps"]) // 4), seed=int(spec["seed"]), data_seed=int(spec["seed"]),
    )
    trainer = DPOTrainer(model=policy, ref_model=reference, args=args, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    final = run_dir / "checkpoints" / "final"; trainer.save_model(final); tokenizer.save_pretrained(final)
    write_json(run_dir / "dpo_log_history.json", trainer.state.log_history)
    return trainer.model
