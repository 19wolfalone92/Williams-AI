"""Shared actual PEFT fit loop; production supervision is audited by private_lora.

Optional ML dependencies are imported only during a fit. CPU is supported here
for a tiny synthetic CI diagnostic, not exposed as a reviewed-data bypass.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

from .lora_integrity import EDITIONS, validation_admission


@dataclass(frozen=True)
class LoraPlan:
    rank: int = 16
    learning_rate: float = 5e-5
    target_profile: str = "attention"
    gradient_accumulation: int = 8
    seed: int = 42

    def __post_init__(self):
        if (type(self.rank) is not int or self.rank not in (4, 8, 16, 32, 64) or
                type(self.learning_rate) not in (int, float) or
                not 1e-6 <= self.learning_rate <= 5e-4 or
                self.target_profile not in ("attention", "attention-mlp") or
                type(self.gradient_accumulation) is not int or
                not 1 <= self.gradient_accumulation <= 64 or
                type(self.seed) is not int or not 0 <= self.seed < 2**31):
            raise ValueError("INVALID_BOUNDED_LORA_PLAN")

    @property
    def targets(self) -> list[str]:
        result = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if self.target_profile == "attention-mlp":
            result += ["gate_proj", "up_proj", "down_proj"]
        return result


def fit_lora(base_model: Any, trainrows: list[dict], validation: dict[str, list[dict]], *,
             workdir: Path, pad_token_id: int, max_steps: int, plan: LoraPlan,
             use_cpu: bool = False) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (EarlyStoppingCallback, Trainer, TrainerCallback,
                              TrainingArguments, set_seed)
    if (type(max_steps) is not int or not 1 <= max_steps <= 10000 or
            set(validation) != set(EDITIONS) or any(not rows for rows in validation.values())
            or not trainrows or type(use_cpu) is not bool):
        raise ValueError("BOUNDED_TRAIN_AND_FOUR_BOOK_VALIDATION_REQUIRED")
    present = {name.rsplit(".", 1)[-1] for name, _ in base_model.named_modules()}
    if not set(plan.targets) <= present:
        raise ValueError("REQUESTED_QWEN_LORA_TARGETS_NOT_PRESENT")
    set_seed(plan.seed)
    model = get_peft_model(base_model, LoraConfig(r=plan.rank, lora_alpha=2 * plan.rank,
        lora_dropout=0.05, target_modules=plan.targets, bias="none",
        task_type=TaskType.CAUSAL_LM))
    model.config.use_cache = False
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any(".lora_" not in name for name in trainable):
        raise ValueError("ONLY_LORA_PARAMETERS_MAY_BE_TRAINABLE")
    initial = {name: p.detach().cpu().clone() for name, p in trainable.items()}

    class Dataset:
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]

    def collate(batch):
        size = max(len(x["input_ids"]) for x in batch)
        def pad(key, filler):
            return torch.tensor([x[key] + [filler] * (size - len(x[key])) for x in batch],
                                dtype=torch.long)
        return {"input_ids": pad("input_ids", pad_token_id),
                "attention_mask": pad("attention_mask", 0), "labels": pad("labels", -100)}

    class FiniteGradients(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            gradients = [p.grad for p in trainable.values() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
                raise ValueError("NONFINITE_OR_MISSING_LORA_GRADIENTS")

        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "grad_norm"):
                if key in (logs or {}) and not math.isfinite(float(logs[key])):
                    raise ValueError("NONFINITE_LORA_TRAINING_LOSS")

    # Include the final step in evaluation/checkpoint selection, even for a
    # short or prime-length run. Never select a checkpoint using test labels.
    interval = min(10, max_steps)
    while max_steps % interval:
        interval -= 1
    bf16 = not use_cpu and torch.cuda.is_bf16_supported()
    args = TrainingArguments(output_dir=str(workdir), max_steps=max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=plan.gradient_accumulation,
        learning_rate=plan.learning_rate, weight_decay=0.01, warmup_ratio=0.05,
        lr_scheduler_type="cosine", max_grad_norm=1.0, optim="adamw_torch",
        eval_strategy="steps", eval_steps=interval, save_strategy="steps", save_steps=interval,
        save_total_limit=2, save_only_model=True, save_safetensors=True,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=1, logging_nan_inf_filter=False, report_to="none", disable_tqdm=True,
        remove_unused_columns=False, label_names=["labels"], seed=plan.seed, data_seed=plan.seed,
        bf16=bf16, fp16=not use_cpu and not bf16, use_cpu=use_cpu,
        dataloader_pin_memory=not use_cpu, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    all_validation = [row for ed in EDITIONS for row in validation[ed]]
    trainer = Trainer(model=model, args=args, train_dataset=Dataset(trainrows),
        eval_dataset=Dataset(all_validation), data_collator=collate,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), FiniteGradients()])

    def scores():
        result = {"ALL": float(trainer.evaluate()["eval_loss"])}
        for ed in EDITIONS:
            result[ed] = float(trainer.evaluate(eval_dataset=Dataset(validation[ed]))["eval_loss"])
        # Fail before training when baseline validation is already invalid.
        validation_admission(result, result)
        return result

    before = scores()
    fitted = trainer.train()
    if not math.isfinite(float(fitted.training_loss)):
        raise ValueError("NONFINITE_LORA_TRAINING_LOSS")
    after = scores()  # Trainer has reloaded its best validation checkpoint.
    steps = trainer.state.global_step
    checkpoint = trainer.state.best_model_checkpoint
    if type(steps) is not int or steps <= 0 or not checkpoint:
        raise ValueError("NO_ACTUAL_LORA_OPTIMIZER_STEP_OR_BEST_CHECKPOINT")
    selected = int(Path(checkpoint).name.removeprefix("checkpoint-"))
    if not 0 < selected <= steps:
        raise ValueError("INVALID_SELECTED_LORA_CHECKPOINT")
    if any(not torch.isfinite(p).all().item() for p in trainable.values()):
        raise ValueError("NONFINITE_LORA_PARAMETERS")
    changed = any(not torch.equal(initial[name], p.detach().cpu())
                  for name, p in trainable.items())
    if not changed:
        raise ValueError("LORA_WEIGHTS_DID_NOT_CHANGE")
    return model, {"optimizer_steps": steps, "selected_checkpoint_step": selected,
        "best_validation_checkpoint_reloaded": True, "adapter_weights_changed": True,
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "base_parameters_frozen": True, "baseline_by_book": before,
        "adapted_by_book": after, "validation_gate": validation_admission(before, after),
        "plan": {**asdict(plan), "alpha": 2 * plan.rank, "dropout": 0.05,
                 "target_modules": plan.targets, "evaluation_interval": interval},
        "early_stopping_patience": 3, "test_labels_not_supplied_to_trainer": True}
