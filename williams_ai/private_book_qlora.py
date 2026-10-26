"""Isolated private Qwen3-4B NF4 QLoRA source-language adaptation on NVIDIA GPU.

Research-only first stage: raw text of user-supplied four PDFs, with immutable
page-separated validation. This is NOT expert chart supervision or complete
vision reading. Model, dataset and adapter stay private and never uploaded.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from .lora_integrity import EDITIONS, local_hf_identity, regular_sha256
from .lora_training import fit_lora, LoraPlan
from .private_lora import _private_file, checked_local_qwen_config
from .private_book_source import load_private_source_corpus, tokenize_source_pages


def train_private_book_qlora(*, private_source_dir: Path, hf_base: Path,
                             new_output: Path, steps: int = 10,
                             max_tokens: int = 512, rank: int = 8,
                             authorize_private_gpu: bool = False) -> dict:
    """Fit REAL pretrained Qwen3-4B LoRA if private data + CUDA are available.

    No automatic exchange access, Android deployment or test-label learning.
    """
    if not authorize_private_gpu:
        raise ValueError("EXPLICIT_PRIVATE_GPU_DATA_PROCESSING_AUTHORIZATION_REQUIRED")
    if type(steps) is not int or not 2 <= steps <= 100:
        raise ValueError("BOUNDED_GPU_TRAINING_STEPS_REQUIRED")
    if rank not in (4, 8, 16) or type(rank) is not int:
        raise ValueError("BOUNDED_GPU_QLORA_RANK_REQUIRED")
    if type(max_tokens) is not int or not 256 <= max_tokens <= 1024:
        raise ValueError("CONSTRAINED_GPU_CONTEXT_REQUIRED")

    source = load_private_source_corpus(private_source_dir)
    base = _private_file(hf_base, must_exist=False)
    target = _private_file(new_output, must_exist=False)
    if (not base.is_dir() or base.is_symlink()
            or base == target or base in target.parents
            or target in base.parents or target.exists() or target.is_symlink()):
        raise ValueError("NEW_PRIVATE_OUTPUT_AND_EXISTING_HF_MODEL_REQUIRED")

    config = checked_local_qwen_config(base)
    import json as _json
    cfg = _json.loads((base / "config.json").read_text())
    if config["model_type"] != "qwen3" or cfg.get("hidden_size") != 2560:
        # Qwen3-4B Instruct: 4B parameter variant hidden_size=2560.
        # Explicit family/shape gate avoids silently training another base.
        raise ValueError("REAL_QWEN3_4B_HF_BASE_REQUIRED")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SINGLE_PRIVATE_NVIDIA_CUDA_GPU_REQUIRED")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    identity = local_hf_identity(base)
    tokenizer = AutoTokenizer.from_pretrained(
        base, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    trainrows = tokenize_source_pages(source["train"], tokenizer, max_tokens=max_tokens)
    validation = {ed: tokenize_source_pages(
        [r for r in source["validation"] if r["edition"] == ed],
        tokenizer, max_tokens=max_tokens) for ed in EDITIONS}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base, local_files_only=True, trust_remote_code=False,
        quantization_config=quant, torch_dtype=dtype,
        device_map={"": 0}, use_safetensors=True)
    if local_hf_identity(base) != identity:
        raise ValueError("MODEL_OR_TOKENIZER_MODIFIED_DURING_GPU_LOAD")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Stages the first session privately; not resumable until independently
    # verified with an actual GPU. This does not modify qlora_gpu_research.
    with tempfile.TemporaryDirectory(prefix=".source-qwen4b-",dir=target.parent) as scratch:
        temp = Path(scratch)
        fitted, fit = fit_lora(model, trainrows, validation, workdir=temp / "checkpoints",
            pad_token_id=tokenizer.pad_token_id, max_steps=steps,
            plan=LoraPlan(rank=rank, learning_rate=8e-5,
                          gradient_accumulation=4, seed=20261027),
            use_cpu=False)
        if local_hf_identity(base) != identity:
            raise ValueError("MODEL_OR_TOKENIZER_CHANGED_DURING_GPU_TRAINING")
        passed = fit["validation_gate"]["accepted"]
        report = {"mode": "PRIVATE_FOUR_BOOK_TEXT_ONLY_QWEN3_4B_NF4_QLORA",
            "weight_training_completed": True,
            "source_only": True, "original_source_text_copied_to_public_git": False,
            "source_audit": source["audit"],
            "base_identity": identity,
            "steps": fit["optimizer_steps"],
            "trainable_parameters": fit["trainable_parameters"],
            "loss_by_book_baseline": fit["baseline_by_book"],
            "loss_by_book_adapted": fit["adapted_by_book"],
            "validation_gate": fit["validation_gate"],
            "no_chart_expert_validation": True,
            "source_book_ocr_and_figures_not_all_reviewed": True,
            "blind_market_holdout_passed": False,
            "live_and_demo_trading_allowed": False,
            "execution_allowed": False, "trade_action": None}
        target.mkdir(mode=0o700, exist_ok=False)
        if passed:
            fitted.save_pretrained(target, safe_serialization=True)
            report["adapter_sha256"] = regular_sha256(target / "adapter_model.safetensors")
        else:
            report["adapter_saved"] = False
            report["rejected_weights_discarded"] = True
        (target / "source_research_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8")
        for path in target.iterdir():
            if path.is_file():
                path.chmod(0o600)
        print(json.dumps({k: v for k, v in report.items()
                          if k not in {"base_identity"}}, ensure_ascii=False, indent=2))
        return report


def main() -> None:
    p = argparse.ArgumentParser(description="PRIVATE, source-only, actual Qwen3-4B GPU QLoRA")
    p.add_argument("--private-source-dir", type=Path, required=True)
    p.add_argument("--hf-base", type=Path, required=True)
    p.add_argument("--new-output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--authorize-private-gpu", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    if a.audit_only:
        result = load_private_source_corpus(a.private_source_dir)["audit"]
    else:
        result = train_private_book_qlora(
            private_source_dir=a.private_source_dir, hf_base=a.hf_base,
            new_output=a.new_output, steps=a.steps, rank=a.rank,
            max_tokens=a.max_tokens, authorize_private_gpu=a.authorize_private_gpu)
    if a.audit_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
