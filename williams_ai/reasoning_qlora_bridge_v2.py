"""Williams AI v2: approved reasoning lessons -> actual QLoRA input contract.

This module does not start GPU work or invent expert answers. The adapter
accepts only fully reviewed train/validation source cases, bound to a frozen
CPU preflight manifest. The sealed test gold stays OUTSIDE the GPU worker.
"""
from __future__ import annotations

import json
from pathlib import Path

from .reasoning_research_v2 import (
    canonical, check_corpus, digest, question_only, verify_case,
)
from .reasoning_training_readiness_v2 import readiness

SYSTEM = ("Research only. Ground every inference in provided closed market "
          "observations and book evidence. State alternative scenarios and "
          "invalidation. Do not assume orders or authorize trading.")


def prepare_gpu_handoff(cases):
    """CPU-side only. Export only reviewed train and validation to GPU."""
    gate = readiness(cases)
    if not gate["ready_for_gpu_experiment"]:
        raise ValueError("REASONING_V2_REVIEWED_CORPUS_NOT_GPU_READY:" +
                         ",".join(gate["reasons"]))
    approved = sorted((row for row in cases
                       if row["split"] in {"train", "validation"}),
                      key=lambda row: (row["split"], row["case_id"]))
    for row in approved:
        if "higher_timeframes" not in row or "book_evidence" not in row:
            raise ValueError("MULTITIMEFRAME_BOOK_EVIDENCE_REQUIRED")
    manifest = {
        "schema": "WILLIAMS_AI_REASONING_V2_TRAIN_VALIDATION_HANDOFF_V1",
        "approved_train_validation_sha256": digest(approved),
        "whole_corpus_sha256": gate["corpus_sha256"],
        "counts": gate["counts"],
        "test_answer_rows_sent_to_gpu": 0,
        "test_data_in_gpu_handoff": False,
        "ready_for_gpu_experiment": True,
        "expert_reviewers_are_declarations_not_authenticated": True,
        "execution_allowed": False,
    }
    return approved, manifest


def validate_gpu_handoff(rows, manifest):
    """Fail closed before constructing training tensors."""
    if (not isinstance(manifest, dict)
            or manifest.get("schema") !=
               "WILLIAMS_AI_REASONING_V2_TRAIN_VALIDATION_HANDOFF_V1"
            or manifest.get("test_data_in_gpu_handoff") is not False
            or manifest.get("test_answer_rows_sent_to_gpu") != 0
            or manifest.get("ready_for_gpu_experiment") is not True):
        raise ValueError("FROZEN_GPU_RESEARCH_PREFLIGHT_REQUIRED")
    if (not isinstance(rows, list) or
            any(not isinstance(row, dict) or
                row.get("split") not in {"train", "validation"}
                for row in rows)):
        raise ValueError("ONLY_TRAIN_AND_VALIDATION_ON_GPU")
    if digest(rows) != manifest.get("approved_train_validation_sha256"):
        raise ValueError("GPU_TRAIN_VALIDATION_DATASET_CHANGED")
    result = check_corpus(rows)
    if (result["counts"].get("train", 0) !=
            manifest["counts"]["train"]["cases"] or
        result["counts"].get("validation", 0) !=
            manifest["counts"]["validation"]["cases"]):
        raise ValueError("GPU_SPLIT_COUNTS_CHANGED")
    if any("higher_timeframes" not in row or "book_evidence" not in row for row in rows):
        raise ValueError("REVIEWED_MARKET_CONTEXT_MISSING")
    return result


def tokenize_verified(rows, tokenizer, *, max_tokens):
    """Answer-only SFT, no truncation and full Qwen chat-turn delimiters."""
    if type(max_tokens) is not int or not 256 <= max_tokens <= 8192:
        raise ValueError("EXPLICIT_TOKEN_CONTEXT_BUDGET_REQUIRED")
    records = []
    for row in rows:
        verify_case(row)
        if row["split"] not in {"train", "validation"}:
            raise ValueError("NEVER_TOKENIZE_TEST_GOLD")
        if "higher_timeframes" not in row or "book_evidence" not in row:
            raise ValueError("REVIEWED_H1_H4_D1_EVIDENCE_REQUIRED")
        prompt = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": canonical(question_only(row))},
        ]
        answer = canonical(row["answer"])
        full = prompt + [{"role": "assistant", "content": answer}]
        before = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True,
            enable_thinking=False)
        complete = tokenizer.apply_chat_template(
            full, tokenize=True, add_generation_prompt=False,
            enable_thinking=False)
        if (not isinstance(before, list) or not isinstance(complete, list)
                or complete[:len(before)] != before
                or not 2 <= len(before) < len(complete) <= max_tokens):
            raise ValueError("ANSWER_ONLY_FULL_TEMPLATE_NO_TRUNCATION_REQUIRED")
        labels = [-100] * len(before) + complete[len(before):]
        records.append({"input_ids": complete, "attention_mask": [1]*len(complete),
                        "labels": labels})
    return records


def load_private_gpu_inputs(data_path, manifest_path):
    from .private_lora import _private_file
    source = _private_file(Path(data_path))
    frozen = _private_file(Path(manifest_path))
    if (source.suffix != ".jsonl" or source.stat().st_size > 100_000_000
            or frozen.stat().st_size > 100_000):
        raise ValueError("BOUNDED_PRIVATE_GPU_INPUTS_REQUIRED")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    manifest = json.loads(frozen.read_text(encoding="utf-8"))
    verify = validate_gpu_handoff(rows, manifest)
    return rows, manifest, verify


def run_reasoning_gpu(*, reviewed_train_validation, handoff_manifest,
                      model_dir, output, source_revision,
                      total_steps=200, session_steps=20, max_tokens=4096,
                      rank=8, learning_rate=5e-5, seed=42,
                      authorized_private_data=False):
    """Actual opt-in, local CUDA QLoRA path using existing resumable trainer.

    Called only when an independently reviewed corpus is genuinely ready.
    Never accesses test gold or routes trading requests.
    """
    if authorized_private_data is not True:
        raise ValueError("EXPLICIT_PRIVATE_DATA_GPU_PERMISSION_REQUIRED")
    rows, manifest, verified = load_private_gpu_inputs(
        reviewed_train_validation, handoff_manifest)
    from .private_lora import _private_file
    from .lora_integrity import local_hf_identity
    from .qlora_gpu_research import (
        _fit_resumable_adapter, _runtime_identity, contract,
    )
    import subprocess
    from .book_index import ROOT
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("ONE_REAL_CUDA_GPU_REQUIRED")
    hf = _private_file(Path(model_dir), must_exist=False)
    dst = _private_file(Path(output), must_exist=False)
    if (not hf.is_dir() or hf.is_symlink() or dst.exists()
            or hf in dst.parents or dst in hf.parents):
        raise ValueError("LOCAL_HF_BASE_AND_NEW_OUTPUT_REQUIRED")
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "williams_ai"],
                           cwd=ROOT)
    if actual != source_revision or clean.returncode != 0:
        raise ValueError("PINNED_CLEAN_TRAINING_SOURCE_REQUIRED")
    identity = local_hf_identity(hf)
    spec = contract(dataset_sha=verified["source_sha256"],
                    base_sha=identity["sha256"], revision=source_revision,
                    max_steps=total_steps, max_tokens=max_tokens, rank=rank,
                    rate=float(learning_rate), seed=seed,
                    runtime=_runtime_identity(), source_only_research=False)
    spec["reviewed_reasoning_schema"] = manifest["schema"]
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(hf), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("QWEN_EOS_TOKEN_REQUIRED")
        tokenizer.pad_token = tokenizer.eos_token
    train = tokenize_verified([row for row in rows if row["split"] == "train"],
                              tokenizer, max_tokens=max_tokens)
    validation = {edition: tokenize_verified(
        [row for row in rows if row["split"] == "validation" and row["edition"] == edition],
        tokenizer, max_tokens=max_tokens) for edition in ("TC1", "NTD", "TC2", "COURSE")}
    if not train or any(not group for group in validation.values()):
        raise ValueError("FOUR_BOOK_DEVELOPMENT_COVERAGE_REQUIRED")
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        str(hf), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, quantization_config=quant,
        device_map={"": 0}, torch_dtype=torch.float16)
    if local_hf_identity(hf)["sha256"] != identity["sha256"]:
        raise ValueError("HF_BASE_CHANGED_DURING_LOADING")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=False)
    model = get_peft_model(
        model, LoraConfig(r=rank, lora_alpha=rank*2, lora_dropout=.05,
                          target_modules=spec["target_modules"],
                          task_type=TaskType.CAUSAL_LM, bias="none"))
    model.config.use_cache = False
    return _fit_resumable_adapter(
        model, train, validation, output=dst, spec=spec,
        pad_token_id=tokenizer.pad_token_id, session_steps=session_steps)


def main():
    import argparse
    p=argparse.ArgumentParser(description="V2 private reviewed QLoRA, no live orders")
    p.add_argument("--reviewed-train-validation", type=Path, required=True)
    p.add_argument("--handoff-manifest", type=Path, required=True)
    p.add_argument("--hf-base", type=Path, required=True)
    p.add_argument("--new-output", type=Path, required=True)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--session-steps", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--authorize-private-data-gpu", action="store_true")
    args=p.parse_args()
    result = run_reasoning_gpu(
        reviewed_train_validation=args.reviewed_train_validation,
        handoff_manifest=args.handoff_manifest, model_dir=args.hf_base,
        output=args.new_output, source_revision=args.source_revision,
        total_steps=args.max_steps, session_steps=args.session_steps,
        max_tokens=args.max_tokens, rank=args.rank,
        learning_rate=args.learning_rate, seed=args.seed,
        authorized_private_data=args.authorize_private_data_gpu)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
