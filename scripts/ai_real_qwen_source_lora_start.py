"""REAL pretrained Qwen3-0.6B LoRA source-only research on public rule exercises.

No private books, human expert chart labels, orders, live/demo keys or phone
model activation. This starts weight training, but is NOT trader qualification.
Run only in opt-in GitHub Actions research workflow.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

# A direct script invocation puts scripts/ (not the repository root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ED = ("TC1", "NTD", "TC2", "COURSE")
MODEL_REPO = "Qwen/Qwen3-0.6B"
# Official Hugging Face LFS SHA256 of the published safetensors file.
MODEL_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for block in iter(lambda: fd.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_public_exercises() -> tuple[list[dict], dict[str, list[dict]], dict]:
    from williams_ai.adaptive_coach import CURRICULUM_PATH, learning_cases
    from williams_ai.book_rules import registry

    cases = learning_cases()  # Validates case structure and page-reviewed rule IDs.
    books = {x["id"] for x in registry()["books"]}
    if set(x["edition"] for x in cases) != set(ED) or books != set(ED):
        raise ValueError("ALL_FOUR_SOURCE_EDITION_EXERCISES_REQUIRED")
    # Entire rule IDs (not random near-duplicate questions) stay in one split.
    held = {ed: set(sorted({x["rule_id"] for x in cases if x["edition"] == ed})[-2:])
            for ed in ED}
    train, validation = [], {ed: [] for ed in ED}
    for case in cases:
        options = "\n".join(f"{key}) {case['choices'][key]}" for key in ("A", "B", "C"))
        row = {
            "edition": case["edition"],
            "rule_id": case["rule_id"],
            "pdf_page": next(card["pdf_pages"][0] for card in registry()["rule_cards"]
                             if card["id"] == case["rule_id"]),
            "question": case["question"] + "\nВарианты:\n" + options
                        + "\nВыбери и объясни один вариант, используя правило книги.",
            "answer": ("Выбор: " + case["correct"] + ". "
                       + case["choices"][case["correct"]] + " "
                       + case["feedback"]),
        }
        if case["rule_id"] in held[case["edition"]]:
            validation[case["edition"]].append(row)
        else:
            train.append(row)
    if not train or any(not v for v in validation.values()):
        raise ValueError("FOUR_EDITION_TRAIN_VALIDATION_REQUIRED")
    rule_train = {x["rule_id"] for x in train}
    rule_val = {x["rule_id"] for ed in ED for x in validation[ed]}
    if rule_train & rule_val or any(ed not in {x["edition"] for x in train} for ed in ED):
        raise ValueError("SOURCE_RULE_LEAKAGE_OR_MISSING_TRAIN_BOOK")
    source = {"public_curriculum_sha256": sha256(CURRICULUM_PATH),
              "train_count": len(train),
              "validation_count": sum(len(x) for x in validation.values()),
              "train_editions": dict(Counter(x["edition"] for x in train)),
              "validation_editions": {k: len(v) for k, v in validation.items()},
              "train_rule_ids_sha256": hashlib.sha256(json.dumps(sorted(rule_train)).encode()).hexdigest(),
              "validation_rule_ids_sha256": hashlib.sha256(json.dumps(sorted(rule_val)).encode()).hexdigest(),
              "rule_ids_not_shared_across_splits": True,
              "source": "PUBLIC_PARAPHRASED_REVIEWED_PAGE_CARDS_NOT_LICENSED_PDFS",
              "independent_expert_chart_review": False}
    return train, validation, source


def train_source_adapter(output: Path, *, steps: int = 6) -> dict:
    if (type(steps) is not int or not 2 <= steps <= 10
            or output.exists() or output.is_symlink()):
        raise ValueError("BOUNDED_NEW_RESEARCH_OUTPUT_REQUIRED")
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from williams_ai.private_lora import _tokenized
    from williams_ai.lora_training import LoraPlan, fit_lora

    if torch.cuda.is_available():
        raise ValueError("THIS_RUN_IS_EXPLICIT_CPU_SOURCE_RESEARCH_ONLY")
    torch.set_num_threads(2)
    train, validation, provenance = build_public_exercises()

    # Download official Qwen model; resolve an immutable revision and check weights.
    revision = HfApi().model_info(MODEL_REPO).sha
    with tempfile.TemporaryDirectory(prefix="williams-public-qwen-", dir="/tmp") as temp:
        root = Path(temp)
        base = Path(snapshot_download(
            MODEL_REPO, revision=revision, local_dir=root / "base",
            allow_patterns=("model.safetensors", "config.json", "tokenizer.json",
                            "tokenizer_config.json", "vocab.json", "merges.txt",
                            "generation_config.json", "special_tokens_map.json")))
        if sha256(base / "model.safetensors") != MODEL_SHA256:
            raise ValueError("UNEXPECTED_PUBLIC_PRETRAINED_QWEN_WEIGHTS")
        tokenizer = AutoTokenizer.from_pretrained(
            base, local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        token_train = _tokenized(train, tokenizer, max_tokens=512)
        token_val = {ed: _tokenized(validation[ed], tokenizer, max_tokens=512)
                     for ed in ED}
        if not token_train or any(not rows for rows in token_val.values()):
            raise ValueError("NONEMPTY_PUBLIC_TRAIN_AND_VALIDATION_REQUIRED")
        model = AutoModelForCausalLM.from_pretrained(
            base, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32)
        # Uses the existing tested PEFT fit/validation implementation; actual
        # pretrained frozen Qwen instead of randomly initialized fake Qwen.
        fitted, metrics = fit_lora(
            model, token_train, token_val, workdir=root / "fit",
            pad_token_id=tokenizer.pad_token_id, max_steps=steps,
            plan=LoraPlan(rank=4, learning_rate=2e-4,
                          gradient_accumulation=1, seed=20261026), use_cpu=True)
        if not metrics["adapter_weights_changed"] or metrics["optimizer_steps"] != steps:
            raise ValueError("PRETRAINED_QWEN_LORA_DID_NOT_TRAIN")
        output.mkdir(parents=True, mode=0o700, exist_ok=False)
        fitted.save_pretrained(output / "adapter", safe_serialization=True)
        if not (output / "adapter" / "adapter_model.safetensors").is_file():
            raise ValueError("REAL_ADAPTER_WEIGHTS_NOT_SAVED")
        report = {
            "mode": "ACTUAL_PRETRAINED_QWEN03_06B_PUBLIC_SOURCE_LORA_RESEARCH",
            "model_repo": MODEL_REPO, "model_revision": revision,
            "model_weights_sha256": MODEL_SHA256,
            "public_training": provenance,
            "steps": metrics["optimizer_steps"],
            "trainable_parameters": metrics["trainable_parameters"],
            "adapter_weights_changed": metrics["adapter_weights_changed"],
            "adapter_weights_sha256": sha256(output / "adapter" / "adapter_model.safetensors"),
            "baseline_validation_loss_by_book": metrics["baseline_by_book"],
            "after_validation_loss_by_book": metrics["adapted_by_book"],
            "validation_gate": metrics["validation_gate"],
            "validation_is_small_public_practice_not_untouched_market_holdout": True,
            "trained_original_private_books": False,
            "model_understands_all_four_books": False,
            "real_market_reasoning_improvement_proven": False,
            "android_gguf_adapter_converted": False,
            "automatic_promotion_to_trading": False,
            "trade_action": None, "execution_allowed": False,
            "source_commit": os.environ.get("SOURCE_HEAD", os.environ.get("GITHUB_SHA")),
        }
        (output / "research_receipt.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research-only actual pretrained Qwen3-0.6B LoRA CPU starting session")
    parser.add_argument("--new-output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    train_source_adapter(args.new_output, steps=args.steps)


if __name__ == "__main__":
    main()
