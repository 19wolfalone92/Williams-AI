"""Bind actual recovered 80-step v5 Kaggle LoRA to a converted GGUF (CPU-only).

Historical v5 Kaggle receipt is NOT compatible with the other local-LoRA
`model_bundle.runtime_identity` receipt schema. Do not forge that schema or
skip checks: create a separate exact historic receipt binding, locally.
Decryption and conversion are performed OFFLINE outside this module using
pinned llama.cpp conversion. No keys, weights or PDFs are exported.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re

BASE_SHA = "f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a"
RAW_ADAPTER_SHA = "8bd26590df627c7452d56e81df6a4572885bfc82515259a7c9de4a3970c850b8"
ORIGINAL_HEAD_SHA = "e3f1b4d34059f10adb1b2791267bb11b79ec67c1"
MODEL_COMMIT = "3fb356690bbdae7353c94efb3eecdce276e6ed1c"
CONVERTER_COMMIT = "970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc"
SCHEMA = "WILLIAMS_V6_HISTORIC_V5_80STEP_GGUF_BINDING_V1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def need(ok: bool, code: str):
    if not ok:
        raise ValueError("V6_V5_LORA_BINDING_" + code)


def hash_file(path: str | Path) -> str:
    p = Path(path).resolve(strict=True)
    need(p.is_file() and not p.is_symlink(), "REGULAR_FILE_REQUIRED")
    sha = sha256()
    with p.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verified_gguf(path: str | Path) -> str:
    p = Path(path)
    need(p.suffix.lower() == ".gguf" and p.is_file()
         and not p.is_symlink(), "ACTUAL_GGUF_FILE_REQUIRED")
    with p.open("rb") as fp:
        need(fp.read(4) == b"GGUF", "INVALID_GGUF_MAGIC")
    return hash_file(p)


def verified_receipt(receipt_path: str | Path, raw_adapter: str | Path | None = None) -> dict:
    path = Path(receipt_path)
    need(path.is_file() and not path.is_symlink()
         and path.stat().st_size < 120000, "PRIVATE_TRAINING_RECEIPT_REQUIRED")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    need(type(receipt) is dict and
         receipt.get("schema") == "WILLIAMS_V5_GENUINE_QWEN3_4B_SOURCE_ONLY_LORA" and
         receipt.get("base_model") == "Qwen/Qwen3-4B" and
         receipt.get("model_commit") == MODEL_COMMIT and
         receipt.get("adapter_sha256") == RAW_ADAPTER_SHA and
         receipt.get("optimizer_steps") == 80 and
         receipt.get("selected_validation_checkpoint") == 80 and
         receipt.get("test_cases_provided_to_trainer") == 0 and
         receipt.get("test_gold_examined_during_checkpoint_selection") is False and
         receipt.get("expert_gold_market_examples") == 0 and
         receipt.get("execution_allowed") is False and
         receipt.get("trading_qualification") is False and
         receipt.get("weights_changed") is True and
         receipt.get("source_pdf_or_ocr_uploaded") is False,
         "HISTORICAL_V5_RECEIPT_NOT_PROVEN")
    before = receipt.get("validation_baseline_per_edition", {})
    after = receipt.get("validation_after_per_edition", {})
    need(type(before) is dict and type(after) is dict and
         all(type(before.get(b)) in (float, int) and
             type(after.get(b)) in (float, int) and
             math.isfinite(before[b]) and math.isfinite(after[b])
             and 0 <= after[b] < before[b]
             for b in ("TC1", "NTD", "TC2", "COURSE")),
         "FOUR_BOOK_VALIDATION_RECEIPT_INVALID")
    if raw_adapter is not None:
        need(hash_file(raw_adapter) == RAW_ADAPTER_SHA,
             "RECOVERED_RAW_ADAPTER_BYTES_MISMATCH")
    return receipt


def create_binding(*, base: Path, lora_gguf: Path, raw_safetensors: Path,
                   training_receipt: Path) -> dict:
    need(verified_gguf(base) == BASE_SHA, "WRONG_BASE_GGUF")
    lora_sha = verified_gguf(lora_gguf)
    verified_receipt(training_receipt, raw_safetensors)
    return {
        "schema": SCHEMA, "base_sha256": BASE_SHA,
        "lora_gguf_sha256": lora_sha, "raw_safetensors_sha256": RAW_ADAPTER_SHA,
        "training_receipt_sha256": hash_file(training_receipt),
        "original_v5_training_head_sha": ORIGINAL_HEAD_SHA,
        "conversion_pinned_llama_cpp_commit": CONVERTER_COMMIT,
        "source_pdf_or_ocr_exported": False,
        "test_gold_exposed_to_conversion": False,
        "weights_changed_or_retrained_in_this_step": False,
        "execution_allowed": False, "gpu_used": False,
        "main_merge_allowed": False,
    }


def verify_binding(*, base: Path, lora_gguf: Path,
                   binding_file: Path) -> str:
    file = Path(binding_file)
    need(file.is_file() and not file.is_symlink()
         and file.stat().st_size < 20000, "SMALL_PRIVATE_BINDING_REQUIRED")
    data = json.loads(file.read_text(encoding="utf-8"))
    fields = {
        "schema", "base_sha256", "lora_gguf_sha256", "raw_safetensors_sha256",
        "training_receipt_sha256", "original_v5_training_head_sha",
        "conversion_pinned_llama_cpp_commit", "source_pdf_or_ocr_exported",
        "test_gold_exposed_to_conversion", "weights_changed_or_retrained_in_this_step",
        "execution_allowed", "gpu_used", "main_merge_allowed"}
    need(type(data) is dict and set(data) == fields and
         data["schema"] == SCHEMA and
         data["base_sha256"] == BASE_SHA and
         data["raw_safetensors_sha256"] == RAW_ADAPTER_SHA and
         data["original_v5_training_head_sha"] == ORIGINAL_HEAD_SHA and
         data["conversion_pinned_llama_cpp_commit"] == CONVERTER_COMMIT and
         all(data[k] is False for k in (
             "source_pdf_or_ocr_exported", "test_gold_exposed_to_conversion",
             "weights_changed_or_retrained_in_this_step", "execution_allowed",
             "gpu_used", "main_merge_allowed")),
         "HISTORIC_BINDING_CONTRACT_CHANGED")
    need(verified_gguf(base) == BASE_SHA, "WRONG_BASE_GGUF")
    digest = verified_gguf(lora_gguf)
    need(data["lora_gguf_sha256"] == digest, "CONVERTED_LORA_SHA_MISMATCH")
    receipt_file = file.parent / "training_receipt.json"
    need(hash_file(receipt_file) == data["training_receipt_sha256"],
         "TRAINING_RECEIPT_CHANGED")
    verified_receipt(receipt_file)
    return digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--lora-gguf", type=Path, required=True)
    p.add_argument("--raw-safetensors", type=Path, required=True)
    p.add_argument("--training-receipt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    need(not args.out.exists(), "DO_NOT_OVERWRITE_PRIVATE_MANIFEST")
    need(args.training_receipt.parent.resolve() == args.out.parent.resolve(),
         "BINDING_NEXT_TO_ORIGINAL_RECEIPT_REQUIRED")
    obj = create_binding(base=args.base, lora_gguf=args.lora_gguf,
                         raw_safetensors=args.raw_safetensors,
                         training_receipt=args.training_receipt)
    with args.out.open("x", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, sort_keys=True)
        fp.write("\n")


if __name__ == "__main__":
    main()
