"""One effective Qwen identity: base weights plus an optional bound GGUF LoRA.

Bindings are local integrity/provenance receipts, not expert certification.
No fallback to an unadapted model is permitted after an adapter load error.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any

from .lora_integrity import local_hf_identity, regular_sha256, strict_json

FORMAT = "WILLIAMS_AI_QWEN_GGUF_LORA_V1"
CONVERTER_COMMIT = "970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc"
_HASH = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: str | Path) -> str:
    return regular_sha256(Path(path).expanduser())


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 100_000:
        raise ValueError("SMALL_LOCAL_ADAPTER_RECEIPT_REQUIRED")
    obj = strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("ADAPTER_RECEIPT_OBJECT_REQUIRED")
    return obj


def _training_receipt(path: Path) -> dict[str, Any]:
    receipt = _json(path)
    before, after = (receipt.get(x) for x in
                     ("baseline_validation_loss", "trained_validation_loss"))
    if (receipt.get("mode") != "LOCAL_RESEARCH_LORA_NOT_TRADING_READY"
            or type(receipt.get("optimizer_steps")) is not int or receipt["optimizer_steps"] <= 0
            or not isinstance(receipt.get("source_dataset_sha256"), str)
            or not _HASH.fullmatch(receipt["source_dataset_sha256"])
            or any(not isinstance(receipt.get(k), str) or not _HASH.fullmatch(receipt[k])
                   for k in ("adapter_weights_sha256", "adapter_config_sha256"))
            or type(receipt.get("source_only_research")) is not bool
            or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (before, after))
            or after >= before
            or receipt.get("test_labels_not_supplied_to_trainer") is not True
            or receipt.get("execution_allowed") is not False
            or receipt.get("trade_action", "INVALID") is not None):
        raise ValueError("COMPLETED_RESEARCH_TRAINING_RECEIPT_REQUIRED")
    return receipt


def _exportable_training_receipt(receipt: dict[str, Any]) -> None:
    """Config-only receipts cannot activate or export a new runtime adapter."""
    from .lora_integrity import validation_admission
    identity = receipt.get("local_base_identity")
    if (receipt.get("training_receipt_version") != 2 or not isinstance(identity, dict)
            or identity.get("format") != "LOCAL_QWEN_SAFETENSORS_IDENTITY_V1"
            or not isinstance(identity.get("files"), dict) or not identity["files"]
            or any(not isinstance(name, str) or Path(name).name != name
                   or not isinstance(digest, str) or not _HASH.fullmatch(digest)
                   for name, digest in identity["files"].items())
            or receipt.get("best_validation_checkpoint_reloaded") is not True
            or receipt.get("adapter_weights_changed") is not True
            or receipt.get("base_parameters_frozen") is not True
            or type(receipt.get("selected_checkpoint_step")) is not int
            or not 0 < receipt["selected_checkpoint_step"] <= receipt["optimizer_steps"]):
        raise ValueError("WEIGHT_BOUND_VALIDATED_TRAINING_RECEIPT_V2_REQUIRED")
    digest = hashlib.sha256(json.dumps(identity["files"], sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if digest != identity.get("sha256") or identity.get("config") != receipt.get("local_base_config"):
        raise ValueError("TRAINED_BASE_IDENTITY_CHANGED")
    gate = validation_admission(receipt.get("baseline_by_book", {}),
                                receipt.get("adapted_by_book", {}))
    if (not gate["accepted"] or gate != receipt.get("validation_gate")
            or receipt["baseline_by_book"]["ALL"] != receipt["baseline_validation_loss"]
            or receipt["adapted_by_book"]["ALL"] != receipt["trained_validation_loss"]):
        raise ValueError("FOUR_BOOK_VALIDATION_GATE_NOT_PASSED")


def _convertible_config(config: dict[str, Any]) -> None:
    # The pinned converter exports alpha/r with a single alpha. It does not
    # implement rsLoRA/DoRA, per-layer alpha, aLoRA or separately saved layers.
    unsupported = ("use_rslora", "use_dora", "lora_bias", "fan_in_fan_out",
                   "rank_pattern", "alpha_pattern", "modules_to_save",
                   "alora_invocation_tokens", "invocation_string")
    if (config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM"
            or config.get("bias", "none") != "none"
            or any(config.get(k) for k in unsupported)
            or config.get("init_lora_weights", True) is not True
            or type(config.get("r")) is not int or not 1 <= config["r"] <= 64
            or type(config.get("lora_alpha")) not in (int, float)
            or not 0 < config["lora_alpha"] <= 128):
        raise ValueError("LORA_CONFIGURATION_NOT_SUPPORTED_BY_PINNED_CONVERTER")


def runtime_identity(base: str | Path, *, adapter: str | Path | None = None,
                     manifest: str | Path | None = None) -> dict[str, Any]:
    """Bind memory to actual effective weights, not a filename or model brand."""
    base_path = Path(base).expanduser()
    if base_path.suffix.lower() != ".gguf":
        raise ValueError("GGUF_BASE_REQUIRED")
    digest = file_sha256(base_path)
    if (adapter is None) != (manifest is None):
        raise ValueError("ADAPTER_AND_BINDING_MANIFEST_REQUIRED_TOGETHER")
    common = {"base_sha256": digest, "adapter_sha256": None, "adapter_scale": None,
              "adapter_loaded": False, "training_origin_independently_verified": False,
              "book_mastery_verified": False, "execution_allowed": False, "trade_action": None}
    if adapter is None:
        # Preserve existing base-only notebooks and study checkpoints.
        return {**common, "model_id": "gguf-" + digest[:24], "adapter_state": "NOT_CONFIGURED"}
    from .private_lora import _private_file
    adapter_path = _private_file(adapter)
    manifest_path = _private_file(manifest)
    binding = _json(manifest_path)
    fields = {"format", "base_sha256", "adapter_sha256", "scale", "training_receipt_sha256"}
    if (set(binding) != fields or binding.get("format") != FORMAT
            or binding.get("base_sha256") != digest
            or type(binding.get("scale")) not in (int, float) or binding["scale"] != 1.0
            or any(not isinstance(binding.get(k), str) or not _HASH.fullmatch(binding[k])
                   for k in ("adapter_sha256", "training_receipt_sha256"))):
        raise ValueError("ADAPTER_BASE_OR_BINDING_MISMATCH")
    if adapter_path.suffix.lower() != ".gguf":
        raise ValueError("CONVERT_HF_LORA_TO_GGUF_BEFORE_INFERENCE")
    with adapter_path.open("rb") as source:
        if source.read(4) != b"GGUF":
            raise ValueError("GGUF_ADAPTER_REQUIRED")
    adapted_digest = file_sha256(adapter_path)
    if adapted_digest != binding["adapter_sha256"]:
        raise ValueError("ADAPTER_BYTES_CHANGED")
    receipt_path = manifest_path.parent / "research_manifest.json"
    if file_sha256(receipt_path) != binding["training_receipt_sha256"]:
        raise ValueError("TRAINING_RECEIPT_CHANGED")
    receipt = _training_receipt(receipt_path)
    _exportable_training_receipt(receipt)
    identity = hashlib.sha256(json.dumps({"base": digest, "adapter": adapted_digest,
        "scale": "1.0"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**common, "model_id": "gguf-" + identity[:24],
            "adapter_sha256": adapted_digest, "adapter_scale": 1.0,
            "adapter_state": "BOUND_RESEARCH_ADAPTER", "adapter_loaded": False,
            "source_only_research": receipt.get("source_only_research") is True,
            "optimizer_steps_declared_in_receipt": receipt["optimizer_steps"]}


def export_runtime_lora(*, trained_adapter: Path, hf_base: Path, base_gguf: Path,
                        llama_cpp: Path, output: Path) -> dict[str, Any]:
    """Convert an already trained private PEFT adapter, offline, with pinned code.

    HF/GGUF base equivalence is the operator's provenance declaration; hashes
    enforce the declared binding but cannot prove who trained the adapter.
    Native tensor/architecture validation still belongs to llama.cpp.
    """
    from .private_lora import _private_file
    trained = _private_file(trained_adapter, must_exist=False)
    target = _private_file(output, must_exist=False)
    base = _private_file(hf_base, must_exist=False)
    if (not trained.is_dir() or not base.is_dir() or target.exists()
            or target == trained or trained in target.parents or target in trained.parents
            or target == base or base in target.parents or target in base.parents):
        raise ValueError("PRIVATE_ADAPTER_BASE_AND_NEW_OUTPUT_REQUIRED")
    receipt_path = trained / "research_manifest.json"
    receipt = _training_receipt(receipt_path)
    receipt_bytes = receipt_path.read_bytes()
    if strict_json(receipt_bytes) != receipt:
        raise ValueError("TRAINING_RECEIPT_CHANGED")
    _exportable_training_receipt(receipt)
    base_identity = local_hf_identity(base)
    if receipt["local_base_identity"] != base_identity:
        raise ValueError("TRAINED_HF_BASE_WEIGHTS_OR_TOKENIZER_MISMATCH")
    weights = trained / "adapter_model.safetensors"
    if not weights.is_file() or weights.is_symlink():
        raise ValueError("SAFETENSORS_LORA_REQUIRED_NO_PICKLE_FALLBACK")
    if (file_sha256(weights) != receipt["adapter_weights_sha256"]
            or file_sha256(trained / "adapter_config.json") != receipt["adapter_config_sha256"]):
        raise ValueError("TRAINED_ADAPTER_ARTIFACT_CHANGED")
    config = _json(trained / "adapter_config.json")
    _convertible_config(config)
    checkout = Path(llama_cpp).expanduser().resolve()
    commit = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    clean = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
                           capture_output=True, text=True, check=True).stdout
    converter = checkout / "convert_lora_to_gguf.py"
    if commit != CONVERTER_COMMIT or clean or not converter.is_file() or converter.is_symlink():
        raise ValueError("CLEAN_PINNED_LLAMA_CPP_CONVERTER_REQUIRED")
    base_digest = file_sha256(base_gguf)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with TemporaryDirectory(prefix=".williams-convert-", dir=target.parent) as scratch:
        private = Path(scratch)
        converted = private / "adapter.gguf"
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1"}
        result = subprocess.run([sys.executable, str(converter), "--base", str(base),
            "--outtype", "f16", "--outfile", str(converted), str(trained)],
            env=env, capture_output=True, timeout=900)
        if result.returncode or not converted.is_file():
            raise RuntimeError("OFFLINE_LORA_CONVERSION_FAILED_OUTPUT_NOT_ACTIVATED")
        if (file_sha256(weights) != receipt["adapter_weights_sha256"]
                or file_sha256(trained / "adapter_config.json") != receipt["adapter_config_sha256"]
                or receipt_path.read_bytes() != receipt_bytes
                or local_hf_identity(base) != base_identity
                or file_sha256(base_gguf) != base_digest):
            raise ValueError("TRAINED_ARTIFACT_CHANGED_DURING_CONVERSION")
        converted.chmod(0o600)
        saved_receipt = private / "research_manifest.json"
        saved_receipt.write_bytes(receipt_bytes); saved_receipt.chmod(0o600)
        binding = {"format": FORMAT, "base_sha256": base_digest,
                   "adapter_sha256": file_sha256(converted), "scale": 1.0,
                   "training_receipt_sha256": file_sha256(saved_receipt)}
        manifest = private / "runtime_manifest.json"
        manifest.write_text(json.dumps(binding, sort_keys=True, indent=2), encoding="utf-8")
        manifest.chmod(0o600)
        runtime_identity(base_gguf, adapter=converted, manifest=manifest)
        # The binding manifest is the last file published; no partial export
        # can be activated, and no pre-existing destination is overwritten.
        target.mkdir(mode=0o700, exist_ok=False)
        try:
            for name in ("adapter.gguf", "research_manifest.json", "runtime_manifest.json"):
                os.replace(private/name, target/name)
        except BaseException:
            shutil.rmtree(target)
            raise
    converted, manifest = target / "adapter.gguf", target / "runtime_manifest.json"
    return runtime_identity(base_gguf, adapter=converted, manifest=manifest)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Private offline Qwen PEFT -> runtime GGUF LoRA")
    for flag in ("trained-adapter", "hf-base", "base-gguf", "llama-cpp", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(export_runtime_lora(trained_adapter=a.trained_adapter,
        hf_base=a.hf_base, base_gguf=a.base_gguf, llama_cpp=a.llama_cpp, output=a.output), indent=2))


if __name__ == "__main__":
    main()
