"""Authenticate the frozen historical v5 adapter before private archival."""
from hashlib import sha256
import json
import math
from pathlib import Path
import struct

EXPECTED_ADAPTER_SHA256 = "8bd26590df627c7452d56e81df6a4572885bfc82515259a7c9de4a3970c850b8"
BASE = "Qwen/Qwen3-4B"
REV = "3fb356690bbdae7353c94efb3eecdce276e6ed1c"
RANK = 8
TARGETS = {"q_proj": (4096, 2560), "k_proj": (1024, 2560),
           "v_proj": (1024, 2560), "o_proj": (2560, 4096)}


def _unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("V5_DUPLICATE_JSON_KEY")
        obj[key] = value
    return obj


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique)


def validate_tensor_bytes(raw):
    """Validate exact PEFT attention matrices and complete finite data payload."""
    if len(raw) < 8:
        raise ValueError("V5_TRUNCATED_SAFETENSORS")
    size = struct.unpack_from("<Q", raw)[0]
    if not 2 <= size <= min(1_000_000, len(raw) - 8):
        raise ValueError("V5_BAD_SAFETENSORS_HEADER")
    header = _json(raw[8:8 + size])
    if not isinstance(header, dict):
        raise ValueError("V5_BAD_SAFETENSORS_HEADER")
    header.pop("__metadata__", None)
    expected = {}
    for layer in range(36):
        for target, (out_dim, in_dim) in TARGETS.items():
            prefix = f"base_model.model.model.layers.{layer}.self_attn.{target}"
            expected[prefix + ".lora_A.weight"] = [RANK, in_dim]
            expected[prefix + ".lora_B.weight"] = [out_dim, RANK]
    if set(header) != set(expected):
        raise ValueError("V5_UNEXPECTED_TENSOR_INVENTORY")
    data = memoryview(raw)[8 + size:]
    spans = []
    for name, shape in expected.items():
        item = header[name]
        if item.get("shape") != shape:
            raise ValueError("V5_WRONG_LORA_SHAPE")
        dtype = item.get("dtype")
        types = {"F32": "<f", "F16": "<e", "BF16": "<H"}
        if dtype not in types:
            raise ValueError("V5_UNSUPPORTED_LORA_DTYPE")
        start, end = item["data_offsets"]
        dt = types[dtype]
        if (type(start) is not int or type(end) is not int or
                not 0 <= start < end <= len(data) or
                end - start != shape[0] * shape[1] * struct.calcsize(dt)):
            raise ValueError("V5_INVALID_TENSOR_OFFSETS")
        values = struct.iter_unpack(dt, data[start:end])
        finite = (all((x & 0x7f80) != 0x7f80 for (x,) in values) if dtype == "BF16"
                  else all(math.isfinite(x) for (x,) in values))
        if not finite:
            raise ValueError("V5_NONFINITE_LORA")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("V5_TENSOR_GAP_OR_OVERLAP")
        cursor = end
    if cursor != len(data):
        raise ValueError("V5_TRAILING_TENSOR_DATA")
    return len(expected)


def verify_recovered_identity(receipt_path: Path, weight_path: Path, config_path: Path):
    receipt = _json(receipt_path.read_bytes())
    raw = weight_path.read_bytes()
    actual = sha256(raw).hexdigest()
    if actual != EXPECTED_ADAPTER_SHA256 or receipt.get("adapter_sha256") != actual:
        raise ValueError("V5_FROZEN_ADAPTER_DIGEST_MISMATCH")
    if (receipt.get("base_model") != BASE or receipt.get("model_commit") != REV or
            receipt.get("optimizer_steps") != 80 or
            receipt.get("selected_validation_checkpoint") != 80):
        raise ValueError("V5_FROZEN_RECEIPT_MISMATCH")
    cfg = _json(config_path.read_bytes())
    if (cfg.get("base_model_name_or_path") != BASE or cfg.get("peft_type") != "LORA" or
            cfg.get("r") != RANK or cfg.get("lora_alpha") != 16 or
            set(cfg.get("target_modules") or []) != set(TARGETS) or
            cfg.get("rank_pattern") or cfg.get("alpha_pattern")):
        raise ValueError("V5_FROZEN_CONFIG_MISMATCH")
    return {"adapter_sha256": actual, "tensor_count": validate_tensor_bytes(raw),
            "all_finite": True, "optimizer_steps": 80}
