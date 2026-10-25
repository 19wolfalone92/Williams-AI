"""Dependency-free integrity and validation gates for private LoRA research."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

EDITIONS = ("TC1", "NTD", "TC2", "COURSE")


def strict_json(text: str | bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("DUPLICATE_JSON_FIELD")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("NONFINITE_JSON_VALUE")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def regular_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("REGULAR_LOCAL_MODEL_FILE_REQUIRED")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_hf_identity(base: Path) -> dict[str, Any]:
    """Fingerprint actual safetensors, shard map and tokenizer, not just config.

    This binds local bytes. It does not authenticate a model publisher or
    establish equivalence with a separately supplied quantized GGUF.
    """
    from .private_lora import checked_local_qwen_config
    config = checked_local_qwen_config(base)
    single, index = base / "model.safetensors", base / "model.safetensors.index.json"
    if single.exists() == index.exists():
        raise ValueError("EXACTLY_ONE_SAFETENSORS_LAYOUT_REQUIRED")
    names = {"config.json", "tokenizer_config.json"}
    if (base / "tokenizer.json").exists():
        names.add("tokenizer.json")
    elif (base / "vocab.json").exists() and (base / "merges.txt").exists():
        names.update(("vocab.json", "merges.txt"))
    else:
        raise ValueError("LOCAL_QWEN_TOKENIZER_REQUIRED")
    if single.exists():
        names.add(single.name)
        if list(base.glob("model-*-of-*.safetensors")):
            raise ValueError("AMBIGUOUS_LOCAL_WEIGHT_FILES")
    else:
        if index.is_symlink() or index.stat().st_size > 20_000_000:
            raise ValueError("BOUNDED_LOCAL_SAFETENSORS_INDEX_REQUIRED")
        obj = strict_json(index.read_bytes())
        mapping = obj.get("weight_map") if isinstance(obj, dict) else None
        if (not isinstance(mapping, dict) or not mapping or
                any(not isinstance(k, str) or not k or not isinstance(v, str)
                    or not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", v)
                    for k, v in mapping.items())):
            raise ValueError("INVALID_LOCAL_SAFETENSORS_SHARD_MAP")
        shards = set(mapping.values())
        if shards != {p.name for p in base.glob("model-*-of-*.safetensors")}:
            raise ValueError("MISSING_OR_UNINDEXED_LOCAL_WEIGHT_SHARD")
        names.update(shards)
        names.add(index.name)
    for optional in ("generation_config.json", "special_tokens_map.json", "added_tokens.json",
                     "vocab.json", "merges.txt", "chat_template.jinja"):
        if (base / optional).exists():
            names.add(optional)
    files = {name: regular_sha256(base / name) for name in sorted(names)}
    if files["config.json"] != config["config_sha256"]:
        raise ValueError("LOCAL_BASE_CHANGED_DURING_FINGERPRINT")
    identity = hashlib.sha256(json.dumps(files, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    return {"format": "LOCAL_QWEN_SAFETENSORS_IDENTITY_V1", "sha256": identity,
            "config": config, "files": files}


def validation_admission(before: dict[str, float], after: dict[str, float], *,
                         minimum_relative_improvement: float = 0.01,
                         maximum_book_regression: float = 0.02) -> dict[str, Any]:
    expected = {"ALL", *EDITIONS}
    if (not isinstance(before, dict) or not isinstance(after, dict) or
            set(before) != expected or set(after) != expected or
            any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
                for scores in (before, after) for x in scores.values())):
        raise ValueError("FINITE_FOUR_BOOK_VALIDATION_REQUIRED")
    if (type(minimum_relative_improvement) not in (int, float) or
            not 0.001 <= minimum_relative_improvement <= 0.2 or
            type(maximum_book_regression) not in (int, float) or
            not 0 <= maximum_book_regression <= 0.1):
        raise ValueError("BOUNDED_VALIDATION_GATE_REQUIRED")
    reasons = []
    improvement = (before["ALL"] - after["ALL"]) / max(before["ALL"], 1e-12)
    if improvement < minimum_relative_improvement:
        reasons.append("INSUFFICIENT_VALIDATION_IMPROVEMENT")
    regressed = [ed for ed in EDITIONS
                 if after[ed] > before[ed] * (1 + maximum_book_regression)]
    if regressed:
        reasons.append("BOOK_VALIDATION_REGRESSION")
    return {"accepted": not reasons, "reasons": reasons, "regressed_books": regressed,
            "relative_improvement": improvement,
            "minimum_relative_improvement": minimum_relative_improvement,
            "maximum_book_regression": maximum_book_regression}
