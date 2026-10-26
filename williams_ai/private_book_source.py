"""Private four-book causal text corpus with immutable page-disjoint holdouts.

No approval/market-label claims. This intentionally does NOT accept a user-
provided 'reviewed' flag in lieu of two independent chart reviewers.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from .book_rules import registry
from .lora_integrity import EDITIONS, strict_json

SPLITS = ("train", "validation", "test")
FIELDS = {"edition", "pdf_page", "text", "extraction_status", "source_sha256"}
FORMAT = "WILLIAMS_USER_PRIVATE_4BOOK_SOURCE_CORPUS_V1"


def _regular(path: Path, max_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("PRIVATE_REGULAR_BOUNDED_SOURCE_REQUIRED")
    return path.read_bytes()


def load_private_source_corpus(directory: Path) -> dict[str, Any]:
    """Audit private page-level corpus WITHOUT sending its text to any service.

    The test cohort is inspected ONLY for integrity and embargo; a caller of
    the training loop is given train/validation but NEVER its test texts.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("PRIVATE_SOURCE_DIRECTORY_REQUIRED")
    from .book_index import ROOT
    if directory == ROOT or ROOT in directory.parents:
        raise ValueError("ORIGINAL_BOOKS_CANNOT_BE_STORED_IN_PUBLIC_REPOSITORY")
    manifest = strict_json(_regular(directory / "manifest.json", 120000))
    if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
            or manifest.get("training_use") != "TEXT_ONLY_CAUSAL_LM_NO_MARKET_HUMAN_LABELS"
            or set(manifest.get("splits", {})) != set(SPLITS)
            or manifest.get("page_boundary_embargo") != 2
            or manifest.get("status") != "SOURCE_ONLY_NOT_EXPERT_APPROVED"
            or manifest.get("all_figures_understood") is not False
            or manifest.get("market_training_or_profitability_verified") is not False):
        raise ValueError("EXPLICIT_SOURCE_ONLY_MANIFEST_REQUIRED")

    expected = {b["id"]: b for b in registry()["books"]}
    if set(expected) != set(EDITIONS) or set(manifest.get("book_registry", {})) != set(EDITIONS):
        raise ValueError("EXACT_FOUR_BOOK_SOURCE_IDS_REQUIRED")
    for edition in EDITIONS:
        record = manifest["book_registry"][edition]
        if record != {"sha256": expected[edition]["sha256"],
                      "pages": expected[edition]["pages"]}:
            raise ValueError("WRONG_OR_UNVERIFIED_ORIGINAL_BOOK_SHA256")

    rows = {s: [] for s in SPLITS}
    pages = {s: {ed: set() for ed in EDITIONS} for s in SPLITS}
    for split in SPLITS:
        payload = _regular(directory / (split + ".jsonl"), 30_000_000)
        if hashlib.sha256(payload).hexdigest() != manifest["splits"][split].get("sha256"):
            raise ValueError("PRIVATE_BOOK_CORPUS_HASH_MISMATCH")
        if len(payload) > 30_000_000:
            raise ValueError("SOURCE_CORPUS_EXCEEDS_LOCAL_BUDGET")
        for line in payload.splitlines():
            if not line.strip():
                continue
            row = strict_json(line)
            if not isinstance(row, dict) or set(row) != FIELDS:
                raise ValueError("SOURCE_PAGE_SCHEMA_MISMATCH")
            ed, page, text = row["edition"], row["pdf_page"], row["text"]
            if (ed not in expected or type(page) is not int
                    or not 1 <= page <= expected[ed]["pages"]
                    or page in pages[split][ed]
                    or not isinstance(text, str) or not 30 <= len(text) <= 250000
                    or row["source_sha256"] != expected[ed]["sha256"]
                    or row["extraction_status"] != (
                        "UNVERIFIED_OCR" if ed == "COURSE" else
                        "PDF_TEXT_NOT_EXPERT_REVIEWED")):
                raise ValueError("INVALID_OR_DUPLICATE_PRIVATE_SOURCE_PAGE")
            pages[split][ed].add(page)
            if split != "test":
                rows[split].append(row)
        actual_counts = {ed: len(pages[split][ed]) for ed in EDITIONS}
        if (manifest["splits"][split].get("rows") != sum(actual_counts.values())
                or manifest["splits"][split].get("books") != actual_counts
                or any(count < 2 for count in actual_counts.values())):
            raise ValueError("FOUR_SOURCE_EDITIONS_IN_EACH_SPLIT_REQUIRED")

    for ed in EDITIONS:
        for i, first in enumerate(SPLITS):
            for second in SPLITS[i + 1:]:
                for a in pages[first][ed]:
                    if any(abs(a - b) <= 2 for b in pages[second][ed]):
                        raise ValueError("SOURCE_PAGE_HOLDOUT_LEAKAGE_WITHIN_TWO_PAGES")
    return {
        "train": rows["train"], "validation": rows["validation"],
        "audit": {
            "source_corpus_format": FORMAT,
            "manifest_sha256": hashlib.sha256(
                _regular(directory / "manifest.json", 120000)).hexdigest(),
            "counts": {s: {ed: len(pages[s][ed]) for ed in EDITIONS} for s in SPLITS},
            "test_texts_loaded_into_training_memory": False,
            "train_and_validation_rules_not_expert_reviewed": True,
            "market_chart_expert_labels_present": False,
            "source_only_not_trading_ready": True,
            "private_book_text_is_never_logged": True,
        }
    }


def tokenize_source_pages(rows: list[dict], tokenizer: Any, *, max_tokens: int = 512
                          ) -> list[dict]:
    """Causal objective; no overlapping windows; response only loss mask.

    The source-text reader is not taught to invent human chart annotations.
    """
    if type(max_tokens) is not int or not 256 <= max_tokens <= 1024:
        raise ValueError("CONSTRAINED_QLORA_CONTEXT_REQUIRED")
    result = []
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int):
        raise ValueError("EOS_TOKEN_REQUIRED")
    for row in rows:
        prefix = (f"Учебный источник: {row['edition']}, PDF страница "
                  f"{row['pdf_page']}. Далее фрагмент оригинального текста.\n")
        prompt = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        sequence = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        width = max_tokens - len(prompt) - 1
        if width < 128:
            raise ValueError("SOURCE_CITATION_PROMPT_TOO_LONG")
        for begin in range(0, len(sequence), width):
            answer = sequence[begin:begin + width] + [eos]
            if len(answer) < 16:
                continue
            ids = prompt + answer
            result.append({"input_ids": ids,
                           "attention_mask": [1] * len(ids),
                           "labels": [-100] * len(prompt) + answer})
    if not result:
        raise ValueError("SOURCE_PAGES_PRODUCED_NO_TRAINING_TOKENS")
    return result
