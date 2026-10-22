"""Source-specific Bill Williams rule memory, with verifiable PDF page pointers.

This is a compact *paraphrased* rule registry, not a model trained on books.
A source's coverage is never inferred from its presence here.
"""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

_REGISTRY = Path(__file__).resolve().parent.parent / "knowledge" / "SOURCE_RULE_CARDS.json"
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)

@lru_cache(maxsize=1)
def registry() -> dict[str, Any]:
    obj = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    assert len({x["id"] for x in obj["rule_cards"]}) == len(obj["rule_cards"])
    assert len({x["id"] for x in obj["books"]}) == 4
    by_book = {x["id"]: x for x in obj["books"]}
    for rule in obj["rule_cards"]:
        assert rule["source"] in by_book
        assert rule["pdf_pages"]
        assert min(rule["pdf_pages"]) >= 1
        assert max(rule["pdf_pages"]) <= by_book[rule["source"]]["pages"]
        assert rule["status"] in {"REVIEWED_PAGES", "PRIOR_AUDIT_NOT_YET_RECHECKED"}
    return obj


def lookup_rules(question: str, *, edition: str | None = None,
                 max_cards: int = 5) -> list[dict[str, Any]]:
    """Offline, sparse search. Return only paraphrases with page references."""
    if not isinstance(question, str) or len(question) > 1500:
        raise ValueError("Invalid question")
    if edition is not None and edition not in {"TC1", "NTD", "TC2", "COURSE"}:
        raise ValueError("Unknown book/profile")
    if not 1 <= max_cards <= 12:
        raise ValueError("max_cards out of range")
    terms = {x.casefold() for x in _WORDS.findall(question) if len(x) >= 2}
    if not terms:
        return []
    ranked = []
    for card in registry()["rule_cards"]:
        if edition and card["source"] != edition:
            continue
        haystack = {x.casefold() for x in _WORDS.findall(
            card["id"] + " " + card["title"] + " " + card["summary"]
        )}
        score = len(terms & haystack)
        if score:
            ranked.append((score, card))
    ranked.sort(key=lambda x: (-x[0], x[1]["source"], x[1]["id"]))
    return [
        {"source": f'{c["source"]}: PDF {",".join(map(str,c["pdf_pages"]))}',
         "rule_id": c["id"], "status": c["status"],
         "excerpt": c["summary"], "confidence": "SOURCE_SCOPED_NOT_MODEL_VALIDATED"}
        for _, c in ranked[:max_cards]
    ]


def coverage() -> dict[str, Any]:
    """Known rule cards are NOT a percentage of full books."""
    pack = registry()
    result = {}
    for book in pack["books"]:
        cards = [x for x in pack["rule_cards"] if x["source"] == book["id"]]
        reviewed = [x for x in cards if x["status"] == "REVIEWED_PAGES"]
        result[book["id"]] = {
            "book_pages": book["pages"],
            "rule_cards": len(cards),
            "directly_page_checked_cards": len(reviewed),
            "book_coverage_percent": None,
            "accuracy_percent": None,
        }
    return result
