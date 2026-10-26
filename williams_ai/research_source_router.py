"""Source-aware research navigation, not proof of model knowledge or trade authority."""
from __future__ import annotations

import re
from typing import Any

# Semantic SIGNAL FAMILIES, not image-template matching.
FAMILIES = {
    "WM1": (("wm1", "первый мудрец", "ангуляц", "divergent bar"), ("TC2.WM1.",)),
    "WM2": (("wm2", "второй мудрец", "super ao"), ("TC2.WM2",)),
    "WM3": (("wm3", "третий мудрец"), ("TC2.WM3",)),
    "ALLIGATOR": (("alligator", "аллигатор", "пасть", "teeth"), ("NTD.ALLIGATOR",)),
    "FRACTAL": (("фрактал", "fractal"), ("TC1.FRACTAL", "NTD.FRACTAL", "TC2.WM3")),
    "MOMENTUM": (("ao", "дивергенц", "осциллятор", "momentum"), ("NTD.AO", "TC2.WM2", "COURSE.AO5345")),
    "ACCELERATION": (("ac", "ускорен", "acceleration"), ("NTD.AC", "NTD.ZONE")),
    "WAVES": (("волн", "wave", "elliott", "abc", "w3", "w4", "w5"), ("TC1.W3", "TC1.W4", "TC1.W5", "TC1.RHYTHM", "COURSE.W4.ZERO", "COURSE.RHYTHM", "COURSE.VISUAL.")),
    "MAGIC_BULLETS": (("волшебн", "пять пуль", "magic bullet", "squat"), ("TC1.MAGIC.",)),
    "CAMPAIGN": (("долив", "пирамид", "кампани", "campaign"), ("TC2.CAMPAIGN", "TC2.REVERSE")),
    "EXIT": (("выход", "закрыти", "стоп", "stop loss", "reverse"), ("NTD.EXITS", "TC2.REVERSE", "TC2.CAMPAIGN")),
    "PSYCHOLOGY": (("психолог", "страх", "жадност", "fomo", "greed"), ("TC1.PSY.",)),
    "BALANCE": (("линия баланса", "balance line"), ("NTD.BALANCE",)),
    "HUMP": (("горб", "hump"), ("COURSE.HUMP",)),
    "THUMB": (("thumb", "большой палец"), ("COURSE.THUMB",)),
}


def source_route(question: str, *, max_cards: int = 12,
                 registry_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Locate reviewed rule IDs and PDF pages, without inventing citations.

    A source pointer is an instruction to OPEN the private original. It is not
    an assertion that Qwen has read the page or that a forecast is correct.
    """
    if not isinstance(question, str) or not question.strip() or len(question) > 1500:
        raise ValueError("BOUNDED_NONEMPTY_QUESTION_REQUIRED")
    if type(max_cards) is not int or not 1 <= max_cards <= 20:
        raise ValueError("INVALID_CARD_LIMIT")
    if registry_data is None:
        from .book_rules import registry
        registry_data = registry()
    rows = registry_data.get("rule_cards") if isinstance(registry_data, dict) else None
    books = registry_data.get("books") if isinstance(registry_data, dict) else None
    if not isinstance(rows, list) or not isinstance(books, list):
        raise ValueError("RULE_REGISTRY_REQUIRED")
    boundaries = {r["id"]: int(r["pages"]) for r in books}
    if set(boundaries) != {"TC1", "NTD", "TC2", "COURSE"}:
        raise ValueError("FOUR_SOURCE_EDITIONS_REQUIRED")
    words = set(re.findall(r"[\w]+", question.casefold()))
    phrase = " " + " ".join(re.findall(r"[\w]+", question.casefold())) + " "
    active = []
    for family, (aliases, _) in FAMILIES.items():
        if any((" " in alias and alias in phrase)
               or (" " not in alias and (alias in words or
                    (len(alias) >= 4 and any(word.startswith(alias) for word in words))))
               for alias in aliases):
            active.append(family)
    found = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("INVALID_RULE_RECORD")
        edition, rid, pages = row.get("source"), row.get("id"), row.get("pdf_pages")
        if (edition not in boundaries or not isinstance(rid, str)
            or not isinstance(pages, list) or not pages
            or any(type(p) is not int or p < 1 or p > boundaries[edition] for p in pages)):
            raise ValueError("INVALID_RULE_SOURCE_REFERENCE")
        if row.get("status") != "REVIEWED_PAGES":
            continue
        hits = [family for family in active if any(
            rid.startswith(prefix) for prefix in FAMILIES[family][1])]
        if hits:
            found.append({"rule_id": rid, "edition": edition,
                "pdf_pages": sorted(set(pages)), "signal_families": hits,
                "source_status": "REVIEWED_PAGES",
                "model_understanding_confirmed": False})
    found.sort(key=lambda r: (-len(r["signal_families"]),r["edition"],r["rule_id"]))
    return {"mode": "REVIEWED_SOURCE_BOOKMARKS_NOT_TRADE_SIGNAL",
            "detected_families": active, "references": found[:max_cards],
            "more_references": max(0,len(found)-max_cards),
            "open_original_private_pages_before_claiming_support": True,
            "nonlinear_chart_requires_competing_hypotheses": True,
            "unverified_market_probability": True,
            "execution_allowed": False, "trade_action": None}
