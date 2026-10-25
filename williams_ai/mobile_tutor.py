"""The Android tutor prompt/protocol mirrored for actual CPU model diagnostics.

Project practice questions and public paraphrases are not an independent expert
exam. External source memory and source-assisted answers do not update weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adaptive_coach import learning_cases
from .book_rules import registry
from .local_model import _final_learning_json

INSTRUCTIONS = Path(__file__).resolve().parents[1] / "knowledge/MOBILE_TUTOR_PROMPT_RU.txt"


def practice_cases(per_edition: int = 2) -> tuple[dict, ...]:
    if type(per_edition) is not int or not 1 <= per_edition <= 4:
        raise ValueError("BOUNDED_MOBILE_PRACTICE_REQUIRED")
    all_cases = learning_cases()
    return tuple(c for ed in ("TC1", "NTD", "TC2", "COURSE")
                 for c in [c for c in all_cases if c["edition"] == ed][:per_edition])


def public_source(case: dict) -> dict:
    card = next(c for c in registry()["rule_cards"] if c["id"] == case["rule_id"])
    if card["source"] != case["edition"] or card["status"] != "REVIEWED_PAGES":
        raise ValueError("CROSS_EDITION_MOBILE_SOURCE")
    return {"book": card["source"], "page": card["pdf_pages"][0], "text": card["summary"]}


def prompt(case: dict, source: dict | None) -> str:
    if source is not None:
        card = next(c for c in registry()["rule_cards"] if c["id"] == case["rule_id"])
        if (source["book"] != case["edition"] or source["page"] not in card["pdf_pages"]
                or not 12 <= len(source["text"]) <= 3000):
            raise ValueError("SOURCE_MUST_MATCH_MOBILE_CASE")
    cited = (f'[{source["book"]}, PDF с. {source["page"]}]\n{source["text"]}' if source
             else "Не предоставлен; ответь по имеющимся знаниям.")
    # Deliberate allowlist: neither correct, feedback, nor an answer-derived
    # explanation can enter this prompt. Rendering matches BookPractice.kt.
    return (INSTRUCTIONS.read_text().strip() + "\nИздание: " + case["edition"] +
            "\nВопрос: " + case["question"] +
            "".join("\n" + k + ": " + case["choices"][k] for k in ("A", "B", "C")) +
            "\nИсточник:\n" + cited + "\n/no_think")


def parse_answer(raw: str, finish_reason: str | None = None) -> dict[str, str]:
    if finish_reason == "length":
        raise ValueError("MODEL_OUTPUT_TRUNCATED")
    if not isinstance(raw, str) or not 1 <= len(raw) <= 4000:
        raise ValueError("INVALID_MOBILE_ANSWER")

    def unique(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError("DUPLICATE_MOBILE_ANSWER_KEY")
            obj[k] = v
        return obj

    answer = json.loads(_final_learning_json(raw), object_pairs_hook=unique)
    if (set(answer) != {"choice", "reason"} or not isinstance(answer["choice"], str)
            or answer["choice"] not in {"A", "B", "C"}
            or not isinstance(answer["reason"], str) or not 1 <= len(answer["reason"]) <= 180):
        raise ValueError("INVALID_MOBILE_ANSWER")
    return answer
