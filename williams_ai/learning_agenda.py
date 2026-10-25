"""Choose original pages from past rule gaps, never reuse invented lessons.

Only already observed, same-model/same-edition events can select source
rules. The agenda is a reading priority, not proof of mastery or training.
Current original text is reloaded, so OCR changes cannot lend stale notes
trading authority. No market outcomes or generated summaries are returned.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sqlite3
from typing import Any

from .book_index import _private_database, _known_book
from .book_rules import registry
from .decision_contract import _as_utc


def prior_reading_focus(memory_path: Path, *, model_id: str, edition: str,
                        as_of: str, source_kind: str = "PRIVATE_INDEXED_PDF") -> dict[str, Any]:
    if source_kind not in {"PRIVATE_INDEXED_PDF","PUBLIC_REVIEWED_RULE_PARAPHRASE"}:
        raise ValueError("UNKNOWN_READING_AGENDA_SOURCE_KIND")
    cutoff = _as_utc(as_of)
    if cutoff is None:
        raise ValueError("READING_AGENDA_REQUIRES_AWARE_CUTOFF")
    known = _known_book(edition)
    allowed = {x["id"] for x in registry()["rule_cards"]
               if x["source"] == edition and x["status"] == "REVIEWED_PAGES"}
    result = {"rule_ids": [], "topic": "NONE", "question": "", "past_requests": 0,
              "source_sha256": known["sha256"],
              "unverified_summaries_used": False, "mastery_certified": False}
    path = _private_database(memory_path)
    if not path.is_file():
        return result
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                          "AND name='decision_study_events'").fetchone():
            return result
        # SQL limits work; exact UTC comparison below also rejects equality
        # and malformed timestamps. Future events cannot crowd out past ones.
        rows = db.execute("SELECT case_sha256,cutoff,diagnostics_json "
            "FROM decision_study_events WHERE model_id=? AND edition=? "
            "AND julianday(cutoff)<julianday(?) "
            "ORDER BY julianday(cutoff) DESC,event_id LIMIT 2000",
            (model_id, edition, as_of)).fetchall()
    counts: dict[str, Counter[str]] = {}
    recency: dict[tuple[str,str], float] = {}
    topics: Counter[str] = Counter()
    questions: dict[str,tuple[float,str]] = {}
    seen: set[str] = set()
    for case, at, raw in rows:
        when = _as_utc(at)
        if when is None or when >= cutoff or case in seen or not isinstance(raw, str) or len(raw) > 20000:
            continue
        try:
            event = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if (not isinstance(event, dict)
                or event.get("source_sha256") != known["sha256"]
                or event.get("source_kind") != source_kind
                or event.get("mode") != "SOURCE_DERIVED_DECISION_LEARNING_NOT_WEIGHT_TRAINING"):
            continue
        rules = event.get("source_rules")
        need = event.get("learning_need")
        failures = event.get("first_failures")
        if (not isinstance(rules, list) or len(rules) > 2
                or any(not isinstance(x, str) or x not in allowed for x in rules)
                or not isinstance(need,str) or need not in {"NONE", "ENTRY", "MANAGEMENT", "EXIT", "WAVE"}
                or not isinstance(failures, list) or len(failures) > 80
                or not (failures or need != "NONE") or not (rules or need == "WAVE")):
            continue
        seen.add(case)
        topic=need if need != "NONE" else "ENTRY"
        topics[topic] += 1
        counts.setdefault(topic,Counter())
        question=event.get("private_learning_question","")
        if (isinstance(question,str) and 0<len(question.strip())<=400
                and (topic not in questions or when>questions[topic][0])):
            questions[topic]=(when,question.strip())
        for rule in set(rules):
            counts[topic][rule] += 1
            recency[(topic,rule)] = max(recency.get((topic,rule), when), when)
    result["past_requests"] = len(seen)
    if topics:
        topic=sorted(topics, key=lambda x: (-topics[x], x))[0]
        result["topic"] = topic
        result["question"]=questions.get(topic,(0,""))[1]
        chosen=counts[topic]
        result["rule_ids"] = sorted(chosen, key=lambda x: (-chosen[x], -recency[(topic,x)], x))[:2]
    return result
