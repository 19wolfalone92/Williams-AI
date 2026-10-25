"""Bounded book consultation inside a real chart decision, not weight training.

A model may declare a rule gap; the independent critic may also detect an
unsupported claim. Read the corresponding source pages, checkpoint one or
two source-derived lessons, and ask for ONE reconsideration at the same
closed cutoff. A valid wait is never retried just to manufacture a signal.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any
import uuid

from .algorithm_guard import audit_reasoning
from .book_index import _private_database
from .book_rules import registry
from .grounded_inference import verified_private_book_pages
from .guided_retry import LESSONS, safe_error_code
from .study import memory, recall, study_batch

LEARNING_NEEDS = {"NONE", "ENTRY", "MANAGEMENT", "EXIT", "WAVE"}
_GAPS = {
    "TC2": {"ENTRY": ("TC2.WM1.TRIGGER", "TC2.WM3"),
            "MANAGEMENT": ("TC2.CAMPAIGN", "TC2.WM1.STOP"),
            "EXIT": ("TC2.CAMPAIGN", "TC2.REVERSE"),
            "WAVE": ("TC2.WM1.ANG",)},
    "NTD": {"ENTRY": ("NTD.FRACTAL", "NTD.AO"),
            "MANAGEMENT": ("NTD.ZONE.ADDED", "NTD.EXITS"),
            "EXIT": ("NTD.EXITS",), "WAVE": ("NTD.ALLIGATOR",)},
    "TC1": {"ENTRY": ("TC1.FRACTAL",),
            "MANAGEMENT": ("TC1.PSY.PLAN",),
            "EXIT": ("TC1.W5", "TC1.MAGIC.DIVERGENCE"),
            "WAVE": ("TC1.W4", "TC1.W5")},
    "COURSE": {"ENTRY": ("COURSE.HUMP", "COURSE.THUMB"),
               "MANAGEMENT": ("COURSE.TRADING_PARTNER",),
               "EXIT": ("COURSE.TRADING_PARTNER",),
               "WAVE": ("COURSE.VISUAL.NESTED_ABC", "COURSE.W4.ZERO")},
}
_QUERIES = {
    "ENTRY": "вход пробой подтверждение фрактал trigger entry",
    "MANAGEMENT": "сопровождение защитный стоп перенос stop trailing pyramid",
    "EXIT": "выход стоп закрытие противоположный сигнал exit stop",
    "WAVE": "волна степень коррекция вложенность wave degree correction",
}


class DeliberateBookModel:
    """Use the supplied actual model; retain a separate auditable learning trace."""

    def __init__(self, model: Any, book_index: Path, study_memory: Path,
                 *, model_id: str):
        if not isinstance(model_id, str) or not re.fullmatch(r"gguf-[a-zA-Z0-9_.:-]{1,115}", model_id):
            raise ValueError("Exact GGUF identity required for deliberate study")
        self.book_index = _private_database(book_index)
        self.study_memory = _private_database(study_memory)
        if self.book_index == self.study_memory:
            raise ValueError("Books and learning memory must be separate")
        self.model, self.model_id = model, model_id
        self.latest: dict[str, Any] | None = None

    def _remember(self, observation: dict[str, Any], edition: str) -> None:
        # Store diagnostics and source IDs, never raw book text, a generated
        # "correct answer", future outcomes, or an expert approval.
        case = hashlib.sha256(json.dumps(observation, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        with memory(self.study_memory) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS decision_study_events(
                event_id TEXT PRIMARY KEY, model_id TEXT NOT NULL,
                edition TEXT NOT NULL, case_sha256 TEXT NOT NULL,
                cutoff TEXT NOT NULL, recorded_at TEXT NOT NULL,
                diagnostics_json TEXT NOT NULL)""")
            db.execute("INSERT INTO decision_study_events VALUES(?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, self.model_id, edition, case,
                 observation["as_of"], datetime.now(timezone.utc).isoformat(),
                 json.dumps(self.latest, ensure_ascii=False, allow_nan=False)))

    def propose_chart_reasoning(self, observation: dict[str, Any], *,
                                edition: str = "TC2", **kwargs: Any) -> dict[str, Any]:
        if (edition not in _GAPS or not isinstance(observation, dict)
                or observation.get("execution_allowed") is not False
                or observation.get("gaps", 0)):
            raise ValueError("Gap-free read-only observations required")
        self.latest = trace = {
            "mode": "SOURCE_DERIVED_DECISION_LEARNING_NOT_WEIGHT_TRAINING",
            "first_failures": [], "learning_need": "NONE",
            "source_rules": [], "source_pages": [], "studied_chunks": 0,
            "failed_study_chunks": 0, "recalled_notes": [],
            "reconsideration_performed": False, "final_failures": [],
            "same_closed_cutoff": True, "independent_review_completed": False,
            "model_weights_updated": False, "execution_allowed": False,
            "trade_action": None, "status": "INITIAL_ANALYSIS",
        }
        first = None
        first_error = None
        try:
            first = self.model.propose_chart_reasoning(deepcopy(observation), edition=edition, **kwargs)
            checked = audit_reasoning(observation, first)
            if first.get("edition") != edition:
                raise ValueError("MODEL_EDITION_MISMATCH")
            need = first.get("learning_need", "NONE")
            if not isinstance(need, str) or need not in LEARNING_NEEDS:
                raise ValueError("INVALID_MODEL_LEARNING_NEED")
            trace["learning_need"] = need
            trace["first_failures"] = checked["hard_failures"]
        except (ValueError, TypeError, KeyError) as exc:
            first_error = exc
            trace["first_failures"] = [safe_error_code(exc)]
        # UNKNOWN is a request for clarification; NO_SETUP can be a perfectly
        # competent wait. Missing price data cannot be supplied by a book.
        if (not trace["first_failures"] and trace["learning_need"] == "NONE"
                and first.get("setup") != "UNKNOWN"):
            trace["status"] = "NO_BOOK_GAP_DECLARED_OR_DETECTED"
            self._remember(observation, edition)
            return first
        cards = {x["id"]: x for x in registry()["rule_cards"]
                 if x["source"] == edition and x["status"] == "REVIEWED_PAGES"}
        focused = []
        for failure in trace["first_failures"]:
            if failure in LESSONS and LESSONS[failure][0] in cards:
                focused.append(LESSONS[failure][0])
        need = trace["learning_need"]
        focused.extend(_GAPS[edition].get(need, ()))
        if isinstance(first, dict) and isinstance(first.get("rule_ids"), list):
            focused.extend(x for x in first["rule_ids"] if isinstance(x, str) and x in cards)
        if not focused:
            focused.extend(_GAPS[edition]["ENTRY"])
        focused = list(dict.fromkeys(focused))[:2]
        trace["source_rules"] = focused
        query = _QUERIES.get(need, _QUERIES["ENTRY"])
        try:
            pages = verified_private_book_pages(self.book_index, edition,
                rule_ids=tuple(focused), question=query, limit=2)
        except (ValueError, OSError, sqlite3.DatabaseError):
            trace["status"] = "SOURCE_UNAVAILABLE_OR_IDENTITY_REJECTED"
            self._remember(observation, edition)
            if first_error:
                raise ValueError("DELIBERATE_STUDY_SOURCE_UNAVAILABLE_AFTER_MODEL_ERROR") from first_error
            return first
        if not pages:
            trace["status"] = "SOURCE_TEXT_MISSING_VISUAL_REVIEW_REQUIRED"
            self._remember(observation, edition)
            if first_error:
                raise ValueError("DELIBERATE_STUDY_SOURCE_TEXT_MISSING_AFTER_MODEL_ERROR") from first_error
            return first
        trace["source_pages"] = [x["source"] for x in pages]
        studied = study_batch(self.book_index, self.study_memory, edition, self.model,
            model_id=self.model_id, budget=2,
            pdf_pages=tuple(dict.fromkeys(x["pdf_page"] for x in pages)),
            chunk_keys=tuple((x["pdf_page"],x["chunk_sha256"]) for x in pages))
        trace["studied_chunks"] = studied["studied_chunks"]
        trace["failed_study_chunks"] = studied["failed_chunks"]
        notes = recall(self.study_memory, query, edition=edition, limit=2,
            model_id=self.model_id, book_index=self.book_index)
        trace["recalled_notes"] = [x["source"] for x in notes]
        # Paraphrases retain their own label; source text has priority. A
        # generated lesson never becomes a correction certified by the critic.
        args = {**kwargs, "primary_pages": pages,
                "prior_lessons": [{"source":f"{edition}:RULE:{x}",
                    "excerpt":cards[x]["summary"][:450]} for x in focused],
                "study_lessons":notes, "model_id":self.model_id}
        trace["reconsideration_performed"] = True
        try:
            second = self.model.propose_chart_reasoning(deepcopy(observation), edition=edition, **args)
            final = audit_reasoning(observation, second)
            if second.get("edition") != edition:
                raise ValueError("MODEL_EDITION_MISMATCH")
            trace["final_failures"] = final["hard_failures"]
            trace["status"] = ("UNSUPPORTED_AFTER_STUDY" if final["hard_failures"] else
                "UNRESOLVED_BOOK_QUESTION" if second.get("learning_need", "NONE") != "NONE"
                or second.get("setup") == "UNKNOWN" else
                "RECONSIDERED_NOT_INDEPENDENTLY_VALIDATED")
        except (ValueError, TypeError, KeyError) as exc:
            trace["final_failures"] = [safe_error_code(exc)]
            trace["status"] = "MODEL_ERROR_AFTER_STUDY"
            self._remember(observation, edition)
            raise
        self._remember(observation, edition)
        return second
