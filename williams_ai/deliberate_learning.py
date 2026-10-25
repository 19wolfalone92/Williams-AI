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
from .learning_agenda import prior_reading_focus
from .study import memory, recall, study_batch

LEARNING_NEEDS = {"NONE", "ENTRY", "MANAGEMENT", "EXIT", "WAVE"}
_GAPS = {
    "TC2": {"ENTRY": ("TC2.WM1.TRIGGER", "TC2.WM3"),
            "MANAGEMENT": ("TC2.CAMPAIGN", "TC2.WM1.STOP"),
            "EXIT": ("TC2.CAMPAIGN", "TC2.REVERSE"),
            "WAVE": ()},
    "NTD": {"ENTRY": ("NTD.FRACTAL", "NTD.AO"),
            "MANAGEMENT": ("NTD.ZONE.ADDED", "NTD.EXITS"),
            "EXIT": ("NTD.EXITS",), "WAVE": ()},
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

    def __init__(self, model: Any, book_index: Path | None, study_memory: Path,
                 *, model_id: str, public_rules_only: bool = False):
        if not isinstance(model_id, str) or not re.fullmatch(r"gguf-[a-zA-Z0-9_.:-]{1,115}", model_id):
            raise ValueError("Exact GGUF identity required for deliberate study")
        if type(public_rules_only) is not bool or (public_rules_only and book_index is not None) or (
                not public_rules_only and book_index is None):
            raise ValueError("Choose private indexed books OR explicitly public rule paraphrases")
        self.book_index = _private_database(book_index) if book_index is not None else None
        self.public_rules_only = public_rules_only
        self.source_kind = "PUBLIC_REVIEWED_RULE_PARAPHRASE" if public_rules_only else "PRIVATE_INDEXED_PDF"
        self.study_memory = _private_database(study_memory)
        if self.book_index == self.study_memory:
            raise ValueError("Books and learning memory must be separate")
        self.model, self.model_id = model, model_id
        self.latest: dict[str, Any] | None = None
        self._private_question = ""

    def _public_rules(self, edition: str, focused: list[str]) -> list[dict[str,str]]:
        cards={x["id"]:x for x in registry()["rule_cards"]
               if x["source"]==edition and x["status"]=="REVIEWED_PAGES"}
        return [{"source":f"{edition}:RULE:{rule}","excerpt":cards[rule]["summary"][:450]}
                for rule in focused]

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
            private_trace={**self.latest}
            if self._private_question:
                # Search data, never an asserted fact. It can quote private
                # material, so retain it only locally, not in latest/report.
                private_trace["private_learning_question"]=self._private_question
            db.execute("INSERT INTO decision_study_events VALUES(?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, self.model_id, edition, case,
                 observation["as_of"], datetime.now(timezone.utc).isoformat(),
                 json.dumps(private_trace, ensure_ascii=False, allow_nan=False)))

    def propose_chart_reasoning(self, observation: dict[str, Any], *,
                                edition: str = "TC2", **kwargs: Any) -> dict[str, Any]:
        if (edition not in _GAPS or not isinstance(observation, dict)
                or observation.get("execution_allowed") is not False
                or observation.get("gaps", 0)):
            raise ValueError("Gap-free read-only observations required")
        if self.public_rules_only and kwargs.get("primary_pages"):
            raise ValueError("Public rule mode cannot masquerade as private PDF reading")
        self._private_question=""
        self.latest = trace = {
            "mode": "SOURCE_DERIVED_DECISION_LEARNING_NOT_WEIGHT_TRAINING",
            "first_failures": [], "learning_need": "NONE",
            "source_rules": [], "source_pages": [], "studied_chunks": 0,
            "failed_study_chunks": 0, "recalled_notes": [],
            "reconsideration_performed": False, "final_failures": [],
            "same_closed_cutoff": True, "independent_review_completed": False,
            "unverified_self_notes_quarantined": True,
            "model_weights_updated": False, "execution_allowed": False,
            "trade_action": None, "status": "INITIAL_ANALYSIS",
            "source_sha256": next(x["sha256"] for x in registry()["books"] if x["id"]==edition),
            "prior_reading_rules": [], "prior_reading_pages": [],
            "prior_reading_public_references": [],
            "prior_question_reused": False,
            "prior_reading_requests": 0, "learning_question_sha256": None,
            "needs_independent_topic_review": False,
            "source_kind": self.source_kind, "public_rule_references": [],
            "private_original_pages_read": False,
        }
        first = None
        first_error = None
        initial_args={**kwargs}
        if "study_lessons" in initial_args:
            initial_args["study_lessons"]=[]
        try:
            agenda=prior_reading_focus(self.study_memory,model_id=self.model_id,
                edition=edition,as_of=observation["as_of"],source_kind=self.source_kind)
            if agenda["rule_ids"] or agenda["topic"]=="WAVE":
                if agenda["topic"]=="WAVE" and not _GAPS[edition]["WAVE"] and not agenda["rule_ids"]:
                    trace["needs_independent_topic_review"]=True
                remembered=([] if self.public_rules_only else verified_private_book_pages(self.book_index,edition,
                    rule_ids=tuple(agenda["rule_ids"]),
                    question=agenda["question"] or _QUERIES.get(agenda["topic"],_QUERIES["ENTRY"]),limit=2))
                existing=initial_args.get("primary_pages") or []
                combined=[]
                for page in remembered+existing:
                    if not any(x["source"]==page["source"] and x["excerpt"]==page["excerpt"] for x in combined):
                        combined.append(page)
                initial_args["primary_pages"]=combined[:3]
                if self.public_rules_only:
                    initial_args["prior_lessons"]=(self._public_rules(edition,agenda["rule_ids"])
                        +(initial_args.get("prior_lessons") or []))[:3]
                    trace["prior_reading_public_references"]=[x["source"] for x in
                        self._public_rules(edition,agenda["rule_ids"])]
                trace["prior_reading_rules"]=agenda["rule_ids"]
                trace["prior_reading_pages"]=[x["source"] for x in remembered]
                trace["private_original_pages_read"]=bool(remembered)
                trace["prior_reading_requests"]=agenda["past_requests"]
                trace["prior_question_reused"]=bool(agenda["question"]) and not self.public_rules_only
        except (ValueError,OSError,sqlite3.DatabaseError):
            trace["prior_reading_unavailable"]=True
        try:
            first = self.model.propose_chart_reasoning(deepcopy(observation), edition=edition, **initial_args)
            if not isinstance(first,dict):
                raise ValueError("INVALID_MODEL_DECISION_OBJECT")
            checked = audit_reasoning(observation, first)
            if first.get("edition") != edition:
                raise ValueError("MODEL_EDITION_MISMATCH")
            need = first.get("learning_need", "NONE")
            if not isinstance(need, str) or need not in LEARNING_NEEDS:
                raise ValueError("INVALID_MODEL_LEARNING_NEED")
            question=first.get("learning_question","")
            if not isinstance(question,str) or len(question)>400:
                raise ValueError("INVALID_MODEL_LEARNING_QUESTION")
            if question.strip():
                trace["learning_question_sha256"]=hashlib.sha256(question.encode()).hexdigest()
                self._private_question=question.strip()
            trace["learning_need"] = need
            trace["first_failures"] = checked["hard_failures"]
        except (ValueError, TypeError, KeyError) as exc:
            first_error = exc
            trace["first_failures"] = [safe_error_code(exc)]
        # UNKNOWN is a request for clarification; NO_SETUP can be a perfectly
        # competent wait. Missing price data cannot be supplied by a book.
        if (not trace["first_failures"] and trace["learning_need"] == "NONE"
                and first.get("setup") != "UNKNOWN"):
            trace["status"] = ("PAST_UNREVIEWED_TOPIC_REQUIRES_INDEPENDENT_REVIEW"
                if trace["needs_independent_topic_review"] else "NO_BOOK_GAP_DECLARED_OR_DETECTED")
            self._remember(observation, edition)
            return first
        cards = {x["id"]: x for x in registry()["rule_cards"]
                 if x["source"] == edition and x["status"] == "REVIEWED_PAGES"}
        unreviewed_wave=trace["learning_need"]=="WAVE" and not _GAPS[edition]["WAVE"]
        # Reading an unmapped chapter is allowed; it cannot certify an
        # Elliott interpretation or substitute another edition's rules.
        trace["needs_independent_topic_review"] = trace["needs_independent_topic_review"] or bool(unreviewed_wave)
        focused = []
        for failure in trace["first_failures"] if not unreviewed_wave else []:
            if failure in LESSONS and LESSONS[failure][0] in cards:
                # The old generic NTD correction points at AO. AC is an
                # independent histogram with its own sign/count rules.
                rule=("NTD.AC" if failure=="NTD_SIGNAL_NOT_OBSERVED"
                      and isinstance(first,dict) and first.get("setup")=="AC" else LESSONS[failure][0])
                focused.append(rule)
        need = trace["learning_need"]
        focused.extend(_GAPS[edition].get(need, ()))
        if not unreviewed_wave and isinstance(first, dict) and isinstance(first.get("rule_ids"), list):
            focused.extend(x for x in first["rule_ids"] if isinstance(x, str) and x in cards)
        if not focused and not unreviewed_wave:
            focused.extend(_GAPS[edition]["ENTRY"])
        focused = list(dict.fromkeys(focused))[:2]
        trace["source_rules"] = focused
        requested=first.get("learning_question","") if isinstance(first,dict) else ""
        query=(requested.strip() if isinstance(requested,str) and 0<len(requested.strip())<=400
               else _QUERIES.get(need,_QUERIES["ENTRY"]))
        try:
            pages = ([] if self.public_rules_only else verified_private_book_pages(self.book_index, edition,
                rule_ids=tuple(focused), question=query, limit=2))
        except (ValueError, OSError, sqlite3.DatabaseError):
            trace["status"] = "SOURCE_UNAVAILABLE_OR_IDENTITY_REJECTED"
            self._remember(observation, edition)
            if first_error:
                raise ValueError("DELIBERATE_STUDY_SOURCE_UNAVAILABLE_AFTER_MODEL_ERROR") from first_error
            return first
        if not pages and (not self.public_rules_only or not focused):
            trace["status"] = ("NO_REVIEWED_RULE_FOR_TOPIC_IN_SELECTED_EDITION"
                if unreviewed_wave else "SOURCE_TEXT_MISSING_VISUAL_REVIEW_REQUIRED")
            self._remember(observation, edition)
            if first_error:
                raise ValueError("DELIBERATE_STUDY_SOURCE_TEXT_MISSING_AFTER_MODEL_ERROR") from first_error
            return first
        trace["source_pages"] = [x["source"] for x in pages]
        trace["private_original_pages_read"]=trace["private_original_pages_read"] or bool(pages)
        if not self.public_rules_only:
            studied = study_batch(self.book_index, self.study_memory, edition, self.model,
                model_id=self.model_id, budget=2,
                pdf_pages=tuple(dict.fromkeys(x["pdf_page"] for x in pages)),
                chunk_keys=tuple((x["pdf_page"],x["chunk_sha256"]) for x in pages))
            trace["studied_chunks"] = studied["studied_chunks"]
            trace["failed_study_chunks"] = studied["failed_chunks"]
            notes = recall(self.study_memory, query, edition=edition, limit=2,
                model_id=self.model_id, book_index=self.book_index)
            trace["recalled_notes"] = [x["source"] for x in notes]
        else:
            trace["public_rule_references"]=[x["source"] for x in self._public_rules(edition,focused)]
        # Paraphrases retain their own label; source text has priority. A
        # generated lesson never becomes a correction certified by the critic.
        args = {**kwargs, "primary_pages": pages,
                "prior_lessons": self._public_rules(edition,focused),
                "study_lessons":[], "model_id":self.model_id}
        trace["reconsideration_performed"] = True
        try:
            second = self.model.propose_chart_reasoning(deepcopy(observation), edition=edition, **args)
            if not isinstance(second,dict):
                raise ValueError("INVALID_MODEL_DECISION_OBJECT")
            final = audit_reasoning(observation, second)
            if second.get("edition") != edition:
                raise ValueError("MODEL_EDITION_MISMATCH")
            trace["final_failures"] = final["hard_failures"]
            trace["status"] = ("UNSUPPORTED_AFTER_STUDY" if final["hard_failures"] else
                "READ_UNREVIEWED_TOPIC_REQUIRES_INDEPENDENT_REVIEW" if unreviewed_wave else
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
