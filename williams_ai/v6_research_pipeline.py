"""Williams AI v6 research-only Qwen + RAG + deterministic-book-guard pipeline.

This is a read-only wrapper around the EXISTING local Qwen GGUF reasoning
and algorithm_guard, not a new strategy or an executable trading adapter.
The model never decides which source is trusted or whether a trade is allowed.

No GPU, broker access, training, embedding model, or future OHLC data.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from .algorithm_guard import audit_reasoning, _timestamp
from .book_rules import registry
from .unprompted_source_application import self_retrieve_source

_EDITIONS = ("TC1", "NTD", "TC2", "COURSE")
_REPO = Path(__file__).resolve().parent.parent
_PROPOSALS = _REPO / "research" / "williams_v6_source_rule_proposals_20261113.json"
_SYMBOL = re.compile(r"[A-Z0-9_:-]{2,40}\Z")
_MAX_REFS = 3


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError("WILLIAMS_V6_" + code)


def _closed_before(value: Any, cutoff: datetime) -> None:
    _need(isinstance(value, str), "MISSING_CLOSED_TIMESTAMP")
    at = _timestamp(value)
    _need(at <= cutoff, "FUTURE_MARKET_EVIDENCE")


def _observation_contract(chart: dict, edition: str) -> None:
    _need(isinstance(chart, dict), "CHART_NOT_A_DICT")
    _need(edition in _EDITIONS, "INVALID_BOOK")
    _need(chart.get("execution_allowed") is False
          and type(chart.get("gaps")) is int and chart["gaps"] == 0,
          "NONCAUSAL_OR_EXECUTABLE_CHART")
    _need(isinstance(chart.get("symbol"), str)
          and bool(_SYMBOL.fullmatch(chart["symbol"])), "INVALID_SYMBOL")
    cutoff = _timestamp(chart.get("as_of"))
    _need(cutoff <= datetime.now(timezone.utc), "FUTURE_CHART_CUTOFF")
    for item in chart.get("candidates", []):
        _need(isinstance(item, dict), "BAD_CANDIDATE")
        for key in ("bar_time", "pivot_time", "confirmed_at",
                    "price_touch_observed_at"):
            if item.get(key) is not None:
                _closed_before(item[key], cutoff)
    native = chart.get("all_binance_timeframes_compact", {})
    _need(isinstance(native, dict), "BAD_NATIVE_FRAMES")
    for frame in native.values():
        _need(isinstance(frame, dict), "BAD_NATIVE_FRAME")
        if frame.get("close") is not None:
            _closed_before(frame["close"], cutoff)
    higher = chart.get("higher_timeframe_context", {})
    _need(isinstance(higher, dict), "BAD_HIGHER_FRAMES")
    for name in ("H4", "D1"):
        frame = higher.get(name, {})
        _need(isinstance(frame, dict), "BAD_HIGHER_FRAME")
        if frame.get("closed_through") is not None:
            _closed_before(frame["closed_through"], cutoff)


def _pending_rule_proposals() -> set[str]:
    """Unapproved source corrections cannot silently become authoritative."""
    data = json.loads(_PROPOSALS.read_text(encoding="utf-8"))
    _need(isinstance(data, dict)
          and data.get("schema") == "WILLIAMS_V6_SOURCE_RULE_PROPOSALS_REVIEW_REQUIRED"
          and data.get("frozen_v5_unchanged") is True
          and data.get("independent_expert_approval_count") == 0,
          "SOURCE_PROPOSALS_NOT_VERIFIED")
    proposals = data.get("proposals")
    _need(isinstance(proposals, list), "SOURCE_PROPOSAL_LIST_INVALID")
    names = set()
    for proposal in proposals:
        _need(isinstance(proposal, dict) and
              isinstance(proposal.get("rule_id"), str) and
              proposal.get("training_allowed") is False,
              "UNAPPROVED_RULE_CANNOT_BE_TRUSTED")
        _need(proposal["rule_id"] not in names, "DUPLICATE_SOURCE_CORRECTION")
        names.add(proposal["rule_id"])
    return names


def _frozen_sources(question: str, edition: str, *, retriever: str) -> list[dict[str, Any]]:
    discovered = self_retrieve_source(question, edition, max_cards=_MAX_REFS,
                                     retriever=retriever)
    books = {b["id"]: b for b in registry()["books"]}
    cards = {c["id"]: c for c in registry()["rule_cards"]
             if c["source"] == edition and c["status"] == "REVIEWED_PAGES"}
    evidence = []
    for match in discovered["retrieved"]:
        card = cards.get(match["rule_id"])
        _need(card is not None and match["excerpt"] == card["summary"] and
              match["pdf_pages"] == card["pdf_pages"] and
              match["source"] == edition + ":RULE:" + card["id"],
              "RETRIEVER_SOURCE_MANIFEST_MISMATCH")
        evidence.append({
            "rule_id": card["id"], "edition": edition,
            "pdf_pages": list(card["pdf_pages"]),
            "source_pdf_sha256": books[edition]["sha256"],
            "review_status": card["status"],
            "excerpt": card["summary"],
        })
    _need(len({x["rule_id"] for x in evidence}) == len(evidence),
          "DUPLICATE_RETRIEVED_SOURCE")
    return evidence


def _envelope(chart: dict, edition: str, *, state: str, sources: list[dict],
              problems: list[str], guard: dict | None = None,
              safe_claim: dict | None = None) -> dict:
    """Only allowlisted research fields returned; never leak a trade command."""
    return {
        "schema": "WILLIAMS_V6_READONLY_GROUNDED_REASONING_V1",
        "edition": edition, "symbol": chart.get("symbol"),
        "as_of": chart.get("as_of"), "state": state,
        "source_evidence": [{k: s[k] for k in (
            "rule_id", "edition", "pdf_pages",
            "source_pdf_sha256", "review_status")} for s in sources],
        "source_citations_not_full_book_mastery": True,
        "claim": safe_claim,
        "deterministic_audit": ({
            "verdict": guard.get("verdict"),
            "hard_failures": list(guard.get("hard_failures", [])),
            "uncertainties": list(guard.get("uncertainties", [])),
        } if isinstance(guard, dict) else None),
        "blockers": list(dict.fromkeys(problems)),
        "expert_market_gold_verified": False,
        "model_quality_gate_passed": False,
        "trade_action": None,
        "execution_allowed": False,
        "gpu_training_allowed": False,
        "main_merge_allowed": False,
    }


def review_chart(chart: dict, *, edition: str, question: str,
                 model: Any = None, retriever: str = "weighted_v2") -> dict:
    """Book-specific study suggestion; only guard-checked claims can be shown.

    `model` must be an OFFLINE, read-only local model exposing
    `propose_chart_reasoning(chart, edition=..., prior_lessons=...)`.
    Even successful audit is PROVISIONAL: never open an order or claim fill.
    No model call occurs until chart/source validation succeeds.
    """
    _observation_contract(chart, edition)
    _need(isinstance(question, str) and 10 <= len(question) <= 1200,
          "INVALID_RESEARCH_QUESTION")
    _need(retriever in ("legacy", "weighted_v2"), "UNPINNED_RETRIEVER")
    pending = _pending_rule_proposals()
    sources = _frozen_sources(question, edition, retriever=retriever)
    if not sources:
        return _envelope(chart, edition, state="ABSTAIN_NO_REVIEWED_SOURCE",
                         sources=[], problems=["NO_SOURCE_CARD_MATCH"])
    if model is None:
        return _envelope(chart, edition, state="SOURCE_ONLY_REVIEW_REQUIRED",
                         sources=sources, problems=["NO_MODEL_INFERENCE_PERFORMED"])
    lessons = [{"source": s["edition"] + ":RULE:" + s["rule_id"],
                "excerpt": s["excerpt"]} for s in sources]
    try:
        proposal = model.propose_chart_reasoning(
            chart, edition=edition, prior_lessons=lessons)
        _need(isinstance(proposal, dict), "INVALID_MODEL_PROPOSAL")
        guard = audit_reasoning(chart, proposal)
    except (ValueError, TypeError, RuntimeError, KeyError, OSError):
        return _envelope(chart, edition, state="ABSTAIN_INVALID_MODEL_OR_GUARD",
                         sources=sources, problems=["MODEL_OR_GUARD_REJECTED"])
    refs = proposal.get("rule_ids", [])
    ids = {s["rule_id"] for s in sources}
    problems = list(guard["hard_failures"])
    if not isinstance(refs, list) or any(type(ref) is not str for ref in refs):
        problems.append("BAD_SOURCE_CITATION_FORMAT")
        refs = []
    elif not set(refs).issubset(ids):
        problems.append("MODEL_CITES_UNRETRIEVED_RULE")
    if any(ref in pending for ref in refs):
        problems.append("SOURCE_CORRECTION_PENDING_SECOND_EXPERT_REVIEW")
    # Inference can identify potential rules, but source citation does not
    # prove the original edition's diagrams or permit an actual exchange fill.
    safe_claim = {
        "setup": proposal.get("setup") if proposal.get("setup") in (
            "WM1", "WM2", "WM3", "TC1_FRACTAL", "TC1_FIVE_MAGIC_BULLETS",
            "AO_ZERO_CROSS", "AO_SAUCER", "AO_TWIN_PEAKS", "AC", "ZONE",
            "BALANCE_LINE", "WAVE_HYPOTHESIS", "NO_SETUP", "UNKNOWN") else "UNKNOWN",
        "side": proposal.get("side") if proposal.get("side") in ("LONG", "SHORT", "NONE") else "NONE",
        "rule_ids": [x for x in refs if x in ids],
        "entry_state": proposal.get("entry_state") if proposal.get("entry_state") in (
            "NO_SETUP", "CLOSED_CANDIDATE", "PENDING_TRIGGER",
            "CLOSED_BAR_TOUCH_ONLY", "UNKNOWN") else "UNKNOWN",
    }
    return _envelope(chart, edition,
        state=("BLOCKED_UNSUPPORTED_RESEARCH_CLAIM" if problems
               else "PROVISIONAL_BOOK_REASONING_NOT_TRADE"),
        sources=sources, problems=problems, guard=guard, safe_claim=safe_claim)
