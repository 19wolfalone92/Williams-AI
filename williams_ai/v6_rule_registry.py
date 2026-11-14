"""Versioned four-book v6 CANDIDATE index, never training or market gold.

Materializes reviewed *paraphrase cards* and separately flagged unapproved
source corrections without changing the frozen v5 registry or SFT manifest.
Nothing here authorizes a new LoRA, expert approval, visual wave labels,
paper position, real order or main merge.
"""
from __future__ import annotations

import argparse
from hashlib import sha1, sha256
import json
from pathlib import Path

from .book_rules import registry

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "knowledge" / "SOURCE_RULE_CARDS.json"
PROPOSALS = ROOT / "research" / "williams_v6_source_rule_proposals_20261113.json"
PINNED_GIT_BLOB = "b09553108aa4ce2c764cb6e489544516003943d8"
BOOKS = ("TC1", "NTD", "TC2", "COURSE")
SCHEMA = "WILLIAMS_V6_CANDIDATE_SOURCE_INDEX_NOT_TRAIN_DATA"


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("V6_SOURCE_INDEX_" + reason)


def _git_blob_sha(raw: bytes) -> str:
    return sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def build_candidate_index() -> dict:
    raw = REGISTRY.read_bytes()
    _require(_git_blob_sha(raw) == PINNED_GIT_BLOB, "FROZEN_V5_SNAPSHOT_CHANGED")
    base = registry()
    proposed_bytes = PROPOSALS.read_bytes()
    proposed = json.loads(proposed_bytes)
    _require(isinstance(proposed, dict) and
             proposed.get("schema") == "WILLIAMS_V6_SOURCE_RULE_PROPOSALS_REVIEW_REQUIRED" and
             proposed.get("parent_registry_blob_sha") == PINNED_GIT_BLOB and
             proposed.get("independent_expert_approval_count") == 0 and
             proposed.get("training_authorized") is False and
             proposed.get("frozen_v5_unchanged") is True,
             "UNAPPROVED_PROPOSALS_MISMATCH")
    cards = {x["id"]: x for x in base["rule_cards"]}
    _require(len(cards) == len(base["rule_cards"]), "DUPLICATE_FROZEN_RULE")
    source_books = {x["id"]: x for x in base["books"]}
    _require(set(source_books) == set(BOOKS), "FOUR_BOOK_PROFILE_REQUIRED")
    issues = {}
    for obj in proposed["proposals"]:
        rid = obj.get("rule_id")
        _require(type(rid) is str and rid in cards and rid not in issues,
                 "UNKNOWN_OR_DUPLICATE_CORRECTION")
        _require(obj.get("training_allowed") is False and
                 obj.get("review_status") != "EXPERT_APPROVED",
                 "CANNOT_PREAPPROVE_SOURCE_CORRECTION")
        pages = obj.get("pages")
        source = cards[rid]["source"]
        _require(type(pages) is list and len(pages) > 0 and
                 all(type(p) is int and 1 <= p <= source_books[source]["pages"]
                     for p in pages), "UNSUPPORTED_CORRECTION_PAGE")
        issues[rid] = obj
    records = []
    for book in BOOKS:
        for card in sorted(base["rule_cards"], key=lambda x:x["id"]):
            if card["source"] != book:
                continue
            candidate = issues.get(card["id"])
            records.append({
                "rule_id": card["id"], "edition": book,
                "category": card.get("category"),
                "pdf_pages_original": list(card["pdf_pages"]),
                "source_pdf_sha256": source_books[book]["sha256"],
                "v5_checked_paraphrase": card["summary"],
                "v5_source_status": card["status"],
                "v6_research_status": (
                    "PENDING_SECOND_INDEPENDENT_SOURCE_REVIEW"
                    if candidate is not None
                    else "V5_PAGE_CHECKED_NOT_INDEPENDENT_EXPERT_MARKET_GOLD"),
                "proposed_new_pages": (
                    list(candidate["pages"]) if candidate else []),
                "proposed_revised_paraphrase": (
                    candidate.get("candidate_summary") if candidate else None),
                "reported_issue": candidate.get("issue") if candidate else None,
                "needs_original_diagram_review": (
                    card["id"] == "COURSE.VISUAL.NESTED_ABC" or
                    card["id"] in {"NTD.ALLIGATOR", "COURSE.W4.ZERO",
                                  "TC2.WM1.ANG"}),
                "independent_expert_reviews": 0,
                "approved_market_chart_labels": 0,
                "training_allowed": False,
                "execution_allowed": False,
            })
    count = {ed: sum(x["edition"] == ed for x in records) for ed in BOOKS}
    _require(all(count.values()), "EMPTY_BOOK")
    _require(len(records) == len(base["rule_cards"]), "MISSING_RULES")
    return {
        "schema": SCHEMA,
        "parent_run_sha": proposed["parent_run_sha"],
        "parent_registry_git_blob": PINNED_GIT_BLOB,
        "parent_registry_sha256": sha256(raw).hexdigest(),
        "unapproved_proposal_sha256": sha256(proposed_bytes).hexdigest(),
        "books": [{
            "id": ed, "title": source_books[ed]["title"],
            "pdf_pages": source_books[ed]["pages"],
            "source_pdf_sha256": source_books[ed]["sha256"]
        } for ed in BOOKS],
        "rule_counts": count,
        "unapproved_corrections_count": len(issues),
        "rules": records,
        "no_original_pdf_or_visual_data_exported": True,
        "test_gold_included": False,
        "independent_visual_wave_gold_count": 0,
        "independently_approved_market_cases": 0,
        "versioned_not_training_dataset": True,
        "no_gpu_training": True,
        "no_real_trading": True,
        "main_merge_allowed": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    data = build_candidate_index()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
