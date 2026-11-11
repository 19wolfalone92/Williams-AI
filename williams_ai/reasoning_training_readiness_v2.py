"""Fail-closed admission before *proposing* a QLoRA reasoning experiment.

This function never starts training. Reviewer identity is a declaration;
real expert approval and legal private book access are operator duties.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .reasoning_research_v2 import ED, SPLITS, check_corpus


def readiness(cases):
    audit = check_corpus(cases)
    minimum = {"train": 64, "validation": 16, "test": 16}
    min_symbols = {"train": 3, "validation": 2, "test": 2}
    min_frames = {"train": 2, "validation": 2, "test": 2}
    split_rows = {name: [r for r in cases if r["split"] == name] for name in SPLITS}
    reasons = []
    book_rules = defaultdict(set)
    counters = {}
    for split, rows in split_rows.items():
        eds = {r["edition"] for r in rows}
        symbols = {r["symbol"] for r in rows}
        frames = {r["timeframe"] for r in rows}
        counts = Counter(r["rule_id"] for r in rows)
        wave = Counter(str(r["answer"]["wave_confirmed"]) for r in rows
                       if r["answer"]["wave_confirmed"] is not None)
        squat = Counter(str(r["answer"]["squat_verified"]) for r in rows
                        if r["answer"]["squat_verified"] is not None)
        counters[split] = {
            "cases": len(rows), "sources": sorted(eds),
            "symbols": len(symbols), "timeframes": len(frames),
            "rules": len(counts),
            "expert_wave_confirmed_true": wave["True"],
            "expert_wave_confirmed_false": wave["False"],
            "expert_squat_true": squat["True"],
            "expert_squat_false": squat["False"],
        }
        if len(rows) < minimum[split]:
            reasons.append(split + "_INSUFFICIENT_REVIEWED_MARKET_CASES")
        if eds != ED:
            reasons.append(split + "_MISSING_FOUR_BOOK_COVERAGE")
        if len(symbols) < min_symbols[split]:
            reasons.append(split + "_INSUFFICIENT_ASSET_DIVERSITY")
        if len(frames) < min_frames[split]:
            reasons.append(split + "_INSUFFICIENT_TIMEFRAME_DIVERSITY")
        if not (wave["True"] >= 2 and wave["False"] >= 2):
            reasons.append(split + "_NO_BALANCED_EXPERT_WAVE_POSITIVE_NEGATIVE")
        if not (squat["True"] >= 2 and squat["False"] >= 2):
            reasons.append(split + "_NO_BALANCED_EXPERT_SQUAT_POSITIVE_NEGATIVE")
        book_rules[split] = {(r["edition"], r["rule_id"]) for r in rows}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if book_rules[a] & book_rules[b]:
            reasons.append("BOOK_RULES_CROSS_" + a.upper() + "_" + b.upper())
    return {
        "schema": "WILLIAMS_AI_REASONING_V2_GPU_READINESS_V1",
        "ready_for_gpu_experiment": not reasons,
        "reasons": sorted(set(reasons)),
        "counts": counters,
        "reviewer_identity_verified_external_to_code": False,
        "test_keys_available_to_training": False,
        "automatic_training_enabled": False,
        "trading_qualification": False,
        "execution_allowed": False,
        "corpus_sha256": audit["source_sha256"],
    }
