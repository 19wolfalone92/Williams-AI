"""Read-only reproducible failure-mode probes for Williams AI 2.0 research.

The probes intentionally demonstrate weaknesses. A passing audit script means
the probes ran as expected; it does NOT mean the learning strategy is ready.
No GPU, no API keys, no books, no trading, no new expert labels.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_williams_ai_reasoning_research_v2 import example
from williams_ai.private_lora import _REQUIRED as EXISTING_QLORA_REQUIRED
from williams_ai.reasoning_research_v2 import (
    HypothesisMemory, adaptive_schedule, check_corpus, digest, export_research,
    grade_blind, prediction_commitment, question_only,
)


def reapprove(row):
    whole_case = digest({k: v for k, v in row.items() if k != "reviewers"})
    for reviewer in row["reviewers"]:
        reviewer["case_sha256"] = whole_case


def probes():
    problems = []
    def finding(code, reproduced, evidence, severity="P0"):
        problems.append({"id": code, "severity": severity,
                         "reproduced": bool(reproduced), "evidence": evidence})

    baseline = example("audit-original", "train", "BTCUSDT")
    modified = deepcopy(baseline)
    modified["question"] += " Additional different training question with identical case ID."
    reapprove(modified)
    old = check_corpus([baseline])["source_sha256"]
    new = check_corpus([modified])["source_sha256"]
    finding("CORPUS_HASH_NOT_CONTENT_BOUND", old == new,
            "Change an approved question; source_sha256 still equals " + old[:12] + "...")

    heldout = example("audit-blind", "test", "SOLUSDT")
    prediction = {
        "case_id": heldout["case_id"], "cutoff": heldout["cutoff"],
        "visible_prefix_sha256": heldout["visible_prefix_sha256"],
        "prediction_at": heldout["cutoff"],
        "answer": {"decision": heldout["answer"]["decision"],
                   "execution_allowed": False}}
    prediction["commitment"] = prediction_commitment(prediction)
    grade = grade_blind([heldout], [prediction])
    finding("BLIND_GRADER_ACCEPTS_NO_REASONING", grade["decision_accuracy"] == 1.0
            and grade["unsafe_execution_claims"] == 0,
            "Decision-only prediction scored 100% and zero unsafe claims; no evidence or counterhypothesis graded.")

    train, _ = export_research([baseline])
    exported = train["train_sft"][0]
    missing = sorted(EXISTING_QLORA_REQUIRED - set(exported))
    finding("NEW_SFT_INCOMPATIBLE_WITH_EXISTING_QLORA", bool(missing),
            "SFT keys=" + ",".join(sorted(exported)) +
            "; missing old reviewed-data fields=" + ",".join(missing))

    no_overlap_failure = [{"case_id": "validation-001",
                           "split": "validation", "rule_id": "NTD.AC",
                           "error_code": "MISUNDERSTOOD_AC"}]
    schedule = adaptive_schedule([baseline], no_overlap_failure, rounds=3)
    lost = all(x["validation_failures_for_rule"] == 0 for x in schedule["schedule"])
    finding("ADAPTIVE_FEEDBACK_DROPPED_BY_RULE_SPLIT", lost,
            "Validation error NTD.AC does not influence train rule TC2.WM1.ANG. "
            "GPU readiness meanwhile forbids the same rule ID crossing splits.")

    quoted = example("quote-a", "train", "BTCUSDT")
    alias = example("quote-b", "test", "BTCUSDC")
    accepted = True
    try: check_corpus([quoted, alias])
    except ValueError: accepted = False
    finding("QUOTE_ALIAS_LEAKAGE_NOT_BLOCKED", accepted,
            "BTCUSDT in train and BTCUSDC in test are accepted as separate symbols; "
            "legacy chart_supervision.check_market_partitions does alias grouping.")

    market = question_only(baseline)
    missing_market = [key for key in ("h4", "d1", "confirmed_swings",
                                     "book_excerpt") if key not in market]
    finding("NO_H4_D1_SWINGS_OR_BOOK_EVIDENCE_IN_PROMPT",
            len(missing_market) == 4 and len(market["visible_numeric_tail"]) == 8,
            "Model prompt exposes only 8 H1 candles and their derived indicators; "
            "omits H4/D1, confirmed swings and actual cited book passage.")

    with tempfile.TemporaryDirectory() as dirname:
        mem = HypothesisMemory(Path(dirname) / "research.sqlite")
        with sqlite3.connect(mem.path) as db:
            cols = {row[1] for row in db.execute("PRAGMA table_info(hypotheses)")}
        mem.append(case_id="same-market-case", symbol="BTCUSDT",
                   cutoff="2026-11-01T01:00:00Z",
                   available_at="2026-11-01T01:00:00Z",
                   snapshot_sha256="a" * 64,
                   hypothesis="Unconfirmed first hypothesis",
                   alternative="Unconfirmed alternative",
                   evidence=["observed"], counterevidence=["uncertain"],
                   invalidation="break of support")
        duplicate_blocked = False
        try:
            mem.append(case_id="same-market-case", symbol="BTCUSDT",
                       cutoff="2026-11-01T01:00:00Z",
                       available_at="2026-11-01T01:00:00Z",
                       snapshot_sha256="a" * 64,
                       hypothesis="Second model alternative",
                       alternative="Second other alternative",
                       evidence=["observed"], counterevidence=["uncertain"],
                       invalidation="retest")
        except sqlite3.IntegrityError:
            duplicate_blocked = True
        finding("NO_MODEL_ISOLATION_IN_HYPOTHESIS_MEMORY",
                "model_id" not in cols and duplicate_blocked,
                "hypotheses PK=case_id only; two models replaying same case collide "
                "and hypotheses cannot be separated by checkpoint/model fingerprint.")

    return {
        "schema": "WILLIAMS_AI_2_STRATEGY_BLOCKER_REPRODUCTIONS_V1",
        "finding_count": len(problems),
        "reproduced_count": sum(x["reproduced"] for x in problems),
        "findings": problems,
        "NOT_MODEL_QUALITY_EXAM": True,
        "execution_allowed": False,
    }


if __name__ == "__main__":
    result = probes()
    report = ROOT / "audit_output/strategy_blockers.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["reproduced_count"] != result["finding_count"]:
        raise SystemExit("AUDIT_PROBE_NOT_REPRODUCED")
