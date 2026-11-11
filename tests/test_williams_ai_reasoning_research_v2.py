"""Executable regression tests for the Williams AI 2.0 research boundary."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from williams_ai.reasoning_training_readiness_v2 import readiness
from williams_ai.reasoning_research_v2 import (
    HypothesisMemory, adaptive_schedule, check_corpus, digest,
    export_research, grade_blind, prediction_commitment, verify_case,
)


def example(case_id="case-001", split="train", symbol="BTCUSDT"):
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    bars = []
    for i in range(220):
        bars.append({"close_time": (start + timedelta(hours=i)).isoformat(),
                     "open": 100.0 + i, "high": 102.0 + i,
                     "low": 99.0 + i, "close": 101.0 + i,
                     "volume": 50.0, "closed": True})
    answer = {
        "hypothesis": "Possible impulse structure is still an unconfirmed hypothesis.",
        "alternative": "The same observations may represent a corrective structure.",
        "evidence": ["Closed candles show an observable higher close."],
        "counterevidence": ["Wave degree is not independently confirmed."],
        "invalidation": "Revisit if the structural low is broken on a closed bar.",
        "decision": "WAIT", "confidence": .3,
        "wave_confirmed": None, "squat_verified": None,
        "execution_allowed": False,
    }
    row = {
        "case_id": case_id, "split": split, "rule_id": "TC2.WM1.ANG",
        "edition": "TC2", "pdf_page": 124, "source_sha256": "055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f",
        "symbol": symbol, "timeframe": "H1",
        "cutoff": bars[-1]["close_time"], "visible_prefix_sha256": digest(bars),
        "bars": bars, "question": "Analyze the closed market snapshot without assuming a fill.",
        "answer": answer,
        "reviewers": [
            {"reviewer_id": "analyst-one", "approved_at": "2026-11-03T00:00:00Z", "case_sha256": "pending-review-seal"},
            {"reviewer_id": "analyst-two", "approved_at": "2026-11-03T00:00:00Z", "case_sha256": "pending-review-seal"},
        ],
    }
    case_sha = digest({k: v for k, v in row.items() if k != "reviewers"})
    for review in row["reviewers"]:
        review["case_sha256"] = case_sha
    return row


class EvidenceGateTests(unittest.TestCase):
    def test_verified_case_and_three_split_export(self):
        rows = [example("first-case", "train", "BTCUSDT"),
                example("second-case", "validation", "ETHUSDT"),
                example("third-case", "test", "SOLUSDT")]
        summary = check_corpus(rows)
        self.assertEqual(summary["case_count"], 3)
        payload, report = export_research(rows)
        self.assertEqual(len(payload["train_sft"]), 1)
        self.assertEqual(len(payload["test_prompts"]), 1)
        self.assertNotIn("answer", payload["test_prompts"][0])
        self.assertNotIn("reviewers", payload["test_prompts"][0])
        self.assertFalse(report["test_gold_exported"])

    def test_refuse_gpu_without_expert_market_coverage(self):
        rows = [example("case-a", "train", "BTCUSDT"),
                example("case-b", "validation", "ETHUSDT"),
                example("case-c", "test", "SOLUSDT")]
        decision = readiness(rows)
        self.assertFalse(decision["ready_for_gpu_experiment"])
        self.assertIn("train_INSUFFICIENT_REVIEWED_MARKET_CASES", decision["reasons"])
        self.assertIn("test_MISSING_FOUR_BOOK_COVERAGE", decision["reasons"])
        self.assertFalse(decision["automatic_training_enabled"])

    def test_no_future_bar_or_open_bar(self):
        row = example()
        row["bars"][-1]["close_time"] = "2026-11-01T02:00:00Z"
        with self.assertRaisesRegex(ValueError, "FUTURE"):
            verify_case(row)
        row = example()
        row["bars"][-1]["closed"] = False
        with self.assertRaisesRegex(ValueError, "NO_OPEN_CANDLES"):
            verify_case(row)

    def test_prefix_and_expert_approval_required(self):
        row = example()
        row["visible_prefix_sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "PREFIX_DIGEST"):
            verify_case(row)
        row = example()
        row["reviewers"][1]["reviewer_id"] = row["reviewers"][0]["reviewer_id"]
        with self.assertRaisesRegex(ValueError, "DISTINCT_REVIEWER"):
            verify_case(row)
        row = example()
        row["reviewers"][1]["case_sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "REVIEW_MUST_APPROVE"):
            verify_case(row)

    def test_prevent_same_symbol_in_train_and_test(self):
        rows = [example("first-case", "train", "BTCUSDT"),
                example("second-case", "test", "BTCUSDT")]
        with self.assertRaisesRegex(ValueError, "REPEATED_PREFIX|WHOLE_SYMBOL"):
            check_corpus(rows)

    def test_dont_train_on_validation_answers(self):
        row = example()
        schedule = adaptive_schedule([row],
            [{"case_id": "validation-1", "split": "validation",
              "rule_id": "TC2.WM1.ANG", "error_code": "MISSED_ALTERNATIVE"}],
            rounds=3)
        self.assertEqual(len(schedule["schedule"]), 3)
        self.assertFalse(schedule["validation_answers_used_as_training"])
        with self.assertRaisesRegex(ValueError, "TRAIN_ONLY"):
            adaptive_schedule([example(split="test")], [], rounds=1)
        with self.assertRaisesRegex(ValueError, "VALIDATION_METADATA"):
            adaptive_schedule([row], [{"case_id": "test-1", "split": "test",
                                      "rule_id": "TC2.WM1", "error_code": "BAD"}])

    def test_memory_never_uses_future_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            mem = HypothesisMemory(Path(folder)/"private.sqlite")
            mem.append(case_id="h-001", symbol="BTCUSDT",
                       cutoff="2026-11-01T01:00:00Z",
                       available_at="2026-11-01T02:00:00Z",
                       snapshot_sha256="a"*64, hypothesis="first possible structure",
                       alternative="another plausible structure", evidence=["bar"],
                       counterevidence=["uncertainty"], invalidation="lose support")
            self.assertEqual(mem.as_of(symbol="BTCUSDT", timestamp="2026-11-01T01:30:00Z"), [])
            rows = mem.as_of(symbol="BTCUSDT", timestamp="2026-11-01T03:00:00Z")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "UNVERIFIED_MODEL_HYPOTHESIS")
            self.assertEqual(mem.path.stat().st_mode & 0o077, 0)

    def test_human_review_never_retroactively_leaks_into_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            mem = HypothesisMemory(Path(folder) / "private.sqlite")
            mem.append(case_id="hist-001", symbol="BTCUSDT",
                       cutoff="2026-11-01T01:00:00Z",
                       available_at="2026-11-01T02:00:00Z",
                       snapshot_sha256="a"*64, hypothesis="initial unconfirmed structure",
                       alternative="possible correction", evidence=["closed candle"],
                       counterevidence=["degree unknown"],
                       invalidation="break of pivot")
            with self.assertRaisesRegex(ValueError, "REVIEW_CANNOT_PRECEDE"):
                mem.review(case_id="hist-001", reviewer_a="expert-one",
                           reviewer_b="expert-two",
                           reviewed_at="2026-11-01T01:30:00Z", verdict="REJECTED",
                           rationale="Later candles contradicted the original thesis.")
            mem.review(case_id="hist-001", reviewer_a="expert-one",
                       reviewer_b="expert-two",
                       reviewed_at="2026-11-04T00:00:00Z", verdict="REJECTED",
                       rationale="Later candles contradicted the original thesis.")
            historical = mem.as_of(symbol="BTCUSDT",
                                   timestamp="2026-11-03T00:00:00Z")
            self.assertEqual(historical[0]["status"], "UNVERIFIED_MODEL_HYPOTHESIS")
            reviewed = mem.as_of(symbol="BTCUSDT",
                                 timestamp="2026-11-05T00:00:00Z")
            self.assertEqual(reviewed[0]["status"], "HUMAN_REVIEW_REJECTED")
            self.assertIn("contradicted", reviewed[0]["review_rationale"])

    def test_blind_prediction_is_snapshot_bound(self):
        case = example("test-1", "test", "BTCUSDT")
        p = {"case_id": "test-1", "cutoff": case["cutoff"],
             "visible_prefix_sha256": case["visible_prefix_sha256"],
             "prediction_at": "2026-11-01T01:01:00Z",
             "answer": {"decision": "WAIT", "wave_confirmed": None,
                        "execution_allowed": False}}
        p["commitment"] = prediction_commitment(p)
        result = grade_blind([case], [p])
        self.assertEqual(result["decision_correct"], 1)
        bad = deepcopy(p); bad["answer"]["decision"] = "REVIEW_ENTRY"
        with self.assertRaisesRegex(ValueError, "COMMITMENT"):
            grade_blind([case], [bad])


if __name__ == "__main__":
    unittest.main()
