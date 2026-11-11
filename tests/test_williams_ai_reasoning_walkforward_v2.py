"""Integration tests for staged historical reasoning, not simulated profitability."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

from tests.test_williams_ai_reasoning_research_v2 import example
from williams_ai.reasoning_research_v2 import HypothesisMemory, digest, verify_case
from williams_ai.reasoning_walkforward_v2 import run_blind_replay, compare_frozen_predictions


class WalkForwardV2Tests(unittest.TestCase):
    def test_no_gold_in_model_input_and_previous_hypotheses_are_recalled(self):
        first = example("test-first", "test", "BTCUSDT")
        second = deepcopy(first)
        second["case_id"] = "test-second"
        extra = deepcopy(first["bars"][-1])
        extra["close_time"] = "2026-11-01T02:00:00+00:00"
        extra["open"], extra["close"], extra["high"], extra["low"] = 102.0, 103.0, 104.0, 100.0
        second["bars"].append(extra)
        second["cutoff"] = extra["close_time"]
        second["visible_prefix_sha256"] = digest(second["bars"])
        observations = []

        def reason(market, history):
            self.assertNotIn("answer", market)
            self.assertNotIn("reviewers", market)
            self.assertTrue(all(x["available_at"] <= market["cutoff"] for x in history))
            observations.append(len(history))
            return deepcopy(first["answer"])

        with tempfile.TemporaryDirectory() as tmp:
            memory = HypothesisMemory(Path(tmp) / "study.sqlite")
            report = run_blind_replay([second, first], reason, memory)
            self.assertEqual(observations, [0, 1])
            self.assertEqual(report["grade"]["cases"], 2)
            self.assertEqual(report["grade"]["decision_correct"], 2)
            self.assertFalse(report["grade"]["cryptographically_attested_prediction_time"])
            paired = compare_frozen_predictions([first, second], report["predictions"], report["predictions"])
            self.assertEqual(paired["accuracy_delta_points"], 0.0)

    def test_reject_fake_book_source(self):
        row = example()
        row["source_sha256"] = "a"*64
        with self.assertRaisesRegex(ValueError, "BOOK_RULE"):
            verify_case(row)

    def test_reject_unapproved_positive_wave_label(self):
        row = example()
        row["answer"]["wave_confirmed"] = True
        with self.assertRaisesRegex(ValueError, "REVIEW_MUST_APPROVE"):
            verify_case(row)


if __name__ == "__main__":
    unittest.main()
