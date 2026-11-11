"""Integration tests for staged historical reasoning, not simulated profitability."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

from tests.test_williams_ai_reasoning_research_v2 import example
from williams_ai.reasoning_research_v2 import HypothesisMemory, digest, verify_case, question_only
from williams_ai.reasoning_walkforward_v2 import run_blind_replay, compare_frozen_predictions, run_prompt_only_replay


class WalkForwardV2Tests(unittest.TestCase):
    def test_no_gold_in_model_input_and_previous_hypotheses_are_recalled(self):
        first = example("test-first", "test", "BTCUSDT")
        second = deepcopy(first)
        second["case_id"] = "test-second"
        extra = deepcopy(first["bars"][-1])
        extra["close_time"] = (datetime.fromisoformat(first["cutoff"]) + timedelta(hours=1)).isoformat()
        extra["open"], extra["close"], extra["high"], extra["low"] = 320.0, 321.0, 322.0, 319.0
        second["bars"].append(extra)
        second["cutoff"] = extra["close_time"]
        second["visible_prefix_sha256"] = digest(second["bars"])
        case_sha = digest({k: val for k, val in second.items() if k != "reviewers"})
        for review in second["reviewers"]:
            review["case_sha256"] = case_sha
        observations = []

        def reason(market, history):
            self.assertNotIn("answer", market)
            self.assertNotIn("reviewers", market)
            self.assertTrue(all(x["available_at"] <= market["cutoff"] for x in history))
            observations.append(len(history))
            return deepcopy(first["answer"])

        with tempfile.TemporaryDirectory() as tmp:
            memory = HypothesisMemory(Path(tmp) / "study.sqlite", run_id="test-replay-0001", model_id="test-model-parent")
            report = run_blind_replay([second, first], reason, memory)
            self.assertEqual(observations, [0, 1])
            self.assertEqual(report["grade"]["cases"], 2)
            self.assertEqual(report["grade"]["decision_correct"], 2)
            self.assertFalse(report["grade"]["cryptographically_attested_prediction_time"])
            paired = compare_frozen_predictions([first, second], report["predictions"], report["predictions"])
            self.assertEqual(paired["accuracy_delta_points"], 0.0)

    def test_strict_prompt_only_worker_without_gold(self):
        from tests.test_williams_ai_reasoning_v2_blocker_fixes import full_case, reseal
        case=full_case()
        case["split"]="test"
        reseal(case)
        seen=[]
        def model(prompt,history):
            self.assertNotIn("answer",prompt)
            self.assertNotIn("reviewers",prompt)
            self.assertTrue(prompt["complete_h1_h4_d1_context"])
            self.assertEqual(history,[])
            seen.append(prompt["case_id"])
            return deepcopy(case["answer"])
        with tempfile.TemporaryDirectory() as folder:
            memory=HypothesisMemory(Path(folder)/"prompts.sqlite",
                                    run_id="prompt-only-run-001",
                                    model_id="qwen-parent-001")
            prompt=question_only(case)
            result=run_prompt_only_replay([prompt],model,memory)
            self.assertEqual(seen,[case["case_id"]])
            self.assertFalse(result["gold_answer_loaded_by_this_function"])
            grade=compare_frozen_predictions([case],result["predictions"],
                                              result["predictions"])
            self.assertEqual(grade["accuracy_delta_points"],0.0)
            with self.assertRaisesRegex(ValueError,"PROMPTS_ONLY"):
                run_prompt_only_replay([{**prompt,"answer":case["answer"]}],
                                       model,memory)

    def test_reject_fake_book_source(self):
        row = example()
        row["source_sha256"] = "a"*64
        case_sha = digest({k: val for k, val in row.items() if k != "reviewers"})
        for review in row["reviewers"]:
            review["case_sha256"] = case_sha
        with self.assertRaisesRegex(ValueError, "BOOK_RULE"):
            verify_case(row)

    def test_reject_unapproved_positive_wave_label(self):
        row = example()
        row["answer"]["wave_confirmed"] = True
        with self.assertRaisesRegex(ValueError, "REVIEW_MUST_APPROVE"):
            verify_case(row)


if __name__ == "__main__":
    unittest.main()
