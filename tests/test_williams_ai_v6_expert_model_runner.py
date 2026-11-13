"""Offline matched Qwen v6 inference contract tests; NO actual model weights."""
import copy
import json
import unittest

from tests import test_williams_ai_independent_exam_v6 as fixture_module
from williams_ai.v6_expert_model_runner import (
    PROTOCOL, combine, decide, infer, prompt_for_case, protocol_sha256,
)


class StubLlama:
    def __init__(self, *, raw=None, finish_reason="stop"):
        self.calls = []
        self.raw = raw
        self.finish_reason = finish_reason

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.raw is None:
            reply = {"decision": "INSUFFICIENT_DATA",
                     "reason": "This chart needs an independent expert review."}
            raw = json.dumps(reply)
        else:
            raw = self.raw
        return {"choices": [{"finish_reason": self.finish_reason,
                             "message": {"content": raw}}]}


class MatchedNoGoldResearchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.V6IndependentIntakeTests("test_frozen_chart_signatures_all_four_book_counts")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cases, self.root = self.fixture.dataset, self.fixture.root
        book_queries = {
            "TC1": "Торговый Хаос: фрактал волна дивергенция 5/34 анализ пяти свечей.",
            "NTD": "NTD зона AO AC Аллигатор ускорение фрактал условия закрытия.",
            "TC2": "TC2 третий Мудрец фрактал пробой зубы Alligator разворот.",
            "COURSE": "Course 5/34/5 momentum осциллятор волна нулевая линия.",
        }
        for case in self.cases["cases"]:
            case["question"] = book_queries[case["edition"]]

    def test_all_books_same_prompt_bound_to_closed_chart_and_no_exam_key(self):
        llm = StubLlama()
        out = infer(llm, self.cases, self.root,
                    name="matched_base", effective_weights_sha256="1" * 64)
        self.assertEqual(len(llm.calls), 32)
        self.assertEqual(len(out["answers"]), 32)
        self.assertEqual(out["protocol_sha256"], protocol_sha256())
        self.assertEqual(out["weights_sha256"], "1" * 64)
        self.assertTrue(out["model_did_not_receive_gold"])
        self.assertFalse(out["gpu_used"])
        self.assertTrue(all(x["decision"] == "INSUFFICIENT_DATA"
                            for x in out["answers"]))
        for call in llm.calls:
            self.assertEqual(call["temperature"], 0.0)
            self.assertEqual(call["seed"], 1942)
            self.assertFalse(call["stream"])
            self.assertEqual(call["response_format"]["type"], "json_object")
            user = json.loads(call["messages"][1]["content"])
            self.assertNotIn("id", user)
            self.assertNotIn("labels", user)
            self.assertNotIn("correct", user)
            self.assertNotIn("expected", user)
            self.assertLessEqual(len(user["closed_bars"]), 96)
            self.assertEqual(len(user["closed_bars"][0]), 6)
            self.assertTrue(all(x["rule_id"].startswith(user["edition"] + ".")
                                for x in user["book_source_cards_not_expert_gold"]))
        self.assertFalse(out["execution_allowed"])

    def test_combine_requires_same_hashes_protocol_and_question_bytes(self):
        a = infer(StubLlama(), self.cases, self.root,
                  name="candidate_v5", effective_weights_sha256="2" * 64)
        b = infer(StubLlama(), self.cases, self.root,
                  name="matched_base", effective_weights_sha256="1" * 64)
        joined = combine(a, b, self.cases, self.root)
        self.assertEqual(joined["schema"], "WILLIAMS_V6_DUAL_MODEL_SUBMISSION_V1")
        self.assertEqual(set(joined["models"]), {"candidate_v5", "matched_base"})
        self.assertEqual(joined["protocol_sha256"], protocol_sha256())
        changed = copy.deepcopy(a)
        changed["prompt_diagnostics"][0]["prompt_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "MODEL_RECEIPT_INPUT_DID_NOT_MATCH_FROZEN_CASE"):
            combine(changed, b, self.cases, self.root)
        # Even BOTH model receipts agreeing on a forged hash cannot
        # bypass independent reconstruction from real frozen chart bytes.
        fake_a, fake_b = copy.deepcopy(a), copy.deepcopy(b)
        fake_a["prompt_diagnostics"][0]["prompt_sha256"] = "e" * 64
        fake_b["prompt_diagnostics"][0]["prompt_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "MODEL_RECEIPT_INPUT_DID_NOT_MATCH_FROZEN_CASE"):
            combine(fake_a, fake_b, self.cases, self.root)
        changed = copy.deepcopy(a)
        changed["protocol_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "INCOMPATIBLE_MODEL_PROTOCOL"):
            combine(changed, b, self.cases, self.root)
        changed = copy.deepcopy(a)
        changed["weights_sha256"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "SAME_WEIGHTS"):
            combine(changed, b, self.cases, self.root)

    def test_parse_malformed_answer_or_length_as_abstain(self):
        case = self.cases["cases"][0]
        chart = json.loads((self.root / case["chart_file"]).read_text())
        for raw in ('{"decision":"LONG_STOP_ENTRY","reason":"x", "reason":"double"}',
                    '{"decision":"EXECUTE_LONG","reason":"invalid"}',
                    "<think>reason</think>{\"decision\":\"NO_SETUP\",\"reason\":\"x\"}"):
            with self.subTest(raw=raw):
                row, meta = decide(StubLlama(raw=raw), case, chart)
                self.assertEqual(row["decision"], "ABSTAIN")
                self.assertEqual(row["finish_reason"], "error")
        row, _ = decide(StubLlama(finish_reason="length"), case, chart)
        self.assertEqual(row["decision"], "ABSTAIN")
        self.assertEqual(row["finish_reason"], "length")

    def test_prompt_does_not_contain_case_id_or_reviewer_labels(self):
        case = self.cases["cases"][0]
        chart = json.loads((self.root / case["chart_file"]).read_text())
        messages, info = prompt_for_case(case, chart)
        self.assertNotIn(case["id"], json.dumps(messages))
        self.assertEqual(len(info["prompt_sha256"]), 64)
        self.assertTrue(PROTOCOL["no_hidden_key_or_training_labels"])


if __name__ == "__main__":
    unittest.main()
