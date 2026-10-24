"""Source-scoped and fully blind TC2 exam: read-only score, no LLModel shortcut."""
from collections import Counter
import unittest

from williams_ai.book_rules import registry
from williams_ai.tc2_trade_lifecycle_exam import (
    lifecycle_cases, grade_lifecycle_under_real_gguf,
)


class Tc2TradeLifecycleExamTests(unittest.TestCase):
    def test_all_twenty_are_valid_blinded_three_choice_tasks(self):
        cases=lifecycle_cases()
        self.assertEqual(len(cases),20)
        self.assertEqual(len({x["public"]["id"] for x in cases}),20)
        self.assertEqual({x["stage"] for x in cases},
                         {"ENTRY","MANAGEMENT","EXIT"})
        reviewed={r["id"] for r in registry()["rule_cards"]
                  if r["source"]=="TC2" and r["status"]=="REVIEWED_PAGES"}
        for row in cases:
            with self.subTest(case=row["public"]["id"]):
                self.assertEqual(set(row["public"]),
                                 {"id","edition","question","choices"})
                self.assertEqual(row["public"]["edition"],"TC2")
                self.assertEqual(set(row["public"]["choices"]),{"A","B","C"})
                self.assertEqual(len(set(row["public"]["choices"].values())),3)
                self.assertIn(row["expected"],"ABC")
                self.assertIn(row["source_rule_id"],reviewed)
                self.assertNotIn("expected",row["public"])
                self.assertNotIn("answer",row["public"])

    def test_deterministic_shuffle_and_changed_seed(self):
        a=lifecycle_cases(101)
        self.assertEqual(a,lifecycle_cases(101))
        b=lifecycle_cases(102)
        self.assertEqual([x["public"]["id"] for x in a],
                         [x["public"]["id"] for x in b])
        self.assertTrue(any(x["public"]["choices"]!=y["public"]["choices"]
                            for x,y in zip(a,b)))
        with self.assertRaises(ValueError):
            lifecycle_cases(True)

    def test_independent_perfect_or_constant_responders_grade_realistically(self):
        cases=lifecycle_cases()
        answer_by_id={x["public"]["id"]:x["expected"] for x in cases}
        class Perfect:
            def answer_exam_case(self, case):
                self.assertion(case)
                return {"choice":answer_by_id[case["id"]],
                        "reason":"Verified standalone rule"}
            def assertion(self, case):
                if "expected" in case or "source_rule_id" in case:
                    raise AssertionError("Correct key leaked into inference")
        perfect=grade_lifecycle_under_real_gguf(Perfect())
        self.assertEqual(perfect["correct"],20)
        self.assertTrue(perfect["research_only_book_cognition_pass"])
        self.assertEqual(sum(x["total"] for x in perfect["stages"].values()),20)
        self.assertFalse(perfect["execution_allowed"])
        self.assertFalse(perfect["actual_Qwen_weights_trained_here"])
        self.assertIsNone(perfect["trade_action"])
        class Constant:
            def answer_exam_case(self, case):
                return {"choice":"A","reason":"blind constant"}
        bad=grade_lifecycle_under_real_gguf(Constant())
        self.assertFalse(bad["research_only_book_cognition_pass"])
        self.assertLess(bad["correct"],20)

    def test_errors_and_abstentions_count_wrong_without_trade(self):
        class Errors:
            def answer_exam_case(self, case):
                if case["id"].endswith("WAIT"):
                    raise RuntimeError("synthetic timeout")
                return {"choice":"","reason":"UNKNOWN"}
        r=grade_lifecycle_under_real_gguf(Errors())
        self.assertEqual(r["correct"],0)
        self.assertEqual(sum(v["abstentions_as_wrong"] for v in r["stages"].values()),20)
        self.assertFalse(r["research_only_book_cognition_pass"])
        self.assertFalse(r["execution_allowed"])


if __name__=="__main__":
    unittest.main()
