"""No answer key leaks in remediation after the REAL Qwen weakness audit."""
import unittest

from williams_ai.book_rules import registry
from williams_ai.targeted_practical_feedback import (
    remediation_cases,paired_source_remediation,
)


class TargetedBookTransferTests(unittest.TestCase):
    def test_all_six_are_book_scoped_and_separate_lesson_from_answer(self):
        tasks=remediation_cases()
        self.assertEqual(len(tasks),6)
        self.assertEqual(len({x["public"]["id"] for x in tasks}),6)
        self.assertEqual(tasks,remediation_cases())
        reviewed={x["id"]:x for x in registry()["rule_cards"]}
        for case in tasks:
            self.assertEqual(case["public"]["edition"],reviewed[case["rule"]]["source"])
            self.assertEqual(case["reminder"]["source"],
                             case["public"]["edition"]+":RULE:"+case["rule"])
            self.assertEqual(case["pages"],reviewed[case["rule"]]["pdf_pages"])
            self.assertEqual(set(case["public"]),{"id","edition","question","choices"})
            self.assertNotIn("expected",case["public"])
            self.assertNotIn("reminder",case["public"])
            self.assertIn(case["expected"],"ABC")
            self.assertEqual(set(case["public"]["choices"]),set("ABC"))

    def test_same_case_assisted_score_is_not_called_true_training(self):
        key={x["public"]["id"]:x["expected"] for x in remediation_cases()}
        class RetrievalStudent:
            def answer_exam_case(self,question,**kw):
                self.check(question,kw)
                if not kw:
                    return {"choice":"","reason":"I need the source"}
                return {"choice":key[question["id"]],"reason":"source rationale"}
            def check(self,question,kwargs):
                if "expected" in question:
                    raise AssertionError("answer key leaked")
                if kwargs and len(kwargs["prior_lessons"])!=1:
                    raise AssertionError("source mismatch")
        result=paired_source_remediation(RetrievalStudent())
        self.assertEqual(result["unguided_correct"],0)
        self.assertEqual(result["retrieval_assisted_correct"],6)
        self.assertEqual(result["delta_source_assistance_not_weight_training"],6)
        self.assertTrue(result["same_questions_in_both_phases_not_independent_blind_holdout"])
        self.assertFalse(result["model_weights_updated"])
        self.assertFalse(result["execution_allowed"])
        self.assertFalse(result["market_profitability_proven"])
        self.assertIsNone(result["trade_action"])

    def test_malformed_model_cannot_pass_by_abstaining(self):
        class Broken:
            def answer_exam_case(self,question,**kw):
                raise RuntimeError("No real model")
        result=paired_source_remediation(Broken())
        self.assertEqual(result["unguided_correct"],0)
        self.assertEqual(result["retrieval_assisted_correct"],0)
        self.assertTrue(all(x["source_assisted"]["failure"]=="REAL_GGUF_CALL_ERROR"
                            for x in result["results"]))

    def test_seed_changes_answer_positions_not_keys_or_sources(self):
        one=remediation_cases(seed=7)
        other=remediation_cases(seed=9)
        self.assertEqual([x["rule"] for x in one],[x["rule"] for x in other])
        self.assertTrue(any(x["public"]["choices"]!=y["public"]["choices"]
                            for x,y in zip(one,other)))
        with self.assertRaises(ValueError):
            remediation_cases(seed=True)


if __name__=="__main__":
    unittest.main()
