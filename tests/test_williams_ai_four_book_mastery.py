"""Four books competence cannot be inferred from 1-book passes or mock grading."""
import unittest
from collections import Counter

from williams_ai.book_rules import registry
from williams_ai.four_book_curriculum import (
    four_book_teaching_passport, single_book_model_lesson,
)
from williams_ai.four_book_blind_exam import (
    four_book_blind_cases, grade_four_book_model,
)


class FourBookCurriculumTests(unittest.TestCase):
    def test_all_books_bound_to_checked_pages_and_separate_competencies(self):
        p=four_book_teaching_passport()
        self.assertEqual(set(p["modules"]),{"TC1","NTD","TC2","COURSE"})
        self.assertFalse(p["execution_allowed"])
        self.assertIsNone(p["book_mastery_percent"])
        self.assertFalse(p["GGUF_weights_trained_here"])
        reviewed={x["id"]:x for x in registry()["rule_cards"]}
        seen=set()
        for edition,lessons in p["modules"].items():
            self.assertEqual(len(lessons),3)
            for lesson in lessons:
                self.assertEqual(lesson["source_ed"],edition)
                self.assertFalse(lesson["expert_chart_acceptance_done"])
                for rule in lesson["reviewed_source_rule_references"]:
                    self.assertEqual(reviewed[rule["id"]]["source"],edition)
                    self.assertEqual(reviewed[rule["id"]]["status"],"REVIEWED_PAGES")
                    self.assertEqual(rule["pdf_pages"],reviewed[rule["id"]]["pdf_pages"])
                    seen.add(rule["id"])
        self.assertGreaterEqual(len(seen),34)

    def test_selected_book_can_never_leak_a_different_books_rules(self):
        for ed in ("TC1","NTD","TC2","COURSE"):
            report=four_book_teaching_passport(edition=ed)
            self.assertEqual(set(report["modules"]),{ed})
            lesson=single_book_model_lesson(ed)
            self.assertFalse(lesson["trading_authority"])
            self.assertTrue(lesson["cross_edition_rule_substitution_forbidden"])
            for objective in lesson["lessons"]:
                self.assertTrue(all(x.startswith(ed+".") for x in objective["rule_ids"]))
        with self.assertRaisesRegex(ValueError,"UNKNOWN_OR_MIXED"):
            single_book_model_lesson("TC2+NTD")


class FourBookBlindExamTests(unittest.TestCase):
    def test_reproducible_balanced_four_books_and_no_answer_leak(self):
        tasks=four_book_blind_cases()
        self.assertEqual(tasks,four_book_blind_cases())
        self.assertEqual(len(tasks),32)
        self.assertEqual(Counter(x["public"]["edition"] for x in tasks),
                         {"TC1":8,"NTD":8,"TC2":8,"COURSE":8})
        self.assertEqual(len({x["public"]["id"] for x in tasks}),32)
        for row in tasks:
            self.assertEqual(set(row["public"]),{"id","edition","question","choices"})
            self.assertEqual(set(row["public"]["choices"]),set("ABC"))
            self.assertIn(row["expected"],"ABC")
            self.assertNotIn("expected",row["public"])
            self.assertNotIn("correct",row["public"])
        self.assertTrue(any(a["public"]["choices"]!=b["public"]["choices"]
                            for a,b in zip(tasks,four_book_blind_cases(seed=12345))))
        with self.assertRaises(ValueError):
            four_book_blind_cases(seed=True)

    def test_perfect_mock_is_not_called_trained_qwen_or_broker_trade(self):
        key={x["public"]["id"]:x["expected"] for x in four_book_blind_cases()}
        class Perfect:
            def answer_exam_case(self,public):
                self.check(public)
                return {"choice":key[public["id"]],"reason":"independent"}
            def check(self,public):
                self_class=public
                if "expected" in self_class or "source_rule_id" in self_class:
                    raise AssertionError("ANSWER_KEY_LEAKED")
        score=grade_four_book_model(Perfect())
        self.assertEqual(score["correct"],32)
        self.assertTrue(score["research_pass_not_book_mastery"])
        self.assertFalse(score["model_weights_trained_here"])
        self.assertFalse(score["expert_wave_validation_completed"])
        self.assertFalse(score["execution_allowed"])
        self.assertIsNone(score["trade_action"])

    def test_abstentions_and_one_letter_guesses_cannot_pass(self):
        class Refuse:
            def answer_exam_case(self,public):
                return {"choice":"","reason":"UNKNOWN"}
        r=grade_four_book_model(Refuse())
        self.assertEqual(r["correct"],0)
        self.assertEqual(sum(s["abstentions_counted_wrong"]
                             for s in r["by_edition"].values()),32)
        self.assertFalse(r["research_pass_not_book_mastery"])
        class AlwaysA:
            def answer_exam_case(self,public):
                return {"choice":"A","reason":"always same letter"}
        same=grade_four_book_model(AlwaysA())
        self.assertFalse(same["research_pass_not_book_mastery"])
        self.assertLess(same["correct"],32)


if __name__=="__main__":
    unittest.main()
