import json
from pathlib import Path
import unittest

from williams_ai.book_rules import registry
from williams_ai.adaptive_coach import learning_cases


class CourseSourceCorrections(unittest.TestCase):
    def test_previously_inverted_hump_rule_is_corrected_at_both_teaching_handoffs(self):
        card = next(c for c in registry()["rule_cards"] if c["id"] == "COURSE.HUMP")
        case = next(c for c in learning_cases() if c["id"] == "CHUMP")
        self.assertIn("НЕ перекрываются", card["summary"])
        self.assertIn("не должны перекрываться", case["choices"][case["correct"]])

    def test_course_minimum_wave_four_is_not_taught_as_completed_correction(self):
        card = next(c for c in registry()["rule_cards"] if c["id"] == "COURSE.W4.ZERO")
        self.assertIn(44, card["pdf_pages"])
        self.assertIn("НЕ её завершение", card["summary"])
        case = next(c for c in learning_cases() if c["id"] == "CW4")
        self.assertIn("Минимальные требования", case["choices"][case["correct"]])

    def test_page_variation_is_explicit_and_old_weights_are_not_falsely_corrected(self):
        audit = json.loads((Path(__file__).resolve().parents[1] / "knowledge/COURSE_SOURCE_RECHECK_20261027.json").read_text())
        self.assertTrue(audit["first_private_gpu_candidate_requires_correction_before_promotion"])
        self.assertFalse(audit["independent_human_expert_review"])
        card = next(c for c in registry()["rule_cards"] if c["id"] == "COURSE.THUMB")
        self.assertIn(69, card["pdf_pages"])
        self.assertIn("в любом сочетании", card["summary"])
        self.assertIn("inside bars не считаются", card["summary"])
