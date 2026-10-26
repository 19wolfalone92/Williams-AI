"""Offline admission tests for the first REAL pretrained Qwen public-source fit.

These tests do NOT download model weights or prove book mastery.
"""
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.ai_real_qwen_source_lora_start import build_public_exercises, train_source_adapter, ED


class PublicSourceWeightTrainingStartupTests(unittest.TestCase):
    def test_four_books_present_and_rule_ids_never_leak_across_splits(self):
        train, validation, provenance = build_public_exercises()
        self.assertTrue(train)
        self.assertEqual(set(validation), set(ED))
        self.assertTrue(all(validation[edition] for edition in ED))
        train_rules = {row["rule_id"] for row in train}
        test_rules = {row["rule_id"] for book in ED for row in validation[book]}
        self.assertFalse(train_rules & test_rules)
        self.assertTrue(provenance["rule_ids_not_shared_across_splits"])
        self.assertFalse(provenance["independent_expert_chart_review"])
        self.assertTrue(all(row["answer"] and row["question"] for row in train))

    def test_deterministic_boundaries_without_hard_coded_answer_key(self):
        a, b, receipt = build_public_exercises()
        x, y, later = build_public_exercises()
        self.assertEqual((a, b, receipt), (x, y, later))
        questions = [r["question"] for r in a]
        self.assertFalse(any("Выбор: " in q for q in questions))
        self.assertEqual(sum(receipt["validation_editions"].values()),
                         sum(len(rows) for rows in b.values()))

    def test_will_not_start_unbounded_or_reuse_existing_output(self):
        for invalid in (0, 1, 11, True, 2.5):
            with self.subTest(steps=invalid), self.assertRaisesRegex(
                    ValueError, "BOUNDED_NEW_RESEARCH"):
                train_source_adapter(Path("/absent-research-output"), steps=invalid)


if __name__ == "__main__":
    unittest.main()
