"""V6 candidate research registry does not convert proposals into training truth."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys
import unittest

from williams_ai.book_rules import registry
from williams_ai.v6_rule_registry import (
    REGISTRY, PROPOSALS, PINNED_GIT_BLOB, _git_blob_sha,
    build_candidate_index, main,
)


class V6CandidateSourceRegistry(unittest.TestCase):
    def test_exact_frozen_parent_registry_blob(self):
        self.assertEqual(_git_blob_sha(REGISTRY.read_bytes()), PINNED_GIT_BLOB)

    def test_all_four_books_and_pending_six_corrections(self):
        before = REGISTRY.read_bytes()
        result = build_candidate_index()
        self.assertEqual(result["unapproved_corrections_count"], 6)
        self.assertEqual(set(result["rule_counts"]),
                         {"TC1", "NTD", "TC2", "COURSE"})
        self.assertEqual(sum(result["rule_counts"].values()),
                         len(registry()["rule_cards"]))
        self.assertEqual(REGISTRY.read_bytes(), before)
        self.assertTrue(all(row["independent_expert_reviews"] == 0 and
                            row["approved_market_chart_labels"] == 0 and
                            not row["training_allowed"] and
                            not row["execution_allowed"] for row in result["rules"]))
        self.assertEqual(result["independently_approved_market_cases"], 0)
        self.assertTrue(result["versioned_not_training_dataset"])
        self.assertFalse(result["test_gold_included"])

    def test_unverified_course_figure_not_promoted(self):
        result = build_candidate_index()
        cards = {x["rule_id"]: x for x in result["rules"]}
        self.assertTrue(cards["COURSE.VISUAL.NESTED_ABC"]["needs_original_diagram_review"])
        self.assertFalse(cards["COURSE.VISUAL.NESTED_ABC"]["training_allowed"])
        self.assertEqual(cards["TC1.RHYTHM"]["v6_research_status"],
                         "PENDING_SECOND_INDEPENDENT_SOURCE_REVIEW")
        self.assertNotEqual(cards["TC1.RHYTHM"]["proposed_revised_paraphrase"],
                            cards["TC1.RHYTHM"]["v5_checked_paraphrase"])

    def test_cannot_use_changed_or_shortened_v5_snapshot(self):
        with TemporaryDirectory() as tmp:
            wrong = Path(tmp) / "registry.json"
            wrong.write_bytes(REGISTRY.read_bytes()[:-1])
            with patch("williams_ai.v6_rule_registry.REGISTRY", wrong):
                with self.assertRaisesRegex(ValueError, "FROZEN_V5_SNAPSHOT_CHANGED"):
                    build_candidate_index()

    def test_forged_approval_cannot_upgrade_rule(self):
        obj = json.loads(PROPOSALS.read_text(encoding="utf-8"))
        obj["independent_expert_approval_count"] = 2
        with TemporaryDirectory() as tmp:
            forged = Path(tmp) / "forged.json"
            forged.write_text(json.dumps(obj), encoding="utf-8")
            with patch("williams_ai.v6_rule_registry.PROPOSALS", forged):
                with self.assertRaisesRegex(ValueError, "UNAPPROVED_PROPOSALS_MISMATCH"):
                    build_candidate_index()

    def test_export_is_source_only_and_non_overwriting(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            with patch.object(sys, "argv", ["v6_registry", "--out", str(path)]):
                main()
            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(result["no_real_trading"])
            self.assertEqual(result["schema"],
                             "WILLIAMS_V6_CANDIDATE_SOURCE_INDEX_NOT_TRAIN_DATA")
            self.assertFalse(result["main_merge_allowed"])
            with patch.object(sys, "argv", ["v6_registry", "--out", str(path)]):
                with self.assertRaises(FileExistsError):
                    main()


if __name__ == "__main__":
    unittest.main()
