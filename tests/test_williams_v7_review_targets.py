import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class V7SourceReviewBacklogTests(unittest.TestCase):
    def test_rule_check_008_references_are_source_rechecked_not_gold(self):
        data = json.loads((ROOT / "research/williams_v7_review_targets_20261115.json").read_text())
        self.assertFalse(data["training_approved"])
        self.assertFalse(data["independent_experts_verified"])
        self.assertEqual(data["expert_reviews_received"], 0)
        self.assertIsNone(data["valid_gold_labels"])
        self.assertTrue(data["no_trading"])
        case = next(x for x in data["cases"] if x["review_id"] == "RULE_CHECK_008")
        review = case["source_recheck"]
        self.assertEqual(review["confirmed_pages"], [131, 149, 150, 151])
        self.assertTrue(review["visual_diagram_page_150_reviewed"])
        self.assertIn("not an opposite-signal", review["excluded_page_164"])
        self.assertFalse(review["independent_experts_verified"])
        self.assertEqual(review["reviewer_signatures_received"], 0)
        self.assertFalse(review["approved_for_SFT"])
        self.assertEqual(case["status"], "SOURCE_RECHECKED_EXPERT_REVIEW_PENDING")


if __name__ == "__main__":
    unittest.main()
