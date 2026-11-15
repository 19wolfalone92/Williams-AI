import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class V7SourceReviewBacklogTests(unittest.TestCase):
    def test_rule_check_005_angulation_is_sha_pinned_qualitative_source_not_gold(self):
        data = json.loads((ROOT / "research/williams_v7_review_targets_20261115.json").read_text())
        self.assertFalse(data["training_approved"])
        self.assertFalse(data["independent_experts_verified"])
        self.assertEqual(data["expert_reviews_received"], 0)
        self.assertIsNone(data["valid_gold_labels"])
        self.assertTrue(data["no_trading"])
        case = next(x for x in data["cases"] if x["review_id"] == "RULE_CHECK_005")
        review = case["source_recheck"]
        self.assertEqual(review["source_pdf_sha256"],
                         "055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f")
        self.assertTrue(review["source_sha256_verified_against_original_library_pdf"])
        self.assertEqual(review["physical_pdf_pages"], [121, 122, 123, 124])
        self.assertEqual(review["printed_pages"], [106, 107, 108, 109])
        self.assertTrue(review["visual_diagrams_reviewed"])
        findings = review["findings"]
        for key in ("definition", "rule_1", "rule_2", "rule_3", "rule_4",
                    "negative_example", "consequence"):
            self.assertTrue(findings[key])
        limits = review["not_established_by_source"]
        self.assertIsNone(limits["numeric_slope_threshold"])
        self.assertIsNone(limits["fixed_lookback_bar_count"])
        self.assertIsNone(limits["regression_formula"])
        self.assertFalse(limits["distance_from_jaw_alone_sufficient"])
        self.assertFalse(review["independent_experts_verified"])
        self.assertEqual(review["reviewer_signatures_received"], 0)
        self.assertFalse(review["approved_for_SFT"])
        self.assertFalse(review["approved_for_trading"])
        self.assertEqual(case["status"], "SOURCE_RECHECKED_EXPERT_REVIEW_PENDING")

    def test_rule_check_006_wm2_is_three_consecutive_ao_direction_bars_not_zero_cross(self):
        data = json.loads((ROOT / "research/williams_v7_review_targets_20261115.json").read_text())
        self.assertFalse(data["training_approved"])
        self.assertFalse(data["independent_experts_verified"])
        self.assertEqual(data["expert_reviews_received"], 0)
        self.assertIsNone(data["valid_gold_labels"])
        self.assertTrue(data["no_trading"])
        case = next(x for x in data["cases"] if x["review_id"] == "RULE_CHECK_006")
        review = case["source_recheck"]
        self.assertEqual(review["source_pdf_sha256"],
                         "055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f")
        self.assertTrue(review["source_sha256_verified_against_original_library_pdf"])
        self.assertEqual(review["physical_pdf_pages"], [135, 136, 137, 138, 139])
        self.assertEqual(review["printed_pages"], [120, 121, 122, 123, 124])
        self.assertTrue(review["visual_diagrams_reviewed"])
        findings = review["findings"]
        for key in ("setup_context", "histogram_color", "long_sequence",
                    "long_price_trigger", "short_sequence", "short_price_trigger",
                    "visual_examples"):
            self.assertTrue(findings[key])
        guards = review["interpretation_guards"]
        self.assertTrue(guards["not_three_alternating_color_changes"])
        self.assertTrue(guards["not_three_bars_above_or_below_zero"])
        self.assertTrue(guards["zero_crossing_not_required_by_this_wm2_rule"])
        self.assertTrue(guards["not_fixed_three_bars_after_wm1_entry"])
        self.assertTrue(guards["count_consecutive_rising_or_falling_ao_histogram_bars"])
        limits = review["source_does_not_establish"]
        self.assertIsNone(limits["universal_maximum_wait_after_wm1"])
        self.assertIsNone(limits["independent_position_size_rule"])
        self.assertIsNone(limits["exchange_fill_confirmation_rule"])
        self.assertFalse(review["independent_experts_verified"])
        self.assertEqual(review["reviewer_signatures_received"], 0)
        self.assertFalse(review["approved_for_SFT"])
        self.assertFalse(review["approved_for_trading"])
        self.assertEqual(case["status"], "SOURCE_RECHECKED_EXPERT_REVIEW_PENDING")

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
        self.assertIn("creative-process", review["excluded_page_164"])
        self.assertIn("not an opposite-signal or position-management rule", review["excluded_page_164"])
        self.assertFalse(review["independent_experts_verified"])
        self.assertEqual(review["reviewer_signatures_received"], 0)
        self.assertFalse(review["approved_for_SFT"])
        self.assertEqual(case["status"], "SOURCE_RECHECKED_EXPERT_REVIEW_PENDING")


if __name__ == "__main__":
    unittest.main()
