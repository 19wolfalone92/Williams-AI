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


    def test_rule_check_007_wm3_teeth_filter_is_at_activation_not_formation(self):
        data = json.loads((ROOT / "research/williams_v7_review_targets_20261115.json").read_text())
        self.assertFalse(data["training_approved"])
        self.assertFalse(data["independent_experts_verified"])
        self.assertEqual(data["expert_reviews_received"], 0)
        self.assertIsNone(data["valid_gold_labels"])
        self.assertTrue(data["no_trading"])
        case = next(x for x in data["cases"] if x["review_id"] == "RULE_CHECK_007")
        review = case["source_recheck"]
        self.assertEqual(review["source_pdf_sha256"],
                         "055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f")
        self.assertTrue(review["source_sha256_recomputed_from_original_library_pdf"])
        self.assertTrue(review["source_sha256_verified_against_pinned_target"])
        self.assertEqual(review["physical_pdf_pages"], [147, 148, 149])
        self.assertEqual(review["printed_pages"], [132, 133, 134])
        self.assertEqual(review["visual_diagrams_reviewed"],
                         ["Figure 11.3", "Figure 11.4", "Figure 11.5", "Figure 11.6"])
        findings = review["findings"]
        for key in ("formation_vs_activation", "buy_stop", "sell_stop",
                    "teeth_filter_at_activation", "formation_location_not_decisive",
                    "pending_lifetime", "valid_invalid_examples"):
            self.assertTrue(findings[key])
        guards = review["interpretation_guards"]
        self.assertTrue(guards["formation_above_or_below_teeth_alone_is_not_sufficient"])
        self.assertTrue(guards["evaluate_teeth_at_activation_not_formation"])
        self.assertTrue(guards["fractal_confirmation_is_not_a_filled_entry"])
        self.assertTrue(guards["pending_signal_lifetime_until_trigger_or_new_fractal"])
        self.assertTrue(guards["do_not_infer_intrabar_order_from_ohlc_without_evidence"])
        self.assertIsNone(review["source_does_not_establish"]["numeric_teeth_tolerance_or_buffer"])
        self.assertFalse(review["independent_experts_verified"])
        self.assertEqual(review["reviewer_signatures_received"], 0)
        self.assertFalse(review["approved_for_SFT"])
        self.assertFalse(review["approved_for_trading"])
        self.assertEqual(case["status"], "SOURCE_RECHECKED_EXPERT_REVIEW_PENDING")


    def test_rule_check_009_course_w4_zero_line_cross_is_minimum_not_completion(self):
        data = json.loads((ROOT / "research/williams_v7_review_targets_20261115.json").read_text())
        self.assertFalse(data["training_approved"])
        self.assertFalse(data["independent_experts_verified"])
        self.assertEqual(data["expert_reviews_received"], 0)
        self.assertIsNone(data["valid_gold_labels"])
        case = next(x for x in data["cases"] if x["review_id"] == "RULE_CHECK_009")
        review = case["source_recheck"]
        self.assertEqual(review["source_pdf_sha256"],
                         "fc57f707378087bfc5146db5e242c51ce8135e0b359f0c607305b364e0decfca")
        self.assertTrue(review["source_sha256_recomputed_from_original_library_pdf"])
        self.assertTrue(review["source_sha256_verified_against_pinned_target"])
        self.assertEqual(review["physical_pdf_pages"], [43, 44, 45])
        self.assertTrue(review["high_resolution_page_scans_reviewed"])
        findings = review["findings"]
        for key in ("wave_degree_window", "less_than_100", "more_than_140",
                    "signal_line_context", "zero_line_trigger",
                    "minimum_not_completion", "structural_context"):
            self.assertTrue(findings[key])
        guards = review["interpretation_guards"]
        self.assertTrue(guards["touch_or_approach_is_not_crossing"])
        self.assertTrue(guards["zero_cross_means_minimum_w4_requirements_not_w4_completion"])
        self.assertTrue(guards["bar_window_applies_to_wave_sequence_not_total_loaded_history"])
        self.assertTrue(guards["no_unconditional_w5_order_from_zero_cross"])
        self.assertFalse(review["independent_experts_verified"])
        self.assertEqual(review["reviewer_signatures_received"], 0)
        self.assertFalse(review["approved_for_SFT"])
        self.assertFalse(review["approved_for_trading"])
        self.assertEqual(case["status"], "SOURCE_RECHECKED_EXPERT_REVIEW_PENDING")

    def test_rule_check_010_course_preserves_nested_wave3_alternative_and_w5_target_zone(self):
        data = json.loads((ROOT / "research/williams_v7_review_targets_20261115.json").read_text())
        self.assertFalse(data["training_approved"])
        self.assertFalse(data["independent_experts_verified"])
        self.assertEqual(data["expert_reviews_received"], 0)
        self.assertIsNone(data["valid_gold_labels"])
        case = next(x for x in data["cases"] if x["review_id"] == "RULE_CHECK_010")
        review = case["source_recheck"]
        self.assertEqual(review["source_pdf_sha256"],
                         "fc57f707378087bfc5146db5e242c51ce8135e0b359f0c607305b364e0decfca")
        self.assertEqual(review["physical_pdf_pages"], [43, 44, 45])
        self.assertTrue(review["source_sha256_recomputed_from_original_library_pdf"])
        findings = review["findings"]
        for key in ("wave_degree_window", "under_100", "over_140",
                    "nested_divergence", "alternative_count", "larger_w4_minimum",
                    "w5_target", "nested_w5_target"):
            self.assertTrue(findings[key])
        guards = review["interpretation_guards"]
        self.assertTrue(guards["non_zero_return_divergence_not_automatically_wave5"])
        self.assertTrue(guards["nested_wave3_of_wave3_alternative_must_be_preserved"])
        self.assertTrue(guards["zero_return_indicates_minimum_w4_requirements_not_completion"])
        self.assertTrue(guards["w5_target_zone_not_guaranteed_take_profit"])
        self.assertFalse(review["independent_experts_verified"])
        self.assertEqual(review["reviewer_signatures_received"], 0)
        self.assertFalse(review["approved_for_SFT"])
        self.assertFalse(review["approved_for_trading"])
        self.assertEqual(case["status"], "SOURCE_RECHECKED_EXPERT_REVIEW_PENDING")


    def test_rule_check_001_tc1_fractal_candidate_start_signal_and_leverage_cancel_are_distinct(self):
        data = json.loads((ROOT / "research/williams_v7_review_targets_20261115.json").read_text())
        self.assertFalse(data["training_approved"])
        self.assertFalse(data["independent_experts_verified"])
        self.assertEqual(data["expert_reviews_received"], 0)
        self.assertIsNone(data["valid_gold_labels"])
        case = next(x for x in data["cases"] if x["review_id"] == "RULE_CHECK_001")
        review = case["source_recheck"]
        self.assertEqual(review["source_pdf_sha256"],
                         "36f6822811088a06254009181480aa1c27af705a76960ecf0342023df803331c")
        self.assertTrue(review["source_sha256_recomputed_from_original_library_pdf"])
        self.assertTrue(review["source_sha256_verified_against_pinned_target"])
        self.assertEqual(review["physical_pdf_pages"], [72, 73, 74])
        findings = review["findings"]
        for key in ("paired_structure", "candidate_confirmation", "signal_initiation",
                    "leverage_decay", "price_invalidation", "fractal_stop",
                    "stop_vs_stop_reverse"):
            self.assertTrue(findings[key])
        guards = review["interpretation_guards"]
        self.assertTrue(guards["three_bar_candidate_not_confirmed_fractal"])
        self.assertTrue(guards["wait_for_two_subsequent_closed_bars"])
        self.assertTrue(guards["do_not_backdate_confirmation_to_central_bar"])
        self.assertTrue(guards["start_and_signal_are_distinct"])
        self.assertTrue(guards["fractal_stop_not_same_as_stop_and_reverse"])
        self.assertFalse(review["independent_experts_verified"])
        self.assertEqual(review["reviewer_signatures_received"], 0)
        self.assertFalse(review["approved_for_SFT"])
        self.assertFalse(review["approved_for_trading"])
        self.assertEqual(case["status"], "SOURCE_RECHECKED_EXPERT_REVIEW_PENDING")


    def test_rule_check_002_tc1_wave_hypothesis_risk_stop_and_realized_exit_are_distinct(self):
        data = json.loads((ROOT / "research/williams_v7_review_targets_20261115.json").read_text())
        self.assertFalse(data["training_approved"])
        self.assertFalse(data["independent_experts_verified"])
        self.assertEqual(data["expert_reviews_received"], 0)
        self.assertIsNone(data["valid_gold_labels"])
        case = next(x for x in data["cases"] if x["review_id"] == "RULE_CHECK_002")
        review = case["source_recheck"]
        self.assertEqual(review["source_pdf_sha256"],
                         "36f6822811088a06254009181480aa1c27af705a76960ecf0342023df803331c")
        self.assertTrue(review["source_sha256_recomputed_from_original_library_pdf"])
        self.assertTrue(review["source_sha256_verified_against_pinned_target"])
        self.assertEqual(review["physical_pdf_pages"], [89, 90, 91, 92, 93])
        self.assertEqual(review["visual_diagrams_reviewed"],
                         ["Figure 9.8", "Figure 9.9", "Figure 9.10", "Figure 9.11"])
        findings = review["findings"]
        for key in ("zero_point_hypothesis", "explicit_uncertainty", "contingency_plan",
                    "risk_stop_separate", "partial_exit_context", "campaign_management",
                    "planned_vs_executed", "examples_not_universal"):
            self.assertTrue(findings[key])
        guards = review["interpretation_guards"]
        self.assertTrue(guards["zero_point_is_not_certain_at_inception"])
        self.assertTrue(guards["planned_stop_not_same_as_trigger_or_fill"])
        self.assertTrue(guards["partial_reduction_not_full_exit"])
        self.assertTrue(guards["wave_hypothesis_stop_order_fill_and_position_are_distinct"])
        self.assertTrue(guards["opposite_fractal_does_not_always_close_whole_position"])
        self.assertFalse(review["independent_experts_verified"])
        self.assertEqual(review["reviewer_signatures_received"], 0)
        self.assertFalse(review["approved_for_SFT"])
        self.assertFalse(review["approved_for_trading"])
        self.assertEqual(case["status"], "SOURCE_RECHECKED_EXPERT_REVIEW_PENDING")


    def test_rule_check_003_ntd_balance_line_base_and_one_two_extremes(self):
        data = json.loads((ROOT / "research/williams_v7_review_targets_20261115.json").read_text())
        self.assertFalse(data["training_approved"])
        self.assertFalse(data["independent_experts_verified"])
        self.assertEqual(data["expert_reviews_received"], 0)
        self.assertIsNone(data["valid_gold_labels"])
        case = next(x for x in data["cases"] if x["review_id"] == "RULE_CHECK_003")
        review = case["source_recheck"]
        self.assertEqual(review["source_pdf_sha256"],
                         "855eb052353c852f208436952b12e46d9a18badf7c1dac7ade7580d57b9f4bb4")
        self.assertTrue(review["source_sha256_recomputed_from_original_library_pdf"])
        self.assertTrue(review["source_sha256_verified_against_pinned_target"])
        self.assertEqual(review["pdf_page_labels_reviewed"], [99, 103, 104])
        findings = review["findings"]
        for key in ("scan_direction", "relevant_extreme", "buy_base", "sell_base",
                    "buy_count", "sell_count", "buy_above_balance_line_special_case",
                    "fresh_highs_need_pullback"):
            self.assertTrue(findings[key])
        guards = review["interpretation_guards"]
        self.assertTrue(guards["base_identified_before_counting_extremes"])
        self.assertTrue(guards["one_or_two_counts_new_extremes_not_elapsed_bars"])
        self.assertTrue(guards["two_required_when_moving_toward_balance_line"])
        self.assertTrue(guards["special_case_evidence_is_buy_side_only"])
        self.assertFalse(review["independent_experts_verified"])
        self.assertEqual(review["reviewer_signatures_received"], 0)
        self.assertFalse(review["approved_for_SFT"])
        self.assertFalse(review["approved_for_trading"])
        self.assertEqual(case["status"], "SOURCE_RECHECKED_EXPERT_REVIEW_PENDING")

    def test_rule_check_004_ntd_pending_buy_stop_moves_only_on_trigger_or_new_base(self):
        data = json.loads((ROOT / "research/williams_v7_review_targets_20261115.json").read_text())
        self.assertFalse(data["training_approved"])
        self.assertFalse(data["independent_experts_verified"])
        self.assertEqual(data["expert_reviews_received"], 0)
        self.assertIsNone(data["valid_gold_labels"])
        case = next(x for x in data["cases"] if x["review_id"] == "RULE_CHECK_004")
        review = case["source_recheck"]
        self.assertEqual(review["source_pdf_sha256"],
                         "855eb052353c852f208436952b12e46d9a18badf7c1dac7ade7580d57b9f4bb4")
        self.assertEqual(review["pdf_page_labels_reviewed"], [103, 104])
        self.assertEqual(review["additional_context_page_reviewed"], [105])
        findings = review["findings"]
        for key in ("pending_buy_stop_stability", "stop_replacement_conditions",
                    "new_base_retest", "visual_non_moving_bars", "scope"):
            self.assertTrue(findings[key])
        guards = review["interpretation_guards"]
        self.assertTrue(guards["do_not_slide_stop_for_every_higher_high"])
        self.assertTrue(guards["reapply_base_selection_before_replacing_stop"])
        self.assertTrue(guards["no_stop_move_before_new_base_confirmed"])
        self.assertTrue(guards["chart_trigger_not_same_as_broker_fill"])
        self.assertFalse(review["independent_experts_verified"])
        self.assertEqual(review["reviewer_signatures_received"], 0)
        self.assertFalse(review["approved_for_SFT"])
        self.assertFalse(review["approved_for_trading"])
        self.assertEqual(case["status"], "SOURCE_RECHECKED_EXPERT_REVIEW_PENDING")


if __name__ == "__main__":
    unittest.main()
