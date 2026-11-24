import json
import unittest

from williams_ai.knowledge_curriculum_v5 import (
    V6_MODULES,
    build,
    build_v6_source,
    registry,
    source_only_admission,
)


class WilliamsAISourceOnlyV6Tests(unittest.TestCase):
    def test_source_only_admission_does_not_claim_expert_market_gold(self):
        gate = source_only_admission()
        self.assertTrue(gate["source_only_sft_permitted"])
        self.assertFalse(gate["expert_market_gold_training_permitted"])
        self.assertEqual(gate["independent_expert_reviews"], 0)
        self.assertEqual(gate["chart_market_gold_rows"], 0)
        self.assertFalse(gate["full_899_page_semantic_review_complete"])
        self.assertFalse(gate["trading_qualification"])
        self.assertFalse(gate["execution_allowed"])
        self.assertEqual(gate["source_lessons_total"], 29)
        self.assertEqual(gate["source_lessons_eligible"], 28)

    def test_source_curriculum_covers_all_fifteen_competencies(self):
        rows, manifest = build_v6_source()
        self.assertEqual(set(manifest["module_coverage"]), set(V6_MODULES))
        self.assertTrue(all(manifest["module_coverage"][m] > 0 for m in V6_MODULES))
        self.assertEqual(manifest["row_count"], len(rows))
        self.assertGreaterEqual(len(rows), 100)
        self.assertEqual(manifest["split_rows"]["train"], len(rows))
        self.assertEqual(manifest["split_rows"]["validation"], 0)
        self.assertEqual(manifest["split_rows"]["test"], 0)
        self.assertTrue(manifest["all_rows_train"])
        self.assertTrue(manifest["old_v5_432_case_database_preserved"])
        self.assertTrue(manifest["old_revealed_exams_for_regression_only"])

    def test_source_conflicts_are_uncertainty_examples_not_gold(self):
        rows, _ = build_v6_source()
        conflicts = [r for r in rows if r["task_type"] == "SOURCE_CONFLICT"]
        self.assertEqual(len(conflicts), 4)
        for row in conflicts:
            answer = json.loads(row["answer"])
            self.assertEqual(answer["status"], "UNRESOLVED_SOURCE_CONFLICT")
            self.assertTrue(answer["do_not_choose_one_version"])
            self.assertFalse(answer["independent_expert_approval"])
            self.assertFalse(row["expert_market_gold"])
            self.assertFalse(row["chart_expert_gold"])

    def test_tc2_english_is_cross_reference_not_a_fifth_book(self):
        pack = registry()
        self.assertEqual(len(pack["books"]), 4)
        tc2 = next(x for x in pack["books"] if x["id"] == "TC2")
        variants = tc2["cross_reference_editions"]
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0]["id"], "TC2_EN")
        self.assertTrue(variants[0]["page_numbers_are_not_interchangeable"])
        self.assertEqual(variants[0]["pages"], 257)

    def test_no_duplicate_case_ids_or_prompt_families(self):
        rows, _ = build_v6_source()
        ids = [r["case_id"] for r in rows]
        prompts = [
            json.dumps(r["prompt"], ensure_ascii=False, sort_keys=True)
            for r in rows
        ]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(prompts), len(set(prompts)))

    def test_v5_curriculum_remains_rebuildable_and_unchanged_in_scope(self):
        rows, manifest = build()
        self.assertEqual(len(rows), 432)
        self.assertEqual(manifest["split_rows"]["train"], 300)
        self.assertEqual(manifest["split_rows"]["validation"], 48)
        self.assertEqual(manifest["split_rows"]["test"], 84)


if __name__ == "__main__":
    unittest.main()
