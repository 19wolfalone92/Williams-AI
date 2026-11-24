import json
import unittest
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from williams_ai.knowledge_curriculum_v5 import (
    V6_MODULES,
    build,
    build_v6_source,
    registry,
    digest,
    export_v6_source_database,
    source_only_admission,
)


class WilliamsAISourceOnlyV6Tests(unittest.TestCase):
    def test_source_only_admission_does_not_claim_expert_market_gold(self):
        gate = source_only_admission()
        self.assertFalse(gate["source_only_sft_permitted"])
        self.assertTrue(gate["candidate_export_permitted"])
        self.assertFalse(gate["expert_market_gold_training_permitted"])
        self.assertEqual(gate["independent_expert_reviews"], 0)
        self.assertEqual(gate["chart_market_gold_rows"], 0)
        self.assertFalse(gate["full_899_page_semantic_review_complete"])
        self.assertFalse(gate["trading_qualification"])
        self.assertFalse(gate["execution_allowed"])
        self.assertEqual(gate["source_lessons_total"], 29)
        self.assertEqual(gate["source_lessons_eligible"], 0)
        self.assertEqual(gate["source_lessons_candidate"], 28)

    def test_source_curriculum_covers_all_fifteen_competencies(self):
        rows, manifest = build_v6_source()
        self.assertEqual(set(manifest["module_coverage"]), set(V6_MODULES))
        self.assertTrue(all(manifest["module_coverage"][m] > 0 for m in V6_MODULES))
        self.assertEqual(manifest["row_count"], len(rows))
        self.assertGreaterEqual(len(rows), 100)
        self.assertEqual(manifest["split_rows"]["train"], 0)
        self.assertEqual(manifest["split_rows"]["candidate"], len(rows))
        self.assertTrue(all(r["split"]=="candidate" and r["eligible_for_sft"] is False for r in rows))
        self.assertEqual(manifest["split_rows"]["validation"], 0)
        self.assertEqual(manifest["split_rows"]["test"], 0)
        self.assertFalse(manifest["all_rows_train"])
        self.assertFalse(manifest["old_v5_database_modified"])
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
        self.assertNotIn("cross_reference_editions", tc2)
        crosswalk=json.loads((Path(__file__).resolve().parents[1]/"knowledge"/"TC2_EN_CROSS_REFERENCE_20261124.json").read_text())
        self.assertEqual(crosswalk["canonical_edition"]["sha256"], tc2["sha256"])
        self.assertEqual(crosswalk["cross_reference_edition"]["pages"], 257)
        self.assertFalse(crosswalk["cross_reference_edition"]["canonical_training_source"])
        self.assertTrue(crosswalk["constraints"]["do_not_merge_page_numbers"])

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

    def test_export_cannot_mark_candidates_as_sft_or_overwrite_existing_db(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/"candidate.sqlite3"
            manifest=export_v6_source_database(p)
            with sqlite3.connect(p) as db:
                self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0],"ok")
                self.assertEqual(db.execute("SELECT count(*) FROM curriculum").fetchone()[0],manifest["row_count"])
                self.assertEqual(db.execute("SELECT count(*) FROM curriculum WHERE eligible_for_sft != 0 OR expert_market_gold != 0 OR execution_allowed != 0").fetchone()[0],0)
            before=p.read_bytes()
            with self.assertRaises(FileExistsError):
                export_v6_source_database(p)
            self.assertEqual(before,p.read_bytes())

    def test_cross_edition_signals_are_not_tc2_mudrec_examples(self):
        rows,_=build_v6_source()
        for row in rows:
            if row["edition"] in {"TC1","NTD","COURSE"} and row["task_type"]!="PRACTICAL":
                self.assertFalse(set(row["module_ids"]) & {"wm1","wm2","wm3","wise_men_interaction"})

    def test_candidate_rows_are_rejected_by_existing_trainer(self):
        from williams_ai.knowledge_training_v5 import tokenize
        rows,_=build_v6_source()
        with self.assertRaisesRegex(ValueError,"NEVER_PASS_TEST_OR_UNVERIFIED_CASE_TO_GPU"):
            tokenize(rows[:1],None)

if __name__ == "__main__":
    unittest.main()
