"""Offline invariants for a research-only Qwen3-4B GPU continuation."""
from __future__ import annotations

import json
import unittest
from decimal import Decimal

from scripts.ai_run_deep_corrective_kaggle import frozen, worker_source
from williams_ai.deep_corrective_gpu import RUN


class DeepCorrectiveContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, cls.design = frozen()

    def test_new_training_has_no_train_validation_test_id_overlap(self):
        all_ids = [row["case_id"] for split in ("train","validation","test")
                   for cohort in RUN.cohorts for row in self.groups[split][cohort]]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(RUN.cohorts), set(self.groups["train"]))
        self.assertFalse(self.design["correction_audit"]["human_expert_wave_labels"])
        self.assertTrue(self.design["correction_audit"]["old_chart_exam_is_revealed_regression"])

    def test_targeted_risk_ac_training_and_new_numeric_test(self):
        train = self.groups["train"]["precision"]
        test = self.groups["test"]["precision"]
        self.assertGreater(len(train), 250)
        new = [r for r in test if r["case_id"].startswith("deep-correction-")]
        self.assertEqual(len(new), 16)
        self.assertEqual(sum(r["rule_id"]=="TC1.PSY.PLAN" for r in new), 8)
        self.assertEqual(sum(r["rule_id"]=="NTD.AC" for r in new), 8)
        self.assertTrue(all(json.loads(r["answer"])["execution_allowed"] is False for r in train+test))

    def test_unacknowledged_stop_prevents_trade_action(self):
        for split in ("train","validation"):
            repairs = [r for r in self.groups[split]["campaign"]
                       if r["case_id"].endswith("-stop-unacknowledged")]
            self.assertGreater(len(repairs), 0)
            self.assertTrue(all(r["operands"]["journal"]["stop_ack"] is False for r in repairs))
            for r in repairs:
                ans=json.loads(r["answer"])
                self.assertEqual(ans["decision"],"AUDIT_PROTECTION")
                self.assertIs(ans["execution_allowed"],False)

    def test_known_qwen_lineage_and_bounded_research(self):
        self.assertEqual(RUN.initial_sha, "d961d7e7406d255e699da3959a3c67606330fee191807627b15de35ac4508ca7")
        self.assertEqual(RUN.limits["optimizer_attempts"], 1024)
        self.assertEqual(RUN.source_repetitions, 2)
        self.assertEqual(RUN.rank_cohorts, (("context","precision"),("waves","campaign")))
        self.assertNotIn("live",RUN.slug)
        self.assertEqual(RUN.protocol["do_sample"],False)

    def test_generated_private_gpu_worker_uses_new_isolated_trainer(self):
        fake={"adapter_config.json":{"url":"https://example.invalid/config","sha256":"0"*64},
              "adapter_model.safetensors":{"url":"https://example.invalid/weights","sha256":"1"*64}}
        worker=worker_source("a"*40,b"{}",b"{}",b"{}",fake,{})
        compile(worker,"deep_qwen_worker","exec")
        self.assertIn("from williams_ai.deep_corrective_gpu import train",worker)
        self.assertIn("Williams_Qwen3_4B_DEEP_CORRECTIVE_20261110.zip",worker)
        self.assertNotIn("from williams_ai.market_decision_gpu import train",worker)
        self.assertNotIn("KAGGLE_API_TOKEN",worker)


if __name__=="__main__":
    unittest.main()
