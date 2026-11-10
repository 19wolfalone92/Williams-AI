"""Offline gate for second Qwen3-4B research run; no model inference or GPU."""
import json
import unittest
from scripts.ai_run_deep_corrective_v2_kaggle import frozen, worker_source, PARENT_KERNEL, PARENT_SOURCE_COMMIT
from williams_ai.deep_corrective_v2_gpu import RUN


class SecondDeepCorrectiveContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, cls.design = frozen()

    def test_parent_is_audited_v1_not_unavailable_sealed_nov7(self):
        self.assertEqual(PARENT_KERNEL, "williams-ai-deep-corrective-20261110-r1")
        self.assertEqual(PARENT_SOURCE_COMMIT, "3f140b36a11628b7cd2ea2add29347903c808562")
        self.assertEqual(RUN.initial_sha, "976349c696879efb180c41e9d9090cf5ff928f93a77bf23a5e17dbedd6c8bd41")
        self.assertEqual(self.design["model_parent_sha256"], RUN.initial_sha)

    def test_bounded_optimizer_and_execution_disabled(self):
        self.assertEqual(RUN.limits["optimizer_attempts"], 1024)
        self.assertEqual(RUN.limits["learning_rate"], 3e-6)
        self.assertEqual(RUN.source_repetitions, 3)
        self.assertEqual(RUN.rank_cohorts, (("context", "precision"), ("waves", "campaign")))
        self.assertFalse(self.design["correction_audit"]["execution_allowed"])
        self.assertFalse(self.design["correction_audit"]["human_expert_wave_labels"])

    def test_train_only_component_teaching_no_future_label_leak(self):
        cohorts=("context","waves","campaign")
        for cohort in cohorts:
            originals=[x for x in self.groups["train"][cohort] if "-v2-components-" not in x["case_id"]]
            micro=[x for x in self.groups["train"][cohort] if "-v2-components-" in x["case_id"]]
            self.assertEqual(len(micro), 2*min(72,len(originals)))
            self.assertGreater(len(originals), 72)
            for row in micro:
                self.assertEqual(row["label_origin"],"TRAIN_ONLY_MACHINE_CALCULATED_COMPONENTS_NOT_EXPERT_TRADING_LABEL")
                self.assertFalse(json.loads(row["answer"])["execution_allowed"])
            for split in ("validation","test"):
                self.assertFalse(any("-v2-components-" in x["case_id"] for x in self.groups[split][cohort]))

    def test_new_holdout_is_distinct_from_prev_revealed_test(self):
        new=[x for x in self.groups["test"]["precision"] if x["case_id"].startswith("deep-v2-")]
        self.assertEqual(len(new),32)
        fresh_train=[r for r in self.groups["train"]["precision"] if r["case_id"].startswith("deep-v2-")]
        self.assertEqual(len(fresh_train),192)
        self.assertEqual(sum(r["rule_id"]=="TC1.PSY.PLAN" for r in fresh_train),96)
        self.assertEqual(sum(r["rule_id"]=="NTD.AC" for r in fresh_train),96)
        self.assertEqual(sum(x["rule_id"]=="NTD.AC" for x in new),16)
        self.assertEqual(sum(x["rule_id"]=="TC1.PSY.PLAN" for x in new),16)
        all_ids=[r["case_id"] for s in ("train","validation","test")
                 for c in RUN.cohorts for r in self.groups[s][c]]
        self.assertEqual(len(all_ids),len(set(all_ids)))
        self.assertFalse(self.design["correction_audit"]["test_answer_keys_sent_to_gpu"])

    def test_worker_uses_new_trainer_and_never_includes_test_keys(self):
        fake={"adapter_config.json":{"url":"https://example.invalid/config","sha256":"0"*64},
              "adapter_model.safetensors":{"url":"https://example.invalid/weights","sha256":"1"*64}}
        worker=worker_source("a"*40,b"{}",b"{}",b"{}",fake,{})
        compile(worker,"deep_corrective_v2_worker","exec")
        self.assertIn("from williams_ai.deep_corrective_v2_gpu import train",worker)
        self.assertIn("Williams_Qwen3_4B_DEEP_CORRECTIVE_V2_20261110.zip",worker)
        self.assertNotIn("KAGGLE_API_TOKEN",worker)


if __name__ == "__main__":
    unittest.main()
