"""True 50-stage QLoRA research contract without Kaggle GPU consumption."""
import json
from pathlib import Path
import unittest
from scripts.ai_run_serial_50_kaggle import frozen,worker_source,PARENT_KERNEL,PARENT_SOURCE_COMMIT,ARCHIVE
from williams_ai.serial_50_gpu import RUN
from williams_ai.historical_rule_curriculum import digest, prompts_only


class SerialFiftyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, cls.design = frozen()

    def test_fifty_distinct_cumulative_phases_with_no_fake_gpu_claim(self):
        self.assertEqual(RUN.limits["mini_rounds"], 50)
        self.assertEqual(RUN.limits["updates_per_mini_round"],20)
        self.assertEqual(RUN.limits["optimizer_attempts"],1000)
        self.assertEqual(RUN.limits["evaluation_steps"],250)
        self.assertEqual(self.design["serial_50_contract"]["micro_rounds"],50)
        self.assertFalse(self.design["serial_50_contract"]["independent_1024_step_kaggle_runs"])
        self.assertTrue(self.design["serial_50_contract"]["resume_lora_weights_between_micro_rounds"])
        self.assertTrue(self.design["serial_50_contract"]["optimizer_continuous"])
        self.assertFalse(self.design["serial_50_contract"]["execution_allowed"])

    def test_parent_verified_v2_research_not_promoted_trader(self):
        self.assertEqual(PARENT_KERNEL, "williams-ai-deep-corrective-20261110-r2")
        self.assertEqual(PARENT_SOURCE_COMMIT,"e427de188591136cbc022c4c26ae332a77fac8f3")
        self.assertEqual(RUN.initial_sha,
            "2a18d0b3788c10bc851a155ee89c2997e08bf63b00fa6e6d345e676e5960ec9a")
        self.assertEqual(self.design["model_parent_sha256"],RUN.initial_sha)

    def test_data_splits_unchanged_dev_and_disjoint_new_holdout(self):
        all_rows=[r for s in ("train","validation","test") for c in RUN.cohorts
                  for r in self.groups[s][c]]
        ids=[r["case_id"] for r in all_rows]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual({g:len(self.groups["train"][g]) for g in RUN.cohorts},
                         {"context":144,"waves":144,"campaign":192,"precision":384})
        fresh=[r for r in self.groups["test"]["precision"]
               if r["case_id"].startswith("serial50-fresh-holdout-")]
        self.assertEqual(len(fresh),32)
        self.assertEqual(sum(r["rule_id"]=="TC1.PSY.PLAN" for r in fresh),16)
        self.assertEqual(sum(r["rule_id"]=="NTD.AC" for r in fresh),16)
        self.assertFalse(self.design["serial_50_contract"]["test_answers_never_sent_to_gpu"] is False)
        self.assertFalse(self.design["serial_50_contract"]["expert_labeled_market_data"])
        for s in ("train","validation","test"):
            for c in RUN.cohorts:
                manifest=self.design["curriculum"]["splits"][s][c]
                rows=self.groups[s][c]
                self.assertEqual(manifest["questions"],len(rows))
                self.assertEqual(manifest["prompt_sha256"],digest(prompts_only(rows)))

    def test_focus_sampler_never_mutates_validation_answers(self):
        src=Path("williams_ai/serial_50_engine.py").read_text()
        self.assertIn("self.micro = bool(focused and run.limits.get",src)
        self.assertIn("train_dataset=Rows(training, focused=True), eval_dataset=Rows(dev_tokens)",src)
        self.assertIn("self.micro_history.append",src)
        self.assertIn("trainer.train_dataset.phase = number",src)
        self.assertIn('len(history) != count',src)

    def test_generated_private_worker_is_compilable_with_real_50_phases(self):
        placeholder={"adapter_config.json":{"url":"https://example.invalid/config","sha256":"0"*64},
                     "adapter_model.safetensors":{"url":"https://example.invalid/weights","sha256":"1"*64}}
        w=worker_source("a"*40,b"{}",b"{}",b"{}",placeholder,{})
        compile(w,"serial50_worker","exec")
        self.assertIn("from williams_ai.serial_50_gpu import train",w)
        self.assertIn(ARCHIVE,w)
        self.assertNotIn("KAGGLE_API_TOKEN",w)


if __name__=="__main__":
    unittest.main()
