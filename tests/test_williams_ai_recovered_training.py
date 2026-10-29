import copy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.ai_continue_own_checkpoint_kaggle import checked_checkpoint,MODEL,REVISION,SOURCE_SHA,TARGETS
from scripts.ai_run_private_deep_kaggle import worker_source
from scripts.ai_real_qwen_source_lora_start import build_public_exercises
from williams_ai.structured_rule_practice import structured_rule_cases,balanced_recovery_curriculum
from williams_ai.structured_transfer_exam import sealed_cases


class RecoveredTrainingTests(unittest.TestCase):
    def fixture(self):
        return ({"r":16,"lora_alpha":32,"bias":"none","target_modules":sorted(TARGETS),"peft_type":"LORA"},
            {"global_step":416,"max_steps":512,"best_model_checkpoint":"/own/checkpoint-416"},
            {"model_repo":MODEL,"model_revision":REVISION,"source_commit":SOURCE_SHA,"rank":16,"execution_allowed":False})

    def test_checkpoint_cannot_masquerade_as_other_base_or_complete_schedule(self):
        config,state,report=self.fixture()
        for key,value in (("model_revision","unverified"),("source_commit","a"*40),("execution_allowed",True)):
            wrong={**report,key:value}
            with self.assertRaisesRegex(ValueError,"PROVENANCE"):
                checked_checkpoint(config,state,wrong,b"")
        for key,value in (("global_step",384),("max_steps",608),("best_model_checkpoint","/own/checkpoint-384")):
            with self.assertRaisesRegex(ValueError,"PARTIAL_416"):
                checked_checkpoint(config,{**state,key:value},report,b"")

    def test_incompatible_adapter_and_changed_weight_bytes_rejected_before_loading(self):
        config,state,report=self.fixture()
        for key,value in (("r",8),("modules_to_save",["lm_head"]),("target_modules",["q_proj"]),("use_dora",True)):
            with self.assertRaisesRegex(ValueError,"PARTIAL_416"):
                checked_checkpoint({**config,key:value},state,report,b"")
        with self.assertRaisesRegex(ValueError,"IMMUTABLE_WEIGHT_SHA"):
            checked_checkpoint(config,state,report,b"not our safetensors")

    @unittest.skipUnless(importlib.util.find_spec("safetensors") and importlib.util.find_spec("numpy"),"optional tensor inspection dependency")
    def test_safe_tensor_admission_verifies_finite_and_updated_lora_matrices(self):
        import numpy as np
        from safetensors.numpy import save
        config,state,report=self.fixture()
        tensors={f"base_model.model.layers.{i}.self_attn.q_proj.lora_{part}.weight":np.ones((1,1),dtype=np.float32)
                 for i in range(252) for part in ("A","B")}
        def check(t):
            raw=save(t)
            with patch("scripts.ai_continue_own_checkpoint_kaggle.WEIGHT_SHA",sha256(raw).hexdigest()),patch("scripts.ai_continue_own_checkpoint_kaggle.PARAMETERS",504):
                return checked_checkpoint(config,state,report,raw)
        proof=check(tensors)
        self.assertFalse(proof["first_schedule_completed"])
        self.assertFalse(proof["optimizer_state_resumed"])
        for invalid in (np.nan,0):
            wrong=copy.deepcopy(tensors)
            next(v for n,v in wrong.items() if ".lora_B." in n)[0,0]=invalid
            with self.assertRaisesRegex(ValueError,"FINITE_ACTUALLY_TRAINED"):
                check(wrong)

    def test_structured_arithmetic_and_rule_counterexamples_are_correct(self):
        rows=structured_rule_cases()
        self.assertEqual([len(rows[s]) for s in ("train","validation")],[112,56])
        first=json.loads(rows["train"][0]["answer"])
        self.assertEqual(first,{"risk_budget":5.615,"quantity":4.66,"execution_allowed":False})
        for rule,field in (("COURSE.HUMP","hump_valid"),("COURSE.W4.ZERO","minimum_w4_met"),("TC2.WM3","confirmed")):
            answers=[json.loads(r["answer"])[field] for r in rows["train"] if r["rule_id"]==rule]
            self.assertEqual(set(answers),{True,False})
        self.assertFalse({r["question"] for r in rows["train"]}&{r["question"] for r in rows["validation"]})

    def test_balanced_curriculum_preserves_all_public_lessons_and_all_books(self):
        train,val,audit=balanced_recovery_curriculum();public,_,_=build_public_exercises()
        self.assertEqual(len(train),278);self.assertEqual(sum(map(len,val.values())),122)
        questions={r["question"] for r in train}
        self.assertTrue({r["question"] for r in public}<=questions)
        self.assertEqual({r["edition"] for r in train},{"TC1","NTD","TC2","COURSE"})
        self.assertFalse(questions&{r["question"] for rows in val.values() for r in rows})
        self.assertFalse(audit["independent_expert_chart_labels"])

    def test_private_worker_handles_partial_weights_without_optimizer_or_exam_keys(self):
        q,_=sealed_cases()
        initial={"format":"OWN_PARTIAL_RANK16_CHECKPOINT_V1","adapter_sha256":"b"*64,
            "provenance":{"first_schedule_completed":False,"completed_optimizer_steps":416},
            "files":{n:{"url":"https://example.invalid/expiring","sha256":"a"*64} for n in ("adapter_config.json","adapter_model.safetensors")}}
        worker=worker_source("a"*40,b"private training only",b"as-of questions",initial,json.dumps(q).encode())
        compile(worker,"recovered_worker","exec")
        self.assertNotIn("KAGGLE_API_TOKEN",worker);self.assertNotIn("WITHHELD_OUTCOME_KEYS",worker)
        self.assertNotIn('"expected":',worker)
        self.assertIn("source_steps=64 if partial",worker)
        self.assertIn("balanced_structured_practice=partial",worker)
        self.assertIn("SEALED_TRANSFER_QUESTION_BYTES_CHANGED",worker)


if __name__=="__main__":unittest.main()
