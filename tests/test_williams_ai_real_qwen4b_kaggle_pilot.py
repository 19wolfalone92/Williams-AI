"""Guard against fake Qwen3-4B GPU training results in the public pilot."""
import re
import tempfile
from pathlib import Path
import unittest

from scripts.ai_real_qwen4b_public_qlora_gpu import (
    MODEL_REPO,MODEL_REVISION,train_public_qlora
)
from scripts.ai_real_qwen_source_lora_start import build_public_exercises


class RealQwen3_4BGpuPublicStudyContract(unittest.TestCase):
    def test_pinned_official_base_is_immutable_4b_model(self):
        self.assertEqual(MODEL_REPO,"Qwen/Qwen3-4B")
        self.assertTrue(re.fullmatch(r"[0-9a-f]{40}",MODEL_REVISION))
        train,validation,provenance=build_public_exercises()
        self.assertGreaterEqual(len(train),20)
        self.assertEqual(set(validation),{"TC1","NTD","TC2","COURSE"})
        self.assertFalse(provenance["independent_expert_chart_review"])
        self.assertTrue(provenance["rule_ids_not_shared_across_splits"])

    def test_rejects_fake_or_oversized_optimizer_steps_before_cuda(self):
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)/"new"
            for invalid in (0,1,13,False,2.0):
                with self.assertRaisesRegex(ValueError,"BOUNDED_NEW"):
                    train_public_qlora(target,steps=invalid)
            target.mkdir()
            with self.assertRaisesRegex(ValueError,"BOUNDED_NEW"):
                train_public_qlora(target,steps=6)


if __name__=="__main__":
    unittest.main()
