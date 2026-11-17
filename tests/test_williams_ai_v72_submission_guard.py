import inspect
import unittest
from pathlib import Path
from scripts.ai_v72_guided_json_8_h1_submit import SLUG,main

class V72SubmitSafety(unittest.TestCase):
    def test_unique_short_diagnostic_slug(self):
        self.assertEqual(SLUG,"williams-v72-eosguard8-20261117")
        self.assertLessEqual(len(SLUG),50)
    def test_worker_uses_frozen_eight_pair_ids_and_eos_guard(self):
        # The real GPU worker imports torch; do not import it in CPU-only CI.
        code=(Path(__file__).resolve().parents[1]/
             "scripts/ai_v72_guided_json_8_h1_worker.py").read_text()
        self.assertIn('IDS=tuple(f"HIST100-{n:02d}-{k}" for n in (1,2) for k in (1,2,3,4))',code)
        self.assertIn("from williams_ai.v72_eos_guard import generate",code)
        self.assertIn("model.disable_adapter()",code)
        self.assertIn('"weights_changed":False',code)
        self.assertIn('"trading_allowed":False',code)
    def test_submit_is_private_uses_same_frozen_lora_never_training(self):
        code=inspect.getsource(main)
        for required in ('meta["is_private"]=="true"',
                         'meta["machine_shape"]=="NvidiaTeslaT4"',
                         'meta["kernel_sources"]==[owner+"/"+TRAINING_NAME]',
                         "prior=api.kernels_status","interegular==0.3.3",
                         "scripts.ai_v72_guided_json_8_h1_worker"):
            self.assertIn(required,code)
        self.assertNotIn("Trainer(",code)
        self.assertNotIn("train_dataset",code)
if __name__=="__main__":
    unittest.main()
