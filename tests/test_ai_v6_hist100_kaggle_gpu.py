"""Safety- and isolation-focused tests for mixed CPU/GPU historical100."""
import unittest
from scripts.ai_v6_hist100_kaggle_gpu_submit import make_metadata, KERNEL_NAME, TRAINING_NAME
from scripts.ai_v6_hist100_kaggle_gpu_worker import MAX_CONTEXT_TOKENS, MAX_NEW_TOKENS


class KaggleGPUContract(unittest.TestCase):
    def test_private_gpu_and_genuine_source_only(self):
        m, worker = make_metadata("my-kaggle-account", "e" * 40)
        self.assertEqual(m["is_private"], "true")
        self.assertEqual(m["enable_gpu"], "true")
        self.assertEqual(m["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(m["kernel_sources"], ["my-kaggle-account/"+TRAINING_NAME])
        self.assertFalse(m["dataset_sources"])
        self.assertFalse(m["model_sources"])
        self.assertIn("scripts.ai_v6_hist100_kaggle_gpu_worker", worker)
        self.assertNotIn("KAGGLE_API_TOKEN", worker)
        self.assertEqual(len(m["kernel_sources"]), 1)
        self.assertEqual(MAX_CONTEXT_TOKENS, 8192)
        self.assertEqual(MAX_NEW_TOKENS, 256)

    def test_invalid_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "INVALID_KAGGLE_OWNER"):
            make_metadata("../escape", "e" * 40)
        with self.assertRaisesRegex(ValueError, "INVALID_SOURCE_REVISION"):
            make_metadata("goodname", "branch-name")


if __name__ == "__main__":
    unittest.main()
