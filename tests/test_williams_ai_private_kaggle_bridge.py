"""Manual Kaggle automation is private, opt-in and never uploads book bytes."""
import json
import unittest
from scripts.ai_private_kaggle_gpu_bridge import generate,kaggle_input_path


class KaggleBridge(unittest.TestCase):
    def data(self):
        return {"kernel_id":"user-1/williams-private-qlora",
            "dataset_source":"user-1/approved-data",
            "model_source":"user-1/qwen-model/pytorch/weights/1",
            "dataset_file":"/kaggle/input/approved-data/reviewed.jsonl",
            "model_directory":"/kaggle/input/qwen-model/weights",
            "revision":"a"*40}

    def test_private_no_tokens_no_uploaded_training_material(self):
        metadata,worker=generate(**self.data())
        self.assertEqual(metadata["is_private"],"true")
        self.assertEqual(metadata["machine_shape"],"NvidiaTeslaT4")
        self.assertEqual(metadata["dataset_sources"],["user-1/approved-data"])
        self.assertEqual(metadata["model_sources"],["user-1/qwen-model/pytorch/weights/1"])
        self.assertNotIn("KAGGLE_API_TOKEN",worker)
        self.assertNotIn("KAGGLE_KEY",worker)
        self.assertIn("--source-only-research",worker)
        self.assertIn("--authorize-cloud-data-processing",worker)

    def test_invalid_shell_or_path_inputs_rejected(self):
        for edit in ({"kernel_id":"user; rm -rf /x"},
                     {"dataset_file":"/tmp/reviewed.jsonl"},
                     {"dataset_file":"/kaggle/input/data/../private.jsonl"},
                     {"model_directory":"/kaggle/input/model\ninjected"},
                     {"revision":"main"},
                     {"model_source":"wrong"}):
            with self.subTest(edit=edit),self.assertRaises(ValueError):
                generate(**{**self.data(),**edit})

    def test_paths_must_be_bound_to_kaggle_input(self):
        with self.assertRaises(ValueError):
            kaggle_input_path("/home/runner/secret")
        self.assertEqual(kaggle_input_path("/kaggle/input/a/b"),
                         "/kaggle/input/a/b")


if __name__=="__main__":unittest.main()
