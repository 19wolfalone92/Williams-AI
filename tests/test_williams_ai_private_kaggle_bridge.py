"""Manual Kaggle automation is private, opt-in and never uploads book bytes."""
import json
from unittest.mock import patch
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
        self.assertEqual(metadata["title"],metadata["id"].split("/")[1])
        self.assertEqual(metadata["dataset_sources"],["user-1/approved-data"])
        self.assertEqual(metadata["model_sources"],["user-1/qwen-model/pytorch/weights/1"])
        self.assertNotIn("KAGGLE_API_TOKEN",worker)
        self.assertNotIn("KAGGLE_KEY",worker)
        self.assertIn("--source-only-research",worker)
        self.assertIn("--authorize-cloud-data-processing",worker)
        compile(worker,"private_kaggle_worker.py","exec")

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

    def test_resume_requires_own_checkpoint_and_explicit_private_input(self):
        for config in ({"directory":"/kaggle/input/state"},
                       {"dataset_source":"user-1/checkpoints","directory":"/tmp/state","trusted_own":True},
                       {"dataset_source":"user-1/checkpoints","directory":"/kaggle/input/state","trusted_own":"true"}):
            with self.subTest(config=config),self.assertRaises(ValueError):
                generate(**self.data(),resume_config=config)
        metadata,worker=generate(**self.data(),resume_config={
            "dataset_source":"user-1/checkpoints","directory":"/kaggle/input/checkpoints/session-001",
            "trusted_own":True})
        self.assertEqual(metadata["dataset_sources"],["user-1/approved-data","user-1/checkpoints"])
        self.assertIn("--resume-directory",worker)
        compile(worker,"private_kaggle_worker.py","exec")

    def test_generated_worker_passes_resume_as_arguments_and_uses_one_gpu(self):
        resume={"dataset_source":"user-1/checkpoints",
                "directory":"/kaggle/input/checkpoints/session-001","trusted_own":True}
        _,worker=generate(**self.data(),resume_config=resume)
        calls=[]
        def run(command,**kwargs):calls.append((command,kwargs))
        with patch("pathlib.Path.is_file",return_value=True), \
             patch("pathlib.Path.is_dir",return_value=True), \
             patch("pathlib.Path.exists",return_value=False), \
             patch("subprocess.run",side_effect=run), \
             patch.dict("os.environ",{},clear=True):
            namespace={"__name__":"__generated_worker__"}
            exec(compile(worker,"private_kaggle_worker.py","exec"),namespace)
            import os
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"],"0")
        final=calls[-1][0]
        self.assertEqual(final[-3:],["--resume-directory",resume["directory"],"--trusted-own-resume"])
        self.assertIn("--authorize-cloud-data-processing",final)
        self.assertEqual(calls[1][0],["git","checkout",self.data()["revision"]])
        self.assertFalse(any(kwargs.get("shell") for _,kwargs in calls))


if __name__=="__main__":unittest.main()
