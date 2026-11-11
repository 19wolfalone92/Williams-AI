"""Fail-closed CPU contract for Qwen3-4B actual public-source adapter market exam."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from williams_ai.qwen4b_public_market_probe import (
    checked_public_adapter_zip,TRAINING_MODE,BASE_MODEL_REPO,
    BASE_REVISION,TRAINING_COMMIT
)


class Qwen4bPublicBlindAdapterTests(unittest.TestCase):
    def make_archive(self,root,**changes):
        weights=b"actual-weights-test-only-"*6000
        r={"mode":TRAINING_MODE,"model_repo":BASE_MODEL_REPO,
           "model_revision":BASE_REVISION,"source_commit":TRAINING_COMMIT,
           "rank":8,"steps":32,
           "trained_official_pretrained_qwen3_4b":True,
           "adapter_weights_changed":True,
           "validation_gate":{"accepted":True,"relative_improvement":0.2},
           "full_original_four_books_trained":False,
           "execution_allowed":False,
           "adapter_model_sha256":sha256(weights).hexdigest()}
        r.update(changes)
        p=Path(root)/"adapter.zip"
        with zipfile.ZipFile(p,"w") as f:
            f.writestr("research_receipt.json",json.dumps(r))
            f.writestr("adapter/adapter_config.json",json.dumps({
                "r":8,"base_model_name_or_path":BASE_MODEL_REPO}))
            f.writestr("adapter/adapter_model.safetensors",weights)
        return p

    def test_valid_genuine_source_contract(self):
        with TemporaryDirectory() as temp:
            result=checked_public_adapter_zip(self.make_archive(temp))
            self.assertEqual(result["rank"],8)
            self.assertEqual(result["steps"],32)
            self.assertFalse(result["execution_allowed"])
            self.assertFalse(result["original_private_books_trained"])

    def test_rejects_false_claims_and_training_regression(self):
        for changes in ({"validation_gate":{"accepted":False}},
                        {"full_original_four_books_trained":True},
                        {"execution_allowed":True},
                        {"source_commit":"different"},
                        {"model_repo":"Qwen/Qwen3-0.6B"},
                        {"adapter_model_sha256":"0"*64},
                        {"rank":64}):
            with self.subTest(changes=changes),TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError,"VALIDATION_REQUIRED"):
                    checked_public_adapter_zip(self.make_archive(temp,**changes))


if __name__=="__main__":unittest.main()
