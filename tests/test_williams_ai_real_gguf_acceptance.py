"""Real-model acceptance contract: mock weights must NOT satisfy real-bytes gate.

A fake model in these tests is a protocol test only; only the separate
public GGUF GitHub Actions job may report real inference.
"""
import tempfile
from pathlib import Path
import unittest

from williams_ai.real_gguf_acceptance import actual_gguf_sha256,public_model_trial


class RealModelContractTests(unittest.TestCase):
    def test_fake_gguf_magic_is_not_sufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"fake.gguf"
            path.write_bytes(b"GGUFfake")
            with self.assertRaisesRegex(ValueError,"unexpectedly tiny"):
                actual_gguf_sha256(path,"0"*64)

    def test_fake_weights_without_magic_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"fake.gguf"
            path.write_bytes(b"not a model")
            with self.assertRaisesRegex(ValueError,"Not genuine GGUF"):
                actual_gguf_sha256(path,"0"*64)

    def test_missing_model_blocked_not_substituted_with_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                actual_gguf_sha256(Path(tmp)/"absent.gguf","0"*64)

    def test_valid_structure_cannot_infer_claim_from_no_verified_weights(self):
        class Fake:
            def answer_exam_case(self,case):
                return {"choice":"B","reason":"fake"}
        with self.assertRaisesRegex(ValueError,"not passed"):
            public_model_trial(Fake(),weight_report={},max_questions=1,
                               history_cases=0)
        with self.assertRaisesRegex(ValueError,"Need 1..10"):
            public_model_trial(Fake(),weight_report={"verified_model_magic":"GGUF"},
                               max_questions=0,history_cases=0)


if __name__=="__main__":
    unittest.main()
