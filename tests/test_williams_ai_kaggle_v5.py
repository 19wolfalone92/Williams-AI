"""No GPU submission in these tests: cryptography, pinned worker and data contract."""
import ast
from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.ai_submit_v5_qlora_kaggle import (
    worker_source,_seal_data,_select_final_adapter,DOMAIN,KERNEL_SLUG,MIN_FREE_GPU_SECONDS)
from williams_ai.knowledge_curriculum_v5 import build


class KaggleV5Contracts(unittest.TestCase):
    def test_pinned_git_commit_and_no_secret_or_private_pdf_text(self):
        worker=worker_source("a"*40)
        ast.parse(worker)
        self.assertIn('REV="'+"a"*40+'"',worker)
        self.assertIn("CUDA_VISIBLE_DEVICES",worker)
        self.assertIn("williams-ai-v5-book-sft",KERNEL_SLUG)
        self.assertIn("v5-training-control.json",worker)
        self.assertIn('knowledge_training_v5',worker)
        self.assertNotIn("KAGGLE_API_TOKEN",worker)
        self.assertNotIn("adapter_model.safetensors",worker)
        with self.assertRaisesRegex(ValueError,"EXACT_TRAINER"):
            worker_source("main")

    def test_final_export_selection_excludes_intermediate_checkpoints(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/"williams_v5_model"
            final=root/"adapter"
            final.mkdir(parents=True)
            for name in ("adapter_model.safetensors","adapter_config.json"):
                (final/name).write_bytes(b"x"*1000001)
            (root/"training_receipt.json").write_text("{}")
            checkpoint=Path(tmp)/".williams-v5-qlora-checkpoints"/"checkpoint-40"
            checkpoint.mkdir(parents=True)
            (checkpoint/"adapter_model.safetensors").write_bytes(b"z"*1000001)
            (checkpoint/"adapter_config.json").write_text("{}")
            receipt,weight,config=_select_final_adapter(Path(tmp))
            self.assertEqual(receipt,root/"training_receipt.json")
            self.assertEqual(weight,final/"adapter_model.safetensors")
            self.assertEqual(config,final/"adapter_config.json")
            (final/"adapter_model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError,"FINAL_KAGGLE"):
                _select_final_adapter(Path(tmp))

    def test_private_envelope_is_random_and_authentic(self):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes
        secret="local-fixture-only-secret-12345678"
        plaintext=b"Test-generated LoRA data, not real weights"
        a=_seal_data(plaintext,secret)
        b=_seal_data(plaintext,secret)
        self.assertNotEqual(a,b)
        prefix=b"WILLIAMSV5A1"
        self.assertTrue(a.startswith(prefix))
        salt=a[len(prefix):len(prefix)+16]
        nonce=a[len(prefix)+16:len(prefix)+28]
        ciphertext=a[len(prefix)+28:]
        key=HKDF(algorithm=hashes.SHA256(),length=32,
                 salt=salt,info=DOMAIN).derive(secret.encode())
        self.assertEqual(AESGCM(key).decrypt(nonce,ciphertext,DOMAIN),plaintext)
        # Always flip an actual authentication-tag bit. Replacing the
        # final byte with literal b"0" was a flaky no-op whenever the
        # original last byte happened to equal 0x30 (~1/256 runs).
        tampered=bytearray(ciphertext)
        tampered[-1]^=1
        from cryptography.exceptions import InvalidTag
        with self.assertRaises(InvalidTag):
            AESGCM(key).decrypt(nonce,bytes(tampered),DOMAIN)

    def test_only_training_paraphrases_available_to_gpu_trainer(self):
        rows,manifest=build()
        self.assertEqual(len(rows),432)
        self.assertEqual(manifest["split_rows"],{
            "train":300,"validation":48,"test":84})
        self.assertEqual(manifest["approved_market_labels"],0)
        self.assertEqual(MIN_FREE_GPU_SECONDS,8500)
        self.assertTrue(all(not row["source_excerpt_in_training"] for row in rows))
        self.assertTrue(all("reviewed" not in row["answer"].lower()
                            or row["task_type"]!="MCQ" for row in rows))


if __name__=="__main__":
    unittest.main()
