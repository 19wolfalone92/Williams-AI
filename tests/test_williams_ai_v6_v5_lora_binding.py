"""No actual adapter/model used: verify historic v5 receipt and fail-closed binding."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from williams_ai.v6_v5_lora_binding import (
    BASE_SHA, RAW_ADAPTER_SHA, SCHEMA, create_binding,
    verified_receipt, verify_binding, hash_file,
)


def valid_receipt():
    return {
        "schema": "WILLIAMS_V5_GENUINE_QWEN3_4B_SOURCE_ONLY_LORA",
        "base_model": "Qwen/Qwen3-4B",
        "model_commit": "3fb356690bbdae7353c94efb3eecdce276e6ed1c",
        "adapter_sha256": RAW_ADAPTER_SHA, "optimizer_steps": 80,
        "selected_validation_checkpoint": 80,
        "test_cases_provided_to_trainer": 0,
        "test_gold_examined_during_checkpoint_selection": False,
        "expert_gold_market_examples": 0,
        "execution_allowed": False, "trading_qualification": False,
        "weights_changed": True, "source_pdf_or_ocr_uploaded": False,
        "validation_baseline_per_edition": {
            "TC1": 4.1, "NTD": 4.0, "TC2": 3.6, "COURSE": 3.5},
        "validation_after_per_edition": {
            "TC1": 1.7, "NTD": 1.6, "TC2": 1.8, "COURSE": 1.2},
    }


class HistoricKaggleV5Binding(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.receipt = self.root / "training_receipt.json"
        self.receipt.write_text(json.dumps(valid_receipt()), encoding="utf-8")

    def test_actual_v5_receipt_80_steps_is_compatible_with_separate_binding(self):
        receipt = verified_receipt(self.receipt)
        self.assertEqual(receipt["optimizer_steps"], 80)
        self.assertEqual(receipt["adapter_sha256"], RAW_ADAPTER_SHA)

    def test_never_accept_one_bad_book_or_false_trading_flag(self):
        for change in (
            {"optimizer_steps": 79},
            {"selected_validation_checkpoint": 60},
            {"execution_allowed": True},
            {"trading_qualification": True},
            {"adapter_sha256": "0" * 64},
            {"test_cases_provided_to_trainer": 1},
        ):
            with self.subTest(change=change):
                obj = valid_receipt()
                obj.update(change)
                self.receipt.write_text(json.dumps(obj), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "HISTORICAL_V5_RECEIPT_NOT_PROVEN"):
                    verified_receipt(self.receipt)
        bad = valid_receipt()
        bad["validation_after_per_edition"]["TC2"] = 9.0
        self.receipt.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "FOUR_BOOK_VALIDATION_RECEIPT_INVALID"):
            verified_receipt(self.receipt)

    def test_creation_and_verification_pin_raw_adapter_and_gguf_sha(self):
        base = self.root / "base.gguf"
        lora = self.root / "adapter.gguf"
        raw = self.root / "adapter_model.safetensors"
        base.write_bytes(b"GGUF" + b"mockbase")
        lora.write_bytes(b"GGUF" + b"mockadapter")
        raw.write_bytes(b"mockraw")
        # Mock impossibly large genuine SHA-pinned fixture files ONLY.
        # Real code rehashes all of the original files on disk.
        fake_converted = "2" * 64
        def gguf(p):
            if Path(p) == base:
                return BASE_SHA
            if Path(p) == lora:
                return fake_converted
            raise AssertionError("unknown file")
        real_hash = hash_file
        def hashes(p):
            return RAW_ADAPTER_SHA if Path(p) == raw else real_hash(p)
        with patch("williams_ai.v6_v5_lora_binding.verified_gguf", side_effect=gguf), \
             patch("williams_ai.v6_v5_lora_binding.hash_file", side_effect=hashes):
            result = create_binding(base=base, lora_gguf=lora,
                                    raw_safetensors=raw, training_receipt=self.receipt)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["lora_gguf_sha256"], fake_converted)
        self.assertFalse(result["gpu_used"])
        manifest = self.root / "v5_historic_bound.json"
        manifest.write_text(json.dumps(result), encoding="utf-8")
        with patch("williams_ai.v6_v5_lora_binding.verified_gguf", side_effect=gguf):
            self.assertEqual(verify_binding(base=base, lora_gguf=lora,
                                            binding_file=manifest), fake_converted)
            altered = copy.deepcopy(result)
            altered["lora_gguf_sha256"] = "3" * 64
            manifest.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CONVERTED_LORA_SHA_MISMATCH"):
                verify_binding(base=base, lora_gguf=lora, binding_file=manifest)
            altered["lora_gguf_sha256"] = fake_converted
            altered["gpu_used"] = True
            manifest.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "HISTORIC_BINDING_CONTRACT_CHANGED"):
                verify_binding(base=base, lora_gguf=lora, binding_file=manifest)


if __name__ == "__main__":
    unittest.main()
