import json
from hashlib import sha256
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from scripts import v5_recovery_integrity as guard


def fixture(dtype="F16", nonfinite=False, wrong_shape=False, overlap=False):
    header, payload = {}, bytearray()
    for layer in range(36):
        for target, (out_dim, in_dim) in guard.TARGETS.items():
            for side, shape in (("A", [8, in_dim]), ("B", [out_dim, 8])):
                name = f"base_model.model.model.layers.{layer}.self_attn.{target}.lora_{side}.weight"
                dt = {"F16": "<e", "F32": "<f", "BF16": "<H"}[dtype]
                arr = bytearray(shape[0] * shape[1] * struct.calcsize(dt))
                first = not header
                if first and nonfinite:
                    struct.pack_into(dt, arr, 0, 0x7f80 if dtype == "BF16" else float("inf"))
                start = len(payload)
                payload.extend(arr)
                header[name] = {"dtype": dtype, "shape": [1, 1] if first and wrong_shape else shape,
                                "data_offsets": [start - 2 if overlap and len(header) == 1 else start,
                                                 len(payload) - 2 if overlap and len(header) == 1 else len(payload)]}
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + payload


class RecoveryIntegrityTests(unittest.TestCase):
    def test_complete_finite_inventory(self):
        self.assertEqual(guard.validate_tensor_bytes(fixture()), 288)

    def test_nonfinite_all_supported_dtypes(self):
        for dtype in ("F16", "F32", "BF16"):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "NONFINITE"):
                guard.validate_tensor_bytes(fixture(dtype, nonfinite=True))

    def test_shape_overlap_trailing_and_truncation(self):
        for raw in (fixture(wrong_shape=True), fixture(overlap=True), fixture() + b"x", fixture()[:-1]):
            with self.assertRaises(ValueError):
                guard.validate_tensor_bytes(raw)

    def test_self_consistent_but_substituted_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w, r, c = (root / n for n in ("weights", "receipt", "config"))
            raw = fixture()
            w.write_bytes(raw)
            r.write_text(json.dumps({"adapter_sha256": sha256(raw).hexdigest(),
                "base_model": guard.BASE, "model_commit": guard.REV,
                "optimizer_steps": 80, "selected_validation_checkpoint": 80}))
            c.write_text(json.dumps({"base_model_name_or_path": guard.BASE, "peft_type": "LORA",
                "r": 8, "lora_alpha": 16, "target_modules": list(guard.TARGETS)}))
            with self.assertRaisesRegex(ValueError, "FROZEN_ADAPTER_DIGEST"):
                guard.verify_recovered_identity(r, w, c)
            # Only a fixture test; never substitute the production frozen SHA.
            with patch.object(guard, "EXPECTED_ADAPTER_SHA256", sha256(raw).hexdigest()):
                self.assertEqual(guard.verify_recovered_identity(r, w, c)["tensor_count"], 288)
                c.write_text('{}')
                with self.assertRaisesRegex(ValueError, "FROZEN_CONFIG"):
                    guard.verify_recovered_identity(r, w, c)


if __name__ == "__main__":
    unittest.main()
