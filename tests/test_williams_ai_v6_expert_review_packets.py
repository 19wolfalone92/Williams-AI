"""Private signer fixtures: self-generated mock expert keys are NOT real experts."""
import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.test_williams_ai_independent_exam_v6 import V6IndependentIntakeTests
from williams_ai.independent_exam_v6 import validate_cases, canonical
from williams_ai.v6_expert_review_packets import (
    template, sign, keypair, trust_scaffold, private_create,
)


class ReviewPacketTests(unittest.TestCase):
    def setUp(self):
        self.fixture = V6IndependentIntakeTests("test_frozen_chart_signatures_all_four_book_counts")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.dataset, self.root = self.fixture.dataset, self.fixture.root
        self.manifest = validate_cases(self.dataset, self.root)
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def test_private_draft_has_placeholder_not_human_gold(self):
        form = template(self.dataset, self.manifest, self.root, "expert-A")
        self.assertEqual(len(form["payload"]["labels"]), 32)
        self.assertTrue(all(not item["reviewable"] for item in form["payload"]["labels"]))
        self.assertTrue(all(item["rationale"].startswith("NOT_REVIEWED")
                            for item in form["payload"]["labels"]))
        self.assertNotIn("predictions", json.dumps(form))
        with self.assertRaisesRegex(ValueError, "REVIEW_STILL_UNEXAMINED"):
            sign(form, self.work / "secret.key")

    def test_key_generation_signing_and_public_trust_default_false(self):
        priv_a, pub_a = self.work / "a.seed", self.work / "a.pub"
        priv_b, pub_b = self.work / "b.seed", self.work / "b.pub"
        keypair(priv_a, pub_a)
        keypair(priv_b, pub_b)
        self.assertNotEqual(pub_a.read_bytes(), pub_b.read_bytes())
        self.assertEqual(len(priv_a.read_bytes()), 32)
        if os.name == "posix":
            self.assertEqual(priv_a.stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, "IDENTITY_PATH_ALREADY_EXISTS"):
            keypair(priv_a, pub_a)
        trust = trust_scaffold("expert-A", pub_a, "expert-B", pub_b)
        self.assertFalse(trust["expert-A"]["externally_vetted_expert"])
        self.assertFalse(trust["expert-B"]["independent_of_training"])
        form = template(self.dataset, self.manifest, self.root, "expert-A")
        for entry in form["payload"]["labels"]:
            entry["rationale"] = (
                "SYNTHETIC MOCK REVIEW: this is not book/chart expert judgement.")
            entry["reviewable"] = False
        signed = sign(form, priv_a)
        self.assertEqual(signed["payload"]["reviewer_id"], "expert-A")
        self.assertEqual(len(signed["signature_hex"]), 128)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(
            pub_a.read_text(encoding="ascii").strip()))
        pub.verify(bytes.fromhex(signed["signature_hex"]), canonical(signed["payload"]))
        tampered = copy.deepcopy(signed)
        tampered["payload"]["labels"][0]["decision"] = "NO_SETUP"
        from cryptography.exceptions import InvalidSignature
        with self.assertRaises(InvalidSignature):
            pub.verify(bytes.fromhex(tampered["signature_hex"]), canonical(
                tampered["payload"]))

    def test_sign_rejects_unreviewed_placeholder_even_with_key(self):
        private_key = self.work / "k.seed"
        public_key = self.work / "k.pub"
        keypair(private_key, public_key)
        form = template(self.dataset, self.manifest, self.root, "expert-A")
        form["payload"]["labels"][0]["reviewable"] = True
        with self.assertRaisesRegex(ValueError, "REVIEW_STILL_UNEXAMINED"):
            sign(form, private_key)

    def test_cannot_overwrite_private_reviewer_secret(self):
        target = self.work / "private_review.json"
        private_create(target, b"first")
        with self.assertRaises(FileExistsError):
            private_create(target, b"second")
        self.assertEqual(target.read_bytes(), b"first")


if __name__ == "__main__":
    unittest.main()
