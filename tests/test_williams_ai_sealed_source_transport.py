import json
import unittest

try:
    from cryptography.exceptions import InvalidTag
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

from scripts.ai_sealed_source_transport import seal, open_envelope, public_key


@unittest.skipUnless(HAS_CRYPTO, "optional cryptography transport dependency")
class PrivateSourceTransportTests(unittest.TestCase):
    def test_authenticated_roundtrip_without_plaintext_or_credential_in_envelope(self):
        secret = "test-only-token-never-a-real-credential"
        data = b"private source text; train and validation only" * 300
        envelope = seal(data, public_key(secret))
        encoded = json.dumps(envelope)
        self.assertNotIn(secret, encoded)
        self.assertNotIn("private source text", encoded)
        self.assertEqual(open_envelope(envelope, secret), data)

    def test_wrong_account_and_authenticated_metadata_fail_closed(self):
        secret = "test-only-token-never-a-real-credential"
        envelope = seal(b"private-source", public_key(secret))
        with self.assertRaises(ValueError):
            open_envelope(envelope, secret + "wrong")
        envelope["plaintext_sha256"] = "a" * 64
        with self.assertRaises(InvalidTag):
            open_envelope(envelope, secret)

    def test_changed_ciphertext_cannot_be_used(self):
        secret = "test-only-token-never-a-real-credential"
        envelope = seal(b"private-source", public_key(secret))
        text = envelope["ciphertext"]
        envelope["ciphertext"] = ("A" if text[0] != "A" else "B") + text[1:]
        with self.assertRaises(InvalidTag):
            open_envelope(envelope, secret)
