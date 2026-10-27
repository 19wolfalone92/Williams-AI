"""Authenticated encryption for this private book-training transfer.

Only an X25519 public key and AES-GCM ciphertext may enter the public branch.
The existing Kaggle secret remains on the Actions runner; neither that secret
nor a decryption key is returned. Test-book texts never leave local storage.
This is a research transport, not model qualification or trading authority.
"""
from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import zlib

FORMAT = "WILLIAMS_PRIVATE_SOURCE_X25519_HKDF_AES256GCM_V1"
CONTEXT = b"Williams-AI/private-four-book-deep-training/20261027/v1"
MAX_BYTES = 8_000_000


def b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def unb64(value: str) -> bytes:
    return base64.b64decode(value, validate=True)


def runner_key(secret: str):
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    if not isinstance(secret, str) or len(secret) < 20:
        raise ValueError("EXISTING_RUNNER_SECRET_REQUIRED")
    material = hmac.digest(secret.encode(), CONTEXT, "sha256")
    return X25519PrivateKey.from_private_bytes(material)


def public_key(secret: str) -> str:
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    return b64(runner_key(secret).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def _symmetric(shared: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=CONTEXT).derive(shared)


def seal(payload: bytes, recipient: str) -> dict:
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    if not 0 < len(payload) <= MAX_BYTES:
        raise ValueError("BOUNDED_PRIVATE_PAYLOAD_REQUIRED")
    ephemeral = X25519PrivateKey.generate()
    peer = X25519PublicKey.from_public_bytes(unb64(recipient))
    nonce = os.urandom(12)
    header = {"format": FORMAT, "recipient": recipient,
              "ephemeral": b64(ephemeral.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)),
              "nonce": b64(nonce), "plaintext_sha256": sha256(payload).hexdigest()}
    aad = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    cipher = AESGCM(_symmetric(ephemeral.exchange(peer))).encrypt(nonce, zlib.compress(payload, 9), aad)
    return {**header, "ciphertext": b64(cipher)}


def open_envelope(envelope: dict, secret: str) -> bytes:
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    fields = {"format", "recipient", "ephemeral", "nonce", "plaintext_sha256", "ciphertext"}
    if (not isinstance(envelope, dict) or set(envelope) != fields
            or envelope["format"] != FORMAT or envelope["recipient"] != public_key(secret)
            or any(not isinstance(v, str) for v in envelope.values())
            or len(envelope["ciphertext"]) > MAX_BYTES * 2):
        raise ValueError("WRONG_SEALED_CORPUS_OR_RECIPIENT")
    header = {k: v for k, v in envelope.items() if k != "ciphertext"}
    aad = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    shared = runner_key(secret).exchange(X25519PublicKey.from_public_bytes(unb64(envelope["ephemeral"])))
    compressed = AESGCM(_symmetric(shared)).decrypt(unb64(envelope["nonce"]), unb64(envelope["ciphertext"]), aad)
    decoder = zlib.decompressobj()
    payload = decoder.decompress(compressed, MAX_BYTES + 1)
    if (len(payload) > MAX_BYTES or decoder.unconsumed_tail or not decoder.eof
            or decoder.unused_data or sha256(payload).hexdigest() != envelope["plaintext_sha256"]):
        raise ValueError("PRIVATE_CORPUS_SIZE_OR_INTEGRITY_FAILURE")
    return payload


def source_payload(directory: Path) -> bytes:
    # Integrity checking includes test-page identities locally, then discards
    # their text. The encrypted GPU payload contains ONLY train and validation.
    from williams_ai.private_book_source import load_private_source_corpus
    audited = load_private_source_corpus(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    test_ids = {ed: [] for ed in manifest["book_registry"]}
    for line in (directory / "test.jsonl").read_text().splitlines():
        row = json.loads(line)
        test_ids[row["edition"]].append(row["pdf_page"])
    files = {s: b64((directory / (s + ".jsonl")).read_bytes()) for s in ("train", "validation")}
    payload = {"format": "PRIVATE_SOURCE_TRAIN_VALIDATION_ONLY_V1", "manifest": manifest,
               "files": files, "withheld_test_page_ids": test_ids, "audit": audited["audit"],
               "test_texts_in_cloud_payload": False, "execution_allowed": False}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("public-key", "seal"))
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--recipient")
    parser.add_argument("--new-output", type=Path)
    args = parser.parse_args()
    if args.mode == "public-key":
        print("PRIVATE_SOURCE_PUBLIC_ENCRYPTION_KEY", public_key(os.environ.get("KAGGLE_API_TOKEN", "")))
    else:
        if not args.new_output or args.new_output.exists() or not args.source_directory or not args.recipient:
            raise ValueError("NEW_SEALED_OUTPUT_AND_EXPLICIT_SOURCE_REQUIRED")
        result = seal(source_payload(args.source_directory), args.recipient)
        args.new_output.parent.mkdir(parents=True, exist_ok=True)
        args.new_output.write_text(json.dumps(result, sort_keys=True))
        print("SEALED_PRIVATE_SOURCE_WRITTEN_TEST_TEXTS_EXCLUDED", args.new_output.stat().st_size)


if __name__ == "__main__":
    main()
