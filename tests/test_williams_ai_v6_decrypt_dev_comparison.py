import base64, hashlib, io, json, os, unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import scripts.ai_v6_decrypt_dev_comparison as mod
from scripts.ai_v6_decrypt_dev_comparison import (
    DOMAIN, ENVELOPE_DOMAIN, MAGIC, build_error_map, build_error_map_from_directory,
    decrypt_sealed, encrypt_error_map, grade
)
from williams_ai.v6_objective_practice import curriculum

SECRET="test-only-kaggle-secret-for-unit-tests"

def seal(data,secret):
    salt=os.urandom(16); nonce=os.urandom(12)
    key=HKDF(algorithm=hashes.SHA256(),length=32,salt=salt,info=DOMAIN).derive(secret.encode())
    return MAGIC+salt+nonce+AESGCM(key).encrypt(nonce,data,DOMAIN)

def group_counts(records,field):
    return {k:dict(Counter(r["grade"] for r in records if r[field]==k)) for k in sorted({r[field] for r in records})}

def archive_fixture():
    rows,manifest=curriculum(); dev=[r for r in rows if r["split"]=="development"]
    results={}; sealed={}
    for model in ("BASE","V5","A_V6"):
        records=[]
        for i,row in enumerate(dev):
            raw=row["answer"]
            if model=="V5" and i==0:
                expected=json.loads(raw)["answer"]; raw=json.dumps({"answer":expected+1 if type(expected) is int else "WRONG"})
            records.append({"id":row["case_id"],"family":row["family"],"edition":row["edition"],
                            "grade":grade(raw,row["answer"]),"raw":raw})
        results[model]={"total":32,"counts":dict(Counter(r["grade"] for r in records)),
            "by_family":group_counts(records,"family"),"by_edition":group_counts(records,"edition")}
        sealed[model]=seal(json.dumps(records).encode(),SECRET)
    report={"schema":"WILLIAMS_V6_RECOVERED_GGUF_DEV_COMPARISON_V1","dataset":manifest,"results":results,
        "new_optimizer_steps":0,"independent_final_exam":False,"historical_chart_exam":False,
        "execution_allowed":False,"qualified":False}
    b=io.BytesIO()
    with ZipFile(b,"w") as z:
        z.writestr("aggregate.json",json.dumps(report))
        z.writestr("BASE.json.sealed",sealed["BASE"]); z.writestr("V5.json.sealed",sealed["V5"])
        z.writestr("A_V6.json.sealed",sealed["A_V6"])
    return b.getvalue()

class DecryptDevelopmentComparisonTests(unittest.TestCase):
    def test_decrypt_and_authenticate(self):
        sealed=seal(b"payload",SECRET); self.assertEqual(decrypt_sealed(sealed,SECRET),b"payload")
        with self.assertRaisesRegex(ValueError,"AUTHENTICATION_FAILED"): decrypt_sealed(sealed,"wrong-secret-long-enough")
    def test_answer_map_and_pairwise(self):
        result=build_error_map(archive_fixture(),SECRET,expected_member_sha256=None)
        self.assertEqual(result["development_cases"],32)
        self.assertEqual(result["models"]["BASE"]["accuracy"],1.0)
        self.assertEqual(len(result["pairwise"]["V5_vs_BASE"]["regressed_cases"]),1)
        self.assertFalse(result["qualification"]); self.assertNotIn('"raw":',json.dumps(result))
    def test_rsa_aes_envelope_round_trip(self):
        report=build_error_map(archive_fixture(),SECRET,expected_member_sha256=None)
        private=rsa.generate_private_key(public_exponent=65537,key_size=3072)
        pub=private.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo)
        envelope=json.loads(encrypt_error_map(report,pub,expected_fingerprint=None))
        data_key=private.decrypt(base64.b64decode(envelope["wrapped_key_b64"]),
            padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=ENVELOPE_DOMAIN))
        plain=AESGCM(data_key).decrypt(base64.b64decode(envelope["nonce_b64"]),
            base64.b64decode(envelope["ciphertext_b64"]),ENVELOPE_DOMAIN)
        self.assertEqual(hashlib.sha256(plain).hexdigest(),envelope["plaintext_sha256"])
        self.assertEqual(json.loads(plain)["development_cases"],32)
    def test_downloaded_directory_and_member_hashes(self):
        payload=archive_fixture()
        with TemporaryDirectory() as d:
            root=Path(d)
            with ZipFile(io.BytesIO(payload)) as z: z.extractall(root)
            expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            with patch.object(mod,"MEMBER_SHA256",expected):
                self.assertEqual(build_error_map_from_directory(root,SECRET)["development_cases"],32)
    def test_crypto_domain_matches_existing_producer(self):
        source=(Path(__file__).resolve().parents[1]/"scripts"/"ai_submit_v5_qlora_kaggle.py").read_text()
        self.assertIn('DOMAIN=b"Williams AI V5 four-book QLoRA private artifact v1"',source)
    def test_archive_hash_mismatch(self):
        with self.assertRaisesRegex(ValueError,"SHA256_MISMATCH"):
            build_error_map(archive_fixture(),SECRET,expected_archive_sha256="0"*64)

if __name__=="__main__": unittest.main()
