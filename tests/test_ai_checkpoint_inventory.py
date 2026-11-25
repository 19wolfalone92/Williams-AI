import base64,hashlib,io,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import padding,rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import scripts.ai_checkpoint_inventory as mod

SECRET="unit-test-kaggle-secret-that-is-long-enough"

def seal(data):
    salt=os.urandom(16);nonce=os.urandom(12)
    key=HKDF(algorithm=hashes.SHA256(),length=32,salt=salt,info=mod.DOMAIN).derive(SECRET.encode())
    return mod.MAGIC+salt+nonce+AESGCM(key).encrypt(nonce,data,mod.DOMAIN)

def v5_fixture(root):
    adapter=b"v5-adapter-bytes";sha=hashlib.sha256(adapter).hexdigest()
    receipt={"adapter_sha256":sha,"optimizer_steps":80}
    b=io.BytesIO()
    with ZipFile(b,"w") as z:
        z.writestr("adapter_model.safetensors",adapter)
        z.writestr("adapter_config.json",json.dumps({"peft_type":"LORA","r":8}))
        z.writestr("training_receipt.json",json.dumps(receipt))
        z.writestr("provenance.json",json.dumps({"research_only":True}))
    encrypted=seal(b.getvalue());(root/"williams-v5-trained-qwen3-4b-lora.zip.sealed").write_bytes(encrypted)
    (root/"v5_training_audit.json").write_text(json.dumps({"sha256_encrypted_archive":hashlib.sha256(encrypted).hexdigest(),"adapter_sha256":sha}))

def v6_fixture(root,include_optimizer=True):
    adapter=b"v6-adapter-bytes";sha=hashlib.sha256(adapter).hexdigest()
    prefix="output/williams_v6_objective_smoke/second/"
    spec={"schema":"WILLIAMS_QLORA_RESEARCH_CONTRACT_V2","max_steps":20,"evaluation_interval":10,
          "quantization":"NF4_DOUBLE_FLOAT16","lora_rank":8,"lora_alpha":16,"lora_dropout":0.05,
          "target_modules":["q_proj","k_proj","v_proj","o_proj"]}
    adapter_cfg={"peft_type":"LORA","r":8,"lora_alpha":16,"lora_dropout":0.05,
                 "target_modules":["q_proj","k_proj","v_proj","o_proj"]}
    checkpoint={"adapter_model.safetensors":adapter,"adapter_config.json":json.dumps(adapter_cfg).encode(),
        "training_args.bin":b"x","scheduler.pt":b"x","rng_state.pth":b"x",
        "trainer_state.json":json.dumps({"global_step":20,"max_steps":20}).encode(),"scaler.pt":b"x"}
    if include_optimizer:checkpoint["optimizer.pt"]=b"x"
    contract_bytes=json.dumps({"contract":spec,"original_baseline":{"ALL":1.0,"TC1":1.0,"NTD":1.0,"TC2":1.0,"COURSE":1.0}},sort_keys=True).encode()
    manifest={"schema":"WILLIAMS_QLORA_CHECKPOINT_SEAL_V2","receipt_sha256":hashlib.sha256(contract_bytes).hexdigest(),
              "checkpoints":{"checkpoint-20":{k:hashlib.sha256(v).hexdigest() for k,v in checkpoint.items()}}}
    b=io.BytesIO()
    with ZipFile(b,"w") as z:
        z.writestr(prefix+"latest_adapter/adapter_model.safetensors",adapter)
        z.writestr(prefix+"latest_adapter/adapter_config.json",json.dumps(adapter_cfg))
        z.writestr(prefix+"research_result.json",json.dumps({"adapter_sha256":sha,"total_optimizer_steps":20,"optimizer_skips_in_this_session":0}))
        z.writestr(prefix+"contract.json",contract_bytes)
        z.writestr(prefix+"checkpoint_sha256.json",json.dumps(manifest))
        for name,data in checkpoint.items():z.writestr(prefix+"checkpoints/checkpoint-20/"+name,data)
    encrypted=seal(b.getvalue());(root/"v6.zip.sealed").write_bytes(encrypted)
    return hashlib.sha256(encrypted).hexdigest(),sha

class CheckpointInventoryTests(unittest.TestCase):
    def test_v5_weights_only_and_v6_optimizer_checkpoint_integrity(self):
        with tempfile.TemporaryDirectory() as t5,tempfile.TemporaryDirectory() as t6:
            p5,p6=Path(t5),Path(t6);v5_fixture(p5);sealed_sha,adapter_sha=v6_fixture(p6,True)
            with patch.object(mod,"V6_SEALED_SHA256",sealed_sha),patch.object(mod,"V6_ADAPTER_SHA256",adapter_sha):
                result=mod.build_inventory(p5,p6,SECRET)
            self.assertFalse(result["v5"]["optimizer_state_available"])
            self.assertTrue(result["v6"]["optimizer_state_available"])
            self.assertEqual(result["v6"]["trainer_global_step"],20)
            self.assertFalse(result["v6"]["exact_resume_compatible"])
    def test_missing_optimizer_is_not_reported_as_resumable(self):
        with tempfile.TemporaryDirectory() as t5,tempfile.TemporaryDirectory() as t6:
            p5,p6=Path(t5),Path(t6);v5_fixture(p5);sealed_sha,adapter_sha=v6_fixture(p6,False)
            with patch.object(mod,"V6_SEALED_SHA256",sealed_sha),patch.object(mod,"V6_ADAPTER_SHA256",adapter_sha):
                result=mod.build_inventory(p5,p6,SECRET)
            self.assertFalse(result["v6"]["optimizer_state_available"])
            self.assertIn("optimizer.pt",result["v6"]["checkpoint_20_missing_files"])
    def test_report_encryption_round_trip(self):
        report={"schema":"test","v5":{"optimizer_state_available":False}}
        private=rsa.generate_private_key(public_exponent=65537,key_size=3072)
        public=private.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo)
        fp=hashlib.sha256(private.public_key().public_bytes(serialization.Encoding.DER,serialization.PublicFormat.SubjectPublicKeyInfo)).hexdigest()
        with patch.object(mod,"PUBLIC_KEY_FINGERPRINT_SHA256",fp): envelope=json.loads(mod.encrypt_report(report,public))
        key=private.decrypt(base64.b64decode(envelope["wrapped_key_b64"]),padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=mod.ENVELOPE_DOMAIN))
        plain=AESGCM(key).decrypt(base64.b64decode(envelope["nonce_b64"]),base64.b64decode(envelope["ciphertext_b64"]),mod.ENVELOPE_DOMAIN)
        self.assertEqual(json.loads(plain),report)

if __name__=="__main__":unittest.main()
