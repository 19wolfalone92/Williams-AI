"""Restore the REAL 50-cycle Qwen3-4B PEFT adapter on a trusted Actions runner.

The secret is consumed only inside GitHub Actions and NEVER printed, committed,
placed in output artifacts or copied to public hosting. Refuse counterfeit
adapter files, archive substitution, zip traversal, and failed research gates.
The result remains strictly research-only and is not deployable for trading.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
import io
import json
import os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

DOMAIN = b"Williams AI private model artifact v1"
CONTEXT = b"Williams-AI/private-four-book-deep-training/20261027/v1"
ADAPTER = ("adapter_model.safetensors", "adapter_config.json")


def recover(encrypted: Path, audit_json: Path, private_output: Path, secret: str) -> dict:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey, X25519PublicKey)
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    if not isinstance(secret,str) or len(secret)<20:
        raise ValueError("ORIGINAL_RUNNER_SECRET_NOT_AVAILABLE")
    if not encrypted.is_file() or encrypted.stat().st_size>400_000_000:
        raise ValueError("PINNED_ENCRYPTED_ARCHIVE_REQUIRED")
    audit=json.loads(audit_json.read_text(encoding="utf-8"))
    if (audit.get("research_gate_passed") is not False or
        audit.get("trading_qualification") is not False or
        audit.get("selected_checkpoint")!="checkpoint-500" or
        audit.get("actual_updates")!=1000 or
        not isinstance(audit.get("candidate_adapter_sha256"),str) or
        len(audit["candidate_adapter_sha256"])!=64 or
        audit.get("execution_allowed") is not False):
        raise ValueError("SOURCE_AUDIT_CANNOT_AUTHORIZE_TRADING_OR_UNPINNED_WEIGHTS")
    envelope=encrypted.read_bytes()
    if len(envelope)<100:
        raise ValueError("INCOMPLETE_SEALED_ARCHIVE")
    private=X25519PrivateKey.from_private_bytes(hmac.digest(secret.encode(),CONTEXT,"sha256"))
    peer=X25519PublicKey.from_public_bytes(envelope[:32])
    key=HKDF(algorithm=hashes.SHA256(),length=32,salt=None,info=DOMAIN).derive(private.exchange(peer))
    # GCM authentication fails closed if either wrong secret or modified data.
    plaintext=AESGCM(key).decrypt(envelope[32:44],envelope[44:],DOMAIN)
    archive_sha=hashlib.sha256(plaintext).hexdigest()
    if archive_sha!=audit.get("archive_sha256"):
        raise ValueError("ORIGINAL_RESEARCH_ZIP_DIGEST_MISMATCH")
    if private_output.exists():
        raise ValueError("NO_OVERWRITE_OF_PRIVATE_TRAINED_ADAPTER")
    with ZipFile(io.BytesIO(plaintext)) as z:
        members=z.infolist()
        names=[m.filename for m in members]
        if (len(names)!=len(set(names)) or len(names)>150 or
            any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts
                or not n or "\\" in n for n in names) or
            sum(m.file_size for m in members)>1_000_000_000):
            raise ValueError("MALICIOUS_OR_UNBOUNDED_TRAINED_ARCHIVE")
        for target in ("candidate_adapter/adapter_model.safetensors",
                       "candidate_adapter/adapter_config.json",
                       "research_receipt.json"):
            if target not in names:
                raise ValueError("EXACT_TRAINED_LORA_FILES_REQUIRED:"+target)
        receipt=json.loads(z.read("research_receipt.json"))
        if (receipt.get("candidate_adapter_sha256")!=audit["candidate_adapter_sha256"] or
            receipt.get("complete_training") is not True or
            receipt.get("complete_test") is not True or
            receipt.get("selected_checkpoint")!="checkpoint-500" or
            receipt.get("model_repo")!="Qwen/Qwen3-4B" or
            receipt.get("execution_allowed") is not False):
            raise ValueError("PINNED_RESEARCH_RECEIPT_REQUIRED")
        adapter=z.read("candidate_adapter/adapter_model.safetensors")
        digest=hashlib.sha256(adapter).hexdigest()
        if digest!=audit["candidate_adapter_sha256"]:
            raise ValueError("TRAINED_LORA_WEIGHTS_SUBSTITUTED")
        config=z.read("candidate_adapter/adapter_config.json")
        values=json.loads(config)
        if not isinstance(values,dict) or values.get("peft_type")!="LORA" or len(config)>20000:
            raise ValueError("PEFT_LORA_CONFIGURATION_REQUIRED")
    private_output.mkdir(mode=0o700,parents=False)
    for name,data in (("adapter_model.safetensors",adapter),
                      ("adapter_config.json",config)):
        fd=os.open(private_output/name,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,"wb") as stream:
            stream.write(data)
    report={"status":"AUTHENTICATED_TRAINED_QWEN3_4B_LORA_RECOVERED",
            "original_archive_sha256":archive_sha,
            "adapter_sha256":digest,
            "selected_checkpoint":"checkpoint-500",
            "optimizer_updates":1000,
            "research_gate_passed":False,
            "model_training_proved":True,
            "actual_model_inference_performed":False,
            "execution_allowed":False}
    fd=os.open(private_output/"recovery_receipt.json",
               os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,"w",encoding="utf-8") as out:
        json.dump(report,out,sort_keys=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sealed",type=Path,required=True)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--private-output",type=Path,required=True)
    a=parser.parse_args()
    result=recover(a.sealed,a.audit,a.private_output,os.environ.get("KAGGLE_API_TOKEN",""))
    print("AUTHENTICATED_PRIVATE_WILLIAMS_CANDIDATE_RECOVERY",
          json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
