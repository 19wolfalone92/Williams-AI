"""Inspect sealed V5/V6 checkpoint archives without publishing plaintext."""
from __future__ import annotations
import argparse, base64, hashlib, io, json, os
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

DOMAIN=b"Williams AI V5 four-book QLoRA private artifact v1"
MAGIC=b"WILLIAMSV5A1"
ENVELOPE_DOMAIN=b"WILLIAMS_AI_DIAGNOSTIC_MAP_ENVELOPE_V1"
PUBLIC_KEY_FINGERPRINT_SHA256="5918e3bf1d07e06f28513ea0ee230b5be0b899e1d6595878449dd05a33934ac5"
V5_RUN=45406594397
V6_RUN=45480042116
V6_SEALED_SHA256="fa026c401c8ea89bcea119fac153d30bf1004874fe16c81cb8de761a852f03e2"
V6_ADAPTER_SHA256="f4979cf0bebbf33b7c8e5746326c1e43762f31fe730e89cf5e39c7a8a155c309"


def decrypt_sealed(raw:bytes,secret:str)->bytes:
    if not isinstance(secret,str) or len(secret)<20: raise ValueError("KAGGLE_SECRET_MISSING")
    if not raw.startswith(MAGIC) or len(raw)<=len(MAGIC)+28: raise ValueError("SEALED_ARCHIVE_HEADER_INVALID")
    n=len(MAGIC); salt=raw[n:n+16]; nonce=raw[n+16:n+28]
    key=HKDF(algorithm=hashes.SHA256(),length=32,salt=salt,info=DOMAIN).derive(secret.encode())
    try: return AESGCM(key).decrypt(nonce,raw[n+28:],DOMAIN)
    except Exception: raise ValueError("SEALED_ARCHIVE_AUTHENTICATION_FAILED") from None


def _read_single(directory:Path,pattern:str)->Path:
    matches=[p for p in directory.rglob(pattern) if p.is_file() and not p.is_symlink()]
    if len(matches)!=1: raise ValueError("EXPECTED_EXACTLY_ONE_"+pattern.upper().replace(".","_").replace("*","STAR"))
    return matches[0]


def _zip_inventory(clear:bytes):
    try:
        with ZipFile(io.BytesIO(clear)) as z:
            names=z.namelist()
            if len(names)!=len(set(names)): raise ValueError("DUPLICATE_INNER_ARCHIVE_PATH")
            info=[]; members={}
            for name in names:
                p=Path(name); zi=z.getinfo(name)
                if p.is_absolute() or ".." in p.parts or ((zi.external_attr>>16)&0o170000)==0o120000:
                    raise ValueError("UNSAFE_INNER_ARCHIVE_MEMBER")
                data=z.read(name); members[name]=data
                info.append({"path":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
            return members,info
    except BadZipFile: raise ValueError("INNER_RECOVERY_ZIP_INVALID") from None


def inspect_v5(directory:Path,secret:str):
    sealed=_read_single(directory,"williams-v5-trained-qwen3-4b-lora.zip.sealed")
    audit_path=_read_single(directory,"v5_training_audit.json")
    audit=json.loads(audit_path.read_bytes())
    raw=sealed.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=audit.get("sha256_encrypted_archive"):
        raise ValueError("V5_SEALED_ARCHIVE_SHA_MISMATCH")
    clear=decrypt_sealed(raw,secret)
    with ZipFile(io.BytesIO(clear)) as z:
        names=set(z.namelist())
        required={"adapter_model.safetensors","adapter_config.json","training_receipt.json","provenance.json"}
        if names!=required: raise ValueError("V5_ARCHIVE_MEMBER_SET_MISMATCH")
        weight=z.read("adapter_model.safetensors")
        receipt=json.loads(z.read("training_receipt.json"))
        cfg=json.loads(z.read("adapter_config.json"))
        files=[{"path":n,"bytes":len(z.read(n)),"sha256":hashlib.sha256(z.read(n)).hexdigest()} for n in sorted(names)]
    adapter_sha=hashlib.sha256(weight).hexdigest()
    if adapter_sha!=receipt.get("adapter_sha256") or adapter_sha!=audit.get("adapter_sha256"):
        raise ValueError("V5_ADAPTER_IDENTITY_MISMATCH")
    return {"artifact_run_id":V5_RUN,"sealed_archive_sha256":hashlib.sha256(raw).hexdigest(),
        "adapter_sha256":adapter_sha,"adapter_bytes":len(weight),"optimizer_state_available":False,
        "training_steps":receipt.get("optimizer_steps"),"peft_type":cfg.get("peft_type"),
        "rank":cfg.get("r"),"inner_files":files,"resume_mode":"FRESH_OPTIMIZER_FROM_VERIFIED_V5_ADAPTER"}


def inspect_v6(directory:Path,secret:str):
    sealed=_read_single(directory,"*.sealed")
    raw=sealed.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=V6_SEALED_SHA256: raise ValueError("V6_SEALED_ARCHIVE_SHA_MISMATCH")
    members,files=_zip_inventory(decrypt_sealed(raw,secret))
    names={x["path"] for x in files}
    prefix="output/williams_v6_objective_smoke/second/"
    weight_name=prefix+"latest_adapter/adapter_model.safetensors"
    config_name=prefix+"latest_adapter/adapter_config.json"
    receipt_name=prefix+"research_result.json"
    for name in (weight_name,config_name,receipt_name):
        if name not in names: raise ValueError("V6_REQUIRED_RECOVERY_MEMBER_MISSING")
    weight=members[weight_name]; receipt=json.loads(members[receipt_name]); cfg=json.loads(members[config_name])
    adapter_sha=hashlib.sha256(weight).hexdigest()
    receipt_adapter_sha=receipt.get("latest_adapter_files",{}).get("adapter_model.safetensors") or receipt.get("adapter_sha256")
    if adapter_sha!=V6_ADAPTER_SHA256 or receipt_adapter_sha!=adapter_sha:
        raise ValueError("V6_RECOVERED_ADAPTER_SHA_MISMATCH")
    ckpt_prefix=prefix+"checkpoints/checkpoint-20/"
    contract_name=prefix+"contract.json"
    manifest_name=prefix+"checkpoint_sha256.json"
    contract=json.loads(members[contract_name]) if contract_name in names else None
    manifest=json.loads(members[manifest_name]) if manifest_name in names else None
    spec=(contract or {}).get("contract",{})
    required={"adapter_model.safetensors","adapter_config.json","training_args.bin",
              "optimizer.pt","scheduler.pt","rng_state.pth","trainer_state.json"}
    if spec.get("quantization")=="NF4_DOUBLE_FLOAT16": required.add("scaler.pt")
    present=sorted(name for name in required if ckpt_prefix+name in names)
    missing=sorted(required-set(present))
    trainer_state={}
    trainer_name=ckpt_prefix+"trainer_state.json"
    if trainer_name in names: trainer_state=json.loads(members[trainer_name])
    checkpoint_record=(manifest or {}).get("checkpoints",{}).get("checkpoint-20",{})
    manifest_valid=bool(manifest and manifest.get("schema")=="WILLIAMS_QLORA_CHECKPOINT_SEAL_V2"
        and contract_name in names and spec.get("schema")=="WILLIAMS_QLORA_RESEARCH_CONTRACT_V2"
        and manifest.get("receipt_sha256")==hashlib.sha256(members[contract_name]).hexdigest()
        and isinstance(checkpoint_record,dict) and required.issubset(checkpoint_record))
    checkpoint_hashes_valid=manifest_valid
    if checkpoint_hashes_valid:
        for name,digest in checkpoint_record.items():
            member=ckpt_prefix+name
            if member not in names or hashlib.sha256(members[member]).hexdigest()!=digest:
                checkpoint_hashes_valid=False; break
    step=trainer_state.get("global_step"); max_steps=trainer_state.get("max_steps")
    step_valid=(type(step) is int and step==20 and type(max_steps) is int
                and max_steps==spec.get("max_steps") and step%spec.get("evaluation_interval",1)==0)
    checkpoint_adapter_sha=hashlib.sha256(members[ckpt_prefix+"adapter_model.safetensors"]).hexdigest() if ckpt_prefix+"adapter_model.safetensors" in names else None
    adapter_cfg=json.loads(members[ckpt_prefix+"adapter_config.json"]) if ckpt_prefix+"adapter_config.json" in names else {}
    config_valid=(adapter_cfg.get("peft_type")=="LORA" and adapter_cfg.get("r")==spec.get("lora_rank")
        and adapter_cfg.get("lora_alpha")==spec.get("lora_alpha")
        and adapter_cfg.get("lora_dropout")==spec.get("lora_dropout")
        and sorted(adapter_cfg.get("target_modules",[]))==sorted(spec.get("target_modules",[])))
    optimizer_complete=(not missing and manifest_valid and checkpoint_hashes_valid and step_valid
        and config_valid and checkpoint_adapter_sha==adapter_sha)
    exact_resume_compatible=optimizer_complete and step<max_steps
    return {"artifact_run_id":V6_RUN,"sealed_archive_sha256":hashlib.sha256(raw).hexdigest(),
        "adapter_sha256":adapter_sha,"adapter_bytes":len(weight),"tensor_count_expected":288,
        "optimizer_state_available":optimizer_complete,"exact_resume_compatible":exact_resume_compatible,
        "checkpoint_20_any_required_file_present":bool(present),"checkpoint_20_complete":not missing,
        "checkpoint_20_required_files_present":present,"checkpoint_20_missing_files":missing,
        "checkpoint_manifest_valid":manifest_valid,"checkpoint_file_hashes_valid":checkpoint_hashes_valid,
        "checkpoint_adapter_matches_latest":checkpoint_adapter_sha==adapter_sha,
        "checkpoint_adapter_config_valid":config_valid,"trainer_global_step":step,
        "trainer_max_steps":max_steps,"contract_max_steps":spec.get("max_steps"),
        "training_steps":receipt.get("total_optimizer_steps"),
        "optimizer_skips":receipt.get("optimizer_skips_in_this_session",receipt.get("optimizer_skips")),
        "peft_type":cfg.get("peft_type"),"rank":cfg.get("r"),"inner_file_count":len(files),
        "inner_files":files,
        "resume_mode":"EXACT_RESUME" if exact_resume_compatible else "FRESH_CORRECTIVE_OPTIMIZER_FROM_VERIFIED_V6_ADAPTER"}


def encrypt_report(report:dict,public_pem:bytes):
    key=serialization.load_pem_public_key(public_pem)
    if not isinstance(key,rsa.RSAPublicKey) or key.key_size<3072: raise ValueError("RSA3072_PUBLIC_KEY_REQUIRED")
    der=key.public_bytes(serialization.Encoding.DER,serialization.PublicFormat.SubjectPublicKeyInfo)
    fp=hashlib.sha256(der).hexdigest()
    if fp!=PUBLIC_KEY_FINGERPRINT_SHA256: raise ValueError("PUBLIC_KEY_FINGERPRINT_MISMATCH")
    plain=json.dumps(report,sort_keys=True,separators=(",",":")).encode()
    data_key=os.urandom(32); nonce=os.urandom(12)
    cipher=AESGCM(data_key).encrypt(nonce,plain,ENVELOPE_DOMAIN)
    wrapped=key.encrypt(data_key,padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=ENVELOPE_DOMAIN))
    envelope={"schema":"WILLIAMS_AI_CHECKPOINT_INVENTORY_ENCRYPTED_V1","public_key_fingerprint_sha256":fp,
        "nonce_b64":base64.b64encode(nonce).decode(),"wrapped_key_b64":base64.b64encode(wrapped).decode(),
        "ciphertext_b64":base64.b64encode(cipher).decode(),"plaintext_sha256":hashlib.sha256(plain).hexdigest()}
    return json.dumps(envelope,sort_keys=True,separators=(",",":")).encode()


def build_inventory(v5_dir:Path,v6_dir:Path,secret:str):
    return {"schema":"WILLIAMS_AI_CHECKPOINT_INVENTORY_V1","v5":inspect_v5(v5_dir,secret),
        "v6":inspect_v6(v6_dir,secret),"candidate_start_policy":{
            "candidate_A":"Continue from V6 adapter weights; use a new optimizer/scheduler unless exact resumable checkpoint compatibility is separately proven.",
            "candidate_B":"Continue from V5 adapter weights with a new optimizer; V5 archive does not contain optimizer state.",
            "never_overwrite_source_adapters":True}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v5-directory",type=Path,required=True)
    p.add_argument("--v6-directory",type=Path,required=True)
    p.add_argument("--public-key",type=Path,required=True)
    p.add_argument("--secret-env",default="KAGGLE_API_TOKEN")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); secret=os.environ.pop(a.secret_env,"")
    try:
        report=build_inventory(a.v5_directory,a.v6_directory,secret)
        encrypted=encrypt_report(report,a.public_key.read_bytes())
        if a.output.exists() or a.output.is_symlink(): raise ValueError("NEW_OUTPUT_REQUIRED")
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes(encrypted)
        print("WILLIAMS_AI_CHECKPOINT_INVENTORY_ENCRYPTED",json.dumps({
            "v5_optimizer_state_available":report["v5"]["optimizer_state_available"],
            "v6_optimizer_state_available":report["v6"]["optimizer_state_available"],
            "v6_exact_resume_compatible":report["v6"]["exact_resume_compatible"],
            "v6_checkpoint_20_complete":report["v6"]["checkpoint_20_complete"],
            "v6_training_steps":report["v6"]["training_steps"],
            "encrypted_sha256":hashlib.sha256(encrypted).hexdigest(),
            "plaintext_emitted":False},sort_keys=True),flush=True)
    except Exception as exc:
        print("WILLIAMS_AI_CHECKPOINT_INVENTORY_FAILED",type(exc).__name__,str(exc),flush=True)
        return 2
    return 0

if __name__=="__main__": raise SystemExit(main())
