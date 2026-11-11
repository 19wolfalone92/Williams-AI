"""One private quota-checked Kaggle NF4 SFT, on clean Qwen3-4B.

Uses PUBLIC approved paraphrased four-book rule cards only. Does not upload
user-supplied book PDFs, OCR, model secret or private error memory. Exports
only numerical receipt and authenticated AES-GCM encrypted trained adapter.
A later CPU exam loads the original adapter after decryption privately.
"""
from __future__ import annotations

from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile,ZIP_STORED
from urllib.request import urlopen

from williams_ai.knowledge_curriculum_v5 import build

KERNEL_SLUG="williams-ai-v5-book-sft-20261111"
DOMAIN=b"Williams AI V5 four-book QLoRA private artifact v1"
MODEL="Qwen/Qwen3-4B"
REV="3fb356690bbdae7353c94efb3eecdce276e6ed1c"
TRAIN_STEPS=80
GPU_TIMEOUT=14000
MIN_FREE_GPU_SECONDS=8500


def worker_source(commit_sha):
    if not isinstance(commit_sha,str) or not re.fullmatch(r"[0-9a-f]{40}",commit_sha):
        raise ValueError("EXACT_TRAINER_SOURCE_COMMIT_SHA_REQUIRED")
    script='''"""Private single-GPU Qwen3-4B real source-card SFT worker."""
import os,subprocess,sys,threading,time,json,signal,urllib.request
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
REV="COMMIT_PLACEHOLDER"
STOP="https://raw.githubusercontent.com/19wolfalone92/Williams-AI/research/williams-ai-curriculum-v5-20261111/experiments/v5-training-control.json"
stop_event=threading.Event()
def watch():
    failures=0
    while not stop_event.wait(45):
        try:
            with urllib.request.urlopen(STOP+"?t="+str(int(time.time()//45)),timeout=12) as r:
                state=json.load(r)
            if state.get("run_slug")!="williams-ai-v5-book-sft-20261111" or type(state.get("stop")) is not bool:
                raise ValueError("UNAUTHORIZED_GPU_STOP_CONFIG")
            if state["stop"]:
                print("GPU_STOP_REQUESTED_CANCEL_TRAINING",flush=True)
                os.kill(os.getpid(),signal.SIGTERM)
            failures=0
        except Exception:
            failures+=1
            if failures>=8:
                print("STOP_WATCH_LOST_FAIL_CLOSED",flush=True)
                os.kill(os.getpid(),signal.SIGTERM)

threading.Thread(target=watch,daemon=True).start()
home=Path("/kaggle/working/williams_v5_source")
if home.exists():
    raise RuntimeError("ISOLATED_SOURCE_DIR_MUST_BE_NEW")
subprocess.run(["git","clone","--quiet",
    "https://github.com/19wolfalone92/Williams-AI.git",str(home)],check=True)
subprocess.run(["git","checkout","--quiet",REV],cwd=home,check=True)
subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check",
    "transformers==4.57.1","peft==0.17.1","accelerate==1.10.1",
    "safetensors==0.6.2","bitsandbytes==0.48.1",
    "huggingface_hub>=0.30,<2","jinja2>=3.1.6,<4"],check=True)
subprocess.run([sys.executable,"-m","williams_ai.knowledge_training_v5",
     "--new-output","/kaggle/working/williams_v5_model",
     "--steps","80","--rank","8","--max-tokens","1536"],
     cwd=home,check=True,env={**os.environ,"PYTHONPATH":str(home)})
stop_event.set()
print("WILLIAMS_V5_SOURCE_CARD_REAL_GPU_COMPLETED",flush=True)
'''
    return script.replace("COMMIT_PLACEHOLDER",commit_sha)


def _seal_data(data:bytes,secret:str)->bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    if not isinstance(secret,str) or len(secret)<20:
        raise ValueError("EXISTING_PRIVATE_KAGGLE_SECRET_REQUIRED")
    salt=os.urandom(16)
    nonce=os.urandom(12)
    key=HKDF(algorithm=hashes.SHA256(),length=32,salt=salt,info=DOMAIN).derive(secret.encode())
    return b"WILLIAMSV5A1"+salt+nonce+AESGCM(key).encrypt(nonce,data,DOMAIN)


def _verify_and_archive(files:Path,secret:str,dest:Path,source_sha:str):
    receipt_files=list(files.rglob("training_receipt.json"))
    weights=list(files.rglob("adapter_model.safetensors"))
    config=list(files.rglob("adapter_config.json"))
    if len(receipt_files)!=1 or len(weights)!=1 or len(config)!=1:
        raise ValueError("ACTUAL_GENUINE_GPU_TRAINED_ADAPTER_FILES_REQUIRED")
    receipt=json.loads(receipt_files[0].read_bytes())
    if (receipt.get("schema")!="WILLIAMS_V5_GENUINE_QWEN3_4B_SOURCE_ONLY_LORA" or
        receipt.get("base_model")!=MODEL or receipt.get("model_commit")!=REV or
        receipt.get("test_cases_provided_to_trainer")!=0 or
        receipt.get("test_gold_examined_during_checkpoint_selection") is not False or
        receipt.get("expert_gold_market_examples")!=0 or
        receipt.get("weights_changed") is not True or
        receipt.get("optimizer_steps",0)<10 or
        receipt.get("execution_allowed") is not False):
        raise ValueError("INCOMPATIBLE_OR_UNSAFE_GPU_TRAINING_RECEIPT")
    if sha256(weights[0].read_bytes()).hexdigest()!=receipt["adapter_sha256"]:
        raise ValueError("PUBLIC_SFT_ADAPTER_DIGEST_DOES_NOT_MATCH_GPU_RECEIPT")
    cfg=json.loads(config[0].read_bytes())
    if cfg.get("peft_type")!="LORA":
        raise ValueError("ACTUAL_PEFT_LORA_CONFIGURATION_REQUIRED")
    cards,manifest=build()
    if (receipt["rule_registry_sha256"]!=manifest["source_registry_sha256"] or
        receipt["dataset_split_prompt_sha256"]!=manifest["split_prompt_sha256"] or
        receipt["dataset_split_gold_sha256"]!=manifest["split_gold_sha256"]):
        raise ValueError("GPU_TRAINED_DIFFERENT_DATASET_SPLITS")
    # A private ZIP is authenticated by its manifest and encrypted before
    # any GitHub artifact; raw weights NEVER leave the runner unsealed.
    buf=io.BytesIO()
    with ZipFile(buf,"w",compression=ZIP_STORED) as z:
        z.writestr("adapter_model.safetensors",weights[0].read_bytes())
        z.writestr("adapter_config.json",config[0].read_bytes())
        z.writestr("training_receipt.json",json.dumps(receipt,sort_keys=True))
        z.writestr("provenance.json",json.dumps(
            {"trainer_source_sha":source_sha,
             "dataset_rule_registry_sha":manifest["source_registry_sha256"],
             "research_only":True,"trade_execution":False},sort_keys=True))
    encrypted=_seal_data(buf.getvalue(),secret)
    dest.mkdir(parents=True,mode=0o700,exist_ok=False)
    (dest/"williams-v5-trained-qwen3-4b-lora.zip.sealed").write_bytes(encrypted)
    public={k:receipt[k] for k in (
        "schema","base_model","model_commit","adapter_sha256",
        "train_case_count","validation_case_counts","optimizer_steps",
        "selected_validation_checkpoint","validation_baseline_per_edition",
        "validation_after_per_edition","validation_gate","execution_allowed",
        "trading_qualification","test_cases_provided_to_trainer")}
    public["source_commit"]=source_sha
    public["encrypted_adapters_only"]=True
    public["sha256_encrypted_archive"]=sha256(encrypted).hexdigest()
    (dest/"v5_training_audit.json").write_text(
        json.dumps(public,ensure_ascii=False,sort_keys=True,indent=2))
    return public


def run():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    token=os.environ.get("KAGGLE_API_TOKEN","")
    sha=os.environ.get("GITHUB_SHA","")
    if len(token)<20 or not re.fullmatch("[0-9a-f]{40}",sha):
        raise ValueError("REPO_SOURCE_AND_CLOUD_SECRET_REQUIRED")
    rows,manifest=build()
    if manifest["approved_market_labels"] or not manifest["test_locked_before_training"]:
        raise ValueError("NO_EXPERT_GOLD_AND_NO_TEST_FIT")
    api=KaggleApi();api.authenticate()
    owner=get_username()
    kernel=owner+"/"+KERNEL_SLUG
    with api.build_kaggle_client() as client:
        quota=client.kernels.kernels_api_client.get_accelerator_quota_statistics(
            ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
        remaining=(quota.total_time_allowed-quota.time_used).total_seconds() if quota else None
    print("WILLIAMS_V5_FREE_KAGGLE_GPU_HOURS",
          round(remaining/3600,2) if remaining is not None else None,flush=True)
    if remaining is None or remaining<MIN_FREE_GPU_SECONDS:
        raise ValueError("INSUFFICIENT_CONFIRMED_FREE_KAGGLE_GPU_QUOTA_NO_SUBMISSION")
    statefile="https://raw.githubusercontent.com/19wolfalone92/Williams-AI/research/williams-ai-curriculum-v5-20261111/experiments/v5-training-control.json"
    with urlopen(statefile+"?preflight="+str(int(time.time())),timeout=20) as source:
        state=json.load(source)
    if state!={"run_slug":KERNEL_SLUG,"stop":False}:
        raise ValueError("USER_EXPLICIT_TRAINING_STOP_OR_CONTROL_MISSING")
    # Fail instead of overwriting a previous Kaggle run with different data.
    try:
        previous=api.kernels_status(kernel)
    except Exception:
        previous=None
    if previous is not None and getattr(previous,"status",None) is not None:
        raise ValueError("TRAINING_KERNEL_ALREADY_EXISTS_DO_NOT_OVERWRITE")
    worker=worker_source(sha)
    if token in worker or len(worker)>45000:
        raise ValueError("CREDENTIAL_EXPOSURE_OR_OVERLARGE_WORKER")
    with tempfile.TemporaryDirectory() as workspace:
        private=Path(workspace)
        (private/"worker.py").write_text(worker)
        (private/"kernel-metadata.json").write_text(json.dumps({
            "id":kernel,"title":KERNEL_SLUG,"code_file":"worker.py",
            "language":"python","kernel_type":"script","is_private":"true",
            "enable_gpu":"true","enable_internet":"true",
            "machine_shape":"NvidiaTeslaT4","dataset_sources":[],
            "model_sources":[],"competition_sources":[],"kernel_sources":[]}))
        push=subprocess.run(["kaggle","kernels","push","-p",workspace,
                "--accelerator","NvidiaTeslaT4","--timeout",str(GPU_TIMEOUT)],
                capture_output=True,text=True,timeout=180)
        if push.returncode:
            # Avoid printing tool's raw stdout/stderr (could include personal
            # metadata); only emit machine-safe classification.
            raise RuntimeError("PRIVATE_KAGGLE_KERNEL_SUBMISSION_FAILED")
    print("WILLIAMS_V5_ACTUAL_GPU_SUBMITTED",sha,flush=True)
    status="UNKNOWN"
    for attempt in range(650):
        status=str(api.kernels_status(kernel).status).upper()
        if attempt%9==0:
            print("WILLIAMS_V5_GPU_STATE",status,flush=True)
        if any(x in status for x in ("COMPLETE","ERROR","CANCEL")):
            break
        time.sleep(20)
    if "COMPLETE" not in status:
        raise RuntimeError("V5_GPU_TRAINING_NOT_COMPLETED_NO_WEIGHTS_CLAIM")
    with tempfile.TemporaryDirectory() as download:
        fetch=subprocess.run(["kaggle","kernels","output",kernel,"-p",download],
            capture_output=True,text=True,timeout=600)
        if fetch.returncode:
            raise RuntimeError("PRIVATE_GPU_MODEL_DOWNLOAD_FAILED")
        dest=Path(os.environ["RUNNER_TEMP"])/"williams_v5_encrypted"
        receipt=_verify_and_archive(Path(download),token,dest,sha)
    print("ACTUAL_WILLIAMS_V5_TRAINING_COMPLETED_AND_SEALED",
          json.dumps({"steps":receipt["optimizer_steps"],
                      "adapter_sha256":receipt["adapter_sha256"],
                      "validation_gate":receipt["validation_gate"],
                      "trading_qualification":False},sort_keys=True),flush=True)


if __name__=="__main__":
    run()
