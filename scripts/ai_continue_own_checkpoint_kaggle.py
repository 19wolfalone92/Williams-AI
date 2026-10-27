"""Continue our verified interrupted checkpoint416 with a fresh optimizer.

Only expiring URLs to two safe adapter files enter the private GPU worker.
The Kaggle credential and all expected exam answers remain on the CPU runner.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import urlopen

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ai_run_private_deep_kaggle import run
from williams_ai.deep_book_research import MODEL,REVISION

WEIGHT_SHA="3b3030afe4c19fdfd533013309f87e5b544f92ab276e03bfa9ce8a54d8f1c6ba"
SOURCE_SHA="6099fb60f4f8f050e0e51f0253c970f20826b833"
PARAMETERS=33030144
TARGETS={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}


def checked_checkpoint(config:dict,state:dict,report:dict,weights:bytes)->dict:
    if (report.get("model_repo")!=MODEL or report.get("model_revision")!=REVISION
        or report.get("source_commit")!=SOURCE_SHA or report.get("rank")!=16
        or report.get("execution_allowed") is not False):
        raise ValueError("OWN_CHECKPOINT_SOURCE_PROVENANCE_FAILED")
    if (config.get("r")!=16 or config.get("lora_alpha")!=32 or config.get("bias")!="none"
        or set(config.get("target_modules",[]))!=TARGETS or config.get("peft_type")!="LORA"
        or config.get("modules_to_save") or config.get("use_dora")
        or state.get("global_step")!=416 or state.get("max_steps")!=512
        or Path(state.get("best_model_checkpoint","")).name!="checkpoint-416"):
        raise ValueError("OWN_COMPATIBLE_PARTIAL_416_CHECKPOINT_REQUIRED")
    if sha256(weights).hexdigest()!=WEIGHT_SHA:
        raise ValueError("OWN_CHECKPOINT_IMMUTABLE_WEIGHT_SHA_FAILED")
    import numpy as np
    from safetensors.numpy import load
    tensors=load(weights)
    if (not tensors or any(".lora_" not in n or x.dtype!=np.float32 or not np.isfinite(x).all() for n,x in tensors.items())
        or sum(x.size for x in tensors.values())!=PARAMETERS
        or sum(".lora_B." in n and np.any(x!=0) for n,x in tensors.items())!=252):
        raise ValueError("OWN_FINITE_ACTUALLY_TRAINED_LORA_TENSORS_REQUIRED")
    return {"source_commit":SOURCE_SHA,"model_repo":MODEL,"model_revision":REVISION,
        "checkpoint":"checkpoint-416","completed_optimizer_steps":416,"requested_source_steps":512,
        "first_schedule_completed":False,"initial_adapter_sha256":WEIGHT_SHA,
        "trainable_parameters":PARAMETERS,"all_tensors_finite":True,"optimizer_state_resumed":False,
        "execution_allowed":False}


def main():
    from kagglehub.auth import get_username
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner=get_username()
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,62}",owner):raise ValueError("OWN_KAGGLE_ACCOUNT_REQUIRED")
    slug="williams-ai-private-originals-deep-20261027"
    status=subprocess.run(["kaggle","kernels","status",owner+"/"+slug],capture_output=True,text=True,timeout=35)
    if status.returncode or not any(w in status.stdout.lower() for w in ("error","failed","cancel")):
        raise RuntimeError("OWN_INTERRUPTED_GPU_MUST_BE_STOPPED_BEFORE_CONTINUING")
    api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as client:
        req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=slug
        response=client.kernels.kernels_api_client.list_kernel_session_output(req)
    files={item.file_name:item for item in response.files}
    prefix="private_checkpoints_originals/checkpoint-416/"
    required={name:prefix+name for name in ("adapter_config.json","adapter_model.safetensors","trainer_state.json")}
    required["research_receipt.json"]="private_deep_qwen4b/research_receipt.json"
    raw={};urls={}
    for name,path in required.items():
        item=files.get(path)
        if item is None or urlsplit(item.url).scheme!="https":raise ValueError("OWN_SAFE_CHECKPOINT_SIGNED_FILES_REQUIRED")
        urls[name]=item.url;raw[name]=urlopen(item.url,timeout=240).read()
    proof=checked_checkpoint(json.loads(raw["adapter_config.json"]),json.loads(raw["trainer_state.json"]),
        json.loads(raw["research_receipt.json"]),raw["adapter_model.safetensors"])
    initial={"format":"OWN_PARTIAL_RANK16_CHECKPOINT_V1","adapter_sha256":WEIGHT_SHA,"provenance":proof,
        "files":{n:{"url":urls[n],"sha256":sha256(raw[n]).hexdigest()} for n in ("adapter_config.json","adapter_model.safetensors")}}
    del raw,urls
    print("VERIFIED_OWN_PARTIAL_416_NEW_OPTIMIZER",json.dumps(proof),flush=True)
    run(sealed_path="experiments/private-source-20261027-course-full.sealed.json",
        slug="williams-ai-recovered-full-course-rank16-20261027",initialization=initial,separate_market=True,
        output_directory="recovered_checkpoint_continued_results",generated_transfer=True)


if __name__=="__main__":main()
