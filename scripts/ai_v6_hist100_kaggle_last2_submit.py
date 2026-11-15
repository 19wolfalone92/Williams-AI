"""One private quota-limited T4 run for ONLY two remaining real H1 cases."""
from __future__ import annotations
from pathlib import Path
import json
import os
import re
import subprocess
import tempfile
from scripts.ai_v6_hist100_kaggle_gpu_submit import make_metadata, TRAINING_NAME

NEW_SLUG="williams-v6-hist100-final-two-gpu-20261115"
MIN_FREE_GPU_SECONDS=1600

def need(ok,why):
    if not ok:
        raise ValueError("HIST100_FINAL2_GPU_SUBMIT_"+why)

def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    sha=os.environ.get("GITHUB_SHA","")
    token=os.environ.get("KAGGLE_API_TOKEN","")
    need(re.fullmatch("[0-9a-f]{40}",sha) is not None and len(token)>=20,
         "NO_AUTHENTICATED_REPO_SOURCE")
    api=KaggleApi();api.authenticate()
    username=get_username()
    meta,_=make_metadata(username,sha)
    meta["id"]=username+"/"+NEW_SLUG
    meta["title"]=NEW_SLUG
    need(meta["kernel_sources"]==[username+"/"+TRAINING_NAME] and
         meta["is_private"]=="true" and meta["enable_gpu"]=="true" and
         meta["machine_shape"]=="NvidiaTeslaT4","KERNEL_INPUTS_OR_RESOURCE_CHANGED")
    try:
        previous=api.kernels_status(meta["id"])
    except Exception:
        previous=None
    need(previous is None,"FINAL_TWO_GPU_KERNEL_EXISTS_NO_RESTART")
    trainer=api.kernels_status(username+"/"+TRAINING_NAME)
    need("COMPLETE" in str(trainer.status).upper(),"ORIGINAL_80STEP_TRAINING_MISSING")
    with api.build_kaggle_client() as client:
        q=client.kernels.kernels_api_client.get_accelerator_quota_statistics(
            ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
    quota=(q.total_time_allowed-q.time_used).total_seconds() if q else None
    need(quota is not None and quota>MIN_FREE_GPU_SECONDS,"NOT_ENOUGH_CONFIRMED_FREE_GPU")
    print("HIST100_FINAL2_CONFIRMED_FREE_GPU_HOURS",round(quota/3600,2),flush=True)
    worker='''"""No keys, weights, labels, future data or training uploaded."""
from pathlib import Path
import subprocess,sys,os
import torch
assert torch.cuda.is_available(),"CUDA_NOT_ALLOCATED"
revision="PINNED_SHA"
checkout=Path("/kaggle/working/williams-hist100-finaltwo")
assert not checkout.exists()
subprocess.run(["git","clone","--quiet",
    "https://github.com/19wolfalone92/Williams-AI.git",str(checkout)],
    check=True,timeout=180)
subprocess.run(["git","checkout","--quiet",revision],cwd=checkout,check=True)
subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check",
    "transformers==4.57.1","peft==0.17.1","accelerate==1.10.1",
    "safetensors==0.6.2","bitsandbytes==0.48.1",
    "huggingface_hub>=0.30,<2","jinja2>=3.1.6,<4"],
    check=True,timeout=520)
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
subprocess.run([sys.executable,"-m","scripts.ai_v6_hist100_kaggle_last2_worker",
    "--plan","research/williams_v6_historical100_selection_20261114.json",
    "--out-dir","/kaggle/working/hist100_only_last2"],
    cwd=checkout,env={**os.environ,"PYTHONPATH":str(checkout)},
    check=True,timeout=3600)
'''.replace("PINNED_SHA",sha)
    need(token not in worker and "KAGGLE_API_TOKEN" not in worker,
         "TOKEN_LEAK")
    with tempfile.TemporaryDirectory(prefix="gpu-hist100-only-two-") as tmp:
        d=Path(tmp)
        (d/"kernel-metadata.json").write_text(json.dumps(meta))
        (d/"worker.py").write_text(worker)
        push=subprocess.run(["kaggle","kernels","push","-p",tmp,
             "--accelerator","NvidiaTeslaT4","--timeout","4800"],
             capture_output=True,text=True,timeout=180)
        need(push.returncode==0,"KAGGLE_FINAL_TWO_SUBMISSION_REJECTED")
    state=str(api.kernels_status(meta["id"]).status).upper()
    print("HIST100_KAGGLE_FINAL_TWO_GPU_SUBMITTED",state,flush=True)
    result={"schema":"WILLIAMS_HIST100_REMAINDER_TWO_GPU_SUBMISSION_RECEIPT_V1",
            "kernel_id":meta["id"],"initial_status":state,
            "source_commit":sha,"no_trading":True,
            "private_trained_v5_kernel_only":True,
            "pinned_case_ids":["HIST100-25-3","HIST100-25-4"],
            "complete":False}
    output=Path(os.environ["RUNNER_TEMP"])/"hist100_last2_gpu_submit.json"
    output.write_text(json.dumps(result,indent=2))
if __name__=="__main__":
    main()
