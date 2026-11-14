"""Safely launch user's private Kaggle T4 inference from GitHub Actions credentials.

Uses one existing PRIVATE v5 training kernel as its only input (no raw weight
upload); no new paid GPU runner, no unverified quota use, no training, no orders.
"""
from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

KERNEL_NAME = "williams-v6-hist100-matched-base-v5-gpu-20261115"
TRAINING_NAME = "williams-ai-v5-book-sft-20261111"
EXPECTED_MODEL_SHA = "8bd26590df627c7452d56e81df6a4572885bfc82515259a7c9de4a3970c850b8"
MIN_FREE_SECONDS = 3 * 3600
TIMEOUT = 11500


def need(cond, code):
    if not cond:
        raise ValueError("HIST100_KAGGLE_GPU_" + code)


def make_metadata(username: str, sha: str):
    need(re.fullmatch(r"[a-zA-Z0-9_-]{2,62}", username) is not None,
         "INVALID_KAGGLE_OWNER")
    need(re.fullmatch("[0-9a-f]{40}", sha) is not None, "INVALID_SOURCE_REVISION")
    kernel_id = f"{username}/{KERNEL_NAME}"
    original = f"{username}/{TRAINING_NAME}"
    metadata = {
        "id": kernel_id, "title": KERNEL_NAME,
        "code_file": "worker.py", "language": "python",
        "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "model_sources": [], "competition_sources": [],
        "kernel_sources": [original],
    }
    # No Kaggle secret or original books included in generated worker.
    worker = '''
"""Isolated private Kaggle GPU inference dispatcher (no keys, no books)."""
from pathlib import Path
import os
import subprocess
import sys
import torch
assert torch.cuda.is_available(), "CUDA_GPU_NOT_ALLOCATED"
SOURCE = "SOURCE_REVISION_REPLACED"
checkout=Path("/kaggle/working/williams-ai-hist100-src")
assert not checkout.exists()
subprocess.run(["git","clone","--quiet",
                "https://github.com/19wolfalone92/Williams-AI.git",
                str(checkout)],check=True,timeout=150)
subprocess.run(["git","checkout","--quiet",SOURCE],cwd=checkout,check=True)
subprocess.run([sys.executable,"-m","pip","install",
    "--disable-pip-version-check",
    "transformers==4.57.1","peft==0.17.1","accelerate==1.10.1",
    "safetensors==0.6.2","bitsandbytes==0.48.1",
    "huggingface_hub>=0.30,<2","jinja2>=3.1.6,<4"],
    cwd=checkout,check=True,timeout=480)
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
subprocess.run([sys.executable,"-m",
    "scripts.ai_v6_hist100_kaggle_gpu_worker",
    "--plan","research/williams_v6_historical100_selection_20261114.json",
    "--out-dir","/kaggle/working/hist100_actual_gpu_only"],
    cwd=checkout,env={**os.environ,"PYTHONPATH":str(checkout)},
    check=True,timeout=10400)
'''.replace("SOURCE_REVISION_REPLACED", sha)
    return metadata, worker


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    revision = os.environ.get("GITHUB_SHA", "")
    need(len(token) >= 20 and token.find("\n") < 0, "NO_PRIVATE_KAGGLE_CREDENTIAL")
    need(re.fullmatch("[0-9a-f]{40}",revision) is not None, "BAD_REPO_SHA")
    api = KaggleApi()
    api.authenticate()
    username = get_username()
    meta, worker = make_metadata(username, revision)
    kernel = meta["id"]
    original = meta["kernel_sources"][0]
    previous = api.kernels_status(original)
    need("COMPLETE" in str(previous.status).upper(),
         "GENUINE_TRAINED_V5_KERNEL_NOT_COMPLETE")
    with api.build_kaggle_client() as client:
        q = client.kernels.kernels_api_client.get_accelerator_quota_statistics(
            ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
    remaining = (q.total_time_allowed-q.time_used).total_seconds() if q else None
    print("HIST100_KAGGLE_FREE_GPU_HOURS",round(remaining/3600,2)
          if remaining is not None else None,flush=True)
    need(remaining is not None and remaining >= MIN_FREE_SECONDS,
         "INSUFFICIENT_CONFIRMED_FREE_GPU_HOURS_USE_CPU_SHARDS")
    # An existing slug may represent a completed or queued GPU job. Never
    # silently update/restart it, preventing wasted GPU quota.
    prior = None
    try:
        prior = api.kernels_status(kernel)
    except Exception:
        pass
    # One controlled retry ONLY when last version had terminal ERROR.
    # Never overwrite a Kaggle job that may still be consuming free GPU quota.
    if prior is not None:
        state=str(prior.status).upper()
        need("ERROR" in state,"GPU_ALREADY_ACTIVE_OR_COMPLETE_DO_NOT_OVERWRITE")
        print("HIST100_GPU_PREVIOUS_ERROR_RETRY_CAUSAL_MOUNT_FIX",flush=True)
    need(token not in worker and "adapter_model.safetensors" not in worker,
         "SECRET_OR_WEIGHTS_EXPOSED")
    with tempfile.TemporaryDirectory(prefix="hist100-private-gpu-submit-") as tmp:
        folder = Path(tmp)
        (folder/"worker.py").write_text(worker,encoding="utf-8")
        (folder/"kernel-metadata.json").write_text(
            json.dumps(meta,indent=2),encoding="utf-8")
        result = subprocess.run(
            ["kaggle","kernels","push","-p",str(folder),
             "--accelerator","NvidiaTeslaT4","--timeout",str(TIMEOUT)],
            capture_output=True,text=True,timeout=200)
        if result.returncode != 0:
            # Kaggle stdout/stderr may contain identifying/private info; do
            # not print them into a public GitHub Actions log.
            raise RuntimeError("PRIVATE_KAGGLE_GPU_SUBMISSION_REJECTED_CPU_REMAINS")
    state = str(api.kernels_status(kernel).status).upper()
    print("HIST100_PRIVATE_GPU_KERNEL_SUBMITTED_STATUS",state,flush=True)
    print("HIST100_KAGGLE_GPU_TRADING_ALLOWED_NO",flush=True)
    print("HIST100_KAGGLE_GPU_TRAINING_ALLOWED_NO",flush=True)
    print("HIST100_PRIVATE_EXISTING_V5_KERNEL_INPUT_ATTACHED_NO_WEIGHTS_EXPORTED",flush=True)
    with open(os.environ["GITHUB_STEP_SUMMARY"],"a",encoding="utf-8") as f:
        f.write("## Williams AI v6 hybrid CPU/GPU historical 100\n\n"
                "Private Kaggle T4 GPU job submitted; first real CUDA allocation "
                "and 100+100 outputs must still be independently verified.\n\n"
                f"[Private Kaggle GPU run](https://www.kaggle.com/code/{kernel})\n\n"
                "The existing CPU shard workflow is unaffected. "
                "No paid GitHub GPU runner was started.\n")
    receipt = {"schema":"WILLIAMS_V6_HIST100_GPU_KAGGLE_SUBMISSION_V1",
               "source_commit_sha":revision,"gpu_type_requested":"NvidiaTeslaT4",
               "kaggle_kernel_id":kernel,"gpu_job_status":state,
               "execution_proven":False,"completed_100_base_100_lora":False,
               "no_gold":True,"no_trading":True,"no_retraining":True,
               "kernel_is_private":True,"trained_adapter_private_kernel_source":True}
    out=Path(os.environ["RUNNER_TEMP"])/"hist100_gpu_submission_receipt.json"
    out.write_text(json.dumps(receipt,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
