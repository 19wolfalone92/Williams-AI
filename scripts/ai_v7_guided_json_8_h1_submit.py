"""Submit at most ONE private T4 pilot; reuse the existing kernel on later checks."""
from __future__ import annotations
import json, os, re, subprocess, tempfile
from pathlib import Path

from scripts.ai_v6_hist100_kaggle_gpu_submit import make_metadata, TRAINING_NAME
SLUG = "williams-v7-guided-json-eight-h1-20261115"


def need(ok, why):
    if not ok:
        raise ValueError("V7_PILOT_KAGGLE_" + why)


def _save(result):
    Path(os.environ["RUNNER_TEMP"], "v7_guided_pilot_submission.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    sha = os.environ.get("GITHUB_SHA", "")
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    need(re.fullmatch("[a-f0-9]{40}", sha) is not None and len(token) >= 20,
         "MISSING_PRIVATE_CREDENTIAL_OR_COMMIT_SHA")
    api = KaggleApi()
    api.authenticate()
    owner = get_username()
    meta, _ = make_metadata(owner, sha)
    meta["id"] = owner + "/" + SLUG
    meta["title"] = SLUG
    need(meta["is_private"] == "true" and
         meta["machine_shape"] == "NvidiaTeslaT4" and
         meta["kernel_sources"] == [owner + "/" + TRAINING_NAME] and
         meta["enable_gpu"] == "true", "UNSAFE_GPU_SOURCE")
    try:
        prior = api.kernels_status(meta["id"])
    except Exception as exc:
        message = str(exc).lower()
        if "404" in message or "not found" in message:
            prior = None
        else:
            raise
    if prior is not None:
        status = str(prior.status).upper()
        _save({
            "schema": "V7_GUIDED_DECODER_EXISTING_KERNEL_REUSE_NOT_RESULTS_V1",
            "kernel_private": True,
            "case_count": 8,
            "new_training": False,
            "new_kernel_submitted": False,
            "existing_kernel_reused": True,
            "status": status,
            "source_commit": sha,
            "trading_allowed": False,
        })
        print("V7_GUIDED_EXISTING_PRIVATE_T4_REUSED_NO_RESUBMISSION", status, flush=True)
        return

    need("COMPLETE" in str(api.kernels_status(owner + "/" + TRAINING_NAME).status).upper(),
         "AUTHENTIC_TRAINING_MISSING")
    with api.build_kaggle_client() as c:
        quota = c.kernels.kernels_api_client.get_accelerator_quota_statistics(
            ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
    sec = (quota.total_time_allowed - quota.time_used).total_seconds() if quota else None
    need(sec is not None and sec > 7200, "CONFIRMED_FREE_GPU_QUOTA_INSUFFICIENT")
    print("V7_PILOT_FREE_GPU_HOURS", round(sec / 3600, 2), flush=True)
    kernel = '''"""Private Kaggle, source pinned; no book scans/weights uploaded."""
import subprocess,sys,os
from pathlib import Path
import torch
assert torch.cuda.is_available(),"GPU_NOT_ALLOCATED"
sha="PINNED_COMMIT_SHA"
root=Path("/kaggle/working/williams_v7_guided_json")
subprocess.run(["git","clone","--quiet",
    "https://github.com/19wolfalone92/Williams-AI.git",str(root)],
    check=True,timeout=180)
subprocess.run(["git","checkout","--quiet",sha],cwd=root,check=True)
subprocess.run([sys.executable,"-m","pip","install",
    "--disable-pip-version-check","transformers==4.57.1",
    "peft==0.17.1","accelerate==1.10.1","safetensors==0.6.2",
    "bitsandbytes==0.48.1","lm-format-enforcer==0.11.3",
    "huggingface_hub>=0.30,<2","jinja2>=3.1.6,<4"],
    check=True,timeout=600)
subprocess.run([sys.executable,"-m","scripts.ai_v7_guided_json_8_h1_worker",
    "--plan","research/williams_v6_historical100_selection_20261114.json",
    "--out-dir","/kaggle/working/v7_8_decoded"],
    cwd=root,env={**os.environ,"PYTHONPATH":str(root),
     "HF_HUB_DISABLE_TELEMETRY":"1","TOKENIZERS_PARALLELISM":"false"},
    check=True,timeout=6400)
'''.replace("PINNED_COMMIT_SHA", sha)
    need(token not in kernel and "KAGGLE_API_TOKEN" not in kernel, "PRIVATE_TOKEN_LEAK")
    with tempfile.TemporaryDirectory(prefix="v7-kaggle-8case-") as tmp:
        d = Path(tmp)
        (d / "worker.py").write_text(kernel)
        (d / "kernel-metadata.json").write_text(json.dumps(meta))
        p = subprocess.run(["kaggle", "kernels", "push", "-p", tmp,
                            "--accelerator", "NvidiaTeslaT4", "--timeout", "7200"],
                           capture_output=True, text=True, timeout=180)
        need(p.returncode == 0, "PRIVATE_T4_SUBMISSION_FAILED")
    status = str(api.kernels_status(meta["id"]).status).upper()
    print("V7_GUIDED_JSON_REAL_T4_GPU_SUBMISSION", status, flush=True)
    _save({
        "schema": "V7_GUIDED_DECODER_GPU_SUBMISSION_NOT_MODEL_RESULTS",
        "kernel_private": True,
        "case_count": 8,
        "new_training": False,
        "new_kernel_submitted": True,
        "existing_kernel_reused": False,
        "status": status,
        "source_commit": sha,
        "trading_allowed": False,
    })


if __name__ == "__main__":
    main()
