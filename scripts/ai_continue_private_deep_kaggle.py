"""Continue only our own checksum-verified private weights on the full Course.

Private signed output URLs avoid the failing Kaggle dataset-attachment path.
The URL is never logged or committed, and the original Kaggle credential is
never sent to a GPU worker. Optimizer continuation is not claimed.
"""
from __future__ import annotations
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import urlopen
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ai_run_private_deep_kaggle import run
from williams_ai.deep_book_research import MODEL, REVISION


def checked_own_previous_result(archive: bytes) -> dict:
    with zipfile.ZipFile(BytesIO(archive)) as z:
        report = json.loads(z.read("research_receipt.json"))
        adapter = z.read("candidate_adapter/adapter_model.safetensors")
    if (report.get("model_repo") != MODEL or report.get("model_revision") != REVISION
            or report.get("source_commit") != "6099fb60f4f8f050e0e51f0253c970f20826b833"
            or report.get("rank") != 16 or report.get("adapter_weights_changed") is not True
            or report.get("execution_allowed") is not False
            or report.get("candidate_adapter_sha256") != sha256(adapter).hexdigest()):
        raise ValueError("OWN_REAL_PRIVATE_RANK16_ADAPTER_NOT_PROVEN")
    return {"archive_sha256": sha256(archive).hexdigest(), "adapter_sha256": sha256(adapter).hexdigest()}


def main():
    from kagglehub.auth import get_username
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner = get_username()
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,62}", owner):
        raise ValueError("OWN_AUTHENTICATED_KAGGLE_ACCOUNT_REQUIRED")
    old_slug = "williams-ai-private-originals-deep-20261027"
    old_kernel = owner + "/" + old_slug
    # Observe only until the current single GPU writer is finished.
    complete = False
    for _ in range(260):
        state = subprocess.run(["kaggle", "kernels", "status", old_kernel], capture_output=True, text=True, timeout=35)
        value = state.stdout.lower()
        if state.returncode == 0 and ("complete" in value or "finished" in value):
            complete = True; break
        if state.returncode == 0 and any(word in value for word in ("error", "failed", "cancel")):
            raise RuntimeError("FIRST_PRIVATE_TRAINING_INCOMPLETE_NO_AUTOMATIC_SECOND_GPU_ALLOCATION")
        time.sleep(20)
    if not complete:
        raise RuntimeError("FIRST_GPU_RUN_NOT_FINISHED_NO_SECOND_TRAINING_STARTED")
    api = KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as client:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = old_slug
        response = client.kernels.kernels_api_client.list_kernel_session_output(req)
    candidates = [item for item in response.files if item.file_name == "Williams_Qwen3_4B_PRIVATE_DEEP_RESEARCH.zip"]
    if len(candidates) != 1 or urlsplit(candidates[0].url).scheme != "https":
        raise ValueError("OWN_PRIVATE_KERNEL_SIGNED_OUTPUT_REQUIRED")
    url = candidates[0].url
    archive = urlopen(url, timeout=240).read()
    initial = {**checked_own_previous_result(archive), "url": url}
    del archive
    print("CONTINUING_ONLY_VERIFIED_OWN_RANK16_WEIGHTS_WITH_NEW_OPTIMIZER", initial["adapter_sha256"], flush=True)
    run(sealed_path="experiments/private-source-20261027-course-full.sealed.json",
        slug="williams-ai-full-course-corrective-rank16-20261027", initialization=initial,
        separate_market=True, output_directory="private_deep_followup_results")


if __name__ == "__main__":
    main()
