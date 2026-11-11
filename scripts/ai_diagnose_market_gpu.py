"""Read numerical and safe exception diagnostics from an owned research kernel."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from scripts.ai_observe_own_training import numerical_progress

_SLUGS = {"williams-ai-closed-chart-transfer-20261107", "williams-ai-closed-chart-transfer-20261107-r2"}
_SAFE_FRAME = re.compile(r'File "[^"\n]*/((?:williams_ai/)?[a-zA-Z0-9_]+\.py)", line (\d+)')
_SAFE_ERROR = re.compile(r"^\s*(?:\[rank\d+\]:\s*)?(\w+(?:Error|Exception)):\s*(.*)$")
_SAFE_CODES = ("EXACT_", "FINITE_", "THREE_", "PINNED_", "OFFICIAL_", "NO_", "BOTH_", "COMPLETE_",
               "FROZEN_", "TWO_", "REAL_", "ONLY_", "UNVERIFIED_", "WEIGHT_", "TRAIN_")


def safe_diagnostic(log: str) -> dict:
    errors = []
    frames = []
    for line in log.splitlines():
        match = _SAFE_FRAME.search(line)
        if match and ("/" in match[1] or match[1] in {"worker.py", "entry.py", "trainer.py"}):
            frames.append([match[1], int(match[2])])
        match = _SAFE_ERROR.match(line)
        if not match:
            continue
        name, detail = match.groups()
        code = None
        if detail.startswith(_SAFE_CODES) and re.fullmatch(r"[A-Z0-9_]{5,110}", detail):
            code = detail
        elif "out of memory" in detail.lower():
            code = "CUDA_OUT_OF_MEMORY"
        elif name == "HTTPError":
            http = re.search(r"HTTP Error ([1-5]\d\d)", detail)
            code = "HTTP_STATUS_" + http[1] if http else "HTTP_STATUS_ERROR"
        elif name == "FileNotFoundError":
            code = "EXPECTED_FILE_MISSING"
        record = {"type": name[:40], "code": code}
        if record not in errors:
            errors.append(record)
    return {"errors": errors[-16:], "trace_locations": frames[-28:],
            "log_lines": len(log.splitlines()), "numerical_progress": numerical_progress(log)[-8:]}


def fetch_diagnostic(api, owner, slug):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    if slug not in _SLUGS:
        raise ValueError("ONLY_OWN_MARKET_RESEARCH_KERNEL_DIAGNOSIS_ALLOWED")
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner
        request.kernel_slug = slug
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
    return {**safe_diagnostic(response.log or ""), "output_file_count": len(response.files)}


if __name__ == "__main__":
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    api = KaggleApi()
    api.authenticate()
    print("SAFE_MARKET_GPU_FAILURE", json.dumps(fetch_diagnostic(api, get_username(), os.environ["WILLIAMS_OBSERVE_SLUG"])))
