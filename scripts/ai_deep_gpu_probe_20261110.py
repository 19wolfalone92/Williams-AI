"""Read-only Kaggle quota / parent / encrypted-corpus preflight. No GPU or order."""
from __future__ import annotations
import json
import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglehub.auth import get_username
from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest

from scripts.ai_run_joint_rule_kaggle import outputs, require_new_kernel
from scripts.ai_sealed_source_transport import open_envelope
from williams_ai.deep_book_research import checked_payload

RUN_SLUG = "williams-ai-deep-corrective-20261110-r1"
PARENT_SLUG = "williams-ai-joint-rules-20261106"
REQUIRED_FREE_GPU_SECONDS = 16200


def main():
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if len(token) < 20:
        raise RuntimeError("KAGGLE_TOKEN_UNAVAILABLE_NO_GPU_SUBMITTED")
    api = KaggleApi();api.authenticate()
    owner = get_username()
    with api.build_kaggle_client() as client:
        quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics(
            ApiGetAcceleratorQuotaStatisticsRequest()
        ).gpu_quota
        free_seconds = int((quota.total_time_allowed - quota.time_used).total_seconds()) if quota else -1
    print("DEEP_GPU_PREFLIGHT_FREE_QUOTA_SECONDS", free_seconds, flush=True)
    if free_seconds < REQUIRED_FREE_GPU_SECONDS:
        raise RuntimeError("INSUFFICIENT_CONFIRMED_FREE_GPU_QUOTA")
    require_new_kernel(api, owner, RUN_SLUG)
    prior = outputs(api, owner, PARENT_SLUG)
    prefix="historical_rule_qwen4b/"
    required={prefix+"research_receipt.json",prefix+"candidate_adapter/adapter_model.safetensors",prefix+"candidate_adapter/adapter_config.json"}
    if not required <= set(prior):
        raise RuntimeError("NOV6_PARENT_ADAPTER_UNAVAILABLE")
    sealed_path=Path("experiments/private-source-20261027-course-full.sealed.json")
    source = open_envelope(json.loads(sealed_path.read_bytes()), token)
    data=checked_payload(json.loads(source))
    if data.get("execution_allowed") is not False:
        raise RuntimeError("PRIVATE_SOURCE_NOT_RESEARCH_ONLY")
    print(json.dumps({"mode":"READ_ONLY_NO_GPU", "free_gpu_seconds":free_seconds,
      "parent_verified_output_inventory":True,
      "encrypted_four_book_corpus_readable":True, "next_slug_unused":True,
      "will_submit_gpu":False, "trading_qualification":False,
      "execution_allowed":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
