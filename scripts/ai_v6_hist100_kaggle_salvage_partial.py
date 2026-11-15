"""Read-only salvage of partial *real* 95+ historical GPU case pairs after Kaggle ERROR.

No new GPU, no kernel edit, no secrets, no weights or books exported. This
reports *partial* evidence only and does not claim the 100-case exam completed.
"""
from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from scripts.ai_v6_hist100_kaggle_gpu_submit import KERNEL_NAME
from scripts.ai_v6_hist100_kaggle_gpu_worker import SCHEMA
from williams_ai.v6_v5_lora_binding import MODEL_COMMIT, RAW_ADAPTER_SHA


def need(ok, why):
    if not ok:
        raise ValueError("GPU_PARTIAL_RECOVERY_" + why)


def verify_partial(result, plan):
    need(result.get("schema") == SCHEMA and
         result.get("complete") is False and
         result.get("no_trading") is True and
         result.get("actual_book_accuracy") is None and
         result.get("expert_gold") is None and
         result.get("qwen_base_revision") == MODEL_COMMIT and
         result.get("verified_genuine_trained_adapter_sha256") == RAW_ADAPTER_SHA and
         result.get("gpu_prompt_is_compacted_vs_cpu_protocol") is True and
         result.get("cuda_is_available") is True and
         result.get("gpu_protocol_not_bit_identical_to_llama_cpp_gguf") is True and
         result.get("source_zip_sha256_verified") is True, "NOT_GENUINE_PARTIAL_GPU")
    count=result.get("completed_paired_cases")
    obs=result.get("observed")
    need(type(count) is int and 1 <= count < 100 and
         type(obs) is dict and
         set(obs) == {"matched_base", "candidate_v5"} and
         all(type(obs[r]) is list and len(obs[r]) == count
             for r in obs) and
         result.get("real_model_inference_rows") == 2*count,
         "RECEIPT_COUNTS_INCONSISTENT")
    chosen={v["id"]:v for v in plan["cases"]}
    need(len(chosen)==100 and
         len(set(x["id"] for x in obs["matched_base"]))==count and
         len(set(x["id"] for x in obs["candidate_v5"]))==count, "ID_DUPLICATE")
    for role in ("matched_base","candidate_v5"):
        for x in obs[role]:
            c=chosen.get(x["id"])
            need(c is not None and x["symbol"]==c["symbol"] and
                 x["edition"]==c["edition"] and
                 x["cutoff_utc"]==c["cutoff_utc"] and
                 x["finish_reason"] in ("stop","length","error") and
                 isinstance(x["prompt_sha256"],str) and len(x["prompt_sha256"])==64 and
                 isinstance(x["chart_sha256"],str) and len(x["chart_sha256"])==64,
                 "CASE_RECEIPT_MISMATCH")
    for a,b in zip(obs["matched_base"], obs["candidate_v5"],strict=True):
        need(all(a[k]==b[k] for k in (
            "id","symbol","edition","family","cutoff_utc","chart_sha256",
            "prompt_sha256","original_cpu_full_prompt_sha256",
            "gpu_prompt_tokens","gpu_visible_closed_h1_bars",
            "retrieved_rule_ids")), "NOT_MATCHED_GPU_CASES")
    return {"verified_real_paired_cases":count,
            "verified_real_model_rows":2*count,
            "complete":False, "missing_paired_cases":100-count,
            "correctness_accuracy":None, "expert_gold":None,
            "raw_gpu_outputs_are_not_gguf_cpu_protocol":True,
            "no_GPU_rerun":True}


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    token=os.environ.get("KAGGLE_API_TOKEN","")
    need(len(token)>=20,"MISSING_SECRET")
    owner=get_username()
    kernel=owner+"/"+KERNEL_NAME
    api=KaggleApi();api.authenticate()
    state=str(api.kernels_status(kernel).status).upper()
    print("KAGGLE_LAST_STATUS_BEFORE_PARTIAL_SALVAGE",state,flush=True)
    need("ERROR" in state,"NOT_TERMINAL_ERROR_KERNEL")
    with tempfile.TemporaryDirectory(prefix="gpu-hist100-partial-private-") as tmp:
        proc=subprocess.run(
            ["kaggle","kernels","output",kernel,"-p",tmp,"-o",
             "--file-pattern",r".*ACTUAL_GPU_HIST100_QWEN_BASE_V5_NO_GOLD\.json$",
             "--page-size","200"],
            capture_output=True,text=True,timeout=540)
        if proc.returncode != 0:
            # A failed Kaggle kernel may not publish outputs. No data fabrication.
            print("HIST100_GPU_PARTIAL_OUTPUT_UNAVAILABLE_FROM_KAGGLE_ERROR",flush=True)
            return
        root=Path(tmp)
        matches=list(root.rglob("ACTUAL_GPU_HIST100_QWEN_BASE_V5_NO_GOLD.json"))
        if len(matches)!=1:
            print("HIST100_GPU_ERROR_KERNEL_HAS_NO_UNIQUE_PARTIAL_JSON",len(matches),flush=True)
            return
        receipt=json.loads(matches[0].read_bytes())
    plan=json.loads(Path("research/williams_v6_historical100_selection_20261114.json").read_bytes())
    verified=verify_partial(receipt,plan)
    output=Path(os.environ["RUNNER_TEMP"])/"hist100_gpu_verified_PARTIAL_not_complete.json"
    with output.open("x",encoding="utf8") as file:
        json.dump(receipt,file,sort_keys=True,ensure_ascii=False,indent=2)
    print("HIST100_GPU_RECOVERED_PARTIAL_NOT_COMPLETE",json.dumps(verified),flush=True)


if __name__=="__main__":
    main()
