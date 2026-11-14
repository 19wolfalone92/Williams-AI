"""Watch ONE previously submitted private Kaggle T4 GPU; export ONLY verified model answers.

Read-only Kaggle polling, no resubmit, no training, no model / book / secret download
to public GitHub artifacts. Fail closed if a claimed 100+100 output is incomplete.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from scripts.ai_v6_hist100_kaggle_gpu_submit import KERNEL_NAME, TRAINING_NAME
from scripts.ai_v6_hist100_kaggle_gpu_worker import SCHEMA
from williams_ai.v6_v5_lora_binding import RAW_ADAPTER_SHA, MODEL_COMMIT
from williams_ai.independent_exam_v6 import canonical


def need(ok, reason):
    if not ok:
        raise ValueError("HIST100_GPU_RESULT_" + reason)


def validate(receipt: dict, plan: dict):
    need(receipt["schema"] == SCHEMA and
         receipt["complete"] is True and
         receipt["cuda_is_available"] is True and
         receipt["cpu_preparation_and_gpu_inference"] is True and
         receipt["source_zip_sha256_verified"] is True,
         "INVALID_GPU_RESULT_HEADER")
    need(receipt["cases_total"] == receipt["completed_paired_cases"] == 100 and
         receipt["real_model_inference_rows"] == 200 and
         receipt["verified_genuine_trained_adapter_sha256"] == RAW_ADAPTER_SHA and
         receipt["qwen_base_revision"] == MODEL_COMMIT and
         receipt["gpu_protocol_not_bit_identical_to_llama_cpp_gguf"] is True and
         receipt["actual_book_accuracy"] is None and
         receipt["expert_gold"] is None and
         receipt["no_trading"] is True and
         receipt["source_train_overlap_independently_audited"] is False,
         "INVALID_GPU_EXECUTION_PROOF")
    rows=receipt["observed"]
    need(set(rows) == {"matched_base","candidate_v5"} and
         all(len(rows[k])==100 for k in rows), "INCOMPLETE_GPU_MODEL_OUTPUTS")
    cases = {r["id"]:r for r in plan["cases"]}
    need(len(cases)==100 and len({x["symbol"] for x in cases.values()})==25,
         "INVALID_FIXED_SELECTION_PLAN")
    for role in ("matched_base", "candidate_v5"):
        need(len(set(x["id"] for x in rows[role]))==100 and
             set(x["id"] for x in rows[role])==set(cases), "INVALID_MODEL_IDS")
        for x in rows[role]:
            c=cases[x["id"]]
            need(x["symbol"]==c["symbol"] and
                 x["edition"]==c["edition"] and
                 x["cutoff_utc"]==c["cutoff_utc"] and
                 isinstance(x["chart_sha256"],str) and len(x["chart_sha256"])==64 and
                 isinstance(x["prompt_sha256"],str) and len(x["prompt_sha256"])==64 and
                 x["finish_reason"] in ("stop","error","length") and
                 x["decision"] is not None, "INVALID_ROW")
    left={x["id"]:x for x in rows["matched_base"]}
    right={x["id"]:x for x in rows["candidate_v5"]}
    for case in cases:
        a,b=left[case],right[case]
        need(all(a[k]==b[k] for k in ("id","symbol","edition","family",
                                     "chart_sha256","prompt_sha256",
                                     "retrieved_rule_ids")), "NOT_MATCHED_PAIRS")
    return {
      "validated_cases": 100, "paired_real_outputs": 200,
      "base_decisions": dict(Counter(x["decision"] for x in rows["matched_base"])),
      "trained_v5_decisions": dict(Counter(x["decision"] for x in rows["candidate_v5"])),
      "decision_disagreements": sum(left[k]["decision"] != right[k]["decision"] for k in cases),
      "reasons_240_base": sum(len(x.get("model_reason") or "") == 240 for x in rows["matched_base"]),
      "reasons_240_v5": sum(len(x.get("model_reason") or "") == 240 for x in rows["candidate_v5"]),
      "accuracy": None, "expert_labels": None, "live_trading_allowed": False,
    }


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    p=argparse.ArgumentParser()
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--max-polls",type=int,default=320)
    p.add_argument("--interval",type=int,default=45)
    a=p.parse_args()
    need(not a.out.exists(), "OUTPUT_EXISTS")
    need(os.environ.get("KAGGLE_API_TOKEN","") and
         len(os.environ["KAGGLE_API_TOKEN"])>=20,
         "PRIVATE_KAGGLE_TOKEN_NOT_AVAILABLE")
    need(1 <= a.max_polls <= 340 and 15 <= a.interval <= 60, "MONITOR_BOUNDS")
    api=KaggleApi(); api.authenticate()
    owner=get_username()
    kernel=owner+"/"+KERNEL_NAME
    terminal=None
    for i in range(a.max_polls):
        state=str(api.kernels_status(kernel).status).upper()
        if i%20==0:
            print("HIST100_PRIVATE_GPU_STATUS",state,"POLL",i,flush=True)
        if "COMPLETE" in state or "ERROR" in state or "FAIL" in state or "CANCEL" in state:
            terminal=state
            break
        need("RUNNING" in state or "QUEUED" in state or "PENDING" in state,
             "UNKNOWN_KAGGLE_EXECUTION_STATE_"+state)
        time.sleep(a.interval)
    need(terminal is not None, "KAGGLE_GPU_STILL_QUEUED_OR_RUNNING_MONITOR_WINDOW_ENDED")
    need("COMPLETE" in terminal, "KAGGLE_GPU_TERMINATED_WITH_"+terminal)
    with tempfile.TemporaryDirectory(prefix="hist100-private-output-") as tmp:
        result=subprocess.run(["kaggle","kernels","output",kernel,"-p",tmp,"-o"],
                              capture_output=True,text=True,timeout=600)
        need(result.returncode == 0, "KAGGLE_OUTPUT_DOWNLOAD_FAILED")
        found=list(Path(tmp).rglob("ACTUAL_GPU_HIST100_QWEN_BASE_V5_NO_GOLD.json"))
        need(len(found)==1, "NO_UNIQUE_REAL_200_MODEL_RESULT")
        content=json.loads(found[0].read_bytes())
    plan=json.loads(a.plan.read_bytes())
    summary=validate(content,plan)
    a.out.write_bytes(canonical(content))
    print("REAL_HIST100_KAGGLE_GPU_100_BASE_PLUS_100_TRAINED_V5_VERIFIED",
          json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
