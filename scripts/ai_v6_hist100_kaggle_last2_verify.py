"""Fetch two genuine GPU inference rows and combine against saved 98-case receipt.

All 200 answers may be reported only when full case SHA coverage is certified.
Two exceptional PEPE cases have a documented 4096-token fallback policy.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import json,os,re,subprocess,tempfile,time
from pathlib import Path
from scripts.ai_v6_hist100_kaggle_gpu_submit import TRAINING_NAME
from scripts.ai_v6_hist100_kaggle_last2_submit import NEW_SLUG
from scripts.ai_v6_hist100_kaggle_last2_worker import CASE_IDS,SCHEMA
from scripts.ai_v6_hist100_kaggle_gpu_worker import SCHEMA as ORIGINAL_SCHEMA
from williams_ai.independent_exam_v6 import canonical
from williams_ai.v6_v5_lora_binding import MODEL_COMMIT,RAW_ADAPTER_SHA

ROLES=("matched_base","candidate_v5")
def need(ok,why):
    if not ok:raise ValueError("HIST100_FINALEXAM_"+why)

def combine(old,last,plan):
    need(old["schema"]==ORIGINAL_SCHEMA and old["complete"] is False and
         old["completed_paired_cases"]==98 and
         old["real_model_inference_rows"]==196 and
         last["schema"]==SCHEMA and last["complete"] is True and
         last["real_paired_cases"]==2 and
         old["original_cpu_prompt_protocol_sha256"]==last["original_cpu_full_protocol_sha256"] and
         old["market_selection_plan_sha256"]==last["original_100_case_selection_sha256"] and
         old["verified_genuine_trained_adapter_sha256"]==last["original_trained_v5_adapter_sha256"]==RAW_ADAPTER_SHA and
         old["qwen_base_revision"]==last["original_base_revision"]==MODEL_COMMIT and
         old["source_zip_sha256_verified"] is True and
         old["no_trading"] is True and last["no_trading"] is True and
         old["expert_gold"] is None and old["actual_book_accuracy"] is None and
         last["no_gold"] is True and last["accuracy"] is None,
         "INVALID_ORIGIN_OR_UNMATCHED_MODELS")
    expected={c["id"]:c for c in plan["cases"]}
    need(len(expected)==100 and len(set(expected))==100 and
         [c["id"] for c in plan["cases"]][-2:]==list(CASE_IDS),
         "FROZEN_100_CASE_PLAN_CHANGED")
    combined={}
    for role in ROLES:
        oldrows=old["observed"][role]
        newrows=last["observed"][role]
        need(len(oldrows)==98 and len(newrows)==2 and
             [x["id"] for x in newrows]==list(CASE_IDS) and
             set(x["id"] for x in oldrows)==set(expected)-set(CASE_IDS),
             "MISSING_OR_DUPLICATE_GPU_PREDICTIONS")
        allrows=oldrows+newrows
        need(len(allrows)==100 and len(set(x["id"] for x in allrows))==100,
             "NOT_EXACTLY_100_UNIQUE_ROWS")
        for case in allrows:
            orig=expected[case["id"]]
            need(case["symbol"]==orig["symbol"] and
                 case["edition"]==orig["edition"] and
                 case["cutoff_utc"]==orig["cutoff_utc"] and
                 isinstance(case["chart_sha256"],str) and len(case["chart_sha256"])==64 and
                 isinstance(case["prompt_sha256"],str) and len(case["prompt_sha256"])==64 and
                 case["finish_reason"] in ("stop","length","error"),
                 "INVALID_CHART_OR_MODEL_OUTPUT")
        combined[role]=allrows
    for a,b in zip(combined["matched_base"],combined["candidate_v5"],strict=True):
        need(all(a[x]==b[x] for x in ("id","edition","family","symbol","cutoff_utc",
                  "chart_sha256","prompt_sha256","retrieved_rule_ids")),
             "TWO_MODEL_INPUTS_NOT_IDENTICAL")
    sums={}
    for role in ROLES:
        arr=combined[role]
        sums[role]={
            "decisions":dict(Counter(x["decision"] for x in arr)),
            "finish_reasons":dict(Counter(x["finish_reason"] for x in arr)),
            "decisions_by_book":{
                book:dict(Counter(x["decision"] for x in arr if x["edition"]==book))
                for book in ("TC1","NTD","TC2","COURSE")},
            "no_expert_labels":True,
            "explanations_present":sum(bool(x.get("model_reason")) for x in arr),
        }
    return {"schema":"WILLIAMS_V6_HIST100_ALL_200_REAL_HF_GPU_MODEL_ANSWERS_RESEARCH_NO_GOLD_V1",
        "complete":True,"historical_charts":100,
        "real_base_answers":100,"real_trained_v5_answers":100,
        "verified_same_prompts_per_paired_model":True,
        "gpu_primary_protocol_98_case_max_tokens":3200,
        "gpu_last_two_fallback_max_tokens":4096,
        "fallback_case_ids":list(CASE_IDS),
        "not_homogeneous_input_length_policy":True,
        "backend":"Qwen3-4B HF/NF4 GPU not same as GGUF CPU",
        "source_selection_sha256":old["market_selection_plan_sha256"],
        "true_trained_adapter_sha256":RAW_ADAPTER_SHA,
        "gold_labels":None,"accuracy":None,"profitability":None,
        "no_lookahead_asserted_by_100_chart_source_validation":True,
        "no_live_trading":True,
        "decision_disagreement_count_NOT_ACCURACY":sum(a["decision"]!=b["decision"]
            for a,b in zip(combined["matched_base"],combined["candidate_v5"],strict=True)),
        "diagnostics":sums,"observed":combined}

def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    token=os.environ.get("KAGGLE_API_TOKEN","")
    need(len(token)>=20,"NO_PRIVATE_TOKEN")
    api=KaggleApi();api.authenticate()
    owner=get_username()
    kernel=owner+"/"+NEW_SLUG
    state=None
    for n in range(180):
        state=str(api.kernels_status(kernel).status).upper()
        if n%12==0:print("HIST100_KAGGLE_FINAL2_GPU_STATUS",state,n,flush=True)
        if any(x in state for x in ("ERROR","FAIL","CANCEL","COMPLETE")):break
        need(any(x in state for x in ("RUNNING","QUEUED","PENDING")), "UNKNOWN_KERNEL_STATE")
        time.sleep(25)
    need("COMPLETE" in state,"NOT_COMPLETED_GPU_LAST2_"+str(state))
    with tempfile.TemporaryDirectory(prefix="hist100-gpu-finish-two-") as tmp:
        p=subprocess.run(["kaggle","kernels","output",kernel,
            "--file-pattern",r".*ACTUAL_HIST100_GPU_LAST_TWO_QWEN_BASE_V5_NO_GOLD\.json$",
            "--page-size","200","-p",tmp,"-o"],capture_output=True,text=True,timeout=540)
        need(p.returncode==0,"KAGGLE_OUTPUT_DOWNLOAD_FAILED")
        files=list(Path(tmp).rglob("ACTUAL_HIST100_GPU_LAST_TWO_QWEN_BASE_V5_NO_GOLD.json"))
        need(len(files)==1,"NO_UNIQUE_TWO_CASE_GPU_RECEIPT")
        last=json.loads(files[0].read_bytes())
    previous=Path("historical98/hist100_gpu_verified_PARTIAL_not_complete.json")
    need(previous.is_file(),"MISSING_VERIFIED_ORIGINAL_98_PARTIAL")
    old=json.loads(previous.read_bytes())
    plan=json.loads(Path("research/williams_v6_historical100_selection_20261114.json").read_bytes())
    obj=combine(old,last,plan)
    out=Path(os.environ["RUNNER_TEMP"])/"WILLIAMS_V6_HIST100_ALL_100_BASE_100_V5_REAL_GPU_NOT_GOLD.json"
    with out.open("x",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,sort_keys=True)
    print("HIST100_RESEARCH_ALL_100_BASE_100_V5_GPU_MODEL_RESULTS_VERIFIED",
           json.dumps({"counts":obj["diagnostics"],
           "disagreements":obj["decision_disagreement_count_NOT_ACCURACY"]},
           ensure_ascii=False),flush=True)
if __name__=="__main__":
    main()
