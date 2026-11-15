"""Read-only verify real v7 8-case paired data and compare against v6 on same cases."""
from __future__ import annotations
from collections import Counter
import json,os,subprocess,tempfile,time
from pathlib import Path
from scripts.ai_v7_guided_json_8_h1_submit import SLUG
from scripts.ai_v7_guided_json_8_h1_worker import IDS,SCHEMA
from williams_ai.v6_v5_lora_binding import MODEL_COMMIT,RAW_ADAPTER_SHA

def need(ok,reason):
    if not ok:raise ValueError("V7_GUIDED_VALIDATION_"+reason)

def validate(v7,v6):
    need(v7.get("schema")==SCHEMA and v7.get("complete") is True and
         v7.get("paired_cases")==8 and
         v7.get("model_base_revision")==MODEL_COMMIT and
         v7.get("raw_v5_adapter_sha256")==RAW_ADAPTER_SHA and
         v7.get("research_only") is True and v7.get("expert_gold") is None and
         v7.get("trading_allowed") is False and
         v7.get("training_started") is False and
         v7.get("weights_unchanged") is True and
         v7.get("gpu_prompt_limit")==3200,
         "INCOMPLETE_OR_UNPINNED_V7_8CASE_PILOT")
    need(v6.get("complete") is True and
         v6.get("source_selection_sha256")==v7["original_market_100_selection_sha256"] and
         v6["true_trained_adapter_sha256"]==RAW_ADAPTER_SHA,
         "NOT_ORIGINAL_GENUINE_V6_GPU")
    stats={}
    for role in ("matched_base","candidate_v5"):
        a=v7["observed"][role];b={x["id"]:x for x in v6["observed"][role]}
        need(len(a)==8 and [x["id"] for x in a]==list(IDS),
             "NOT_THE_FROZEN_EIGHT")
        for x in a:
            prev=b[x["id"]]
            need(x["chart_sha256"]==prev["chart_sha256"] and
                 x["original_cpu_full_prompt_sha256"]==prev["original_cpu_full_prompt_sha256"] and
                 x["prompt_sha256"]==prev["prompt_sha256"] and
                 x["gpu_visible_bars"]==prev["gpu_visible_closed_h1_bars"] and
                 x["tokens_in"]==prev["gpu_prompt_tokens"] and
                 x["trade_action"] is None and x["book_gold"] is None and
                 x["weights_changed"] is False,
                 "INPUT_DIFFERENCE_OR_TRADING_LEAK")
            if x["finish_reason"]!="stop":
                need(x["decision"] is None and not x["valid_model_decision"],
                     "TECHNICAL_ERROR_LAUNDERED_AS_DECISION")
        stats[role]={
          "v6_stop":sum(b[x["id"]]["finish_reason"]=="stop" for x in a),
          "v7_stop":sum(x["finish_reason"]=="stop" for x in a),
          "v7_failure_codes":dict(Counter(x["failure_code"] for x in a if x["failure_code"])),
          "v7_decisions":dict(Counter(x["decision"] for x in a if x["decision"])),
          "v6_decisions":dict(Counter(b[x["id"]]["decision"] for x in a)),
        }
    return {"schema":"WILLIAMS_V7_REAL_GUIDED_JSON_8_FROZEN_SAME_INPUT_NO_GOLD_V1",
        "completed":True,"genuine_v5_lora_same_as_v6":True,
        "compared_on_same_eight_original_gpu_prompts":True,
        "model_accuracy":None,"expert_gold":None,"no_live_trading":True,
        "training_executed":False,
        "format_comparison_not_book_quality":stats}

def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    need(len(os.environ.get("KAGGLE_API_TOKEN",""))>=20,"KAGGLE_SECRET_MISSING")
    api=KaggleApi();api.authenticate()
    kernel=get_username()+"/"+SLUG
    status=None
    for i in range(235):
        status=str(api.kernels_status(kernel).status).upper()
        if i%15==0:print("V7_GUIDED_GPU_STATE",status,i,flush=True)
        if any(x in status for x in ("COMPLETE","ERROR","CANCEL","FAIL")): break
        need(any(x in status for x in ("RUNNING","QUEUED","PENDING")),
             "UNKNOWN_KERNEL_STATE")
        time.sleep(25)
    need("COMPLETE" in status,"V7_REAL_T4_NOT_COMPLETED_"+str(status))
    with tempfile.TemporaryDirectory(prefix="v7-real-kaggle-8-") as tmp:
        p=subprocess.run(["kaggle","kernels","output",kernel,"-p",tmp,"-o",
            "--file-pattern",r".*ACTUAL_V7_SCHEMA_GUIDED_EIGHT_PAIRED_MODEL_ANSWERS\.json$",
            "--page-size","200"],capture_output=True,text=True,timeout=480)
        need(p.returncode==0,"V7_OUTPUT_FETCH_FAILED")
        files=list(Path(tmp).rglob("ACTUAL_V7_SCHEMA_GUIDED_EIGHT_PAIRED_MODEL_ANSWERS.json"))
        need(len(files)==1,"MISSING_REAL_V7_EIGHT_RECEIPT")
        v7=json.loads(files[0].read_bytes())
    v6=json.loads(Path("baseline_v6/WILLIAMS_V6_HIST100_ALL_100_BASE_100_V5_REAL_GPU_NOT_GOLD.json").read_bytes())
    final=validate(v7,v6)
    root=Path(os.environ["RUNNER_TEMP"])
    (root/"v7_GUIDED_REAL_8_MODEL_ANSWERS.json").write_text(json.dumps(v7,ensure_ascii=False,indent=2))
    (root/"v7_VS_V6_IDENTICAL_EIGHT_PROMPTS_NO_GOLD.json").write_text(json.dumps(final,ensure_ascii=False,indent=2))
    print("V7_GUIDED_REAL_8_PAIRED_FORMAT_COMPARISON",json.dumps(final,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
