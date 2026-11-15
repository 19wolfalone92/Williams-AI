"""Read-only one-shot status/result fetch for the already submitted v7.1 private T4 kernel.

Never submits a kernel, never retrains weights, never runs new inference.
Third read-only status poll after completion of the 29-lesson curriculum audit. This job may only inspect/download the existing kernel; no execution trigger, resubmission or new inference.
"""
from __future__ import annotations
from collections import Counter
import json,os,re,subprocess,tempfile
from pathlib import Path
from scripts.ai_v71_guided_json_8_h1_submit import SLUG
from williams_ai.v71_guided_json_diagnostics import ENFORCER_COMMIT
from williams_ai.v6_v5_lora_binding import MODEL_COMMIT,RAW_ADAPTER_SHA

EXPECTED=tuple(f"HIST100-{n:02d}-{k}" for n in (1,2) for k in (1,2,3,4))
MODEL_ROLES=("matched_base","candidate_v5")

def need(ok,why):
 if not ok:raise ValueError("V71_STATUS_"+why)

def _summary(rows):
 return {
  "rows":len(rows),
  "finish_reason":dict(Counter(x["finish_reason"] for x in rows)),
  "valid_decisions":sum(x["valid_model_decision"] for x in rows),
  "technical_failures":dict(Counter(x["failure_code"] for x in rows if x.get("failure_code"))),
  "decisions":dict(Counter(x["decision"] for x in rows if x.get("valid_model_decision"))),
  "missing_eos":sum(not x.get("diagnostic",{}).get("eos_seen",False) for x in rows),
  "raw_generations_with_nonempty_text":sum(bool(x.get("diagnostic",{}).get("raw_without_special_tokens","")) for x in rows),
  "mean_output_tokens":round(sum(x["tokens_out"] for x in rows)/len(rows),2) if rows else None,
 }

def compare(v71,v6):
 need(v71.get("schema")=="WILLIAMS_V71_GUIDED_JSON_TOKEN_DIAGNOSTICS_NO_GOLD_V1"
      and v71.get("complete") is True and v71.get("paired_cases")==8,
      "V71_NOT_COMPLETE_8_PAIRS")
 need(v71.get("enforcer_commit")==ENFORCER_COMMIT and
      v71.get("enforcer_decode_patch_present") is True and
      v71.get("base_revision")==MODEL_COMMIT and
      v71.get("v5_adapter_sha256")==RAW_ADAPTER_SHA and
      v71.get("expert_gold") is None and v71.get("training_started") is False and
      v71.get("weights_changed") is False and v71.get("trading_allowed") is False,
      "V71_SOURCE_OR_SAFETY_IDENTITY")
 need(v6.get("complete") is True and v6.get("source_selection_sha256")==v71["selection_sha256"]
      and v6.get("true_trained_adapter_sha256")==RAW_ADAPTER_SHA,
      "V6_BASELINE_NOT_SAME_ORIGINAL_CORPUS")
 summary={}
 for role in MODEL_ROLES:
  rows=v71["observed"].get(role)
  old={x["id"]:x for x in v6["observed"][role]}
  need(isinstance(rows,list) and [x["id"] for x in rows]==list(EXPECTED),"V71_CASE_SET")
  for x in rows:
   prev=old.get(x["id"])
   need(prev is not None and
        x["chart_sha256"]==prev["chart_sha256"] and
        x["original_cpu_full_prompt_sha256"]==prev["original_cpu_full_prompt_sha256"] and
        x["prompt_sha256"]==prev["prompt_sha256"] and
        x["gpu_visible_bars"]==prev["gpu_visible_closed_h1_bars"] and
        x["tokens_in"]==prev["gpu_prompt_tokens"],
        "ORIGINAL_V6_PROMPT_OR_CHART_NOT_IDENTICAL")
   need(x.get("trade_action") is None and x.get("book_gold") is None and
        x.get("weights_changed") is False,"UNSAFE_RESULT_FIELDS")
   if x["finish_reason"]!="stop":
    need(x["decision"] is None and x["valid_model_decision"] is False,
         "TECHNICAL_FAILURE_LAUNDERED")
   else:
    need(x["valid_model_decision"] is True and
         x["decision"] in {"LONG_STOP_ENTRY","SHORT_STOP_ENTRY","NO_SETUP",
          "INSUFFICIENT_DATA","EXIT_LONG","EXIT_SHORT","HOLD",
          "WAVE_CONFIRMED","WAVE_HYPOTHESIS","WAVE_INVALID","ABSTAIN"},
         "INVALID_V71_MODEL_DECISION")
  v6rows=[old[c] for c in EXPECTED]
  summary[role]={
   "v6_finish":dict(Counter(x["finish_reason"] for x in v6rows)),
   "v6_valid_stop":sum(x["finish_reason"]=="stop" for x in v6rows),
   "v71":_summary(rows),
   "same_valid_decision_on_both":sum(
      x["finish_reason"]=="stop" and old[x["id"]]["finish_reason"]=="stop"
      and x["decision"]==old[x["id"]]["decision"] for x in rows),
   "decision_changes_among_both_valid":sum(
      x["finish_reason"]=="stop" and old[x["id"]]["finish_reason"]=="stop"
      and x["decision"]!=old[x["id"]]["decision"] for x in rows),
  }
 return {"schema":"WILLIAMS_V71_PATCHED_LMFE_SAME_8_INPUT_COMPARISON_NO_GOLD_V1",
   "completed":True,"same_original_8_charts_and_gpu_prompts":True,
   "upstream_lmfe_fix_sha":ENFORCER_COMMIT,
   "genuine_existing_v5_lora_used":True,
   "book_accuracy":None,"expert_gold":None,"financial_profitability":None,
   "training_performed":False,"weights_changed":False,"trading_allowed":False,
   "format_reliability_comparison_not_book_correctness":summary}

def main():
 from kaggle.api.kaggle_api_extended import KaggleApi
 from kagglehub.auth import get_username
 need(len(os.environ.get("KAGGLE_API_TOKEN",""))>=20,"KAGGLE_CREDENTIAL_MISSING")
 api=KaggleApi();api.authenticate()
 kernel=get_username()+"/"+SLUG
 status=str(api.kernels_status(kernel).status).upper()
 result={"schema":"WILLIAMS_V71_EXISTING_PRIVATE_T4_STATUS_V1",
  "kernel":kernel,"status":status,"new_kernel_submitted":False,
  "new_inference_started":False,"new_training_started":False,
  "book_accuracy":None,"trading_allowed":False}
 out=Path(os.environ["RUNNER_TEMP"])
 if "COMPLETE" not in status:
  (out/"v71_existing_kernel_status.json").write_text(json.dumps(result,indent=2))
  print("V71_EXISTING_KERNEL_NOT_COMPLETE",status,flush=True)
  return
 with tempfile.TemporaryDirectory(prefix="v71-status-fetch-") as tmp:
  p=subprocess.run(["kaggle","kernels","output",kernel,"-p",tmp,"-o",
   "--file-pattern",r".*ACTUAL_V71_PATCHED_LMFE_8_PAIRED_DIAGNOSTICS_NO_GOLD\.json$",
   "--page-size","200"],capture_output=True,text=True,timeout=480)
  files=list(Path(tmp).rglob("ACTUAL_V71_PATCHED_LMFE_8_PAIRED_DIAGNOSTICS_NO_GOLD.json"))
  if p.returncode!=0 or len(files)!=1:
   result["output_download_ok"]=False
   result["output_error"]=(p.stderr or p.stdout)[-3000:]
  else:
   v71=json.loads(files[0].read_bytes())
   baseline=Path("baseline_v6/WILLIAMS_V6_HIST100_ALL_100_BASE_100_V5_REAL_GPU_NOT_GOLD.json")
   v6=json.loads(baseline.read_bytes())
   comparison=compare(v71,v6)
   (out/"v71_patched_lmfe_8_actual_answers_NO_GOLD.json").write_text(
       json.dumps(v71,ensure_ascii=False,indent=2))
   (out/"v71_same_8_inputs_vs_v6_NO_GOLD.json").write_text(
       json.dumps(comparison,ensure_ascii=False,indent=2))
   result["output_download_ok"]=True
   result["comparison"]=comparison
 (out/"v71_existing_kernel_status.json").write_text(
   json.dumps(result,ensure_ascii=False,indent=2))
 print("V71_PRIVATE_T4_EXISTING_KERNEL_AUDIT",json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
