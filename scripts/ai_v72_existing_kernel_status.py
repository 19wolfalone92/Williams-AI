"""READ-ONLY check of the one already submitted v7.2 T4 kernel; NO submit/train.

Private output and logs never pass through public text logs.  The final
artifact includes model answers only if the exact matched source identities
and 8×2 completion contract pass.
"""
from __future__ import annotations
from collections import Counter
import json,os,re,subprocess,tempfile
from pathlib import Path
from scripts.ai_v71_existing_kernel_status import (
    EXPECTED,MODEL_ROLES,compare as compare_v71,
    _safe_kernel_failure_diagnostic,_read_existing_kernel_error
)
from scripts.ai_v72_guided_json_8_h1_submit import SLUG
from williams_ai.v72_eos_guard import SCHEMA,ENFORCER_COMMIT
from williams_ai.v6_v5_lora_binding import MODEL_COMMIT,RAW_ADAPTER_SHA

def need(ok,why):
    if not ok:
        raise ValueError("V72_STATUS_"+why)

def _safe_output_error(p,file_count):
    return {"cli_exit_ok":p.returncode==0,
            "matching_file_count":file_count,
            "allowlisted_diagnostic":_safe_kernel_failure_diagnostic(
                ((p.stderr or p.stdout) or "")[-100000:])}

def _summary(rows):
    return {
        "rows":len(rows),
        "finish":dict(Counter(x["finish_reason"] for x in rows)),
        "valid_json_responses":sum(x.get("valid_model_decision") is True for x in rows),
        "technical_failure_codes":dict(Counter(x.get("failure_code")
            for x in rows if x.get("failure_code"))),
        "valid_model_decisions":dict(Counter(x["decision"]
            for x in rows if x.get("valid_model_decision"))),
        "eos_block_events":sum(x["metrics"]["eos_blocked"] for x in rows),
        "tokens_generated":sum(x["tokens_out"] for x in rows)
    }

def compare_v72(v72,v71,v6):
    # Explicitly verify the ORIGINAL v7.1 against v6 before comparing v7.2.
    old_summary=compare_v71(v71,v6)
    need(type(v72) is dict and v72.get("schema")==SCHEMA
         and v72.get("complete") is True
         and v72.get("paired_cases")==8
         and v72.get("eos_early_termination_guard") is True,
         "NOT_EXACT_8_COMPLETE_EOS_GUARDED_PAIRS")
    need(v72.get("base_revision")==MODEL_COMMIT
         and v72.get("v5_adapter_sha256")==RAW_ADAPTER_SHA
         and v72.get("enforcer_commit")==ENFORCER_COMMIT
         and v72.get("enforcer_decode_patch_present") is True
         and v72.get("selection_sha256")==v6["source_selection_sha256"]
         and v72.get("original_v6_protocol_sha256")==v71["original_v6_protocol_sha256"]
         and v72.get("training_started") is False
         and v72.get("weights_changed") is False
         and v72.get("trading_allowed") is False
         and v72.get("expert_gold") is None,
         "MODEL_SOURCE_OR_SAFETY_IDENTITY")
    summary={}
    for role in MODEL_ROLES:
        rows=v72["observed"].get(role)
        prev={x["id"]:x for x in v71["observed"][role]}
        v6old={x["id"]:x for x in v6["observed"][role]}
        need(type(rows) is list and [x["id"] for x in rows]==list(EXPECTED),
             "EIGHT_FROZEN_CASE_IDS")
        for row in rows:
            pid=row["id"];source=prev[pid];legacy=v6old[pid]
            for key in ("id","chart_sha256","prompt_sha256",
                        "original_cpu_full_prompt_sha256","cutoff_utc",
                        "edition","family","gpu_visible_bars","tokens_in"):
                need(row.get(key)==source.get(key),
                     "MISMATCHED_V71_INPUT_"+key.upper())
            need(row["chart_sha256"]==legacy["chart_sha256"]
                 and row["prompt_sha256"]==legacy["prompt_sha256"]
                 and row.get("eos_guard_activated") is True,
                 "MISMATCHED_V6_SOURCE_OR_GUARD")
            need(row.get("weights_changed") is False
                 and row.get("trade_action") is None
                 and row.get("book_gold") is None,
                 "UNSAFE_OUTPUT")
            need(row.get("finish_reason") in {"stop","length","error"},
                 "INVALID_FINISH")
            need(type(row.get("metrics")) is dict
                 and type(row["metrics"].get("eos_blocked")) is int
                 and row["metrics"]["eos_blocked"]>=0,
                 "NO_EOS_GUARD_METRICS")
            if row["finish_reason"]=="stop":
                need(row.get("valid_model_decision") is True
                     and type(row.get("decision")) is str
                     and row["decision"] in {"LONG_STOP_ENTRY","SHORT_STOP_ENTRY",
                         "NO_SETUP","INSUFFICIENT_DATA","EXIT_LONG","EXIT_SHORT",
                         "HOLD","WAVE_CONFIRMED","WAVE_HYPOTHESIS","WAVE_INVALID","ABSTAIN"},
                     "PREMATURE_OR_INVALID_STOP")
            else:
                need(row.get("decision") is None and
                     row.get("valid_model_decision") is False,
                     "TECHNICAL_FAILURE_AS_SIGNAL")
        summary[role]={
          "v6":old_summary["format_reliability_comparison_not_book_correctness"][role]["v6_finish"],
          "v71":old_summary["format_reliability_comparison_not_book_correctness"][role]["v71"],
          "v72":_summary(rows),
          "v71_to_v72_technical_stop_gain":sum(r["finish_reason"]=="stop" for r in rows)-
                 sum(r["finish_reason"]=="stop" for r in prev.values())
        }
    return {
       "schema":"WILLIAMS_V72_SOURCE_MATCHED_8X2_TECHNICAL_COMPARISON_NOT_GOLD_V1",
       "complete":True,"original_8_h1_sha_and_prompts_identical":True,
       "real_v5_adapter_matched":True,"book_accuracy":None,"gold_labels":0,
       "weights_changed":False,"training_performed":False,
       "trading_allowed":False,
       "comparison_of_technical_json_responses_only":summary}

def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    need(len(os.environ.get("KAGGLE_API_TOKEN",""))>=20,"NO_KAGGLE_TOKEN")
    api=KaggleApi();api.authenticate()
    kernel=get_username()+"/"+SLUG
    try:
        status=str(api.kernels_status(kernel).status).upper()
    except Exception as exc:
        status="STATUS_API_UNAVAILABLE"
        error_type=type(exc).__name__
    else:
        error_type=None
    out=Path(os.environ["RUNNER_TEMP"])
    result={"schema":"WILLIAMS_V72_EXISTING_T4_READ_ONLY_STATUS_V1",
       "kernel":kernel,"status":status,"book_accuracy":None,
       "new_kernel_submitted":False,"new_training":False,"trading_allowed":False}
    if error_type:
        result["status_error_type"]=error_type
    if "COMPLETE" not in status:
        if "ERROR" in status:
            result["safe_failure_diagnostic"]=_read_existing_kernel_error(kernel)
        (out/"v72_existing_kernel_status.json").write_text(
            json.dumps(result,ensure_ascii=False,indent=2))
        print("V72_EXISTING_KERNEL_PENDING_OR_ERROR",json.dumps(result,ensure_ascii=False))
        return
    with tempfile.TemporaryDirectory(prefix="v72-existing-status-") as tmp:
        p=subprocess.run(["kaggle","kernels","output",kernel,"-p",tmp,"-o",
            "--file-pattern",r".*ACTUAL_V72_EOSGUARD_8_PAIRED_DIAGNOSTICS_NO_GOLD\\.json$",
            "--page-size","200"],capture_output=True,text=True,timeout=480)
        # Some SDK versions implement --file-pattern differently; only
        # trust the EXACT output filename, never infer from a wildcard.
        files=list(Path(tmp).rglob("ACTUAL_V72_EOSGUARD_8_PAIRED_DIAGNOSTICS_NO_GOLD.json"))
        if p.returncode!=0 or len(files)!=1:
            result["output_download_ok"]=False
            result["output_safe_diagnostic"]=_safe_output_error(p,len(files))
        else:
            try:
                new=json.loads(files[0].read_bytes())
                old=json.loads(Path("baseline_v71/v71_patched_lmfe_8_actual_answers_NO_GOLD.json").read_bytes())
                base=json.loads(Path("baseline_v6/WILLIAMS_V6_HIST100_ALL_100_BASE_100_V5_REAL_GPU_NOT_GOLD.json").read_bytes())
                cmp=compare_v72(new,old,base)
            except (ValueError,KeyError,TypeError,OSError) as exc:
                result["output_download_ok"]=True
                result["comparison_verified"]=False
                reason=str(exc)
                result["safe_failure_code"]=(reason if re.fullmatch(r"V72_STATUS_[A-Z0-9_]+",reason)
                     else "RESULT_CONTRACT_INVALID")
            else:
                (out/"v72_real_paired_8x2_NO_GOLD.json").write_text(
                    json.dumps(new,ensure_ascii=False,indent=2))
                (out/"v72_source_matched_comparison_NO_GOLD.json").write_text(
                    json.dumps(cmp,ensure_ascii=False,indent=2))
                result["output_download_ok"]=True
                result["comparison_verified"]=True
                result["comparison"]=cmp
    (out/"v72_existing_kernel_status.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2))
    print("V72_EXISTING_KERNEL_ONLY_VERIFICATION",json.dumps(result,ensure_ascii=False))

if __name__=="__main__":
    main()
