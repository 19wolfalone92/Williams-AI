"""One-shot status check for already-submitted private T4 v7 pilot.

Never submits another kernel, never starts training, never starts inference.
If kernel is still QUEUED/RUNNING, write honest status and exit successfully.
"""
from __future__ import annotations
import json,os,subprocess,tempfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglehub.auth import get_username
from scripts.ai_v7_guided_json_8_h1_submit import SLUG
from scripts.ai_v7_guided_json_8_h1_monitor import validate

def main():
    if len(os.environ.get("KAGGLE_API_TOKEN",""))<20:
        raise RuntimeError("KAGGLE_SECRET_MISSING")
    api=KaggleApi();api.authenticate()
    kernel=get_username()+"/"+SLUG
    status=str(api.kernels_status(kernel).status).upper()
    result={"schema":"V7_SINGLE_CHECK_PRIVATE_T4_STATUS_V1",
        "kernel":kernel,"status":status,"new_kernel_submitted":False,
        "new_inference_started":False,"weights_trained":False,
        "results_present":False,"book_accuracy":None,"trading_allowed":False}
    root=Path(os.environ["RUNNER_TEMP"])
    if "COMPLETE" not in status:
        (root/"v7_one_shot_kernel_status.json").write_text(
            json.dumps(result,ensure_ascii=False,indent=2))
        print("V7_EXISTING_KERNEL_STILL_NOT_COMPLETE",status,flush=True)
        return
    with tempfile.TemporaryDirectory(prefix="v7-status-only-") as tmp:
        p=subprocess.run(["kaggle","kernels","output",kernel,"-p",tmp,"-o",
            "--file-pattern",r".*ACTUAL_V7_SCHEMA_GUIDED_EIGHT_PAIRED_MODEL_ANSWERS\.json$",
            "--page-size","200"],capture_output=True,text=True,timeout=480)
        if p.returncode:
            result["output_error"]=p.stderr[-3000:]
        files=list(Path(tmp).rglob("ACTUAL_V7_SCHEMA_GUIDED_EIGHT_PAIRED_MODEL_ANSWERS.json"))
        if len(files)==1:
            v7=json.loads(files[0].read_bytes())
            baseline=Path("baseline_v6/WILLIAMS_V6_HIST100_ALL_100_BASE_100_V5_REAL_GPU_NOT_GOLD.json")
            v6=json.loads(baseline.read_bytes())
            comparison=validate(v7,v6)
            (root/"v7_GUIDED_REAL_8_MODEL_ANSWERS.json").write_text(
                json.dumps(v7,ensure_ascii=False,indent=2))
            (root/"v7_VS_V6_IDENTICAL_EIGHT_PROMPTS_NO_GOLD.json").write_text(
                json.dumps(comparison,ensure_ascii=False,indent=2))
            result["results_present"]=True
            result["comparison_schema"]=comparison["schema"]
            result["summary"]=comparison["format_comparison_not_book_quality"]
        else:
            result["output_download_found"]=len(files)
    (root/"v7_one_shot_kernel_status.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2))
    print("V7_EXISTING_KERNEL_STATUS_AND_RESULT_FETCH",json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
