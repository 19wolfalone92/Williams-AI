"""Read/verify the existing pinned GPU run; never submit another kernel."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ai_run_joint_rule_kaggle import groups_and_design, ARCHIVE
from scripts.ai_verify_joint_rules import verify_archive
from williams_ai.joint_rule_curriculum import RUN_SLUG

PINNED_WORKER = "26b5cabe8c11dde70c8f1c9792223d457da5ad1a"


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    dest=Path(os.environ["RUNNER_TEMP"])/"joint_rule_recovered_results"
    dest.mkdir(mode=0o700)
    groups,design=groups_and_design()
    api=KaggleApi();api.authenticate();owner=get_username()
    state="UNKNOWN"
    # Bounded CPU observation. It has no submission or paid-fallback path.
    for i in range(730):
        state=str(api.kernels_status(owner+"/"+RUN_SLUG).status).upper()
        if i%15==0:print("EXISTING_JOINT_GPU_STATE",state,flush=True)
        if any(s in state for s in ("COMPLETE","ERROR","CANCEL")):break
        time.sleep(20)
    result=subprocess.run(["kaggle","kernels","output",owner+"/"+RUN_SLUG,"-p",str(dest),
        "--file-pattern",r".*(JOINT_RULES_20261106[.]zip|joint_rule_failure[.]json)$"],
        capture_output=True,text=True,timeout=420)
    archive=dest/ARCHIVE
    if result.returncode or not archive.is_file() or "COMPLETE" not in state:
        raise RuntimeError("EXISTING_JOINT_GPU_INCOMPLETE_NO_TRAINING_SUCCESS_CLAIM")
    proof=verify_archive(archive,dest,revision=PINNED_WORKER,design=design,groups=groups)
    print("INDEPENDENTLY_VERIFIED_EXISTING_JOINT_GPU",json.dumps({"source_commit":proof["source_commit"],
        "actual_updates":proof["actual_updates"],"selected_checkpoint":proof["selected_checkpoint"],
        "candidate_adapter_sha256":proof["candidate_adapter_sha256"],
        "correct":{g:{s:v["correct"] for s,v in scores.items()} for g,scores in proof["grades"].items()},
        "pilot_research_gates_passed":proof["pilot_research_gates_passed"],"execution_allowed":False}),flush=True)


if __name__=="__main__":main()
