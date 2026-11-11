"""Recover an existing own GPU result after controller timeout; NO new fit."""
from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from williams_ai.deep_book_research import MODEL, REVISION


def main():
    from kagglehub.auth import get_username
    kernel=get_username()+"/williams-ai-private-originals-deep-20261027"
    final="UNKNOWN"
    for _ in range(120):
        status=subprocess.run(["kaggle","kernels","status",kernel],capture_output=True,text=True,timeout=35)
        text=status.stdout.lower()
        if status.returncode==0 and ("complete" in text or "finished" in text):final="COMPLETE";break
        if status.returncode==0 and any(w in text for w in ("error","failed","cancel")):final="ERROR";break
        time.sleep(20)
    dest=Path(os.environ["RUNNER_TEMP"])/"private_first_recovered_results"
    dest.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory() as td:
        result=subprocess.run(["kaggle","kernels","output",kernel,"-p",td,
            "--file-pattern",r".*(PRIVATE_DEEP_RESEARCH[.]zip|deep_research_failure[.]json)$"],
            capture_output=True,text=True,timeout=300)
        if result.returncode:raise RuntimeError("EXISTING_PRIVATE_GPU_RESULT_DOWNLOAD_FAILED")
        archive=next(Path(td).rglob("Williams_Qwen3_4B_PRIVATE_DEEP_RESEARCH.zip"),None)
        if archive is None:
            # Preserve coherent own safe-tensor checkpoints if the GPU time cap
            # interrupted an epoch. No optimizer pickle is downloaded or loaded.
            checkpoint_dir=dest/"own_partial_checkpoints"
            checkpoint_dir.mkdir(mode=0o700)
            subprocess.run(["kaggle","kernels","output",kernel,"-p",str(checkpoint_dir),
                "--file-pattern",r".*(adapter_model[.]safetensors|adapter_config[.]json|trainer_state[.]json|research_receipt[.]json|deep_research_failure[.]json)$"],
                capture_output=True,text=True,timeout=420)
            weights=list(checkpoint_dir.rglob("adapter_model.safetensors"))
            receipt={"final_kaggle_state":final,"complete_candidate_archive":False,
                "own_safe_tensor_checkpoints_preserved":len(weights),
                "training_success_claimed":False,"execution_allowed":False}
            (dest/"recovery_receipt.json").write_text(json.dumps(receipt,indent=2))
            print("PRIVATE_GPU_TIMEOUT_RECOVERY_PARTIAL_WEIGHTS",len(weights),flush=True)
            raise RuntimeError("GPU_RESULT_INCOMPLETE_OWN_CHECKPOINTS_PRESERVED")
        shutil.copy2(archive,dest/archive.name)
    with zipfile.ZipFile(dest/"Williams_Qwen3_4B_PRIVATE_DEEP_RESEARCH.zip") as z:
        if any(Path(n).is_absolute() or ".." in Path(n).parts for n in z.namelist()):raise ValueError("UNSAFE_OWN_RESULT_ARCHIVE")
        report=json.loads(z.read("research_receipt.json"))
        if report.get("model_repo")!=MODEL or report.get("model_revision")!=REVISION or report.get("source_commit")!="6099fb60f4f8f050e0e51f0253c970f20826b833" or report.get("execution_allowed") is not False:
            raise ValueError("OWN_PRIVATE_SOURCE_RESULT_PROVENANCE_FAILED")
        hashes={}
        for stage in ("source","candidate"):
            member=stage+"_adapter/adapter_model.safetensors"
            if member in z.namelist():
                digest=sha256(z.read(member)).hexdigest()
                if report.get(stage+"_adapter_sha256") and report[stage+"_adapter_sha256"]!=digest:
                    raise ValueError("OWN_SAVED_ADAPTER_HASH_FAILED")
                hashes[stage]=digest
        for name in ("research_receipt.json","market_precommit.json","practice_choice_exam.json"):
            if name in z.namelist():(dest/name).write_bytes(z.read(name))
    if (dest/"market_precommit.json").exists():
        sealed=Path(os.environ["RUNNER_TEMP"])/"recovered_first_market_seal"
        subprocess.run([sys.executable,"scripts/ai_seal_15pair_blind_charts.py","--new-output",str(sealed)],check=True)
        pre=json.loads((dest/"market_precommit.json").read_bytes())
        if pre["receipt"]["market_questions_file_sha256"]!=sha256((sealed/"model_questions.jsonl").read_bytes()).hexdigest():
            raise ValueError("RECOVERED_MARKET_COHORT_HASH_FAILED")
        from williams_ai.qwen4b_blind_evaluation import grade_pair
        grade=grade_pair(predictions=dest/"market_precommit.json",withheld_outcomes=sealed/"WITHHELD_OUTCOME_KEYS.jsonl",new_report=dest/"market_blind_grade.json")
        print("RECOVERED_FIRST_MARKET_GRADE",json.dumps(grade),flush=True)
    proof={"final_kaggle_state":final,"gpu_training_rerun":False,"actual_adapter_sha256":hashes,
        "source_optimizer_steps":report.get("source_optimizer_steps"),"practice_optimizer_steps":report.get("practice_optimizer_steps"),
        "source_validation":report.get("final_source_validation_gate"),"practice_validation":report.get("final_practice_validation_gate"),
        "execution_allowed":False}
    (dest/"recovery_receipt.json").write_text(json.dumps(proof,indent=2))
    print("RECOVERED_EXISTING_PRIVATE_GPU_WEIGHTS",json.dumps(proof),flush=True)
    if final!="COMPLETE" or "candidate" not in hashes or report.get("adapter_weights_changed") is not True:
        raise RuntimeError("OWN_PARTIAL_RESULT_SAVED_BUT_TRAINING_NOT_COMPLETE")


if __name__=="__main__":main()
