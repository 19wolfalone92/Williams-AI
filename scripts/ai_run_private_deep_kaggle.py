"""Submit user-authorized private originals research using the existing secret.

No new credentials, public plaintext, private-dataset attachment, original
test pages, market outcome labels, orders, or automatic model deployment.
"""
from __future__ import annotations
import base64
from hashlib import sha256
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ai_sealed_source_transport import open_envelope
from williams_ai.deep_book_research import checked_payload


def packed(payload: bytes) -> str:
    return base64.b64encode(lzma.compress(payload, preset=6)).decode()


def worker_source(revision: str, private_payload: bytes, questions: bytes) -> str:
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("PINNED_SOURCE_REVISION_REQUIRED")
    template = '''"""PRIVATE user-authorized original-source study. Never publish this worker."""
import base64,hashlib,json,lzma,os,subprocess,sys,zipfile
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
revision=__REVISION__
os.environ["TRAINING_SOURCE_SHA"]=revision
repo=Path("/tmp/williams-private-originals-source")
subprocess.run(["git","clone","-q","https://github.com/19wolfalone92/Williams-AI.git",str(repo)],check=True)
subprocess.run(["git","checkout","-q",revision],cwd=repo,check=True)
# Remove complete market archives before any model import or inference.
shutil=__import__("shutil")
shutil.rmtree(repo/"examples")
subprocess.run([sys.executable,"-m","pip","install","--quiet","--disable-pip-version-check",
    "transformers==4.57.1","peft==0.17.1","accelerate==1.10.1",
    "bitsandbytes==0.48.1","safetensors==0.6.2"],check=True)
private=Path("/tmp/private_source_payload.json")
private.write_bytes(lzma.decompress(base64.b64decode(__PRIVATE_PAYLOAD__)))
private.chmod(0o600)
question=Path("/tmp/sealed_asof_questions.jsonl")
question.write_bytes(lzma.decompress(base64.b64decode(__ASOF_QUESTIONS__)))
question.chmod(0o600)
if hashlib.sha256(question.read_bytes()).hexdigest()!=__QUESTION_SHA__:
    raise RuntimeError("SEALED_QUESTION_BYTES_CHANGED")
sys.path.insert(0,str(repo))
from williams_ai.deep_book_research import train
out=Path("/kaggle/working/private_deep_qwen4b")
failure=None
try:
    train(private,question,out,source_steps=512,practice_steps=96)
except Exception as error:
    failure=type(error).__name__
    Path("/kaggle/working/deep_research_failure.json").write_text(json.dumps({
        "error_type":failure,"training_success_claimed":False,"execution_allowed":False}))
    print("PRIVATE_DEEP_RESEARCH_STAGE_FAILED",failure,flush=True)
finally:
    private.unlink(missing_ok=True)
    if out.is_dir():
        archive=Path("/kaggle/working/Williams_Qwen3_4B_PRIVATE_DEEP_RESEARCH.zip")
        with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    rel=path.relative_to(out)
                    # Only weight/tokenizer artifacts and numerical receipts.
                    if rel.parts[0] in ("source_adapter","candidate_adapter") or (len(rel.parts)==1 and path.suffix==".json"):
                        z.write(path,str(rel))
        print("PRIVATE_DEEP_RESEARCH_ARCHIVE_BYTES",archive.stat().st_size,flush=True)
if failure:
    raise RuntimeError("PRIVATE_RESEARCH_INCOMPLETE_CHECK_FAILURE_RECEIPT")
print("REAL_PRIVATE_ORIGINALS_TRAINING_AND_PREDICTION_SEAL_COMPLETE",flush=True)
'''
    return (template.replace("__REVISION__", repr(revision))
        .replace("__PRIVATE_PAYLOAD__", repr(packed(private_payload)))
        .replace("__ASOF_QUESTIONS__", repr(packed(questions)))
        .replace("__QUESTION_SHA__", repr(sha256(questions).hexdigest())))


def run():
    from kagglehub.auth import get_username
    secret = os.environ.get("KAGGLE_API_TOKEN", "")
    revision = os.environ["GITHUB_SHA"]
    root = Path(os.environ["RUNNER_TEMP"])
    sealed = Path("experiments/private-source-20261027.sealed.json")
    payload = open_envelope(json.loads(sealed.read_bytes()), secret)
    audited = checked_payload(json.loads(payload))
    print("PRIVATE_SOURCE_COUNTS_NO_TEXT", audited["audit"]["source_page_counts"], flush=True)
    market = root / "strict_market_sealed"
    subprocess.run([sys.executable, "scripts/ai_seal_15pair_blind_charts.py", "--new-output", str(market)], check=True)
    questions = (market / "model_questions.jsonl").read_bytes()
    username = get_username()
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,62}", username):
        raise ValueError("AUTHENTICATED_KAGGLE_ACCOUNT_REQUIRED")
    slug = "williams-ai-private-originals-deep-20261027"
    kernel = username + "/" + slug
    metadata = {"id": kernel, "title": slug, "code_file": "worker.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "model_sources": [],
        "competition_sources": [], "kernel_sources": []}
    worker = worker_source(revision, payload, questions)
    if secret in worker or "WITHHELD_OUTCOME_KEYS" in worker or len(worker) > 3_000_000:
        raise ValueError("PRIVATE_WORKER_SECRET_FUTURE_KEY_OR_SIZE_VIOLATION")
    del payload
    with tempfile.TemporaryDirectory(prefix="private-source-kaggle-") as temp:
        d = Path(temp)
        (d / "kernel-metadata.json").write_text(json.dumps(metadata))
        (d / "worker.py").write_text(worker)
        (d / "worker.py").chmod(0o600)
        pushed = subprocess.run(["kaggle", "kernels", "push", "-p", str(d),
            "--accelerator", "NvidiaTeslaT4", "--timeout", "5400"], capture_output=True, text=True, timeout=180)
        if pushed.returncode:
            # Never print the worker, HTTP payload, token, or book text.
            print("PRIVATE_ORIGINALS_GPU_SUBMISSION_REJECTED_RETURN_CODE", pushed.returncode, flush=True)
            raise RuntimeError("PRIVATE_GPU_SUBMISSION_REJECTED_NO_SUCCESS_CLAIM")
    print("PRIVATE_ORIGINALS_KAGGLE_PAGE", "https://www.kaggle.com/code/" + kernel, flush=True)
    status_final = "UNKNOWN"
    for _ in range(220):
        status = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True, text=True, timeout=35)
        raw = status.stdout.lower()
        if status.returncode == 0 and ("complete" in raw or "finished" in raw):
            status_final = "COMPLETE"; break
        if status.returncode == 0 and any(word in raw for word in ("error", "failed", "cancel")):
            status_final = "ERROR"; break
        time.sleep(20)
    print("PRIVATE_ORIGINALS_KAGGLE_FINAL_STATE", status_final, flush=True)
    dest = root / "private_deep_results"
    dest.mkdir(mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="private-source-result-") as temp:
        downloaded = subprocess.run(["kaggle", "kernels", "output", kernel, "-p", temp,
            "--file-pattern", r".*(PRIVATE_DEEP_RESEARCH[.]zip|deep_research_failure[.]json)$"],
            capture_output=True, text=True, timeout=300)
        if downloaded.returncode:
            raise RuntimeError("PRIVATE_RESEARCH_OUTPUT_DOWNLOAD_FAILED")
        for path in Path(temp).rglob("*"):
            if path.is_file() and path.name in ("Williams_Qwen3_4B_PRIVATE_DEEP_RESEARCH.zip", "deep_research_failure.json"):
                shutil.copy2(path, dest / path.name)
    archive = dest / "Williams_Qwen3_4B_PRIVATE_DEEP_RESEARCH.zip"
    if not archive.is_file():
        raise RuntimeError("NO_SAVED_TRAINED_ADAPTER_ARCHIVE")
    with zipfile.ZipFile(archive) as z:
        if any(Path(n).is_absolute() or ".." in Path(n).parts for n in z.namelist()):
            raise ValueError("UNSAFE_TRAINED_ARCHIVE")
        report = json.loads(z.read("research_receipt.json"))
        for stage in ("source", "candidate"):
            name = stage + "_adapter/adapter_model.safetensors"
            if name in z.namelist():
                actual = sha256(z.read(name)).hexdigest()
                declared = report.get(stage + "_adapter_sha256")
                if declared and declared != actual:
                    raise ValueError("ADAPTER_ARCHIVE_SHA_MISMATCH")
                print("REAL_PRIVATE_4B_ADAPTER_SHA256", stage, actual, flush=True)
        if "market_precommit.json" in z.namelist():
            precommit = dest / "market_precommit.json"
            precommit.write_bytes(z.read("market_precommit.json"))
            sealed_prediction = json.loads(precommit.read_bytes())
            if sealed_prediction["receipt"]["market_questions_file_sha256"] != sha256(questions).hexdigest():
                raise ValueError("MARKET_PREDICTION_COHORT_CHANGED")
            from williams_ai.qwen4b_blind_evaluation import grade_pair
            grade = grade_pair(predictions=precommit, withheld_outcomes=market / "WITHHELD_OUTCOME_KEYS.jsonl",
                new_report=dest / "market_blind_grade.json")
            print("REAL_PRIVATE_4B_UNSEEN_MARKET_GRADE", json.dumps(grade), flush=True)
    (dest / "research_receipt.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("REAL_PRIVATE_4B_OPTIMIZER_STEPS", report.get("source_optimizer_steps"), report.get("practice_optimizer_steps"), flush=True)
    print("REAL_PRIVATE_4B_SOURCE_VALIDATION", report.get("final_source_validation_gate"), flush=True)
    print("REAL_PRIVATE_4B_PRACTICE_VALIDATION", report.get("final_practice_validation_gate"), flush=True)
    if status_final != "COMPLETE" or report.get("adapter_weights_changed") is not True or "blind_prediction_receipt" not in report:
        raise RuntimeError("PRIVATE_DEEP_RESEARCH_INCOMPLETE_PARTIAL_EVIDENCE_SAVED")


if __name__ == "__main__":
    run()
