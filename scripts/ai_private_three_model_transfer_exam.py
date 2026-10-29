"""Observe two completed own GPU fits, then compare actual generated answers."""
from __future__ import annotations
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import urlopen
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ai_run_private_deep_kaggle import packed
from williams_ai.deep_book_research import MODEL, REVISION
from williams_ai.structured_transfer_exam import sealed_cases, grade


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner=get_username()
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,62}",owner):raise ValueError("OWN_KAGGLE_ACCOUNT_REQUIRED")
    final_slug="williams-ai-full-course-corrective-rank16-20261027"
    for attempt in range(420):
        status=subprocess.run(["kaggle","kernels","status",owner+"/"+final_slug],capture_output=True,text=True,timeout=35)
        value=status.stdout.lower()
        if status.returncode==0 and ("complete" in value or "finished" in value):break
        if status.returncode==0 and any(word in value for word in ("error","failed","cancel")):
            raise RuntimeError("CORRECTIVE_GPU_NOT_COMPLETE_NO_EXAM_ALLOCATION")
        if attempt%6==0:
            try:
                with urlopen("https://api.github.com/repos/19wolfalone92/Williams-AI/actions/runs/40479971545",timeout=15) as response:
                    parent=json.load(response)
                if parent.get("status")=="completed" and parent.get("conclusion")!="success":
                    raise RuntimeError("CORRECTIVE_WORKFLOW_FAILED_NO_EXAM_GPU_STARTED")
            except (OSError,ValueError):pass
        time.sleep(20)
    else:raise RuntimeError("TWO_OWN_GPU_RUNS_NOT_FINISHED_NO_EXAM_STARTED")
    api=KaggleApi();api.authenticate()
    adapters={}
    for name,slug,source_sha in (("first_private","williams-ai-private-originals-deep-20261027","6099fb60f4f8f050e0e51f0253c970f20826b833"),
                                  ("corrected",final_slug,"c0f862cd5c30cc12f786fbe7b1b86097b108894f")):
        with api.build_kaggle_client() as client:
            req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=slug
            outputs=client.kernels.kernels_api_client.list_kernel_session_output(req)
        files=[f for f in outputs.files if f.file_name=="Williams_Qwen3_4B_PRIVATE_DEEP_RESEARCH.zip"]
        if len(files)!=1 or urlsplit(files[0].url).scheme!="https":raise ValueError("OWN_CHECKED_ADAPTER_OUTPUT_REQUIRED")
        raw=urlopen(files[0].url,timeout=240).read()
        with zipfile.ZipFile(BytesIO(raw)) as z:
            receipt=json.loads(z.read("research_receipt.json"))
            weight=z.read("candidate_adapter/adapter_model.safetensors")
        digest=sha256(weight).hexdigest()
        if (receipt.get("model_repo")!=MODEL or receipt.get("model_revision")!=REVISION
            or receipt.get("source_commit")!=source_sha or receipt.get("rank")!=16
            or receipt.get("candidate_adapter_sha256")!=digest or receipt.get("execution_allowed") is not False):
            raise ValueError("PREVIOUS_OWN_ADAPTER_IDENTITY_MISMATCH")
        adapters[name]={"url":files[0].url,"archive_sha256":sha256(raw).hexdigest(),"adapter_sha256":digest}
        del raw,weight
    questions,keys=sealed_cases()
    raw_questions=json.dumps(questions,sort_keys=True,ensure_ascii=False).encode()
    question_sha=sha256(raw_questions).hexdigest()
    revision=os.environ["GITHUB_SHA"]
    worker='''"""Private three-model inference only; NO answer keys or training."""
import base64,hashlib,json,lzma,os,subprocess,sys,zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
repo=Path("/tmp/williams-transfer-exam-source")
subprocess.run(["git","clone","-q","https://github.com/19wolfalone92/Williams-AI.git",str(repo)],check=True)
subprocess.run(["git","checkout","-q",__REVISION__],cwd=repo,check=True)
__import__("shutil").rmtree(repo/"examples")
subprocess.run([sys.executable,"-m","pip","install","--quiet","--disable-pip-version-check",
    "transformers==4.57.1","peft==0.17.1","accelerate==1.10.1","bitsandbytes==0.48.1","safetensors==0.6.2"],check=True)
adapter_config=__ADAPTERS__
directories={}
for name,config in adapter_config.items():
    raw=urlopen(config["url"],timeout=240).read()
    if hashlib.sha256(raw).hexdigest()!=config["archive_sha256"]:raise ValueError("PRIVATE_ADAPTER_ARCHIVE_HASH_FAILED")
    folder=Path("/tmp")/("transfer_own_"+name);folder.mkdir(mode=0o700)
    with zipfile.ZipFile(BytesIO(raw)) as z:
        for member in z.namelist():
            rel=Path(member)
            if rel.parts and rel.parts[0]=="candidate_adapter":
                if len(rel.parts)!=2 or rel.name in (".",".."):raise ValueError("PRIVATE_ADAPTER_PATH_FAILED")
                (folder/rel.name).write_bytes(z.read(member))
    if hashlib.sha256((folder/"adapter_model.safetensors").read_bytes()).hexdigest()!=config["adapter_sha256"]:
        raise ValueError("PRIVATE_ADAPTER_WEIGHTS_HASH_FAILED")
    directories[name]=folder
    del raw
questions=Path("/tmp/transfer_questions_only.json")
questions.write_bytes(lzma.decompress(base64.b64decode(__QUESTIONS__)))
if hashlib.sha256(questions.read_bytes()).hexdigest()!=__QUESTION_SHA__:raise ValueError("TRANSFER_QUESTIONS_HASH_FAILED")
sys.path.insert(0,str(repo))
from williams_ai.structured_transfer_exam import predict
predict(questions,directories,Path("/kaggle/working/three_model_transfer_precommit.json"))
print("REAL_THREE_MODEL_GENERATED_TEXT_EXAM_PRECOMMITTED",flush=True)
'''.replace("__REVISION__",repr(revision)).replace("__ADAPTERS__",repr(adapters)).replace("__QUESTIONS__",repr(packed(raw_questions))).replace("__QUESTION_SHA__",repr(question_sha))
    if os.environ["KAGGLE_API_TOKEN"] in worker or "\"expected\"" in worker:raise ValueError("NO_CREDENTIALS_OR_ANSWER_KEYS_IN_GPU_EXAM")
    slug="williams-ai-three-model-transfer-exam-20261027"
    kernel=owner+"/"+slug
    meta={"id":kernel,"title":slug,"code_file":"worker.py","language":"python","kernel_type":"script",
        "is_private":"true","enable_gpu":"true","enable_internet":"true","machine_shape":"NvidiaTeslaT4",
        "dataset_sources":[],"model_sources":[],"kernel_sources":[],"competition_sources":[]}
    with tempfile.TemporaryDirectory() as td:
        p=Path(td);(p/"worker.py").write_text(worker);(p/"kernel-metadata.json").write_text(json.dumps(meta))
        submit=subprocess.run(["kaggle","kernels","push","-p",td,"--accelerator","NvidiaTeslaT4","--timeout","2400"],capture_output=True,text=True,timeout=180)
        if submit.returncode:raise RuntimeError("PRIVATE_TRANSFER_EXAM_SUBMISSION_REJECTED")
    print("PRIVATE_TRANSFER_EXAM_ACCEPTED_NO_TRAINING_OR_EXPECTED_ANSWERS",flush=True)
    for _ in range(110):
        status=subprocess.run(["kaggle","kernels","status",kernel],capture_output=True,text=True,timeout=35)
        value=status.stdout.lower()
        if status.returncode==0 and ("complete" in value or "finished" in value):break
        if status.returncode==0 and any(word in value for word in ("error","failed","cancel")):raise RuntimeError("PRIVATE_TRANSFER_EXAM_WORKER_FAILED")
        time.sleep(20)
    else:raise RuntimeError("PRIVATE_TRANSFER_EXAM_COMPLETION_NOT_CONFIRMED")
    dest=Path(os.environ["RUNNER_TEMP"])/"private_three_model_transfer_results";dest.mkdir(mode=0o700)
    get=subprocess.run(["kaggle","kernels","output",kernel,"-p",str(dest),"--file-pattern",r".*three_model_transfer_precommit[.]json$"],capture_output=True,text=True,timeout=120)
    if get.returncode:raise RuntimeError("PRIVATE_TRANSFER_EXAM_PRECOMMIT_DOWNLOAD_FAILED")
    precommit=json.loads((dest/"three_model_transfer_precommit.json").read_bytes())
    report=grade(questions,keys,precommit)
    report["adapter_sha256"]={name:config["adapter_sha256"] for name,config in adapters.items()}
    (dest/"three_model_transfer_grade.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("REAL_THREE_MODEL_GENERATED_TEXT_TRANSFER",json.dumps({s:{k:v for k,v in r.items() if k!="case_results"} for s,r in report["stages"].items()}),flush=True)


if __name__=="__main__":main()
