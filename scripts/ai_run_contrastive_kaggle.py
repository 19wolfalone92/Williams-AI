"""Train joint replay on two T4s from our immutable416 checkpoint.

Existing Kaggle credential stays on this CPU runner. Private books, signed
own-output URLs and test prompts enter only the PRIVATE Kaggle worker.
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
from urllib.parse import urlsplit
from urllib.request import urlopen
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ai_continue_own_checkpoint_kaggle import checked_checkpoint,WEIGHT_SHA
from scripts.ai_sealed_source_transport import open_envelope
from williams_ai.deep_book_research import checked_payload
from williams_ai.contrastive_book_practice import paired_cases,question_rows,questions_digest,score_responses
from williams_ai.contrastive_gpu_research import RUN_SLUG,CONTROL_URL,checked_stop_control,checked_baselines

GPU_SECONDS=10800
POLL_SECONDS=20
REFERENCE_RECEIPT_SHA="30fc73e921d550a033d138131bc7a47156f940d1e5fce00643d87720a4ef5e65"


def packed(raw:bytes)->str:return base64.b64encode(lzma.compress(raw,preset=6)).decode()


def safe_submission_error(raw:str,credential:str)->str:
    if credential:raw=raw.replace(credential,"<CREDENTIAL>")
    raw=re.sub(r"https?://[^\s\"']+","<URL>",raw)
    raw=re.sub(r"[A-Za-z0-9_+/=.-]{40,}","<LONG_TOKEN>",raw)
    return raw[-1200:]


def worker_source(revision:str,payload:bytes,questions:bytes,initial:dict,reference:dict)->str:
    if not re.fullmatch(r"[a-f0-9]{40}",revision):raise ValueError("IMMUTABLE_RESEARCH_SOURCE_REQUIRED")
    template='''import base64,hashlib,json,lzma,os,re,shutil,subprocess,sys,zipfile
from pathlib import Path
from urllib.request import urlopen
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"
os.environ["TRAINING_SOURCE_SHA"]=__REVISION__
repo=Path("/tmp/own-contrastive-research-source")
subprocess.run(["git","clone","-q","https://github.com/19wolfalone92/Williams-AI.git",str(repo)],check=True)
subprocess.run(["git","checkout","-q",__REVISION__],cwd=repo,check=True)
shutil.rmtree(repo/"examples")
subprocess.run([sys.executable,"-m","pip","install","--quiet","--disable-pip-version-check",
    "transformers==4.57.1","peft==0.17.1","accelerate==1.10.1","bitsandbytes==0.48.1","safetensors==0.6.2"],check=True)
private=Path("/tmp/contrastive_source_train_validation.json")
private.write_bytes(lzma.decompress(base64.b64decode(__PAYLOAD__)));private.chmod(0o600)
questions=Path("/tmp/contrastive_56_prompts_only.json")
questions.write_bytes(lzma.decompress(base64.b64decode(__QUESTIONS__)));questions.chmod(0o600)
if hashlib.sha256(questions.read_bytes()).hexdigest()!=__QUESTION_SHA__:raise RuntimeError("SEALED_PROMPTS_CHANGED")
reference=Path("/tmp/own_same_cohort_baselines.json");reference.write_text(json.dumps(__REFERENCE__))
config=__INITIAL__
adapter=Path("/tmp/own_verified416");adapter.mkdir(mode=0o700)
if set(config["files"])!={"adapter_config.json","adapter_model.safetensors"}:raise RuntimeError("OWN_SAFE_FILES_ONLY")
for name,meta in config["files"].items():
    raw=urlopen(meta["url"],timeout=240).read()
    if hashlib.sha256(raw).hexdigest()!=meta["sha256"]:raise RuntimeError("OWN_CHECKPOINT_FILE_SHA_CHANGED")
    (adapter/name).write_bytes(raw)
if hashlib.sha256((adapter/"adapter_model.safetensors").read_bytes()).hexdigest()!=__INITIAL_SHA__:
    raise RuntimeError("OWN416_WEIGHT_SHA_CHANGED")
entry=Path("/tmp/run_own_dual_t4.py")
entry.write_text("from pathlib import Path\\nimport sys\\nsys.path.insert(0,"+repr(str(repo))+")\\nfrom williams_ai.contrastive_gpu_research import train_joint\\ntrain_joint(Path("+repr(str(private))+"),Path("+repr(str(questions))+"),Path("+repr(str(adapter))+"),Path("+repr(str(reference))+"),Path('/kaggle/working/contrastive_qwen4b'))\\n")
out=Path("/kaggle/working/contrastive_qwen4b")
failed=None
try:
    subprocess.run([sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2",str(entry)],check=True)
except Exception as error:
    failed=type(error).__name__
    Path("/kaggle/working/contrastive_failure.json").write_text(json.dumps({"error_type":failed,"complete_training_claimed":False,"execution_allowed":False}))
finally:
    private.unlink(missing_ok=True)
    if out.is_dir():
        with zipfile.ZipFile("/kaggle/working/Williams_Qwen3_4B_CONTRASTIVE_DUAL_T4.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    rel=p.relative_to(out)
                    if rel.parts[0] in ("source_adapter","candidate_adapter") or (len(rel.parts)==1 and p.suffix==".json"):z.write(p,str(rel))
if failed:raise RuntimeError("OWN_DUAL_T4_RESEARCH_INCOMPLETE_PARTIAL_EVIDENCE_SAVED")
print("OWN_DUAL_T4_TRAINING_PROCESS_FINISHED_CHECK_RECEIPT",flush=True)
'''
    return (template.replace("__REVISION__",repr(revision)).replace("__PAYLOAD__",repr(packed(payload)))
        .replace("__QUESTIONS__",repr(packed(questions))).replace("__QUESTION_SHA__",repr(sha256(questions).hexdigest()))
        .replace("__REFERENCE__",repr(reference)).replace("__INITIAL__",repr(initial)).replace("__INITIAL_SHA__",repr(WEIGHT_SHA)))


def checked_result_archive(path:Path,destination:Path,questions:list[dict],cases:list[dict])->dict:
    with zipfile.ZipFile(path) as z:
        if any(Path(n).is_absolute() or ".." in Path(n).parts for n in z.namelist()):raise ValueError("UNSAFE_OWN_TRAINED_ARCHIVE")
        report=json.loads(z.read("research_receipt.json"))
        if report.get("initial_adapter_sha256")!=WEIGHT_SHA or report.get("execution_allowed") is not False or report.get("world_size")!=2:
            raise ValueError("OWN_DUAL_GPU_RESULT_PROVENANCE_REQUIRED")
        for stage in ("source","candidate"):
            actual=sha256(z.read(stage+"_adapter/adapter_model.safetensors")).hexdigest()
            if actual!=report[stage+"_adapter_sha256"]:raise ValueError("SAVED_ADAPTER_SHA_FAILED")
        if "contrastive_test_precommit.json" in z.namelist() and not report.get("training_stopped_by_user_or_control_unavailable"):
            pre=json.loads(z.read("contrastive_test_precommit.json"))
            if pre["questions_sha256"]!=questions_digest(questions) or set(pre["stages"])!={"base","initial416","candidate"}:
                raise ValueError("COMPLETE_SEALED_NEW_PARAMETER_TEST_REQUIRED")
            grades={stage:score_responses(cases,rows) for stage,rows in pre["stages"].items()}
            grade={"mode":"SEALED_56_NEW_PARAMETER_RULE_TRANSFER_NOT_EXPERT_MARKET","stages":grades,
                "inference_mode":pre["inference_mode"],"expected_keys_on_cpu_only":True,"trading_qualification":False,"execution_allowed":False}
            candidate=grades["candidate"]
            grade["further_research_transfer_gate"]={"accepted":candidate["accuracy"]>=0.8
                and candidate["accuracy"]>=max(grades["base"]["accuracy"],grades["initial416"]["accuracy"])+0.05
                and all(v["correct"]/v["total"]>=0.75 for v in candidate["per_rule"].values())
                and not candidate["constant_fields_despite_varied_expected_values"],
                "minimum_accuracy":0.8,"minimum_each_rule_accuracy":0.75,"minimum_gain":0.05,
                "trading_qualification":False}
            grade["all_research_gates_accepted"]=(report["final_source_validation_gate"]["accepted"]
                and report["final_practice_validation_gate"]["accepted"] and grade["further_research_transfer_gate"]["accepted"])
            (destination/"contrastive_test_grade.json").write_text(json.dumps(grade,ensure_ascii=False,indent=2))
            (destination/"contrastive_test_precommit.json").write_text(json.dumps(pre,ensure_ascii=False,indent=2))
            print("OWN_DUAL_T4_NEW_PARAMETER_GRADE",json.dumps({s:{"correct":r["correct"],"total":r["total"],"per_rule":r["per_rule"]} for s,r in grades.items()}),flush=True)
    (destination/"research_receipt.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def main():
    from kagglehub.auth import get_username
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    secret=os.environ["KAGGLE_API_TOKEN"];revision=os.environ["GITHUB_SHA"];owner=get_username()
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,62}",owner):raise ValueError("OWN_KAGGLE_ACCOUNT_REQUIRED")
    api=KaggleApi();api.authenticate()
    with urlopen(CONTROL_URL+"?preflight="+str(int(time.time())),timeout=20) as response:
        if checked_stop_control(json.load(response)):raise RuntimeError("USER_STOP_FLAG_NO_TRAINING_SUBMITTED")
    for slug in ("williams-ai-private-originals-deep-20261027","williams-ai-recovered-full-course-rank16-20261027"):
        state=str(api.kernels_status(owner+"/"+slug).status).lower()
        if not any(s in state for s in ("complete","error","cancel")):raise RuntimeError("EXISTING_OWN_GPU_MUST_BE_STOPPED")
    def files(slug):
        found={};page=None;seen=set()
        with api.build_kaggle_client() as client:
            while True:
                request=ApiListKernelSessionOutputRequest();request.user_name=owner;request.kernel_slug=slug
                if page:request.page_token=page
                response=client.kernels.kernels_api_client.list_kernel_session_output(request)
                found.update({f.file_name:f for f in response.files})
                page=getattr(response,"next_page_token",None)
                if not page:break
                if page in seen:raise ValueError("OWN_OUTPUT_PAGINATION_LOOP")
                seen.add(page)
        return found
    own=files("williams-ai-private-originals-deep-20261027");raw={};urls={}
    required={n:"private_checkpoints_originals/checkpoint-416/"+n for n in ("adapter_model.safetensors","adapter_config.json","trainer_state.json")}
    required["research_receipt.json"]="private_deep_qwen4b/research_receipt.json"
    for name,path in required.items():
        item=own.get(path)
        if item is None or urlsplit(item.url).scheme!="https":raise ValueError("OWN_SAFE_SIGNED_OUTPUT_REQUIRED")
        urls[name]=item.url;raw[name]=urlopen(item.url,timeout=240).read()
    proof=checked_checkpoint(json.loads(raw["adapter_config.json"]),json.loads(raw["trainer_state.json"]),json.loads(raw["research_receipt.json"]),raw["adapter_model.safetensors"])
    initial={"files":{n:{"url":urls[n],"sha256":sha256(raw[n]).hexdigest()} for n in ("adapter_config.json","adapter_model.safetensors")}}
    previous=files("williams-ai-recovered-full-course-rank16-20261027")
    reference_bytes=urlopen(previous["private_deep_qwen4b/research_receipt.json"].url,timeout=60).read()
    if sha256(reference_bytes).hexdigest()!=REFERENCE_RECEIPT_SHA:raise ValueError("EXACT_OWN_BASELINE_RECEIPT_SHA_REQUIRED")
    reference=json.loads(reference_bytes)
    payload=open_envelope(json.loads(Path("experiments/private-source-20261027-course-full.sealed.json").read_bytes()),secret)
    audited=checked_payload(json.loads(payload));checked_baselines(reference,audited["audit"])
    cases=paired_cases("test");questions=question_rows(cases);qbytes=json.dumps(questions,ensure_ascii=False).encode()
    worker=worker_source(revision,payload,qbytes,initial,reference)
    if secret in worker or "WITHHELD_OUTCOME_KEYS" in worker or len(worker)>3_000_000:raise ValueError("PRIVATE_WORKER_SECRET_OR_SIZE_VIOLATION")
    compile(worker,"private_dual_t4_worker","exec")
    del raw,urls,payload
    print("VERIFIED_OWN416_CONTRASTIVE_TWO_GPU_PLAN",json.dumps({"checkpoint_sha256":WEIGHT_SHA,"rank":16,"originals":audited["audit"]["source_page_counts"],"test_prompts":56,"expected_test_keys_sent_to_gpu":False}),flush=True)
    if len(RUN_SLUG)>50:raise ValueError("KAGGLE_KERNEL_TITLE_MAX50_REQUIRED")
    metadata={"id":owner+"/"+RUN_SLUG,"title":RUN_SLUG,"code_file":"worker.py","language":"python","kernel_type":"script",
        "is_private":"true","enable_gpu":"true","enable_internet":"true","machine_shape":"NvidiaTeslaT4", "dataset_sources":[],"model_sources":[],"competition_sources":[],"kernel_sources":[]}
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"worker.py").write_text(worker);(root/"worker.py").chmod(0o600)
        (root/"kernel-metadata.json").write_text(json.dumps(metadata))
        pushed=subprocess.run(["kaggle","kernels","push","-p",td,"--accelerator","NvidiaTeslaT4","--timeout",str(GPU_SECONDS)],capture_output=True,text=True,timeout=180)
        if pushed.returncode:
            print("PRIVATE_TWO_GPU_SUBMISSION_VALIDATION",safe_submission_error(pushed.stdout+"\n"+pushed.stderr,secret),flush=True)
            raise RuntimeError("PRIVATE_TWO_GPU_SUBMISSION_REJECTED")
    print("PRIVATE_DUAL_T4_KAGGLE_PAGE","https://www.kaggle.com/code/"+owner+"/"+RUN_SLUG,flush=True)
    state="UNKNOWN"
    for attempt in range((GPU_SECONDS+600)//POLL_SECONDS):
        state=str(api.kernels_status(owner+"/"+RUN_SLUG).status).upper()
        if attempt%15==0:print("PRIVATE_DUAL_T4_CURRENT_STATE",state,flush=True)
        if any(s in state for s in ("COMPLETE","ERROR","CANCEL")):break
        time.sleep(POLL_SECONDS)
    print("PRIVATE_DUAL_T4_FINAL_STATE",state,flush=True)
    dest=Path(os.environ["RUNNER_TEMP"])/"contrastive_dual_t4_results";dest.mkdir(mode=0o700)
    result=subprocess.run(["kaggle","kernels","output",owner+"/"+RUN_SLUG,"-p",str(dest),"--file-pattern",r".*(CONTRASTIVE_DUAL_T4[.]zip|contrastive_failure[.]json)$"],capture_output=True,text=True,timeout=420)
    archive=dest/"Williams_Qwen3_4B_CONTRASTIVE_DUAL_T4.zip"
    if result.returncode or not archive.is_file():
        subprocess.run(["kaggle","kernels","output",owner+"/"+RUN_SLUG,"-p",str(dest/"safe_partial_checkpoints"),"--file-pattern",r".*(adapter_model[.]safetensors|adapter_config[.]json|trainer_state[.]json|research_receipt[.]json)$"],capture_output=True,text=True,timeout=420)
        raise RuntimeError("INCOMPLETE_GPU_SAFE_CHECKPOINTS_REQUESTED_NO_COMPLETION_CLAIM")
    report=checked_result_archive(archive,dest,questions,cases)
    print("REAL_TWO_GPU_WEIGHT_SHA256",report.get("candidate_adapter_sha256"),flush=True)
    print("REAL_TWO_GPU_OPTIMIZER_STEPS",report.get("joint_optimizer_steps"),report.get("selected_checkpoint"),flush=True)
    print("REAL_TWO_GPU_FOUR_BOOK_GATE",json.dumps(report.get("final_source_validation_gate")),flush=True)
    print("REAL_TWO_GPU_PRACTICE_GATE",json.dumps(report.get("final_practice_validation_gate")),flush=True)
    if report.get("training_stopped_by_user_or_control_unavailable"):
        print("USER_STOPPED_OWN_CHECKPOINT_PRESERVED",flush=True);return
    if "COMPLETE" not in state or report.get("complete_training_and_test") is not True:raise RuntimeError("INCOMPLETE_TRAINING_OR_EXAM_PARTIAL_EVIDENCE_SAVED")


if __name__=="__main__":main()
