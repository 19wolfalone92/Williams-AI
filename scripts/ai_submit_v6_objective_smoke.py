"""One private, quota-bounded Kaggle Qwen3-4B technical pilot, no market orders.

Only this unique run slug may be submitted; never overwrite a kernel. No
secret is included in the worker source. Public Actions artifacts contain
only aggregate receipts and an authenticated encrypted private checkpoint ZIP.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

from slugify import slugify
from zipfile import ZipFile,ZIP_DEFLATED

from scripts.ai_submit_v5_qlora_kaggle import _seal_data
from scripts.v5_recovery_integrity import validate_tensor_bytes
from williams_ai.qlora_gpu_research import seal_checkpoints,latest_checkpoint,digest_file
from williams_ai.v6_objective_practice import curriculum
from williams_ai.knowledge_training_v5 import BASE,REV

SLUG="williams-ai-v6-objective-smoke-20261124"
CONTROL="https://raw.githubusercontent.com/19wolfalone92/Williams-AI/research/williams-ai-source-sft-v6-20261124/experiments/v6-objective-smoke-control.json"
GPU_TIMEOUT=2700
MIN_QUOTA=3600


def check_control():
    with urlopen(CONTROL+"?t="+str(int(time.time())),timeout=20) as response:
        state=json.load(response)
    if state!={"run_slug":SLUG,"stop":False}:
        raise ValueError("V6_SMOKE_USER_STOP_OR_CONTROL_INVALID")


def worker_source(sha):
    if not re.fullmatch(r"[0-9a-f]{40}",sha):raise ValueError("EXACT_SOURCE_SHA_REQUIRED")
    script = '''import json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
SOURCE="SOURCE_PLACEHOLDER"
CONTROL="CONTROL_PLACEHOLDER"
SLUG="SLUG_PLACEHOLDER"
def authorized():
    with urllib.request.urlopen(CONTROL+"?t="+str(int(time.time())),timeout=15) as r:
        state=json.load(r)
    if state=={"run_slug":SLUG,"stop":True}:return False
    if state!={"run_slug":SLUG,"stop":False}:raise ValueError("INVALID_CONTROL")
    return True
def run(cmd,cwd=None):
    if not authorized():raise RuntimeError("USER_STOP")
    p=subprocess.Popen(cmd,cwd=cwd,start_new_session=True)
    next_check=time.monotonic()+30
    failures=0
    try:
        while p.poll() is None:
            if time.monotonic()>=next_check:
                try:
                    allowed=authorized()
                    failures=0
                except Exception:
                    failures+=1
                    allowed=failures<3
                if not allowed:raise RuntimeError("STOP_OR_CONTROL_LOST")
                next_check=time.monotonic()+30
            time.sleep(2)
        if p.returncode:raise RuntimeError("WORKER_COMMAND_FAILED_"+str(p.returncode))
    finally:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
home=Path("/kaggle/temp/williams-v6-source")
if home.exists():raise RuntimeError("NEW_SOURCE_DIR_REQUIRED")
run(["git","clone","--quiet","https://github.com/19wolfalone92/Williams-AI.git",str(home)])
run(["git","checkout","--quiet",SOURCE],cwd=home)
run([sys.executable,"-m","pip","install","--disable-pip-version-check",
    "transformers==4.57.1","peft==0.17.1","accelerate==1.10.1",
    "safetensors==0.6.2","bitsandbytes==0.48.1","huggingface_hub>=0.30,<2","jinja2>=3.1.6,<4"])
run([sys.executable,"-m","scripts.ai_v6_objective_gpu_smoke",
    "--output","/kaggle/working/williams_v6_objective_smoke","--source-sha",SOURCE],cwd=home)
print("ACTUAL_V6_OBJECTIVE_CUDA_SMOKE_COMPLETED",flush=True)
'''
    return script.replace("SOURCE_PLACEHOLDER",sha).replace("CONTROL_PLACEHOLDER",CONTROL).replace("SLUG_PLACEHOLDER",SLUG)


def _inventory_identity(owner,item):
    """Resolve listed personal kernel identity using Kaggle's canonical slug rule."""
    raw_id=getattr(item,"id",None)
    item_id=raw_id if type(raw_id) is int and raw_id>0 else None
    ref=getattr(item,"ref",None)
    bare=getattr(item,"slug",None)
    inferred=False

    def normalize_slug(value):
        if not isinstance(value,str) or not value:
            raise ValueError("INVENTORY_KERNEL_SLUG_INVALID")
        parts=value.split("/")
        if len(parts)==2:
            if parts[0].casefold()!=owner:
                raise ValueError("INVENTORY_OWNER_OR_SLUG_MISMATCH")
            value=parts[1]
        elif len(parts)!=1:
            raise ValueError("INVENTORY_KERNEL_SLUG_INVALID")
        if len(value)<5 or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]*",value):
            raise ValueError("INVENTORY_KERNEL_SLUG_INVALID")
        return value.casefold()

    if isinstance(ref,str) and ref:
        parts=ref.split("/")
        if len(parts)!=2 or parts[0].casefold()!=owner:
            raise ValueError("INVENTORY_OWNER_OR_REF_MISMATCH")
        slug=normalize_slug(parts[1])
        if bare and normalize_slug(bare)!=slug:
            raise ValueError("INVENTORY_REF_SLUG_CONFLICT")
    elif ref not in (None,""):
        raise ValueError("INVENTORY_REF_INVALID")
    elif isinstance(bare,str) and bare:
        slug=normalize_slug(bare)
    elif bare not in (None,""):
        raise ValueError("INVENTORY_SLUG_INVALID")
    else:
        title=getattr(item,"title",None)
        if not isinstance(title,str) or not title.strip():
            raise ValueError("INVENTORY_REF_SLUG_AND_TITLE_MISSING")
        slug=normalize_slug(slugify(title))
        inferred=True
    return owner+"/"+slug,item_id,inferred


def own_inventory_absent(api,kernel):
    """Use complete authenticated personal inventory, never a public search.

    Kaggle CLI's documented title-to-slug normalization resolves metadata rows
    that omit both ref and slug. A positive numeric id is checked when present;
    a unique owner-scoped canonical slug remains the resource identifier when
    the SDK omits that optional metadata field. Ambiguity remains fail-closed.
    """
    parts=kernel.casefold().split("/")
    if len(parts)!=2 or not all(parts):
        raise ValueError("KAGGLE_KERNEL_REFERENCE_INVALID")
    owner,slug=parts
    kernel=owner+"/"+slug
    witness=owner+"/williams-ai-v5-book-sft-20261111"
    seen=set()
    token=None
    refs=set()
    ids={}
    inferred_count=0
    missing_id_count=0
    try:
        for _ in range(100):
            page=api.kernels_list_with_response(mine=True,user=owner,page_size=100,page_token=token)
            if page is None or not isinstance(page.kernels,list):
                raise ValueError("INVALID_PRIVATE_INVENTORY")
            for item in page.kernels:
                if item is None:
                    raise ValueError("NULL_PRIVATE_INVENTORY_ENTRY")
                ref,item_id,inferred=_inventory_identity(owner,item)
                if ref in refs:
                    raise ValueError("INVENTORY_DUPLICATE_SLUG")
                refs.add(ref)
                if item_id is None:
                    missing_id_count+=1
                else:
                    if item_id in ids and ids[item_id]!=ref:
                        raise ValueError("INVENTORY_DUPLICATE_ID_DIFFERENT_SLUGS")
                    ids[item_id]=ref
                inferred_count+=int(inferred)
            token=page.next_page_token
            if not token:
                if witness not in refs:
                    raise ValueError("PRIVATE_V5_VISIBILITY_WITNESS_MISSING")
                print("V6_PRIVATE_INVENTORY_VERIFIED",
                      json.dumps({"kernels":len(refs),"title_slug_inferred":inferred_count,
                                  "numeric_id_missing":missing_id_count}),
                      flush=True)
                return kernel not in refs
            if not isinstance(token,str) or token in seen:
                raise ValueError("INVALID_INVENTORY_PAGINATION")
            seen.add(token)
    except Exception as exc:
        print("V6_INVENTORY_REJECTED",type(exc).__name__,flush=True)
        raise RuntimeError("KAGGLE_KERNEL_ABSENCE_NOT_VERIFIED") from None
    raise RuntimeError("KAGGLE_KERNEL_ABSENCE_NOT_VERIFIED")


def status_absent(api,kernel):
    try:
        api.kernels_status(kernel)
    except Exception as exc:
        code=getattr(exc,"status",None) or getattr(exc,"status_code",None)
        response=getattr(exc,"response",None)
        if code is None and response is not None:code=getattr(response,"status_code",None)
        if str(code)=="404":return True
        # Official Kaggle CLI v2.2.5 wraps 401/403 into ValueError for a
        # missing/inaccessible slug. A denied lookup alone is not absence.
        # Only an independently authenticated, complete own inventory
        # with a known private notebook witness may establish absence.
        if (isinstance(exc,ValueError) and
            "Permission 'kernels.get' was denied" in str(exc) and
            kernel in str(exc)):
            return own_inventory_absent(api,kernel)
        print("V6_KAGGLE_STATUS_LOOKUP_UNVERIFIED",type(exc).__name__,flush=True)
        raise RuntimeError("KAGGLE_KERNEL_ABSENCE_NOT_VERIFIED") from None
    return False


def verify_and_seal(root,secret,dest,source_sha):
    summary=root/"aggregate.json"
    if summary.is_symlink() or not summary.is_file():raise ValueError("NO_COMPLETED_GPU_AGGREGATE")
    report=json.loads(summary.read_text())
    manifest=curriculum()[1]
    if (report.get("schema")!="WILLIAMS_V6_REAL_OBJECTIVE_CUDA_SMOKE_V1" or
        report.get("source_commit")!=source_sha or report.get("base_model")!=BASE or
        report.get("base_revision")!=REV or report.get("dataset")!=manifest or
        report.get("new_optimizer_steps")!=20 or report.get("session_steps")!=[10,10] or
        report.get("optimizer_skips")!=0 or report.get("full_optimizer_state_resume") is not True or
        report.get("reloaded_adapter") is not True or report.get("all_finite") is not True or
        report.get("independent_final_exam") is not False or report.get("execution_allowed") is not False):
        raise ValueError("INCOMPLETE_OR_WRONG_REAL_GPU_RECEIPT")
    for part in ("first","second"):
        phase=root/part
        original=json.loads((phase/"checkpoint_sha256.json").read_text())
        seal_checkpoints(phase)
        if original!=json.loads((phase/"checkpoint_sha256.json").read_text()):
            raise ValueError("DOWNLOADED_CHECKPOINT_SEAL_MISMATCH")
    first,_=latest_checkpoint(root/"first")
    if digest_file(first/"adapter_model.safetensors")!=report["first_checkpoint_sha256"]:
        raise ValueError("FIRST_CHECKPOINT_DIGEST_MISMATCH")
    weight=root/"second"/"latest_adapter"/"adapter_model.safetensors"
    raw=weight.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=report["adapter_sha256"] or validate_tensor_bytes(raw)!=report["tensor_count"]:
        raise ValueError("FINAL_ADAPTER_INVENTORY_OR_HASH_MISMATCH")
    dest.mkdir(mode=0o700,parents=True,exist_ok=False)
    # Export only this private run's verified checkpoints, receipt and
    # predictions. No public base, credentials, books or repository tree.
    buf=io.BytesIO()
    with ZipFile(buf,"w",compression=ZIP_DEFLATED,compresslevel=3) as z:
        for file in sorted(root.rglob("*")):
            if file.is_symlink():raise ValueError("LINKED_PRIVATE_OUTPUT")
            if not file.is_file():continue
            rel=file.relative_to(root)
            if rel.parts[0] in ("first","second","aggregate.json","private_predictions.json"):
                z.write(file,str(rel))
    encrypted=_seal_data(buf.getvalue(),secret)
    (dest/"v6-objective-checkpoints.zip.sealed").write_bytes(encrypted)
    public={**report,"encrypted_archive_sha256":hashlib.sha256(encrypted).hexdigest(),
            "private_checkpoints_encrypted":True}
    (dest/"aggregate.json").write_text(json.dumps(public,sort_keys=True,indent=2))
    return public


def run():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    token=os.environ.get("KAGGLE_API_TOKEN","")
    sha=os.environ.get("GITHUB_SHA","")
    if len(token)<20 or not re.fullmatch(r"[0-9a-f]{40}",sha):
        raise ValueError("EXACT_SOURCE_AND_EXISTING_PRIVATE_CREDENTIAL_REQUIRED")
    check_control()
    api=KaggleApi();api.authenticate()
    owner=get_username()
    if not owner or not isinstance(owner,str):raise ValueError("KAGGLE_OWNER_UNVERIFIED")
    kernel=owner+"/"+SLUG
    with api.build_kaggle_client() as client:
        q=client.kernels.kernels_api_client.get_accelerator_quota_statistics(
            ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
        remaining=(q.total_time_allowed-q.time_used).total_seconds() if q else None
    print("V6_PRIMARY_FREE_GPU_QUOTA_SECONDS",remaining,flush=True)
    if remaining is None or remaining<MIN_QUOTA:
        raise ValueError("INSUFFICIENT_CONFIRMED_FREE_GPU_QUOTA")
    if not status_absent(api,kernel):
        raise ValueError("V6_GPU_KERNEL_ALREADY_EXISTS_NO_OVERWRITE")
    worker=worker_source(sha)
    if token in worker:raise ValueError("CREDENTIAL_IN_WORKER")
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        (folder/"worker.py").write_text(worker)
        (folder/"kernel-metadata.json").write_text(json.dumps({
            "id":kernel,"title":SLUG,"code_file":"worker.py","language":"python",
            "kernel_type":"script","is_private":"true","enable_gpu":"true",
            "enable_internet":"true","machine_shape":"NvidiaTeslaT4",
            "dataset_sources":[],"model_sources":[],"competition_sources":[],"kernel_sources":[]}))
        result=subprocess.run(["kaggle","kernels","push","-p",directory,
            "--accelerator","NvidiaTeslaT4","--timeout",str(GPU_TIMEOUT)],
            capture_output=True,text=True,timeout=180)
        if result.returncode:raise RuntimeError("PRIVATE_V6_KAGGLE_SUBMISSION_FAILED")
    print("V6_OBJECTIVE_GPU_SUBMITTED",sha,flush=True)
    status="UNKNOWN"
    for i in range(155):
        state=str(api.kernels_status(kernel).status).upper()
        if state!=status or i%6==0:print("V6_OBJECTIVE_GPU_STATE",state,flush=True)
        status=state
        if any(word in status for word in ("COMPLETE","ERROR","CANCEL")):break
        time.sleep(20)
    if "COMPLETE" not in status:raise RuntimeError("V6_GPU_NOT_COMPLETE_NO_TRAINING_CLAIM")
    with tempfile.TemporaryDirectory() as directory:
        result=subprocess.run(["kaggle","kernels","output",kernel,"-p",directory],
                              capture_output=True,text=True,timeout=600)
        if result.returncode:raise RuntimeError("V6_PRIVATE_OUTPUT_DOWNLOAD_FAILED")
        public=verify_and_seal(Path(directory)/"williams_v6_objective_smoke",token,
                        Path(os.environ["RUNNER_TEMP"])/"williams-v6-objective-sealed",sha)
    print("V6_REAL_GPU_VERIFIED_SEALED",json.dumps(public,sort_keys=True),flush=True)


if __name__=="__main__":run()
