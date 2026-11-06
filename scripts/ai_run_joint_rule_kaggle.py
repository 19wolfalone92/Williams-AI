"""Submit one bounded private free-GPU corrective run, then verify its bytes."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ai_run_historical_rule_kaggle import worker_source as historical_worker
from scripts.ai_run_restorative_kaggle import safe_submission_error
from scripts.ai_sealed_source_transport import open_envelope
from scripts.ai_verify_joint_rules import audit_groups, verify_archive
from williams_ai.joint_rule_curriculum import (MODE, RUN_SLUG, INITIAL_SHA, LIMITS,
    boundary_cases, numeric_cases, manifest)
from williams_ai.historical_rule_curriculum import canonical, digest, load_pack, cases_from_pack, check_partitions, prompts_only
from williams_ai.deep_book_research import checked_payload
from williams_ai.restorative_book_practice import INITIAL_SHA as FOCUSED_SHA, PREVIOUS_RECEIPT_SHA

CONTROL_URL = "https://raw.githubusercontent.com/19wolfalone92/Williams-AI/research/private-four-book-deep-training-20261027/experiments/joint-rule-training-control.json"
ARCHIVE = "Williams_Qwen3_4B_JOINT_RULES_20261106.zip"


def groups_and_design():
    packs = {s:load_pack(Path("examples/historical_rule_20261029")/(s+".json.gz")) for s in ("train","validation","test")}
    check_partitions(packs)
    historical = {s:cases_from_pack(p) for s,p in packs.items()}
    groups = {s:{"historical":historical[s],"numeric":numeric_cases(s),"boundary":boundary_cases(s)} for s in packs}
    design = manifest(historical)
    committed = json.loads(Path("experiments/joint-rule-curriculum-20261106-design.json").read_bytes())
    if design!=committed: raise ValueError("PRECOMMITTED_JOINT_DESIGN_CHANGED")
    for data in groups.values(): audit_groups(data)
    return groups, design


def worker_source(revision, source, curriculum, questions, initial, focused, reference):
    return (historical_worker(revision,source,curriculum,questions,initial,focused,reference)
        .replace("historical_rule_gpu import train", "joint_rule_gpu import train")
        .replace("HISTORICAL_RULES_20261029", "JOINT_RULES_20261106")
        .replace("historical_rule_failure.json", "joint_rule_failure.json")
        .replace("HISTORICAL_GPU", "JOINT_GPU"))


def outputs(api, owner, slug):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    state = str(api.kernels_status(owner+"/"+slug).status).upper()
    if "COMPLETE" not in state: raise ValueError("VERIFIED_PARENT_KERNEL_MUST_BE_COMPLETE")
    result = {}; page = None; seen = set()
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest();request.user_name=owner;request.kernel_slug=slug
            if page:request.page_token=page
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            result.update({f.file_name:f for f in response.files})
            page = getattr(response,"next_page_token",None)
            if not page:break
            if page in seen:raise ValueError("OWN_OUTPUT_PAGINATION_LOOP")
            seen.add(page)
    return result


def require_new_kernel(api, owner):
    """Status of a nonexistent private kernel can return 403, so list our own.

    Never reinterpret an authorization error as proof that a run is absent.
    """
    for page in range(1, 21):
        rows = api.kernels_list(mine=True, search=RUN_SLUG, page=page, page_size=100)
        if not isinstance(rows, list) or any(not isinstance(getattr(row, "ref", None), str) for row in rows):
            raise ValueError("EXPLICIT_OWN_KERNEL_INVENTORY_REQUIRED")
        if any(row.ref.casefold()==(owner+"/"+RUN_SLUG).casefold() for row in rows):
            raise RuntimeError("JOINT_KERNEL_ALREADY_EXISTS_NO_AUTOMATIC_DUPLICATE")
        if len(rows)<100:return
    raise ValueError("COMPLETE_OWN_KERNEL_INVENTORY_REQUIRED")


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    token = os.environ["KAGGLE_API_TOKEN"];revision = os.environ["GITHUB_SHA"]
    dest = Path(os.environ["RUNNER_TEMP"])/"joint_rule_results";dest.mkdir(mode=0o700)
    groups,design = groups_and_design()
    (dest/"frozen_design.json").write_bytes(canonical(design))
    owner=get_username();api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as client:
        quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
        remaining = (quota.total_time_allowed-quota.time_used).total_seconds() if quota else None
    (dest/"gpu_preflight.json").write_bytes(canonical({"free_gpu_hours_remaining":None if remaining is None else remaining/3600,"paid_fallback":False}))
    print("JOINT_FREE_GPU_HOURS",None if remaining is None else round(remaining/3600,4),flush=True)
    if remaining is None or remaining<LIMITS["minimum_free_gpu_seconds"]:
        raise RuntimeError("CONFIRMED_FREE_GPU_RESERVE_REQUIRED_NO_PAID_FALLBACK")
    with urlopen(CONTROL_URL+"?preflight="+str(int(time.time())),timeout=20) as response:control=json.load(response)
    if control!={"run_slug":RUN_SLUG,"stop":False}:raise RuntimeError("USER_STOP_NO_JOINT_GPU_SUBMISSION")
    # Re-running an existing pending kernel would lose its checkpoint state.
    require_new_kernel(api, owner)
    prior=outputs(api,owner,"williams-ai-historical-books-20261029")
    focused=outputs(api,owner,"williams-ai-focused-2t4-20261027")
    raw_reference=urlopen(focused["focused_qwen4b/research_receipt.json"].url,timeout=60).read()
    if sha256(raw_reference).hexdigest()!=PREVIOUS_RECEIPT_SHA:raise ValueError("EXACT_FOCUSED_REFERENCE_RECEIPT_REQUIRED")
    reference=json.loads(raw_reference)
    def adapter(files,prefix,expected):
        result={}
        for name in ("adapter_config.json","adapter_model.safetensors"):
            item=files[prefix+"/"+name];raw=urlopen(item.url,timeout=240).read();actual=sha256(raw).hexdigest()
            if name.endswith(".safetensors") and actual!=expected:raise ValueError("EXACT_OWN_JOINT_PARENT_ADAPTER_REQUIRED")
            result[name]={"url":item.url,"sha256":actual}
        return result
    initial=adapter(prior,"historical_rule_qwen4b/candidate_adapter",INITIAL_SHA)
    control_adapter=adapter(focused,"focused_qwen4b/candidate_adapter",FOCUSED_SHA)
    source=open_envelope(json.loads(Path("experiments/private-source-20261027-course-full.sealed.json").read_bytes()),token)
    if checked_payload(json.loads(source))["audit"]!=reference["source_audit"]:raise ValueError("UNCHANGED_ORIGINAL_BOOK_PARTITIONS_REQUIRED")
    train_payload={"schema":MODE,"train":groups["train"],"validation":groups["validation"],
                   "audit":{s:design["questions_sha256"][s] for s in ("train","validation")}}
    questions={g:prompts_only(rows) for g,rows in groups["test"].items()}
    questions["sha256"]={g:digest(rows) for g,rows in questions.items()}
    worker=worker_source(revision,source,canonical(train_payload),canonical(questions),initial,control_adapter,reference)
    if token in worker or len(worker)>5_000_000:raise ValueError("NO_ACCOUNT_CREDENTIALS_IN_PRIVATE_WORKER")
    compile(worker,"joint_rule_worker","exec")
    metadata={"id":owner+"/"+RUN_SLUG,"title":RUN_SLUG,"code_file":"worker.py","language":"python","kernel_type":"script",
              "is_private":"true","enable_gpu":"true","enable_internet":"true","machine_shape":"NvidiaTeslaT4",
              "dataset_sources":[],"model_sources":[],"competition_sources":[],"kernel_sources":[]}
    with tempfile.TemporaryDirectory() as td:
        folder=Path(td);(folder/"worker.py").write_text(worker);(folder/"worker.py").chmod(0o600)
        (folder/"kernel-metadata.json").write_text(json.dumps(metadata))
        result=subprocess.run(["kaggle","kernels","push","-p",td,"--accelerator","NvidiaTeslaT4","--timeout",str(LIMITS["maximum_kernel_seconds"])],capture_output=True,text=True,timeout=180)
        if result.returncode:
            print(safe_submission_error(result.stdout+result.stderr,token),flush=True)
            raise RuntimeError("PRIVATE_JOINT_GPU_SUBMISSION_REJECTED")
    print("PRIVATE_JOINT_FREE_DUAL_T4_SUBMITTED",json.dumps({"source_commit":revision,"initial_sha256":INITIAL_SHA,
          "training_cases":design["counts"]["train"],"limits":LIMITS,"execution_allowed":False}),flush=True)
    for i in range(750):
        state=str(api.kernels_status(owner+"/"+RUN_SLUG).status).upper()
        if i%15==0:print("JOINT_GPU_STATE",state,flush=True)
        if any(s in state for s in ("COMPLETE","ERROR","CANCEL")):break
        time.sleep(20)
    result=subprocess.run(["kaggle","kernels","output",owner+"/"+RUN_SLUG,"-p",str(dest),"--file-pattern",r".*(JOINT_RULES_20261106[.]zip|joint_rule_failure[.]json)$"],capture_output=True,text=True,timeout=420)
    archive=dest/ARCHIVE
    if result.returncode or not archive.is_file() or "COMPLETE" not in state:
        raise RuntimeError("JOINT_GPU_INCOMPLETE_PARTIAL_EVIDENCE_ONLY")
    proof=verify_archive(archive,dest,revision=revision,design=design,groups=groups)
    print("VERIFIED_JOINT_RULE_GPU",json.dumps({"actual_updates":proof["actual_updates"],"selected_checkpoint":proof["selected_checkpoint"],
          "correct":{g:{s:v["correct"] for s,v in scores.items()} for g,scores in proof["grades"].items()},
          "pilot_research_gates_passed":proof["pilot_research_gates_passed"]}),flush=True)


if __name__=="__main__":main()
