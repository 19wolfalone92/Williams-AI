"""Private free-GPU pilot combining original train pages and real OHLCV rules."""
from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ai_run_restorative_kaggle import packed, safe_submission_error
from scripts.ai_sealed_source_transport import open_envelope
from scripts.ai_verify_historical_rule_curriculum import independent_key, audit as audit_curriculum
from scripts.ai_verify_restorative_result import independent_key as numeric_key
from williams_ai.historical_rule_curriculum import (SCHEMA, FAMILIES, canonical, digest, load_pack,
    cases_from_pack, audit_cases, prompts_only, score)
from williams_ai.historical_rule_gpu import RUN_SLUG, CONTROL_URL, STAGES, PROTOCOL, selection
from williams_ai.restorative_book_practice import (INITIAL_SHA, PREVIOUS40_SHA, PREVIOUS_RECEIPT_SHA, restorative_cases)
from williams_ai.contrastive_book_practice import question_rows, score_responses
from williams_ai.deep_book_research import checked_payload, MODEL, REVISION, EDITIONS
from williams_ai.lora_integrity import validation_admission


def numeric_cases():
    return restorative_cases("test", seed_override=83047831, case_namespace="historical-books-20261029")


def worker_source(revision, source, curriculum, questions, initial, previous, reference):
    if not re.fullmatch(r"[a-f0-9]{40}", revision): raise ValueError("IMMUTABLE_WORKER_SOURCE_REQUIRED")
    template = '''import base64,hashlib,json,lzma,os,shutil,subprocess,sys,zipfile
from pathlib import Path
from urllib.request import urlopen
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"
os.environ["TRAINING_SOURCE_SHA"]=__REVISION__
repo=Path('/tmp/historical-rule-research')
subprocess.run(['git','clone','-q','https://github.com/19wolfalone92/Williams-AI.git',str(repo)],check=True)
subprocess.run(['git','checkout','-q',__REVISION__],cwd=repo,check=True)
shutil.rmtree(repo/'examples')
shutil.rmtree(repo/'experiments')
subprocess.run([sys.executable,'-m','pip','install','--quiet','--disable-pip-version-check','transformers==4.57.1','peft==0.17.1','accelerate==1.10.1','bitsandbytes==0.48.1','safetensors==0.6.2'],check=True)
private=Path('/tmp/historical-private-inputs');private.mkdir(mode=0o700)
for name,encoded in __PAYLOADS__.items():
    p=private/name;p.write_bytes(lzma.decompress(base64.b64decode(encoded)));p.chmod(0o600)
for name,config in __ADAPTERS__.items():
    root=private/name;root.mkdir(mode=0o700)
    if set(config)!={'adapter_config.json','adapter_model.safetensors'}:raise RuntimeError('ONLY_OWN_ADAPTER_FILES_ALLOWED')
    for filename,meta in config.items():
        raw=urlopen(meta['url'],timeout=240).read()
        if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise RuntimeError('OWN_ADAPTER_SHA_CHANGED')
        (root/filename).write_bytes(raw)
out=Path('/kaggle/working/historical_rule_qwen4b')
entry=private/'entry.py'
entry.write_text("from pathlib import Path\\nimport sys\\nsys.path.insert(0,"+repr(str(repo))+")\\nfrom williams_ai.historical_rule_gpu import train\\ntrain(*[Path(p) for p in "+repr([str(private/n) for n in ['source.json','curriculum.json','questions.json','initial','previous','reference.json']]+[str(out)])+"])\\n")
failure=None
try:subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(entry)],check=True)
except Exception as e:
    failure=type(e).__name__
    Path('/kaggle/working/historical_rule_failure.json').write_text(json.dumps({'error_type':failure,'actual_complete_training_claimed':False,'execution_allowed':False}))
finally:
    shutil.rmtree(private)
    if out.is_dir():
        with zipfile.ZipFile('/kaggle/working/Williams_Qwen3_4B_HISTORICAL_RULES_20261029.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file():
                    rel=p.relative_to(out)
                    if rel.parts[0] in {'source_adapter','candidate_adapter'} or (len(rel.parts)==1 and p.suffix=='.json'):z.write(p,str(rel))
if failure:raise RuntimeError('HISTORICAL_GPU_INCOMPLETE_PARTIAL_EVIDENCE_SAVED')
print('HISTORICAL_GPU_PROCESS_FINISHED_CHECK_RECEIPT',flush=True)
'''
    payloads = {"source.json": source, "curriculum.json": curriculum, "questions.json": questions, "reference.json": canonical(reference)}
    return (template.replace("__REVISION__", repr(revision))
        .replace("__PAYLOADS__", repr({k: packed(v) for k, v in payloads.items()}))
        .replace("__ADAPTERS__", repr({"initial": initial, "previous": previous})))


def verify_archive(archive, dest, *, revision, manifest, historical, numeric):
    import numpy as np
    from safetensors.numpy import load
    with zipfile.ZipFile(archive) as z:
        if len(set(z.namelist())) != len(z.namelist()) or any(Path(n).is_absolute() or ".." in Path(n).parts for n in z.namelist()):
            raise ValueError("SAFE_UNIQUE_RESULT_ARCHIVE_REQUIRED")
        receipt = json.loads(z.read("research_receipt.json"))
        if (receipt["source_commit"] != revision or receipt["model_repo"] != MODEL or receipt["model_revision"] != REVISION
                or receipt["initial_adapter_sha256"] != INITIAL_SHA or receipt["world_size"] != 2
                or receipt["execution_allowed"] is not False or receipt["complete_training"] is not True or receipt["complete_test"] is not True):
            raise ValueError("COMPLETE_REAL_HISTORICAL_GPU_PROVENANCE_REQUIRED")
        data = {s: z.read(s+"_adapter/adapter_model.safetensors") for s in ("source", "candidate")}
        if sha256(data["source"]).hexdigest() != INITIAL_SHA or sha256(data["candidate"]).hexdigest() != receipt["candidate_adapter_sha256"]:
            raise ValueError("SAVED_SOURCE_AND_CANDIDATE_WEIGHT_SHA_REQUIRED")
        tensors = {s: load(raw) for s, raw in data.items()}
        if set(tensors["source"]) != set(tensors["candidate"]): raise ValueError("UNCHANGED_LORA_MATRIX_SET_REQUIRED")
        for values in tensors.values():
            if len(values) != 504 or sum(v.size for v in values.values()) != 33030144 or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in values.values()):
                raise ValueError("ALL_504_FINITE_F32_LORA_MATRICES_REQUIRED")
        changed = sum(int(np.count_nonzero(v != tensors["source"][n])) for n, v in tensors["candidate"].items())
        if not changed: raise ValueError("REAL_LEARNED_WEIGHT_CHANGES_REQUIRED")
        if receipt["actual_updates"] <= 0 or receipt["optimizer_attempted_steps"] != receipt["actual_updates"] + receipt["amp_skipped_steps"]:
            raise ValueError("ACTUAL_OPTIMIZER_COUNTS_REQUIRED")
        development = cases_from_pack(load_pack(Path(manifest["packs"]["validation"]["path"])))
        for row in receipt["checkpoints"]:
            raw_grade = score(development, json.loads(z.read("development_"+str(row["optimizer_step"])+".json")))
            if raw_grade != row["grade"]: raise ValueError("SAVED_DEVELOPMENT_ANSWERS_MUST_RECOMPUTE")
            refs = receipt["references"]
            # Separate CPU arithmetic; do not trust a cached selection metric.
            regression = max(after[e]/before[e]-1 for after, before in (
                (row["source_loss"], refs["focused69_source"]), (row["source_loss"], refs["previous40_source"]),
                (row["practice_loss"], refs["focused69_practice"]), (row["practice_loss"], refs["previous40_practice"])) for e in EDITIONS)
            minimum = min(v["correct"]/v["total"] for v in raw_grade["per_rule"].values())
            collapsed = len(raw_grade["constant_fields_despite_varied_expected_values"])
            metric = -3-regression if regression>.02 else minimum+.1*raw_grade["accuracy"]-.1*collapsed
            recalculated = {"score": metric, "max_any_reference_book_regression": regression,
                "minimum_rule_accuracy": minimum, "collapsed_rules": collapsed, "preserved_within_two_percent": regression<=.02}
            if recalculated != row["selection"]: raise ValueError("DEVELOPMENT_SELECTION_MUST_RECOMPUTE")
        if (any(type(receipt[k]) is not int for k in ("actual_updates", "optimizer_attempted_steps", "amp_skipped_steps", "selected_actual_updates"))
                or not 0 < receipt["selected_actual_updates"] <= receipt["actual_updates"] <= receipt["optimizer_attempted_steps"]):
            raise ValueError("REAL_SELECTED_UPDATE_COUNTER_REQUIRED")
        best = max(receipt["checkpoints"], key=lambda x: x["selection"]["score"])
        if receipt["selected_checkpoint"] != "checkpoint-"+str(best["optimizer_step"]): raise ValueError("DEVELOPMENT_ONLY_CHECKPOINT_SELECTION_REQUIRED")
        for filename, field, total in (("source_validation_window_losses.json", "source_final_loss", 178), ("practice_validation_window_losses.json", "practice_final_loss", 122)):
            windows = json.loads(z.read(filename)); seen = set(); grouped = {e: [] for e in EDITIONS}
            for edition, index, loss in windows["window_losses"]:
                if (edition not in grouped or type(index) is not int or not 0 <= index < windows["window_counts"][edition]
                        or (edition, index) in seen or not math_is_finite_nonnegative(loss)):
                    raise ValueError("UNIQUE_FINITE_BOOK_WINDOW_LOSSES_REQUIRED")
                seen.add((edition, index)); grouped[edition].append(loss)
            if len(seen) != total: raise ValueError("FULL_ORIGINAL_VALIDATION_REQUIRED")
            from math import isclose
            values = {e: sum(grouped[e])/len(grouped[e]) for e in EDITIONS}
            values["ALL"] = sum(v for group in grouped.values() for v in group)/total
            if any(not isclose(v, receipt[field][e], rel_tol=1e-12, abs_tol=1e-12) for e, v in values.items()):
                raise ValueError("ORIGINAL_BOOK_LOSS_RECOMPUTATION_FAILED")
        predictions = json.loads(z.read("final_predictions.json"))
        identity = predictions["receipt"]
        if (identity["source_commit"] != revision or identity["candidate_adapter_sha256"] != receipt["candidate_adapter_sha256"]
                or identity["protocol"] != PROTOCOL or identity["expected_keys_loaded_on_gpu"] is not False
                or identity["test_questions_sha256"] != {"historical": digest(prompts_only(historical)), "numeric": digest(question_rows(numeric))}):
            raise ValueError("FROZEN_PROMPTS_ONLY_FINAL_PREDICTIONS_REQUIRED")
        if any(set(predictions[c]) != set(STAGES) for c in ("historical", "numeric")):
            raise ValueError("ALL_FOUR_COMPARISON_MODELS_REQUIRED")
        grades = {"historical": {s: score(historical, predictions["historical"][s]) for s in STAGES},
                  "numeric": {s: score_responses(numeric, predictions["numeric"][s]) for s in STAGES}}
        small = [r for family in ("TC1.PSY.PLAN", "NTD.AC") for r in [r for r in historical if r["rule_id"] == family][:4]]
        grades["thinking_diagnostic"] = {s: score(small, p) for s, p in predictions["thinking_diagnostic"].items()}
        gates = {name: validation_admission(before, receipt["source_final_loss"] if name.endswith("source") else receipt["practice_final_loss"]) for name, before in receipt["references"].items()}
        if gates != receipt["text_gates"]: raise ValueError("ALL_FOUR_ORIGINAL_TEXT_GATES_MUST_RECOMPUTE")
        transfer = {}
        for cohort in ("historical", "numeric"):
            candidate = grades[cohort]["candidate"]
            transfer[cohort] = (candidate["accuracy"] >= .8
                and min(v["correct"]/v["total"] for v in candidate["per_rule"].values()) >= .75
                and candidate["accuracy"] >= max(grades[cohort][s]["accuracy"] for s in STAGES if s != "candidate") + .05
                and not candidate["constant_fields_despite_varied_expected_values"])
        proof = {"mode": SCHEMA, "source_commit": revision, "archive_sha256": sha256(Path(archive).read_bytes()).hexdigest(),
            "candidate_adapter_sha256": receipt["candidate_adapter_sha256"], "changed_values": changed,
            "all_504_matrices_finite": True, "actual_updates": receipt["actual_updates"],
            "selected_checkpoint": receipt["selected_checkpoint"], "selected_actual_updates": receipt["selected_actual_updates"],
            "corpus_manifest_sha256": digest(manifest), "grades": grades, "text_gates": gates, "transfer_gates": transfer,
            "pilot_research_gates_passed": all(transfer.values()) and all(g["accepted"] for g in gates.values()),
            "prior_five_model_admission_not_replaced": True, "expert_chart_supervision_passed": False,
            "trading_qualification": False, "execution_allowed": False}
        for name in ("research_receipt.json", "final_predictions.json", "source_validation_window_losses.json", "practice_validation_window_losses.json"):
            (dest/name).write_bytes(z.read(name))
        (dest/"independent_historical_rule_audit.json").write_bytes(canonical(proof))
        return proof


def math_is_finite_nonnegative(value):
    from math import isfinite
    return type(value) in (int, float) and isfinite(value) and value >= 0


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest, ApiListKernelSessionOutputRequest
    token = os.environ["KAGGLE_API_TOKEN"]; revision = os.environ["GITHUB_SHA"]
    dest = Path(os.environ["RUNNER_TEMP"])/"historical_rule_results"; dest.mkdir(mode=0o700)
    manifest = json.loads(Path("experiments/historical-rule-curriculum-20261029-design.json").read_bytes())
    root = Path("examples/historical_rule_20261029")
    for split, info in manifest["packs"].items():
        if sha256(Path(info["path"]).read_bytes()).hexdigest() != info["sha256"]: raise ValueError("FROZEN_PUBLIC_MARKET_ARCHIVE_CHANGED")
    audited = audit_curriculum(root); (dest/"independent_curriculum_audit.json").write_bytes(canonical(audited))
    packs = {s: load_pack(root/(s+".json.gz")) for s in ("train", "validation", "test")}
    rows = {s: cases_from_pack(p) for s, p in packs.items()}
    if any(audit_cases(rows[s]) != manifest["cases"][s] for s in rows): raise ValueError("FROZEN_RULE_QUESTIONS_CHANGED")
    numeric = numeric_cases()
    if digest(question_rows(numeric)) != manifest["numeric_test_questions_sha256"]: raise ValueError("FROZEN_FRESH_NUMERIC_EXAM_CHANGED")
    for row, question in zip(numeric, question_rows(numeric)):
        if numeric_key(question) != json.loads(row["answer"]): raise ValueError("INDEPENDENT_NUMERIC_KEYS_FAILED")
    owner = get_username(); api = KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as client:
        quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
        remaining = (quota.total_time_allowed-quota.time_used).total_seconds() if quota else None
    (dest/"gpu_preflight.json").write_bytes(canonical({"free_gpu_hours_remaining": None if remaining is None else remaining/3600, "paid_fallback": False}))
    print("FREE_GPU_HOURS_REMAINING", None if remaining is None else round(remaining/3600, 4), flush=True)
    if remaining is not None and remaining < 10800: raise RuntimeError("THREE_FREE_GPU_HOURS_REQUIRED_NO_PAID_FALLBACK")
    with urlopen(CONTROL_URL+"?preflight="+str(int(time.time())), timeout=20) as response: control = json.load(response)
    if control != {"run_slug": RUN_SLUG, "stop": False}: raise RuntimeError("USER_STOP_NO_GPU_SUBMISSION")
    slug = "williams-ai-focused-2t4-20261027"
    if "complete" not in str(api.kernels_status(owner+"/"+slug).status).lower(): raise ValueError("COMPLETED_VERIFIED_FOCUSED69_REQUIRED")
    found = {}; page = None; seen = set()
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest(); request.user_name = owner; request.kernel_slug = slug
            if page: request.page_token = page
            result = client.kernels.kernels_api_client.list_kernel_session_output(request)
            found.update({f.file_name: f for f in result.files}); page = getattr(result, "next_page_token", None)
            if not page: break
            if page in seen: raise ValueError("OWN_OUTPUT_PAGINATION_LOOP")
            seen.add(page)
    raw_reference = urlopen(found["focused_qwen4b/research_receipt.json"].url, timeout=60).read()
    if sha256(raw_reference).hexdigest() != PREVIOUS_RECEIPT_SHA: raise ValueError("EXACT_VERIFIED_FOCUSED_RECEIPT_REQUIRED")
    reference = json.loads(raw_reference)
    def adapter(prefix, expected):
        config = {}
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            item = found["focused_qwen4b/"+prefix+"/"+name]
            data = urlopen(item.url, timeout=240).read(); actual = sha256(data).hexdigest()
            if name.endswith(".safetensors") and actual != expected: raise ValueError("EXACT_OWN_ADAPTER_SHA_REQUIRED")
            config[name] = {"url": item.url, "sha256": actual}
        return config
    initial = adapter("candidate_adapter", INITIAL_SHA); previous = adapter("source_adapter", PREVIOUS40_SHA)
    source = open_envelope(json.loads(Path("experiments/private-source-20261027-course-full.sealed.json").read_bytes()), token)
    if checked_payload(json.loads(source))["audit"] != reference["source_audit"]: raise ValueError("ORIGINAL_SOURCE_PARTITION_CHANGED")
    curriculum = canonical({"schema": SCHEMA, "train": rows["train"], "validation": rows["validation"], "audit": {s: audit_cases(rows[s]) for s in ("train", "validation")}})
    prompts = {"historical": prompts_only(rows["test"]), "numeric": question_rows(numeric)}
    prompts["sha256"] = {k: digest(v) for k, v in prompts.items()}
    worker = worker_source(revision, source, curriculum, canonical(prompts), initial, previous, reference)
    if token in worker or len(worker) > 3_000_000: raise ValueError("NO_ACCOUNT_CREDENTIAL_IN_GPU_WORKER")
    compile(worker, "historical_rule_worker", "exec")
    metadata = {"id": owner+"/"+RUN_SLUG, "title": RUN_SLUG, "code_file": "worker.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [], "model_sources": [], "competition_sources": [], "kernel_sources": []}
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td); (folder/"worker.py").write_text(worker); (folder/"worker.py").chmod(0o600)
        (folder/"kernel-metadata.json").write_text(json.dumps(metadata))
        result = subprocess.run(["kaggle", "kernels", "push", "-p", td, "--accelerator", "NvidiaTeslaT4", "--timeout", "8400"], capture_output=True, text=True, timeout=180)
        if result.returncode:
            print(safe_submission_error(result.stdout+result.stderr, token), flush=True); raise RuntimeError("PRIVATE_HISTORICAL_GPU_SUBMISSION_REJECTED")
    state = "UNKNOWN"
    for i in range(465):
        state = str(api.kernels_status(owner+"/"+RUN_SLUG).status).upper()
        if i % 15 == 0: print("HISTORICAL_BOOKS_GPU_STATE", state, flush=True)
        if any(s in state for s in ("COMPLETE", "ERROR", "CANCEL")): break
        time.sleep(20)
    result = subprocess.run(["kaggle", "kernels", "output", owner+"/"+RUN_SLUG, "-p", str(dest), "--file-pattern", r".*(HISTORICAL_RULES_20261029[.]zip|historical_rule_failure[.]json)$"], capture_output=True, text=True, timeout=420)
    archive = dest/"Williams_Qwen3_4B_HISTORICAL_RULES_20261029.zip"
    if result.returncode or not archive.is_file() or "COMPLETE" not in state:
        raise RuntimeError("HISTORICAL_GPU_INCOMPLETE_NO_SUCCESS_CLAIM")
    proof = verify_archive(archive, dest, revision=revision, manifest=manifest, historical=rows["test"], numeric=numeric)
    print("VERIFIED_HISTORICAL_RULE_GPU", json.dumps({"actual_updates": proof["actual_updates"], "selected_checkpoint": proof["selected_checkpoint"],
        "historical_correct": {s: g["correct"] for s, g in proof["grades"]["historical"].items()},
        "numeric_correct": {s: g["correct"] for s, g in proof["grades"]["numeric"].items()}, "pilot_research_gates_passed": proof["pilot_research_gates_passed"]}), flush=True)


if __name__ == "__main__": main()
