"""Submit 50 sequential 20-step QLoRA micro-trainings in ONE bounded free Kaggle dual-T4 session.\nEach phase transfers weights to the next; one final untouched holdout CPU evaluation.\n"""
from __future__ import annotations

import gzip
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
from scripts.ai_run_joint_rule_kaggle import outputs, require_new_kernel
from scripts.ai_run_restorative_kaggle import safe_submission_error
from scripts.ai_sealed_source_transport import open_envelope
from scripts.ai_verify_serial_50 import audit_keys, verify
from williams_ai.deep_book_research import checked_payload
from williams_ai.historical_rule_curriculum import canonical, digest, prompts_only
from williams_ai.market_decision_curriculum import build_curriculum
from williams_ai.serial_50_gpu import RUN

DESIGN = Path("experiments/market-decision-20261107-design.json")
DATA = Path("examples/market_decision_20261107")
START = Path("experiments/serial-50-start.json")
PARENT_KERNEL = "williams-ai-deep-corrective-20261110-r2"
PARENT_SOURCE_COMMIT = "e427de188591136cbc022c4c26ae332a77fac8f3"
PREFIX = "historical_rule_qwen4b"
ARCHIVE = "Williams_Qwen3_4B_SERIAL_50_20261110.zip"
DOMAIN = b"Williams AI private model artifact v1"


def frozen():
    """Frozen train/dev/test, plus never disclosed numeric final exam in this run.

    Old chart and arithmetic exams are disclosed regression cases. The 32 new
    deterministic arithmetic samples are retained as a *separate* test supplement.
    None of the test answers is ever packaged into the Kaggle worker.
    """
    from scripts.ai_run_deep_corrective_v2_kaggle import frozen as v2_frozen
    from williams_ai.joint_rule_curriculum import boundary_cases
    from williams_ai.restorative_book_practice import restorative_cases
    groups, design = v2_frozen()
    # Source upload cap is ~1MB. Sample TRAIN deterministically and
    # keep EVERY original validation + test untouched, including four books.
    sizes = {"context": 144, "waves": 144, "campaign": 192, "precision": 384}
    for cohort, count in sizes.items():
        eligible = groups["train"][cohort]
        if len(eligible) < count:
            raise ValueError("INSUFFICIENT_REPRESENTATIVE_TRAIN_ROWS")
        groups["train"][cohort] = sorted(
            eligible,
            key=lambda r: digest({"id": r["case_id"], "seed": "serial50-train-20261110"})
        )[:count]
    # Risk and AC are critical: ensure both are represented in the
    # 384-row precision selection, before any GPU work.
    by_rule = {rule:sum(r["rule_id"] == rule for r in groups["train"]["precision"])
               for rule in ("TC1.PSY.PLAN", "NTD.AC")}
    if min(by_rule.values()) < 100:
        raise ValueError("RISK_AC_TRAINING_BALANCE_REQUIRED")
    new = []
    for factory, salt in ((boundary_cases, 2026111503),
                          (restorative_cases, 2026111619)):
        for attempt in range(24):
            try:
                rows = factory("test", seed_override=salt + 1009*attempt,
                               case_namespace="serial50-fresh-holdout")
                break
            except ValueError as exc:
                if str(exc) != "UNIQUE_JOINT_QUESTIONS_REQUIRED":
                    raise
        else:
            raise ValueError("FRESH_HOLDOUT_SEED_EXHAUSTED")
        new.extend([r for r in rows if r["rule_id"] in ("TC1.PSY.PLAN","NTD.AC")])
    fresh_risk = [r for r in new if r["rule_id"]=="TC1.PSY.PLAN"][:16]
    fresh_ac = [r for r in new if r["rule_id"]=="NTD.AC"][:16]
    if len(fresh_risk) != 16 or len(fresh_ac) != 16:
        raise ValueError("FRESH_32_FINAL_TEST_KEYS_REQUIRED")
    groups["test"]["precision"].extend(fresh_risk + fresh_ac)
    ids=[r["case_id"] for split in ("train","validation","test")
         for c in RUN.cohorts for r in groups[split][c]]
    if len(ids) != len(set(ids)):
        raise ValueError("SERIAL_50_NO_TRAIN_DEV_TEST_CASE_ID_OVERLAP")
    design["mode"] = RUN.mode
    design["limits"] = RUN.limits
    design["protocol"] = RUN.protocol
    design["model_parent_sha256"] = RUN.initial_sha
    design["exam_cohorts_by_gpu_rank"] = [list(x) for x in RUN.rank_cohorts]
    design["curriculum"]["splits"] = {
        split:{cohort:{"questions":len(rows),
                       "prompt_sha256":digest(prompts_only(rows)),
                       "answer_sha256":digest([r["answer"] for r in rows])}
               for cohort,rows in groups[split].items()}
        for split in ("train","validation","test")
    }
    design["serial_50_contract"]={
        "experiment":"ONE_KAGGLE_SESSION_FIFTY_SEQUENTIAL_FOCUSED_20_STEP_TRAININGS",
        "independent_1024_step_kaggle_runs":False,
        "micro_rounds":50,"optimizer_attempts_per_round":20,
        "total_max_optimizer_attempts":1000,
        "focus_rotation":["precision","context","waves","campaign","books"],
        "resume_lora_weights_between_micro_rounds":True,
        "optimizer_continuous":True,
        "new_numeric_exam_questions":32,
        "test_answers_never_sent_to_gpu":True,
        "old_chart_and_numeric_test_is_revealed_regression":True,
        "expert_labeled_market_data":False,
        "execution_allowed":False
    }
    audit_keys(groups)
    return groups, design

def worker_source(revision, source, curriculum, questions, initial, reference):
    return (historical_worker(revision, source, curriculum, questions, initial, initial, reference)
            .replace("from williams_ai.historical_rule_gpu import train", "from williams_ai.serial_50_gpu import train")
            .replace("HISTORICAL_RULES_20261029", "SERIAL_50_20261110")
            .replace("historical_rule_failure.json", "serial_50_failure.json")
            .replace("HISTORICAL_GPU", "DEEP_CORRECTIVE_GPU")
            .replace("    if set(config)!=", "    if name=='previous':\n        shutil.rmtree(root);root.symlink_to(private/'initial',target_is_directory=True);continue\n    if set(config)!="))


def _key(shared):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=DOMAIN).derive(shared)


def seal_archive(source, target, public_key):
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    ephemeral = X25519PrivateKey.generate()
    sender = ephemeral.public_key().public_bytes_raw()
    nonce = os.urandom(12)
    key = _key(ephemeral.exchange(X25519PublicKey.from_public_bytes(bytes.fromhex(public_key))))
    target.write_bytes(sender + nonce + AESGCM(key).encrypt(nonce, source.read_bytes(), DOMAIN))


def open_archive(source, target, private_key_file):
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    encrypted = source.read_bytes()
    receiver = X25519PrivateKey.from_private_bytes(private_key_file.read_bytes())
    key = _key(receiver.exchange(X25519PublicKey.from_public_bytes(encrypted[:32])))
    payload = AESGCM(key).decrypt(encrypted[32:44], encrypted[44:], DOMAIN)
    with target.open("xb") as out:
        out.write(payload)


def _parent(api, owner):
    prior = outputs(api, owner, PARENT_KERNEL)
    with urlopen(prior[PREFIX + "/research_receipt.json"].url, timeout=60) as response:
        raw = response.read()
    if not raw or len(raw) > 1_000_000:
        raise ValueError("BOUNDED_PREVIOUS_RECEIPT_REQUIRED")
    receipt = json.loads(raw)
    if (receipt.get("candidate_adapter_sha256") != RUN.initial_sha
            or receipt.get("source_commit") != PARENT_SOURCE_COMMIT
            or receipt.get("actual_updates") != 1024
            or receipt.get("selected_checkpoint") != "checkpoint-768"
            or receipt.get("world_size") != 2
            or any("T4" not in x for x in receipt.get("gpu_names", ()))
            or receipt.get("complete_training") is not True
            or receipt.get("complete_test") is not True
            or receipt.get("execution_allowed") is not False
            or receipt.get("trading_qualification") is not False):
        raise ValueError("VERIFIED_PARENT_CANDIDATE_REQUIRED")
    initial = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        url = prior[PREFIX + "/candidate_adapter/" + name].url
        with urlopen(url, timeout=240) as response:
            actual = sha256(response.read()).hexdigest()
        if name.endswith(".safetensors") and actual != RUN.initial_sha:
            raise ValueError("PREVIOUS_ADAPTER_WEIGHT_HASH_MISMATCH")
        initial[name] = {"url": url, "sha256": actual}
    reference = {"candidate_adapter_sha256": RUN.initial_sha, "source_audit": receipt["source_audit"],
                 "source_final_loss": receipt["source_final_loss"], "practice_final_loss": receipt["practice_final_loss"],
                 "source_baseline_loss": receipt["source_final_loss"], "practice_baseline_loss": receipt["practice_final_loss"],
                 "validation_references": {**receipt["references"], "parent_source": receipt["source_final_loss"],
                                           "parent_practice": receipt["practice_final_loss"]}}
    return initial, reference


def main(*, recover=False):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest

    token = os.environ["KAGGLE_API_TOKEN"]
    revision = os.environ["GITHUB_SHA"]
    dest = Path(os.environ["RUNNER_TEMP"]) / "serial_50_export"
    dest.mkdir(mode=0o700)
    groups, design = frozen()
    (dest / "frozen_design.json").write_bytes(canonical(design))
    config = json.loads(START.read_bytes())
    if (set(config) != {"run_slug", "recipient_key_mode"}
            or config["run_slug"] != RUN.slug
            or config["recipient_key_mode"] != "KAGGLE_RUNNER_SECRET_DERIVED_X25519"):
        raise ValueError("EXPLICIT_ONE_RUN_AND_RECOVERABLE_ENCRYPTION_REQUIRED")
    from scripts.ai_sealed_source_transport import runner_key
    # Never log key material. Derived key is recoverable by the account owner
    # while the pre-existing Kaggle Actions secret remains unchanged.
    recipient_public_key = runner_key(token).public_key().public_bytes_raw().hex()
    if len(bytes.fromhex(recipient_public_key)) != 32:
        raise ValueError("EXPLICIT_PRIVATE_ONE_RUN_DESTINATION_REQUIRED")
    api = KaggleApi()
    api.authenticate()
    owner = get_username()
    kernel = owner + "/" + RUN.slug
    if not recover:
        with api.build_kaggle_client() as client:
            quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
            remaining = (quota.total_time_allowed - quota.time_used).total_seconds() if quota else None
        print("SERIAL_50_FREE_GPU_HOURS", round(remaining / 3600, 3) if remaining is not None else None, flush=True)
        if remaining is None or remaining < RUN.limits["minimum_free_gpu_seconds"]:
            raise ValueError("CONFIRMED_FREE_GPU_RESERVE_REQUIRED")
        with urlopen(RUN.control_url + "?preflight=" + str(int(time.time())), timeout=20) as response:
            if json.load(response) != {"run_slug": RUN.slug, "stop": False}:
                raise ValueError("USER_STOP_NO_GPU_SUBMISSION")
        require_new_kernel(api, owner, RUN.slug)
        initial, reference = _parent(api, owner)
        source = open_envelope(json.loads(Path("experiments/private-source-20261027-course-full.sealed.json").read_bytes()), token)
        if checked_payload(json.loads(source))["audit"] != reference["source_audit"]:
            raise ValueError("UNCHANGED_PRIVATE_BOOK_PARTITIONS_REQUIRED")
        payload = {"schema": RUN.mode, "train": groups["train"], "validation": groups["validation"],
                   "audit": {s: {g: digest(prompts_only(rows)) for g, rows in groups[s].items()}
                             for s in ("train", "validation")}}
        questions = {g: prompts_only(rows) for g, rows in groups["test"].items()}
        questions["sha256"] = {g: digest(rows) for g, rows in questions.items()}
        worker = worker_source(revision, source, canonical(payload), canonical(questions), initial, reference)
        if token in worker or len(worker) > 5_000_000:
            raise ValueError("NO_ACCOUNT_CREDENTIALS_IN_PRIVATE_WORKER")
        compile(worker, "serial_50_worker", "exec")
        worker_bytes = len(worker.encode("utf-8"))
        print("SERIAL_50_PRIVATE_WORKER_BYTES", worker_bytes, flush=True)
        print("SERIAL_50_SPLIT_ROWS", json.dumps({
            split: {cohort: len(rows) for cohort, rows in groups[split].items()}
            for split in ("train", "validation", "test")
        }, sort_keys=True), flush=True)
        if os.environ.get("WILLIAMS_GPU_DRY_RUN") == "1":
            print("SERIAL_50_DRY_RUN_NO_SUBMISSION", json.dumps({
                "code_bytes": worker_bytes, "under_975k": worker_bytes < 975000,
                "existing_kernel_absent": True, "free_quota_verified": True,
                "parent_adapter_verified": True, "private_book_corpus_verified": True,
                "execution_allowed": False}, sort_keys=True), flush=True)
            return
        if worker_bytes >= 975000:
            raise ValueError("KAGGLE_SOURCE_SIZE_TOO_LARGE_MUST_SHRINK_PRIVATE_PAYLOAD")
        metadata = {"id": kernel, "title": RUN.slug, "code_file": "worker.py", "language": "python",
                    "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
                    "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                    "model_sources": [], "competition_sources": [], "kernel_sources": []}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "worker.py").write_text(worker)
            (root / "worker.py").chmod(0o600)
            (root / "kernel-metadata.json").write_text(json.dumps(metadata))
            push = subprocess.run(["kaggle", "kernels", "push", "-p", td, "--accelerator", "NvidiaTeslaT4",
                                  "--timeout", str(RUN.limits["maximum_kernel_seconds"])],
                                  capture_output=True, text=True, timeout=180)
            if push.returncode:
                print(safe_submission_error(push.stdout + push.stderr, token), flush=True)
                raise RuntimeError("PRIVATE_FREE_GPU_SUBMISSION_REJECTED")
        print("PRIVATE_DUAL_T4_SERIAL_50_SUBMITTED", revision, flush=True)
    else:
        revision = os.environ["WILLIAMS_PINNED_GPU_SOURCE"]
    state = "UNKNOWN"
    for attempt in range(720):
        state = str(api.kernels_status(kernel).status).upper()
        if attempt % 9 == 0:
            print("SERIAL_50_GPU_STATE", state, flush=True)
        if any(word in state for word in ("COMPLETE", "ERROR", "CANCEL")):
            break
        time.sleep(20)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        result = subprocess.run(["kaggle", "kernels", "output", kernel, "-p", td, "--file-pattern",
                                 r".*(SERIAL_50_20261110[.]zip|serial_50_(?:rank[01]_)?failure[.]json)$"],
                                capture_output=True, text=True, timeout=420)
        archive = root / ARCHIVE
        if result.returncode or not archive.is_file() or "COMPLETE" not in state:
            # Only safe numeric diagnostic metadata, no partial model promoted.
            for failure in root.glob("serial_50_*failure.json"):
                (dest / failure.name).write_bytes(failure.read_bytes())
            raise RuntimeError("SERIAL_50_GPU_INCOMPLETE_NO_TRAINING_CLAIM")
        proof = verify(archive, groups, design, revision)
        (dest / "independent_market_decision_audit.json").write_bytes(canonical(proof))
        seal_archive(archive, dest / (ARCHIVE + ".sealed"), recipient_public_key)
    print("INDEPENDENTLY_VERIFIED_SERIAL_50_TRAINING", json.dumps({
        "updates": proof["actual_updates"], "checkpoint": proof["selected_checkpoint"],
        "scores": {g: {s: v["correct"] for s, v in scores.items()} for g, scores in proof["grades"].items()},
        "research_gate_passed": proof["research_gate_passed"]}), flush=True)


if __name__ == "__main__":
    main(recover=sys.argv[1:] == ["--recover"])
