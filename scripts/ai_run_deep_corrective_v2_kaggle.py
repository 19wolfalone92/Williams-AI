"""Submit one bounded free GPU continuation and export independently graded evidence."""
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
from scripts.ai_verify_deep_corrective_v2 import audit_keys, verify
from williams_ai.deep_book_research import checked_payload
from williams_ai.historical_rule_curriculum import canonical, digest, prompts_only
from williams_ai.market_decision_curriculum import build_curriculum
from williams_ai.deep_corrective_v2_gpu import RUN

DESIGN = Path("experiments/market-decision-20261107-design.json")
DATA = Path("examples/market_decision_20261107")
START = Path("experiments/deep-corrective-v2-start.json")
PARENT_KERNEL = "williams-ai-deep-corrective-20261110-r1"
PARENT_SOURCE_COMMIT = "3f140b36a11628b7cd2ea2add29347903c808562"
PREFIX = "historical_rule_qwen4b"
ARCHIVE = "Williams_Qwen3_4B_DEEP_CORRECTIVE_V2_20261110.zip"
DOMAIN = b"Williams AI private model artifact v1"


def frozen():
    # V1's already disclosed chart/precision exam remains regression only.
    # This iteration prepares NEW numeric holdouts and authored TRAIN-only
    # decomposition lessons; no evaluation label or future candle is fed to GPU.
    from copy import deepcopy
    from scripts.ai_run_deep_corrective_kaggle import frozen as frozen_v1
    from williams_ai.joint_rule_curriculum import boundary_cases
    from williams_ai.restorative_book_practice import restorative_cases
    groups, design = frozen_v1()
    # The 1,728 precision examples in the previous attempt contained
    # heavily repeated machine-authored risk/AC templates. Draw an immutable
    # representative sample from TRAIN only (320 per rule), preserving every
    # validation and test case unchanged. This is *not* market-data thinning.
    old_precision = groups["train"]["precision"]
    selected_precision = []
    for rule in ("TC1.PSY.PLAN", "NTD.AC"):
        cohort_rows = sorted(
            (r for r in old_precision if r["rule_id"] == rule),
            key=lambda row: digest({"id": row["case_id"], "salt": "v2-compact-precision"})
        )
        if len(cohort_rows) < 320:
            raise ValueError("NOT_ENOUGH_RISK_AC_SOURCE_EXAMPLES")
        selected_precision.extend(cohort_rows[:320])
    groups["train"]["precision"] = selected_precision
    for cohort, subsets in {
        "context": (("ac_side", "ac_required", "ntd_next"),
                    ("h1_stack", "h4_d1_agree", "course_rhythm")),
        "waves": (("compatible", "reject_a", "reject_b"),
                  ("w4_zero_a", "w4_zero_b", "squat_verified", "wave_confirmed")),
        "campaign": (("decision", "rule", "side"),
                     ("trigger", "stop")),
    }.items():
        lessons = []
        # Only TRAIN rows. Source chart windows retain original as-of cutoff.
        # Sample a deterministic, stratified-by-cohort subset instead of
        # embedding duplicate component prompts for every historical case.
        # Kaggle limits a kernel source to <1 MB. Previously 1,079,099 bytes
        # triggered HTTP 400 before GPU ever started. We preserve every
        # original training row and all test/validation questions unchanged.
        originals = tuple(groups["train"][cohort])
        selected = sorted(
            originals, key=lambda row: digest(
                {"cohort": cohort, "case_id": row["case_id"], "seed": "v2-micro-20261110"}
            )
        )[:24]
        for row in selected:
            expected = json.loads(row["answer"])
            for index, fields in enumerate(subsets):
                if not set(fields) <= set(expected):
                    raise ValueError("INCOMPLETE_CORRECTIVE_COMPONENT_REFERENCE")
                lesson = deepcopy(row)
                lesson["case_id"] += "-v2-components-" + str(index)
                answer = {name: expected[name] for name in fields}
                answer["execution_allowed"] = False
                prompt = row["question"].rsplit("\nВерни только JSON", 1)[0]
                lesson["question"] = (
                    "Учебная декомпозиция правила. Рассчитай ТОЛЬКО поля "
                    + ", ".join(fields) + ",execution_allowed. "
                    + "Не представляй это как торговый приказ.\n"
                    + prompt + "\nОтвет строго JSON без других полей; execution_allowed=false."
                )
                lesson["answer"] = canonical(answer).decode()
                lesson["label_origin"] = (
                    "TRAIN_ONLY_MACHINE_CALCULATED_COMPONENTS_NOT_EXPERT_TRADING_LABEL"
                )
                lessons.append(lesson)
        groups["train"][cohort].extend(lessons)
    for split, seed in (("train", 2026111101),
                        ("validation", 2026111102),
                        ("test", 2026111103)):
        # Discard parameter seeds that create duplicate authored questions;
        # this decision depends on TRAIN/TEST sample uniqueness only, never
        # on held-out model answers or observed evaluation scores.
        for attempt in range(32):
            try:
                boundary = boundary_cases(
                    split, seed_override=seed + 1009*attempt,
                    case_namespace="deep-v2-boundary")
                break
            except ValueError as exc:
                if str(exc) != "UNIQUE_JOINT_QUESTIONS_REQUIRED":
                    raise
        else:
            raise ValueError("NO_UNIQUE_PRECOMMITTED_NUMERIC_BOUNDARY_SEED")
        numerical = restorative_cases(split, seed_override=seed+970,
                                      case_namespace="deep-v2-numeric")
        rows = [r for r in boundary+numerical
                if r["rule_id"] in ("TC1.PSY.PLAN", "NTD.AC")]
        if split == "train":
            # The previous 2304 precision train samples already include
            # large amounts of machine-generated risk/AC arithmetic.
            # Retain 192 diverse new V2 lessons, avoiding duplicate-heavy
            # supervised overfitting and the Kaggle source upload cap.
            risk = sorted(
                (r for r in rows if r["rule_id"] == "TC1.PSY.PLAN"),
                key=lambda r: digest(r["case_id"])
            )[:64]
            ac = sorted(
                (r for r in rows if r["rule_id"] == "NTD.AC"),
                key=lambda r: digest(r["case_id"])
            )[:64]
            rows = risk + ac
            if len(rows) != 128:
                raise ValueError("V2_NUMERIC_CURRICULUM_NOT_DIVERSE_ENOUGH")
        if split == "test":
            # New machine-checked arithmetic holdout; entirely withheld keys.
            risk = [r for r in rows if r["rule_id"] == "TC1.PSY.PLAN"][:16]
            ac = [r for r in rows if r["rule_id"] == "NTD.AC"][:16]
            rows = risk+ac
            if len(rows) != 32:
                raise ValueError("EXACT_32_NEW_NUMERIC_HOLDOUT_REQUIRED")
        groups[split]["precision"].extend(rows)
    ids = [r["case_id"] for split in ("train", "validation", "test")
           for cohort in RUN.cohorts for r in groups[split][cohort]]
    if len(ids) != len(set(ids)):
        raise ValueError("NO_OVERLAPPING_CASE_IDS_ACROSS_CORRECTIVE_SPLITS")
    design["mode"] = RUN.mode
    design["limits"] = RUN.limits
    design["protocol"] = RUN.protocol
    design["model_parent_sha256"] = RUN.initial_sha
    design["exam_cohorts_by_gpu_rank"] = [list(x) for x in RUN.rank_cohorts]
    design["curriculum"]["splits"] = {
        split: {cohort: {"questions": len(rows),
                        "prompt_sha256": digest(prompts_only(rows)),
                        "answer_sha256": digest([r["answer"] for r in rows])}
                for cohort, rows in groups[split].items()}
        for split in ("train", "validation", "test")
    }
    design["correction_audit"] = {
        "mode": "V2_TRAIN_ONLY_DECOMPOSITION_AND_NEW_PRECISION_HOLDOUT",
        "historical_chart_exam_revealed_regression_only": True,
        "all_micro_lessons_train_only": True,
        "new_holdout_seed": 2026111103,
        "new_numeric_holdout_count": 32,
        "human_expert_wave_labels": False,
        "test_answer_keys_sent_to_gpu": False,
        "execution_allowed": False,
    }
    audit_keys(groups)
    return groups, design

def worker_source(revision, source, curriculum, questions, initial, reference):
    return (historical_worker(revision, source, curriculum, questions, initial, initial, reference)
            .replace("from williams_ai.historical_rule_gpu import train", "from williams_ai.deep_corrective_v2_gpu import train")
            .replace("HISTORICAL_RULES_20261029", "DEEP_CORRECTIVE_V2_20261110")
            .replace("historical_rule_failure.json", "deep_corrective_v2_failure.json")
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
            or receipt.get("selected_checkpoint") != "checkpoint-1024"
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
    dest = Path(os.environ["RUNNER_TEMP"]) / "deep_corrective_v2_export"
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
        print("DEEP_CORRECTIVE_V2_FREE_GPU_HOURS", round(remaining / 3600, 3) if remaining is not None else None, flush=True)
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
        compile(worker, "deep_corrective_v2_worker", "exec")
        worker_bytes = len(worker.encode("utf-8"))
        print("KAGGLE_PRIVATE_WORKER_BYTES", worker_bytes, flush=True)
        print("KAGGLE_PRIVATE_TRAIN_SPLIT_ROWS", json.dumps({
            split: {cohort: len(rows) for cohort, rows in groups[split].items()}
            for split in ("train", "validation", "test")
        }, sort_keys=True), flush=True)
        if os.environ.get("WILLIAMS_GPU_DRY_RUN") == "1":
            print("DRY_RUN_NO_KAGGLE_SUBMISSION", json.dumps({
                "code_bytes": worker_bytes, "under_950k": worker_bytes < 950000,
                "existing_kernel_absent": True, "free_quota_verified": True,
                "parent_adapter_verified": True, "private_book_corpus_verified": True,
                "execution_allowed": False}, sort_keys=True), flush=True)
            return
        if worker_bytes >= 950000:
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
        print("PRIVATE_DUAL_T4_DEEP_CORRECTIVE_V2_SUBMITTED", revision, flush=True)
    else:
        revision = os.environ["WILLIAMS_PINNED_GPU_SOURCE"]
    state = "UNKNOWN"
    for attempt in range(720):
        state = str(api.kernels_status(kernel).status).upper()
        if attempt % 9 == 0:
            print("DEEP_CORRECTIVE_V2_GPU_STATE", state, flush=True)
        if any(word in state for word in ("COMPLETE", "ERROR", "CANCEL")):
            break
        time.sleep(20)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        result = subprocess.run(["kaggle", "kernels", "output", kernel, "-p", td, "--file-pattern",
                                 r".*(DEEP_CORRECTIVE_V2_20261110[.]zip|deep_corrective_v2_(?:rank[01]_)?failure[.]json)$"],
                                capture_output=True, text=True, timeout=420)
        archive = root / ARCHIVE
        if result.returncode or not archive.is_file() or "COMPLETE" not in state:
            # Only safe numeric diagnostic metadata, no partial model promoted.
            for failure in root.glob("deep_corrective_v2_*failure.json"):
                (dest / failure.name).write_bytes(failure.read_bytes())
            raise RuntimeError("DEEP_CORRECTIVE_V2_GPU_INCOMPLETE_NO_TRAINING_CLAIM")
        proof = verify(archive, groups, design, revision)
        (dest / "independent_market_decision_audit.json").write_bytes(canonical(proof))
        seal_archive(archive, dest / (ARCHIVE + ".sealed"), recipient_public_key)
    print("INDEPENDENTLY_VERIFIED_DEEP_CORRECTIVE_V2_TRAINING", json.dumps({
        "updates": proof["actual_updates"], "checkpoint": proof["selected_checkpoint"],
        "scores": {g: {s: v["correct"] for s, v in scores.items()} for g, scores in proof["grades"].items()},
        "research_gate_passed": proof["research_gate_passed"]}), flush=True)


if __name__ == "__main__":
    main(recover=sys.argv[1:] == ["--recover"])
