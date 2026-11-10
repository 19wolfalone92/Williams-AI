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
from scripts.ai_verify_deep_corrective import audit_keys, verify
from williams_ai.deep_book_research import checked_payload
from williams_ai.historical_rule_curriculum import canonical, digest, prompts_only
from williams_ai.market_decision_curriculum import build_curriculum
from williams_ai.deep_corrective_gpu import RUN

DESIGN = Path("experiments/market-decision-20261107-design.json")
DATA = Path("examples/market_decision_20261107")
START = Path("experiments/deep-corrective-start.json")
PARENT_KERNEL = "williams-ai-joint-rules-20261106"
PARENT_RECEIPT = "92a31ef53c29a9e8360dfffd339fe46469260475500f2a9472d42180bc7c72f2"
PREFIX = "historical_rule_qwen4b"
ARCHIVE = "Williams_Qwen3_4B_DEEP_CORRECTIVE_20261110.zip"
DOMAIN = b"Williams AI private model artifact v1"


def frozen():
    design = json.loads(DESIGN.read_bytes())
    history = {}
    for split in ("train", "validation", "test"):
        path = DATA / (split + ".json.gz")
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != design["pack_sha256"][split]:
            raise ValueError("FROZEN_BINANCE_PACK_SHA256_MISMATCH")
        history[split] = json.loads(gzip.decompress(raw))
    groups, regenerated = build_curriculum(history)
    if regenerated != design["curriculum"]:
        raise ValueError("PRECOMMITTED_CHART_PROMPTS_OR_KEYS_CHANGED")
    # All pre-existing market windows remain causal. Correct only the TRAIN
    # and development sets, and add an independent numeric TEST supplement.
    # Historical 2026 chart results are a revealed regression, NOT blind market gold.
    from copy import deepcopy
    from williams_ai.joint_rule_curriculum import boundary_cases
    from williams_ai.restorative_book_practice import restorative_cases
    from williams_ai.market_decision_curriculum import _campaign, _CAMPAIGN_QUESTION
    for split, seed in (("train", 2026111001), ("validation", 2026111002), ("test", 2026111003)):
        boundary = boundary_cases(split, seed_override=seed,
                                  case_namespace="deep-correction-boundary")
        restorative = restorative_cases(split, seed_override=seed + 320,
                                        case_namespace="deep-correction-restorative")
        direct = [row for row in boundary + restorative
                  if row["rule_id"] in ("TC1.PSY.PLAN", "NTD.AC")]
        if split == "test":
            # The other chart cohorts remain historical regression cases.
            risk = [r for r in direct if r["rule_id"] == "TC1.PSY.PLAN"][:8]
            ac = [r for r in direct if r["rule_id"] == "NTD.AC"][:8]
            direct = risk + ac
        groups[split]["precision"].extend(direct)
        if split != "test":
            stop_rows=[]
            # Never use validation/test chart labels to construct lessons.
            for original in groups[split]["campaign"]:
                journal=original["operands"]["journal"]
                if not journal["fills"] or not journal["stop_ack"]:
                    continue
                row=deepcopy(original)
                row["case_id"] += "-stop-unacknowledged"
                row["operands"]["journal"]["stop_ack"]=False
                expected=_campaign(row["operands"])
                if expected["decision"] != "AUDIT_PROTECTION":
                    raise ValueError("UNACKNOWLEDGED_STOP_MUST_BLOCK_ALL_ACTIONS")
                row["question"]=_CAMPAIGN_QUESTION + "\n" + canonical(row["operands"]).decode() + "\nВерни только JSON с полями " + ",".join(expected) + ",execution_allowed. execution_allowed=false."
                row["answer"]=canonical({**expected, "execution_allowed":False}).decode()
                row["label_origin"]="MACHINE_CHECKED_STOP_ACK_VARIANT_NOT_INDEPENDENT_MARKET_CHART"
                stop_rows.append(row)
                if split=="validation" and len(stop_rows)>=8:
                    break
            groups[split]["campaign"].extend(stop_rows)
    all_ids = [r["case_id"] for split in ("train","validation","test")
               for cohort in RUN.cohorts for r in groups[split][cohort]]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("CORRECTIVE_TRAIN_VALIDATION_TEST_IDS_MUST_BE_DISJOINT")
    design["curriculum"]["splits"] = {
        split: {cohort: {"questions":len(rows),
                       "prompt_sha256":digest(prompts_only(rows)),
                       "answer_sha256":digest([r["answer"] for r in rows])}
                for cohort,rows in groups[split].items()}
        for split in ("train","validation","test")
    }
    design["correction_audit"]={
        "mode":"AUTHOR_VERIFIED_DETERMINISTIC_TRAINING_AND_NEW_NUMERIC_EXAM",
        "seeds":{"train":2026111001,"validation":2026111002,"test":2026111003},
        "new_stop_ack_training_only":True,"human_expert_wave_labels":False,
        "old_chart_exam_is_revealed_regression":True,
        "new_numeric_exam_keys_never_sent_to_gpu":True,
        "execution_allowed":False}
    audit_keys(groups)
    return groups, design


def worker_source(revision, source, curriculum, questions, initial, reference):
    return (historical_worker(revision, source, curriculum, questions, initial, initial, reference)
            .replace("from williams_ai.historical_rule_gpu import train", "from williams_ai.deep_corrective_gpu import train")
            .replace("HISTORICAL_RULES_20261029", "DEEP_CORRECTIVE_20261110")
            .replace("historical_rule_failure.json", "deep_corrective_failure.json")
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
    if sha256(raw).hexdigest() != PARENT_RECEIPT:
        raise ValueError("EXACT_PREVIOUS_COMPLETED_RESEARCH_RECEIPT_REQUIRED")
    receipt = json.loads(raw)
    if (receipt["candidate_adapter_sha256"] != RUN.initial_sha or receipt["complete_test"] is not True):
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
    dest = Path(os.environ["RUNNER_TEMP"]) / "deep_corrective_export"
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
        print("DEEP_CORRECTIVE_FREE_GPU_HOURS", round(remaining / 3600, 3) if remaining is not None else None, flush=True)
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
        compile(worker, "market_decision_worker", "exec")
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
        print("PRIVATE_DUAL_T4_DEEP_CORRECTIVE_SUBMITTED", revision, flush=True)
    else:
        revision = os.environ["WILLIAMS_PINNED_GPU_SOURCE"]
    state = "UNKNOWN"
    for attempt in range(720):
        state = str(api.kernels_status(kernel).status).upper()
        if attempt % 9 == 0:
            print("DEEP_CORRECTIVE_GPU_STATE", state, flush=True)
        if any(word in state for word in ("COMPLETE", "ERROR", "CANCEL")):
            break
        time.sleep(20)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        result = subprocess.run(["kaggle", "kernels", "output", kernel, "-p", td, "--file-pattern",
                                 r".*(DEEP_CORRECTIVE_20261110[.]zip|deep_corrective_(?:rank[01]_)?failure[.]json)$"],
                                capture_output=True, text=True, timeout=420)
        archive = root / ARCHIVE
        if result.returncode or not archive.is_file() or "COMPLETE" not in state:
            # Only safe numeric diagnostic metadata, no partial model promoted.
            for failure in root.glob("deep_corrective_*failure.json"):
                (dest / failure.name).write_bytes(failure.read_bytes())
            raise RuntimeError("DEEP_CORRECTIVE_GPU_INCOMPLETE_NO_TRAINING_CLAIM")
        proof = verify(archive, groups, design, revision)
        (dest / "independent_market_decision_audit.json").write_bytes(canonical(proof))
        seal_archive(archive, dest / (ARCHIVE + ".sealed"), recipient_public_key)
    print("INDEPENDENTLY_VERIFIED_DEEP_CORRECTIVE_TRAINING", json.dumps({
        "updates": proof["actual_updates"], "checkpoint": proof["selected_checkpoint"],
        "scores": {g: {s: v["correct"] for s, v in scores.items()} for g, scores in proof["grades"].items()},
        "research_gate_passed": proof["research_gate_passed"]}), flush=True)


if __name__ == "__main__":
    main(recover=sys.argv[1:] == ["--recover"])
