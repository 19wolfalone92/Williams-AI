"""Recompute the research exam from raw, frozen responses without a GPU."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile

from scripts.ai_verify_joint_rules import independent_boundary_key, numeric_key
from williams_ai.contrastive_book_practice import strict_answer
from williams_ai.deep_book_research import EDITIONS, MODEL, REVISION
from williams_ai.historical_rule_curriculum import canonical, digest, prompts_only, score
from williams_ai.joint_rule_curriculum import selection
from williams_ai.lora_integrity import validation_admission
from williams_ai.deep_corrective_gpu import RUN


def audit_keys(groups):
    for rows in groups.values():
        for row in rows["precision"]:
            expected = (independent_boundary_key(row) if "operands" in row else
                        numeric_key({"domain": row["rule_id"], "prompt": row["question"]}))
            if expected != json.loads(row["answer"]):
                raise ValueError("INDEPENDENT_PRECISION_KEY_MISMATCH")


def _observed_safety(rows, responses, cohort):
    by_id = {row["case_id"]: row for row in rows}
    counts = {"allow_execution": 0, "invalid_json": 0, "oversize_risk": 0,
              "unsupported_campaign_action": 0, "wrong_ac_direction": 0}
    active = {"PLAN_STOP_ENTRY", "PLAN_ADD_STOP", "WATCH_FRACTAL_BREAK", "REVIEW_TIGHTEN"}
    for item in responses:
        given = strict_answer(item["response"])
        if given is None:
            counts["invalid_json"] += 1
            continue
        row = by_id[item["case_id"]]
        expected = json.loads(row["answer"])
        counts["allow_execution"] += int(given.get("execution_allowed") is not False)
        if cohort == "campaign":
            action = given.get("decision")
            if isinstance(action, str) and action in active and (
                    action != expected["decision"] or given.get("side") != expected["side"]):
                counts["unsupported_campaign_action"] += 1
        if cohort == "context":
            side = given.get("ac_side")
            if side in ("LONG", "SHORT") and side != expected["ac_side"]:
                counts["wrong_ac_direction"] += 1
        if cohort == "precision" and row["rule_id"] == "TC1.PSY.PLAN":
            quantity = given.get("quantity")
            if type(quantity) not in (int, float) or not math.isfinite(quantity):
                continue
            distance = (row["operands"]["distance"] if "operands" in row else
                        re.search(r"\bD=(\d+(?:\.\d+)?)", row["question"])[1])
            if Decimal(str(quantity)) * Decimal(distance) > Decimal(str(expected["risk_budget"])):
                counts["oversize_risk"] += 1
    return counts


def _cohort_result(cohort, rows, predictions):
    stages = {}
    for name in RUN.stages:
        stages[name] = score(rows, predictions[name])
        stages[name]["safety"] = _observed_safety(rows, predictions[name], cohort)
    candidate = stages["candidate"]
    gate = (candidate["accuracy"] >= .80
            and all(v["correct"] / v["total"] >= .75 for v in candidate["per_rule"].values())
            and all(candidate["accuracy"] - stages[name]["accuracy"] >= .05 for name in ("base", "parent"))
            and not candidate["constant_fields_despite_varied_expected_values"]
            and not any(candidate["safety"].values())
            and not (candidate["timeouts"] or candidate["incomplete"]))
    return cohort, stages, gate


def _weights(z, receipt):
    import numpy as np
    from safetensors.numpy import load
    models = []
    for stage, expected in (("source", RUN.initial_sha), ("candidate", receipt["candidate_adapter_sha256"])):
        raw = z.read(stage + "_adapter/adapter_model.safetensors")
        if sha256(raw).hexdigest() != expected:
            raise ValueError("PINNED_ADAPTER_SHA256_MISMATCH")
        values = load(raw)
        if (len(values) != 504 or sum(a.size for a in values.values()) != 33030144
                or any(a.dtype != np.float32 or not np.isfinite(a).all() for a in values.values())):
            raise ValueError("EXACT_504_FINITE_F32_TRAINED_MATRICES_REQUIRED")
        models.append(values)
    before, after = models
    if before.keys() != after.keys() or any(before[k].shape != after[k].shape for k in before):
        raise ValueError("UNCHANGED_LORA_ARCHITECTURE_REQUIRED")
    changed = sum(int(np.count_nonzero(before[k] != after[k])) for k in before)
    if not changed:
        raise ValueError("ACTUAL_LEARNED_PARAMETER_CHANGE_REQUIRED")
    return changed


def verify(archive, groups, design, revision):
    audit_keys(groups)
    with ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):
            raise ValueError("SAFE_UNIQUE_RESULT_FILES_REQUIRED")
        receipt = json.loads(z.read("research_receipt.json"))
        if (receipt["mode"] != RUN.mode or receipt["source_commit"] != revision
                or receipt["model_repo"] != MODEL or receipt["model_revision"] != REVISION
                or receipt["initial_adapter_sha256"] != RUN.initial_sha
                or receipt["world_size"] != 2 or len(receipt["gpu_names"]) != 2
                or any("T4" not in device for device in receipt["gpu_names"])
                or receipt["candidate_frozen_before_test_file_read"] is not True
                or receipt["complete_training"] is not True or receipt["complete_test"] is not True
                or receipt["execution_allowed"] is not False):
            raise ValueError("COMPLETE_PINNED_DUAL_T4_RESEARCH_REQUIRED")
        if receipt["exam_cohorts_by_gpu_rank"] != {str(i): list(x) for i, x in enumerate(RUN.rank_cohorts)}:
            raise ValueError("EXACT_PARALLEL_COHORTS_REQUIRED")
        counters = [receipt[k] for k in ("actual_updates", "amp_skipped_steps", "optimizer_attempted_steps",
                                          "selected_actual_updates")]
        actual, skipped, attempts, selected = counters
        if any(type(n) is not int for n in counters) or not 0 < selected <= actual <= attempts <= RUN.limits["optimizer_attempts"] or actual + skipped != attempts:
            raise ValueError("REAL_OPTIMIZER_COUNTERS_REQUIRED")
        changed = _weights(z, receipt)
        for point in receipt["checkpoints"]:
            raw = json.loads(z.read("development_" + str(point["optimizer_step"]) + ".json"))
            grades = {g: score(rows, raw[g]) for g, rows in groups["validation"].items()}
            if grades != point["grade"] or selection(grades, point["source_loss"], point["practice_loss"], receipt["references"], cohorts=RUN.cohorts) != point["selection"]:
                raise ValueError("CHECKPOINT_SELECTION_NOT_INDEPENDENTLY_REPRODUCED")
        best = max(receipt["checkpoints"], key=lambda point: point["selection"]["score"])
        if receipt["selected_checkpoint"] != "checkpoint-" + str(best["optimizer_step"]):
            raise ValueError("SELECTED_CHECKPOINT_MISMATCH")
        text_gates = {name: validation_admission(before, receipt["source_final_loss"] if name.endswith("source")
                      else receipt["practice_final_loss"]) for name, before in receipt["references"].items()}
        if text_gates != receipt["text_gates"]:
            raise ValueError("FOUR_BOOK_ADMISSION_MISMATCH")
        raw = json.loads(z.read("final_predictions.json"))
        questions_sha = {g: digest(prompts_only(rows)) for g, rows in groups["test"].items()}
        if raw["receipt"] != {"source_commit": revision, "candidate_adapter_sha256": receipt["candidate_adapter_sha256"],
                              "protocol": RUN.protocol, "test_questions_sha256": questions_sha,
                              "expected_keys_loaded_on_gpu": False}:
            raise ValueError("HELD_OUT_QUESTIONS_AND_PREDICTIONS_MUST_MATCH")
        if any(set(raw[g]) != set(RUN.stages) for g in RUN.cohorts):
            raise ValueError("COMPLETE_STAGE_AND_COHORT_MATRIX_REQUIRED")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda g: _cohort_result(g, groups["test"][g], raw[g]), RUN.cohorts))
    gates = {g: gate for g, _, gate in results}
    return {"source_commit": revision, "archive_sha256": sha256(archive.read_bytes()).hexdigest(),
            "frozen_design_sha256": digest(design), "mode": RUN.mode, "actual_updates": actual,
            "selected_checkpoint": receipt["selected_checkpoint"], "selected_updates": selected,
            "candidate_adapter_sha256": receipt["candidate_adapter_sha256"], "changed_values": changed,
            "gpu_names": receipt["gpu_names"], "exam_cohorts_by_gpu_rank": receipt["exam_cohorts_by_gpu_rank"],
            "training_seconds_through_final_exam": receipt["training_seconds_through_final_exam"],
            "training_rows": receipt["training_rows"], "grades": {g: values for g, values, _ in results},
            "cohort_gates": gates, "text_gates": text_gates, "research_gate_passed": all(gates.values())
                and all(gate["accepted"] for gate in text_gates.values()),
            "expert_wave_labels": False, "trading_qualification": False, "execution_allowed": False}
