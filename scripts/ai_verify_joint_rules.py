"""Independent Decimal answer keys and CPU audit of the November run."""
from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ai_verify_restorative_result import independent_key as numeric_key
from scripts.ai_verify_historical_rule_curriculum import independent_key as historical_key
from williams_ai.joint_rule_curriculum import MODE, INITIAL_SHA, PROTOCOL, LIMITS
from williams_ai.historical_rule_curriculum import canonical, digest, prompts_only, score
from williams_ai.deep_book_research import MODEL, REVISION, EDITIONS
from williams_ai.restorative_book_practice import INITIAL_SHA as FOCUSED_SHA
from williams_ai.lora_integrity import validation_admission


def independent_boundary_key(row):
    """Read operands, never the supplied training answer or explanation."""
    o = row["operands"]; rule = row["rule_id"]
    dec = lambda key: Decimal(str(o[key]))
    if rule == "TC1.PSY.PLAN":
        budget = dec("equity")*dec("risk_fraction")
        per_lot = dec("distance")*dec("quantity_step")
        # Integer-ratio arithmetic, independent of Decimal division rounding.
        bn, bd = budget.as_integer_ratio(); ln, ld = per_lot.as_integer_ratio()
        lots = (bn*ld)//(bd*ln)
        q = Decimal(lots)*dec("quantity_step")
        if not q*dec("distance") <= budget < (q+dec("quantity_step"))*dec("distance"):
            raise ValueError("RISK_BUDGET_AND_NEXT_LOT_INVARIANT_FAILED")
        answer = {"risk_budget": float(budget), "quantity": float(q)}
    elif rule == "NTD.AC":
        values = [Decimal(str(v)) for v in o["ac_oldest_to_latest"]]
        required = None if values[-1]==0 else (2 if (values[-1]>0)==(o["side"]=="LONG") else 3)
        transitions = [b>a if o["side"]=="LONG" else b<a for a,b in zip(values,values[1:])]
        count = 0
        for good in transitions[::-1]:
            if not good: break
            count += 1
        answer = {"required_bars": required, "signal_ready": required is not None and count>=required}
    elif rule == "NTD.ALLIGATOR":
        jaw, teeth, lips, close = (dec(k) for k in ("visible_jaw", "visible_teeth", "visible_lips", "close"))
        answer = {"order": "UP_STACK" if lips>teeth>jaw else "DOWN_STACK" if jaw>teeth>lips else "MIXED",
                  "above_all": all(close>x for x in (jaw,teeth,lips)), "below_all": all(close<x for x in (jaw,teeth,lips))}
    elif rule == "NTD.AO":
        a,b = dec("previous_ao"),dec("current_ao")
        answer = {"above_zero": b>0, "rising": b>a, "crossed_zero": (a<0<b) or (b<0<a)}
    elif rule == "TC2.WM1.TRIGGER":
        lo,hi,t = dec("last_bar_low"),dec("last_bar_high"),dec("research_tick")
        answer = {"entry": float(hi+t if o["side"]=="LONG" else lo-t),
                  "stop": float(lo-t if o["side"]=="LONG" else hi+t)}
    elif rule == "TC2.WM3":
        xs = [Decimal(str(v)) for v in o["closed_extrema_oldest_first"]]
        possible = len(xs)==5 and o["right_closed"]==2 and o["pivot_offset"]==2
        if possible:
            possible = all(xs[2]>v if o["kind"]=="HIGH" else xs[2]<v for j,v in enumerate(xs) if j!=2)
        answer = {"confirmed": possible}
    elif rule == "COURSE.AO5345":
        h,s = dec("histogram"),dec("own_sma5")
        answer = {"momentum": "LONG" if h>s else "SHORT" if s>h else "NEUTRAL"}
    elif rule == "COURSE.HUMP":
        answer = {"hump_valid": o["second_formation_inclusive_start"]-o["first_formation_inclusive_end"]>=1}
    elif rule == "COURSE.W4.ZERO":
        a,b = dec("previous_5_34"),dec("current_5_34")
        answer = {"zero_transition_seen": (a<0<b) or (b<0<a), "w4_finished": False}
    else:
        raise ValueError("UNKNOWN_BOUNDARY_FAMILY")
    return {**answer, "execution_allowed": False}


def audit_groups(groups):
    for name, rows in groups.items():
        for row in rows:
            key = (independent_boundary_key(row) if name=="boundary" else
                   numeric_key({"domain":row["rule_id"],"prompt":row["question"]}) if name=="numeric" else historical_key(row))
            if key != json.loads(row["answer"]):
                raise ValueError("INDEPENDENT_JOINT_KEY_MISMATCH:"+row["case_id"])
    return {g:{"count":len(rows), "questions_sha256":digest(prompts_only(rows))} for g,rows in groups.items()}


def verify_archive(path, dest, *, revision, design, groups):
    import numpy as np
    from safetensors.numpy import load
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(set(names))!=len(names) or any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):
            raise ValueError("SAFE_UNIQUE_JOINT_RESULT_ARCHIVE_REQUIRED")
        r = json.loads(z.read("research_receipt.json"))
        if (r["mode"]!=MODE or r["source_commit"]!=revision or r["model_repo"]!=MODEL or r["model_revision"]!=REVISION
            or r["initial_adapter_sha256"]!=INITIAL_SHA or r["focused69_adapter_sha256"]!=FOCUSED_SHA or r["world_size"]!=2
            or r["complete_training"] is not True or r["complete_test"] is not True or r["execution_allowed"] is not False):
            raise ValueError("COMPLETE_PINNED_REAL_JOINT_GPU_PROVENANCE_REQUIRED")
        tensors = {}
        for stage, expected in (("source", INITIAL_SHA), ("candidate", r["candidate_adapter_sha256"])):
            raw = z.read(stage+"_adapter/adapter_model.safetensors")
            if sha256(raw).hexdigest()!=expected: raise ValueError("JOINT_ADAPTER_SHA_MISMATCH")
            values = load(raw)
            if len(values)!=504 or sum(a.size for a in values.values())!=33030144 or any(a.dtype!=np.float32 or not np.isfinite(a).all() for a in values.values()):
                raise ValueError("ALL_504_FINITE_F32_JOINT_LORA_MATRICES_REQUIRED")
            tensors[stage] = values
        if set(tensors["source"])!=set(tensors["candidate"]): raise ValueError("SAME_LORA_MATRIX_SET_REQUIRED")
        changed = sum(int(np.count_nonzero(a!=tensors["source"][n])) for n,a in tensors["candidate"].items())
        if not changed: raise ValueError("ACTUAL_CHANGED_JOINT_WEIGHTS_REQUIRED")
        if any(type(r[k]) is not int for k in ("actual_updates","optimizer_attempted_steps","amp_skipped_steps","selected_actual_updates")):
            raise ValueError("INTEGER_REAL_JOINT_OPTIMIZER_COUNTS_REQUIRED")
        if (r["optimizer_attempted_steps"]!=r["actual_updates"]+r["amp_skipped_steps"]
            or not 0<r["selected_actual_updates"]<=r["actual_updates"]<=r["optimizer_attempted_steps"]<=LIMITS["optimizer_attempts"]):
            raise ValueError("BOUNDED_ACTUAL_JOINT_OPTIMIZER_COUNTS_REQUIRED")
        for point in r["checkpoints"]:
            responses = json.loads(z.read("development_"+str(point["optimizer_step"])+".json"))
            grades = {g:score(rows,responses[g]) for g,rows in groups["validation"].items()}
            if grades!=point["grade"]: raise ValueError("DEVELOPMENT_GENERATIONS_MUST_RECOMPUTE")
            refs = r["references"]
            regression = max(after[e]/before[e]-1 for after,before in (
                (point["source_loss"],refs["focused69_source"]), (point["source_loss"],refs["previous40_source"]),
                (point["practice_loss"],refs["focused69_practice"]), (point["practice_loss"],refs["previous40_practice"])) for e in EDITIONS)
            minimum = min(x["correct"]/x["total"] for g in grades.values() for x in g["per_rule"].values())
            # The private training payload is canonical JSON: boundary,
            # historical, numeric. Preserve that summation order exactly.
            macro = sum(grades[name]["accuracy"] for name in sorted(grades))/len(grades)
            collapsed = sum(len(g["constant_fields_despite_varied_expected_values"]) for g in grades.values())
            value = -3-regression if regression>.02 else minimum+.2*macro-.05*collapsed
            expected = {"score":value,"minimum_rule_accuracy":minimum,"macro_accuracy":macro,"collapsed_rules":collapsed,
                        "max_any_reference_book_regression":regression,"preserved_within_two_percent":regression<=.02}
            if expected!=point["selection"]: raise ValueError("INDEPENDENT_JOINT_CHECKPOINT_SELECTION_FAILED")
        best = max(r["checkpoints"],key=lambda p:p["selection"]["score"])
        if r["selected_checkpoint"]!="checkpoint-"+str(best["optimizer_step"]):
            raise ValueError("DEVELOPMENT_ONLY_JOINT_CHECKPOINT_REQUIRED")
        for file,field,total in (("source_validation_window_losses.json","source_final_loss",178),
                                 ("practice_validation_window_losses.json","practice_final_loss",122)):
            rows = json.loads(z.read(file)); seen = set(); by_edition = {e:[] for e in EDITIONS}
            for edition,index,loss in rows["window_losses"]:
                if (edition not in by_edition or type(index) is not int or not 0<=index<rows["window_counts"][edition]
                    or (edition,index) in seen or type(loss) not in (int,float) or not math.isfinite(loss) or loss<0):
                    raise ValueError("UNIQUE_FINITE_ALL_BOOK_WINDOW_LOSSES_REQUIRED")
                seen.add((edition,index));by_edition[edition].append(loss)
            if len(seen)!=total: raise ValueError("ALL_ORIGINAL_DEVELOPMENT_WINDOWS_REQUIRED")
            values = {e:sum(v)/len(v) for e,v in by_edition.items()}
            values["ALL"] = sum(sum(v) for v in by_edition.values())/total
            if any(not math.isclose(v,r[field][e],rel_tol=1e-12,abs_tol=1e-12) for e,v in values.items()):
                raise ValueError("INDEPENDENT_JOINT_SOURCE_LOSS_MISMATCH")
        predictions = json.loads(z.read("final_predictions.json")); identity = predictions["receipt"]
        if (identity["source_commit"]!=revision or identity["candidate_adapter_sha256"]!=r["candidate_adapter_sha256"]
            or identity["protocol"]!=PROTOCOL or identity["expected_keys_loaded_on_gpu"] is not False
            or identity["test_questions_sha256"]!=design["questions_sha256"]["test"]
            or r["candidate_frozen_before_test_file_read"] is not True):
            raise ValueError("FROZEN_PROMPTS_ONLY_JOINT_FINAL_EXAM_REQUIRED")
        stages = {"base","focused69","historical93","candidate"}
        if any(set(predictions[g])!=stages for g in groups["test"]):
            raise ValueError("SAME_FOUR_JOINT_COMPARISONS_REQUIRED")
        grades = {g:{stage:score(rows,predictions[g][stage]) for stage in stages} for g,rows in groups["test"].items()}
        gates = {name:validation_admission(before,r["source_final_loss"] if name.endswith("source") else r["practice_final_loss"])
                 for name,before in r["references"].items()}
        if gates!=r["text_gates"]: raise ValueError("UNMODIFIED_JOINT_BOOK_GATES_REQUIRED")
        transfer = {}
        for group in ("numeric","boundary"):
            candidate = grades[group]["candidate"]
            transfer[group] = (candidate["accuracy"]>=.8 and min(v["correct"]/v["total"] for v in candidate["per_rule"].values())>=.75
                and candidate["accuracy"]>=max(grades[group][s]["accuracy"] for s in stages-{"candidate"})+.05
                and not candidate["constant_fields_despite_varied_expected_values"])
        proof = {"mode":MODE,"source_commit":revision,"archive_sha256":sha256(Path(path).read_bytes()).hexdigest(),
                 "candidate_adapter_sha256":r["candidate_adapter_sha256"],"changed_values":changed,
                 "actual_updates":r["actual_updates"],"selected_checkpoint":r["selected_checkpoint"],
                 "selected_actual_updates":r["selected_actual_updates"],"design_sha256":digest(design),
                 "grades":grades,"text_gates":gates,"transfer_gates":transfer,
                 "pilot_research_gates_passed":all(transfer.values()) and all(v["accepted"] for v in gates.values()),
                 "historical_test_is_revealed_regression":True,"fresh_test_is_synthetic_not_expert_charts":True,
                 "expert_chart_supervision_passed":False,"trading_qualification":False,"execution_allowed":False}
        for name in ("research_receipt.json","final_predictions.json","source_validation_window_losses.json","practice_validation_window_losses.json"):
            (dest/name).write_bytes(z.read(name))
        (dest/"independent_joint_rule_audit.json").write_bytes(canonical(proof))
        return proof
