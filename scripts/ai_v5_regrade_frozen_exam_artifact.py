"""Offline v5 artifact forensics: reconstruct 28 MCQ keys from frozen source.

Run at the exact curriculum source commit or one with byte-identical
knowledge_curriculum_v5.py and SOURCE_RULE_CARDS.json. NO model calls,
NO GPU, NO credentials, NO execution. This examines known synthetic tasks,
not an untouched expert dataset or a live-trading qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from williams_ai.knowledge_curriculum_v5 import build, digest
from williams_ai.v5_postfit_exam import research_quality_gate

EDITIONS=("TC1","NTD","TC2","COURSE")
EXAM="williams_v5_knowledge_exam.json"
TRAIN="williams_v5_encrypted/v5_training_audit.json"
MANIFEST="williams-v5-train-dev-test-manifest.json"
SEAL="williams_v5_encrypted/williams-v5-trained-qwen3-4b-lora.zip.sealed"


def _need(condition, message):
    if not condition:
        raise ValueError("V5_AUDIT_"+message)


def regrade_28(grades:dict, frozen_rows:list)->dict:
    """Regrade ALL 28 test MCQs; never trust stored correct booleans."""
    gold={x["case_id"]:(json.loads(x["answer"])["choice"],x["edition"],x["rule_id"],x["variant"])
          for x in frozen_rows
          if x["split"]=="test" and x["task_type"]=="MCQ"}
    _need(len(gold)==28,"UNEXPECTED_TEST_GOLD_COUNT")
    observed=grades.get("case_pass_fail")
    _need(isinstance(observed,list) and len(observed)==28,"UNEXPECTED_CASE_COUNT")
    _need(len({x.get("id") for x in observed})==28,"DUPLICATE_CASE_ID")
    _need({x["id"] for x in observed}==set(gold),"CASE_GOLD_MISMATCH")
    counts=Counter();correct=Counter();abst=Counter();fallback=Counter()
    wrong=[];paired={}
    for row in observed:
        id_=row["id"]
        expected,ed,rule,variant=gold[id_]
        choice=row.get("chosen")
        _need(row.get("edition")==ed,"CASE_BOOK_CHANGED")
        _need(choice in ("","A","B","C"),"INVALID_CHOICE")
        _need(type(row.get("abstained")) is bool and
              row["abstained"]==(choice==""),"INVALID_ABSTENTION")
        computed=(choice==expected)
        _need(type(row.get("correct")) is bool and
              row["correct"]==computed,"SELF_REPORTED_GRADE_MISMATCH")
        counts[ed]+=1
        correct[ed]+=computed
        abst[ed]+=(choice=="")
        fallback[ed]+=row.get("json_fallback_used") is True
        paired[id_]={"edition":ed,"choice":choice,"correct":computed}
        if not computed:
            wrong.append({"id":id_,"edition":ed,"rule_id":rule,"variant":variant,
                          "expected":expected,"chosen":choice,"abstained":choice=="",
                          "finish_reason":row.get("initial_finish_reason"),
                          "json_fallback_used":row.get("json_fallback_used")})
    _need(dict(counts)=={"TC1":8,"NTD":8,"TC2":8,"COURSE":4},
          "WRONG_BOOK_DISTRIBUTION")
    per=grades.get("by_edition",{})
    for ed in EDITIONS:
        _need(per.get(ed)=={"correct":correct[ed],"total":counts[ed],
                            "abstained":abst[ed]},
              "CASE_AGGREGATION_CHANGED_"+ed)
    _need(grades.get("correct")==sum(correct.values()) and grades.get("total")==28,
          "OVERALL_AGGREGATION_CHANGED")
    return {"correct":sum(correct.values()),"total":28,
            "by_edition":{ed:{"correct":correct[ed],"total":counts[ed],
                              "abstained":abst[ed],"fallback_used":fallback[ed]}
                          for ed in EDITIONS},
            "wrong":wrong,"paired":paired}


def audit_archive(archive:Path)->dict:
    with ZipFile(archive) as z:
        _need(len(z.namelist())==len(set(z.namelist())),"DUPLICATE_ZIP_MEMBER")
        _need(all(x in z.namelist() for x in (EXAM,TRAIN,MANIFEST,SEAL)),
              "MISSING_EVIDENCE")
        exam=json.loads(z.read(EXAM))
        training=json.loads(z.read(TRAIN))
        saved_manifest=json.loads(z.read(MANIFEST))
        h=sha256()
        with z.open(SEAL) as sealed:
            for piece in iter(lambda:sealed.read(1024*1024),b""):
                h.update(piece)
    _need(h.hexdigest()==training["sha256_encrypted_archive"],
          "SEALED_MODEL_SHA_MISMATCH")
    _need(exam.get("adapter_sha256")==training.get("adapter_sha256"),
          "ADAPTER_RECEIPT_SHA_MISMATCH")
    _need(training.get("optimizer_steps")==80 and
          training.get("selected_validation_checkpoint")==80,
          "EXPECTED_80_STEPS_AND_CHECKPOINT")
    _need(training.get("test_cases_provided_to_trainer")==0,
          "TRAIN_TEST_LEAKAGE")
    _need(exam.get("execution_allowed") is False and
          exam.get("trading_qualification") is False,
          "TRADING_NOT_DISABLED")
    rows,current_manifest=build()
    _need(current_manifest==saved_manifest and
          digest(saved_manifest)==exam.get("source_curriculum_manifest_sha256"),
          "CURRICULUM_VERSION_NOT_FROZEN_IDENTICAL")
    _need(saved_manifest.get("approved_market_labels")==0 and
          saved_manifest.get("independently_reviewed_diagrams")==0,
          "UNVERIFIED_MARKET_LABEL_STATE_CHANGED")
    _need(exam.get("same_v5_inference_protocol_used_for_candidate_and_base") is True and
          exam.get("old_checkpoint500_not_same_inference_protocol") is True,
          "PROTOCOL_COMPARISON_WARNING_MISSING")
    observed=exam["candidate_vs_base"]
    comparisons={}
    for label in ("candidate_v5","public_base_control"):
        p=observed[label]
        comparisons[label]=regrade_28(p["new_rule_disjoint_28_case_exam"],rows)
        for test in ("prior_16_case_closed_book","prior_16_case_with_source"):
            cohort=p[test]
            _need(cohort["total"]==16 and
                  all(cohort["by_edition"][ed]["total"]==4 for ed in EDITIONS) and
                  sum(cohort["by_edition"][ed]["correct"] for ed in EDITIONS)==cohort["correct"] and
                  sum(cohort["by_edition"][ed]["abstentions_counted_wrong"] for ed in EDITIONS)==cohort["abstentions"],
                  "INCONSISTENT_16_CASE_AGGREGATES")
    c=comparisons["candidate_v5"]["paired"]
    b=comparisons["public_base_control"]["paired"]
    _need(set(c)==set(b),"PAIRED_CASE_MISMATCH")
    changed=[{"id":id_,"edition":c[id_]["edition"],
              "v5_correct":c[id_]["correct"],"base_correct":b[id_]["correct"]}
             for id_ in sorted(c) if c[id_]["correct"]!=b[id_]["correct"]]
    gate=research_quality_gate(observed)
    _need(gate["accepted_for_further_research"] is False,
          "UNEXPECTED_RESEARCH_GATE_ACCEPTANCE")
    return {"source_run_id":43229923254,
            "optimizer_steps_attested":training["optimizer_steps"],
            "adapter_sha256_attested":training["adapter_sha256"],
            "encrypted_archive_sha256_verified":h.hexdigest(),
            "validation_baseline":training["validation_baseline_per_edition"],
            "validation_after":training["validation_after_per_edition"],
            "candidate":{k:v for k,v in comparisons["candidate_v5"].items() if k!="paired"},
            "base":{k:v for k,v in comparisons["public_base_control"].items() if k!="paired"},
            "changed_cases":changed,
            "gate_recomputed_all_four_editions":gate,
            "disclaimers":["Synthetic developer gold only; not independent chart expertise.",
                           "16-item cohorts lack question-level rows in this historic artifact.",
                           "Actual adapter plaintext SHA cannot be rehashed from a sealed archive.",
                           "Raw completion missing for TC2.REVERSE abstention.",
                           "Older checkpoint500 RAG/decoder protocol differs.",
                           "No deployment, GPU retraining or trading authorized."]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact_zip",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    result=audit_archive(a.artifact_zip)
    output=json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\\n"
    if a.output:
        if a.output.exists():
            raise FileExistsError("V5_AUDIT_DO_NOT_OVERWRITE_OUTPUT")
        a.output.write_text(output,encoding="utf-8")
    else:
        print(output)


if __name__=="__main__":
    main()
