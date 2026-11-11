"""Causal reviewed-error curriculum *actually consumed* by QLoRA.

Only expert-approved TRAIN examples may be oversampled; validation contributes
error-code metadata, never the gold answer or its full market snapshot.
A small cap prevents forgetting the old four-book curriculum.
"""
from __future__ import annotations

from .reasoning_research_v2 import digest
from .reasoning_feedback_v2 import ReviewedFailureJournal


def adaptive_training_cases(train_cases,journal,*,as_of,rounds=None,seed=42):
    if not isinstance(journal,ReviewedFailureJournal):
        raise ValueError("MODEL_SCOPED_REVIEWED_ERROR_JOURNAL_REQUIRED")
    if (not isinstance(train_cases,list) or len(train_cases)<8
            or any(case.get("split")!="train" for case in train_cases)):
        raise ValueError("APPROVED_TRAIN_ONLY_SET_REQUIRED")
    if rounds is None:
        rounds=max(1,min(100,len(train_cases)//4))
    if (type(rounds) is not int or
            not 1<=rounds<=min(100,max(1,len(train_cases)//2))):
        raise ValueError("BOUNDED_REVIEWED_ERROR_OVERSAMPLING")
    result=journal.schedule_from_previous_errors(
        train_cases,timestamp=as_of,rounds=rounds)
    catalog={r["case_id"]:r for r in train_cases}
    if len(catalog)!=len(train_cases):
        raise ValueError("DUPLICATE_TRAIN_CASES_FORBIDDEN")
    selection=[]
    for entry in result["schedule"]:
        ident=entry["train_case_id"]
        if ident not in catalog or catalog[ident]["split"]!="train":
            raise ValueError("FORGED_TRAIN_CASE_FROM_FEEDBACK")
        selection.append(catalog[ident])
    expanded=list(train_cases)+selection
    return expanded,{
        "schema":"WILLIAMS_AI_REVIEWED_ADAPTIVE_GRADIENT_CURRICULUM_V3",
        "baseline_train_examples":len(train_cases),
        "extra_reviewed_train_exposures":len(selection),
        "original_four_book_train_rows_preserved":True,
        "validation_gold_in_train":False,"test_gold_in_train":False,
        "error_reviewed_as_of":as_of,
        "reviewed_error_count":result["previously_reviewed_failure_count"],
        "journal_run_id":journal.run_id,
        "journal_model_fingerprint":journal.model_id,
        "schedule_sha256":digest(result["schedule"]),
        "expanded_training_case_sha256":digest([
            (row["case_id"],row["source_sha256"],row["visible_prefix_sha256"])
            for row in expanded]),
        "training_gradient_exposure_changed":len(selection)>0,
    }
