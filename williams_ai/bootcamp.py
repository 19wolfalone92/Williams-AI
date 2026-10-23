"""Run the COMPLETE bounded research-teaching LOOP with a REAL local GGUF.

Private four books -> cited study summaries -> source-scoped practice and
private error correction -> separate multiple-choice diagnostic -> historical
H1/H4/D1 model reasoning scored by deterministic rule critic.

This is RAG + external-memory tutoring, not GGUF weight fine-tuning and not a
certification of profitable or canonical full TC2/NTD/Course execution.
All stages are synchronous, offline and READ_ONLY. No future candle to model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adaptive_coach import coach_practice
from .book_index import index_report
from .book_rules import registry
from .competency_exam import grade_exam
from .causal_paper_lab import FROZEN_DATA,frozen_markets
from .reasoning_batch import examine_many
from .study_cli import _model_identity,train_private_curriculum


def validate_book_readiness(book_db:Path)->dict[str,Any]:
    """Show whether the pinned original indexes are ready; not if model knows."""
    state=index_report(book_db)
    found={v["book"]:v for v in state["volumes"]}
    needed={b["id"]:b for b in registry()["books"]}
    missing=sorted(set(needed)-set(found))
    issues=[]
    for ed,item in found.items():
        if ed not in needed:
            issues.append("UNKNOWN_EDITION:"+str(ed))
            continue
        if item["pages"]!=needed[ed]["pages"]:
            issues.append("PAGE_COUNT_MISMATCH:"+ed)
        if item["pages_without_text"]>0:
            issues.append("MISSING_OR_UNREVIEWED_PAGE_TEXT:"+ed)
    return {
        "mode":"PRIVATE_BOOK_READINESS",
        "indexed_editions":sorted(found),
        "unindexed_editions":missing,
        "total_expected_pdf_pages":sum(b["pages"] for b in needed.values()),
        "pages_with_extracted_or_ocr_text":sum(
            min(v["pages_with_text"],needed[k]["pages"])
            for k,v in found.items() if k in needed
        ),
        "issues":issues,
        "readiness_for_partial_local_RAG":not missing and not any(
            x.startswith("PAGE_COUNT_MISMATCH") for x in issues),
        "verified_model_book_mastery_percent":None,
        "execution_allowed":False,
    }


def bootcamp(
    model:Any, *,model_id:str,book_index:Path,study_memory:Path,
    coaching_memory:Path,study_budget_per_book:int=4,
    study_rounds:int=1,practice_cases:int=29,
    historical_markets:dict[str,dict[str,Any]]|None=None,
    historical_editions:tuple[str,...]=("TC1","NTD","TC2","COURSE"),
    checkpoints:tuple[int,...]=(899,),
)->dict[str,Any]:
    """A bounded SINGLE supervised research cycle. Never optimize on test PnL."""
    ready=validate_book_readiness(book_index)
    if not ready["readiness_for_partial_local_RAG"]:
        raise ValueError("All pinned private books must be indexed first")
    books=train_private_curriculum(
        book_index,study_memory,model,model_id=model_id,
        budget_per_book=study_budget_per_book,rounds=study_rounds)
    practice=coach_practice(
        model,coaching_memory,model_id=model_id,max_cases=practice_cases)
    # The public curated 10-question test is diagnostic only. It is NOT
    # independent: answers are version-controlled in the public repository.
    from .competency_exam import public_question
    exam=grade_exam(model.answer_exam_case)
    markets=historical_markets or dict(sorted(frozen_markets().items()))
    # All testing uses only the candle prefix up to each decision cutoff.
    historical=examine_many(
        markets,model,editions=historical_editions,
        checkpoints=checkpoints,
        book_index=book_index,coach_notebook=coaching_memory,
        model_id=model_id)
    return {
        "mode":"ACTUAL_GGUF_BOUNDED_BOOK_STUDY_AND_CAUSAL_PRACTICE",
        "model_fingerprint_id":model_id,
        "readiness":ready,
        "source_study":{
            "new_chunks":books["new_study_chunks"],
            "rejected_or_retry_chunks":books["failed_or_retry_chunks"],
            "books":books["progress"]["books"],
        },
        "adaptive_practice":{
            "cases":practice["cases"],
            "initial_pass":practice["initial_pass"],
            "initial_pass_percent":practice["initial_pass_percent"],
            "same_case_retries_correct":practice["corrected_same_case_retries"],
            "private_misconception_topics":practice["private_misconception_topics"],
        },
        "public_curated_diagnostic_exam":{
            "total_cases":exam["total_cases"],
            "correct_cases":exam["correct_cases"],
            "score_percent":exam["score_on_curated_cases_percent"],
            "not_independent_blind_exam":True,
        },
        "historical_chart_reasoning":{
            "tested_cases":historical["tested_cases"],
            "source_consistent_cases":historical["source_consistent_cases"],
            "unsupported_claim_cases":historical["unsupported_claim_cases"],
            "malformed_model_cases":historical["malformed_model_cases"],
            "abstained_or_unknown_cases":historical["abstained_or_unknown_cases"],
            "substantive_claim_cases":historical["substantive_claim_cases"],
            "source_consistency_percent":historical["source_consistency_percent"],
            "failure_counts":historical["failure_counts"],
        },
        "knowledge_mode":"RAG_AND_PRIVATE_ERROR_MEMORY_NOT_WEIGHT_TRAINING",
        "actual_language_model_weights_changed":False,
        "independent_blind_generalization_percent":None,
        "full_book_mastery_percent":None,
        "trade_action":None,"execution_allowed":False,
        "notes":[
            "Only direct local GGUF inference can be called a real student run.",
            "A reanswered practice question cannot prove independent understanding.",
            "Published public questions cannot be used as true private holdout.",
            "Trading claims remain forbidden without Binance Demo order and SL receipts.",
            "Book diagrams require a separate actually verified vision model.",
        ],
    }


def main()->None:
    p=argparse.ArgumentParser(description="Williams AI local four-book to market bootcamp")
    p.add_argument("--book-index",type=Path,required=True)
    p.add_argument("--study-memory",type=Path)
    p.add_argument("--coach-memory",type=Path)
    p.add_argument("--local-model",type=Path)
    p.add_argument("--dry-run",action="store_true",help="Check private source index; no model")
    p.add_argument("--study-budget-per-book",type=int,default=4)
    p.add_argument("--study-rounds",type=int,default=1)
    p.add_argument("--practice-cases",type=int,default=29)
    p.add_argument("--pairs",type=int,default=15)
    p.add_argument("--checkpoints",choices=[1,2,4],type=int,default=1)
    p.add_argument("--gpu-layers",type=int,default=0)
    a=p.parse_args()
    if a.dry_run:
        result=validate_book_readiness(a.book_index)
    else:
        if not (a.study_memory and a.coach_memory and a.local_model):
            p.error("Full study requires --local-model, --study-memory, --coach-memory")
        from .local_model import LocalWilliamsModel
        model=LocalWilliamsModel(a.local_model,n_ctx=8192,
                                 gpu_layers=a.gpu_layers)
        all_markets=frozen_markets(FROZEN_DATA)
        if not 1<=a.pairs<=len(all_markets):
            p.error("--pairs must be 1..15")
        markets=dict(sorted(all_markets.items())[:a.pairs])
        checkpoints={1:(899,),2:(599,999),4:(299,599,899,1199)}[a.checkpoints]
        result=bootcamp(
            model,model_id=_model_identity(a.local_model),
            book_index=a.book_index,study_memory=a.study_memory,
            coaching_memory=a.coach_memory,
            study_budget_per_book=a.study_budget_per_book,
            study_rounds=a.study_rounds,practice_cases=a.practice_cases,
            historical_markets=markets,checkpoints=checkpoints)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
