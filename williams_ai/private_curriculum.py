"""Offline four-original-book read-through with model-specific provenance.

A passage examined by the real local GGUF is NOT evidence of understanding.
Complete source indexing is NOT a completed model pass, and any OCR/image
gap means full-book text coverage is *not* established. No private book
content, model weights, secrets or individual lessons are printed or uploaded.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .book_index import _private_database, index_report
from .book_rules import registry
from .study import study_report
from .study_cli import _model_identity, train_private_curriculum


def complete_private_readthrough(
    book_index: Path, study_memory: Path, model: Any, *,
    model_id: str, batch_budget: int = 30, rounds: int = 50,
) -> dict[str, Any]:
    """Resumable, bounded REAL-inference driver; caller supplies local model.

    The existing study_batch already checks citations against exact indexed
    passage text, stores lessons by edition, source hash and model hash,
    and excludes duplicate chunks on subsequent runs.
    """
    if (_private_database(book_index) == _private_database(study_memory)
            or not isinstance(model_id,str) or not model_id
            or not isinstance(batch_budget,int) or isinstance(batch_budget,bool)
            or not 1<=batch_budget<=100
            or not isinstance(rounds,int) or isinstance(rounds,bool)
            or not 1<=rounds<=50):
        raise ValueError("INVALID_LOCAL_PRIVATE_BOOK_STUDY_CONFIGURATION")
    volumes=index_report(book_index)["volumes"]
    expected={x["id"]:x for x in registry()["books"]}
    found={x["book"]:x for x in volumes}
    if set(found)!=set(expected) or any(
            found[k]["pages"]!=x["pages"] for k,x in expected.items()):
        raise ValueError("ALL_FOUR_VERIFIED_BOOKS_REQUIRED")
    run=train_private_curriculum(
        Path(book_index),Path(study_memory),model,model_id=model_id,
        budget_per_book=batch_budget,rounds=rounds,
        editions=("TC1","NTD","TC2","COURSE"),
    )
    progress=study_report(book_index,study_memory,model_id=model_id)
    lines=[]
    for book in progress["books"]:
        source=found[book["edition"]]
        missing_source_pages=source["pages_without_text"]
        missing_chunks=book["indexed_text_chunks_remaining"]
        complete_text=missing_source_pages==0 and missing_chunks==0
        lines.append({
            "edition":book["edition"],
            "total_pdf_pages":book["total_pages"],
            "pages_with_source_text":book["pages_with_text"],
            "pages_without_source_text":missing_source_pages,
            "indexed_chunks":book["source_chunks"],
            "model_processed_chunks":book["studied_chunks"],
            "remaining_indexed_chunks":missing_chunks,
            "text_pass_completed_for_this_specific_model":complete_text,
            "actual_understanding_independently_verified":False,
        })
    all_text=all(x["text_pass_completed_for_this_specific_model"] for x in lines)
    source_gaps=sum(x["pages_without_source_text"] for x in lines)
    remaining=sum(x["remaining_indexed_chunks"] for x in lines)
    # This passport is stable across reruns, without publishing book passages.
    manifest=json.dumps({
        "model_id":model_id,
        "sources":sorted((x["id"],x["sha256"]) for x in expected.values()),
        "text_chunk_counts":sorted((x["edition"],x["indexed_chunks"]) for x in lines),
    },sort_keys=True)
    return {
        "mode":"SAME_GGUF_PRIVATE_ORIGINALS_BOUNDED_FULL_READTHROUGH",
        "model_id":model_id,
        "readthrough_protocol_sha256":sha256(manifest.encode()).hexdigest(),
        "newly_studied_chunks":run["new_study_chunks"],
        "failed_chunk_attempts":run["failed_or_retry_chunks"],
        "chunks_needing_retry_or_review":progress["chunks_needing_retry_or_review"],
        "total_original_pdf_pages":sum(x["total_pdf_pages"] for x in lines),
        "source_text_pages_missing":source_gaps,
        "indexed_chunks_remaining_for_model":remaining,
        "books":lines,
        "readthrough_status":(
            "COMPLETE_SOURCE_TEXT_PASS_NOT_BOOK_MASTERY" if all_text
            else "BLOCKED_SOURCE_SCAN_OR_IMAGE_GAPS" if remaining==0 and source_gaps
            else "INCOMPLETE_CONTINUE_SAME_MODEL_AND_PRIVATE_MEMORY"),
        "real_weight_updates_occurred":False,
        "book_mastery_percent":None,
        "independent_wave_chart_competence_percent":None,
        "trade_action":None,
        "execution_allowed":False,
    }


def main()->None:
    parser=argparse.ArgumentParser(description="Private resumable GGUF read-through of all four verified Williams PDFs")
    parser.add_argument("--book-index",type=Path,required=True)
    parser.add_argument("--study-memory",type=Path,required=True)
    parser.add_argument("--local-model",type=Path,required=True)
    parser.add_argument("--expected-sha256",required=True,
                        help="Externally verified expected GGUF SHA256; self-digest is not provenance")
    parser.add_argument("--batch-budget",type=int,default=30)
    parser.add_argument("--rounds",type=int,default=50)
    parser.add_argument("--gpu-layers",type=int,default=0)
    args=parser.parse_args()
    # Verify ordinary GGUF contents before claim of real model study.
    from .real_gguf_acceptance import actual_gguf_sha256
    actual_gguf_sha256(args.local_model,args.expected_sha256)
    from .local_model import LocalWilliamsModel
    model=LocalWilliamsModel(args.local_model,n_ctx=8192,
                             gpu_layers=args.gpu_layers)
    data=complete_private_readthrough(
        args.book_index,args.study_memory,model,
        model_id=_model_identity(args.local_model),
        batch_budget=args.batch_budget,rounds=args.rounds)
    print(json.dumps(data,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
