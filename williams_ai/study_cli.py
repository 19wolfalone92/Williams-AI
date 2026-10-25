"""Start real resumable four-book *inference-based study* for a LOCAL GGUF.

This builds a PRIVATE indexed memory of source-supported lesson candidates.
It is NOT a language model weight update or proof of book comprehension.
No exchange API, automation, internet or user credentials are used.

python -m williams_ai.study_cli --book-index /private/books.sqlite3 \
  --study-memory /private/lessons.sqlite3 --local-model /private/weights.gguf \
  --budget-per-book 10 --rounds 3

python -m williams_ai.study_cli --book-index /private/books.sqlite3 \
  --study-memory /private/lessons.sqlite3 --report
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from .study import study_batch,study_report
from .book_rules import registry


def _model_identity(path:Path, *, lora_adapter:Path|None=None,
                    lora_manifest:Path|None=None)->str:
    """Bind an experience notebook to ACTUAL GGUF bytes, not just filename."""
    if not path.is_file() or path.is_symlink() or path.suffix.lower()!=".gguf":
        raise FileNotFoundError("Local regular GGUF weights required")
    from .model_bundle import runtime_identity
    return runtime_identity(path,adapter=lora_adapter,manifest=lora_manifest)["model_id"]


def train_private_curriculum(
    book_index:Path,study_memory:Path,model:Any, *,
    model_id:str,editions:tuple[str,...]=("TC1","NTD","TC2","COURSE"),
    budget_per_book:int=10,rounds:int=1,
    pdf_pages:tuple[int,...]|None=None,
)->dict[str,Any]:
    """Bounded synchronous study of already indexed privately owned originals.

    Every new chunk passes through a local model, verifies that supporting
    spans exist on that book page, then persists an UNVERIFIED lesson. A
    mistake or lack of model ability does not become an accepted book rule.
    """
    allowed={x["id"] for x in registry()["books"]}
    if not isinstance(editions,tuple) or not editions or len(set(editions))!=len(editions):
        raise ValueError("Invalid edition selection")
    if not set(editions)<=allowed:
        raise ValueError("Unknown book")
    if pdf_pages is not None and len(editions)!=1:
        raise ValueError("Focused PDF pages require exactly one edition")
    if not isinstance(budget_per_book,int) or not 1<=budget_per_book<=100:
        raise ValueError("budget_per_book must be 1..100")
    if not isinstance(rounds,int) or not 1<=rounds<=50:
        raise ValueError("rounds must be 1..50")
    tasks=[]
    total_studied=total_failed=0
    failure_codes:Counter[str]=Counter()
    for cycle in range(rounds):
        gained=0
        for edition in editions:
            batch=study_batch(
                book_index,study_memory,edition,model,
                model_id=model_id,budget=budget_per_book,pdf_pages=pdf_pages)
            tasks.append({"round":cycle+1,"edition":edition,
                          "new_page_chunks":batch["studied_chunks"],
                          "rejected_or_retry_chunks":batch["failed_chunks"],
                          "failure_codes":batch["failure_codes"]})
            gained+=batch["studied_chunks"]
            total_studied+=batch["studied_chunks"]
            total_failed+=batch["failed_chunks"]
            failure_codes.update(batch["failure_codes"])
        if gained==0:
            # No infinite self-reinforcing retry of invalid lessons.
            break
    return {
        "mode":"LOCAL_PRIVATE_BOOK_STUDY_SOURCE_DERIVED_NOT_WEIGHT_TRAINING",
        "model_fingerprint_id":model_id,
        "editions":list(editions),
        "focused_pdf_pages":list(pdf_pages) if pdf_pages is not None else None,
        "new_study_chunks":total_studied,
        "failed_or_retry_chunks":total_failed,
        "failure_codes":dict(sorted(failure_codes.items())),
        "cycles":tasks,
        "progress":study_report(book_index,study_memory,model_id=model_id),
        "language_model_weights_modified":False,
        "full_books_understood_percent":None,
        "execution_allowed":False,
        "trade_action":None,
    }


def main()->None:
    p=argparse.ArgumentParser(description="Run bounded offline GGUF private book lessons")
    p.add_argument("--book-index",type=Path,required=True)
    p.add_argument("--study-memory",type=Path,required=True)
    p.add_argument("--report",action="store_true")
    p.add_argument("--local-model",type=Path)
    p.add_argument("--lora-adapter",type=Path)
    p.add_argument("--lora-manifest",type=Path)
    p.add_argument("--gpu-layers",type=int,default=0)
    p.add_argument("--learning-mode",choices=["legacy_json","auto","reasoning"],default="legacy_json")
    p.add_argument("--reasoning-budget",type=int,default=2048)
    p.add_argument("--edition",choices=["TC1","NTD","TC2","COURSE","ALL"],default="ALL")
    p.add_argument("--budget-per-book",type=int,default=10)
    p.add_argument("--rounds",type=int,default=1)
    p.add_argument("--pdf-page",type=int,action="append",
                   help="Repeat for focused source study; requires a single --edition")
    a=p.parse_args()
    if (a.lora_adapter or a.lora_manifest) and not a.local_model:
        p.error("LoRA options require --local-model even for a model-scoped report")
    if a.pdf_page and a.edition=="ALL":
        p.error("--pdf-page requires a single --edition")
    if a.report:
        data=study_report(a.book_index,a.study_memory,
            model_id=_model_identity(a.local_model,lora_adapter=a.lora_adapter,
                lora_manifest=a.lora_manifest) if a.local_model else None)
    else:
        if a.local_model is None:
            p.error("--local-model is required to process books")
        from .local_model import LocalWilliamsModel
        data=train_private_curriculum(
            a.book_index,a.study_memory,
            LocalWilliamsModel(a.local_model,gpu_layers=a.gpu_layers,
                               learning_mode=a.learning_mode,reasoning_budget=a.reasoning_budget,
                               lora_adapter=a.lora_adapter,lora_manifest=a.lora_manifest),
            model_id=_model_identity(a.local_model,lora_adapter=a.lora_adapter,
                lora_manifest=a.lora_manifest),
            editions=("TC1","NTD","TC2","COURSE") if a.edition=="ALL" else (a.edition,),
            budget_per_book=a.budget_per_book,rounds=a.rounds,
            pdf_pages=tuple(a.pdf_page) if a.pdf_page else None)
    print(json.dumps(data,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
