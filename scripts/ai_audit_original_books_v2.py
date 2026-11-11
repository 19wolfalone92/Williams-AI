"""Verify ORIGINAL private Bill Williams PDFs and cited case excerpts.

Read-only local tool; never uploads any licensed PDF, OCR, image or source text.
A PDF hash proves file identity, not the semantic correctness of a rule.
For scanned pages without extractable text report NEEDS_PAGE_IMAGE_REVIEW:
do not silently call OCR/AI paraphrases verbatim source quotes.

python scripts/ai_audit_original_books_v2.py --books /private/original_books \
  --out /private/source_provenance_audit.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from williams_ai.book_rules import registry

def audit_pdf_sources(books_folder, reviewed_cases=None):
    from pypdf import PdfReader
    from williams_ai.book_index import verify_original
    folder=Path(books_folder).expanduser().resolve()
    if not folder.is_dir() or ROOT==folder or ROOT in folder.parents:
        raise ValueError("PRIVATE_ORIGINAL_BOOK_DIRECTORY_REQUIRED")
    sources=registry()["books"]
    report=[]
    page_cache={}
    for src in sources:
        pdf=folder/src["filename"]
        match=verify_original(src["id"],pdf)
        reader=PdfReader(str(pdf),strict=True)
        if len(reader.pages)!=src["pages"]:
            raise ValueError("BOOK_PAGE_COUNT_MISMATCH")
        blank=[]
        texts={}
        for i,page in enumerate(reader.pages,1):
            value=page.extract_text() or ""
            if not value.strip():blank.append(i)
            texts[i]=value
        page_cache[src["id"]]=texts
        report.append({"edition":src["id"],"file_sha256_matches_registry":True,
                       "pages":len(reader.pages),"extractable_text_pages":len(reader.pages)-len(blank),
                       "image_only_or_empty_pages":blank,
                       "visual_figure_meaning_not_machine_verified":True})
    citations=Counter()
    invalid=[]
    if reviewed_cases is not None:
        from williams_ai.reasoning_research_v2 import verify_case
        for case in reviewed_cases:
            verify_case(case)
            ev=case.get("book_evidence")
            if ev is None:
                citations["NOT_SUPPLIED"]+=1
                continue
            text=page_cache[case["edition"]][case["pdf_page"]]
            norm=lambda s:re.sub(r"\s+"," ",s).strip().casefold()
            if not text.strip():
                citations["IMAGE_ONLY_REQUIRES_MANUAL_PAGE_REVIEW"]+=1
            elif norm(ev["excerpt"]) in norm(text):
                citations["EXACT_TEXT_EXCERPT_PRESENT_ON_PINNED_PAGE"]+=1
            else:
                citations["NOT_VERBATIM_REVIEWED_PARAPHRASE_OR_EXTRACTION_ERROR"]+=1
                invalid.append(case["case_id"])
    return {"schema":"WILLIAMS_AI_ORIGINAL_SOURCE_PROVENANCE_V2",
            "books":report, "citation_status":dict(sorted(citations.items())),
            "not_verbatim_case_ids":invalid[:100],
            "scanned_course_requires_visual_review":True,
            "all_rule_meanings_verified":False,
            "execution_allowed":False}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--books",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--reviewed-cases",type=Path)
    args=parser.parse_args()
    cases=None
    if args.reviewed_cases:
        from williams_ai.private_lora import _private_file
        src=_private_file(args.reviewed_cases)
        cases=[json.loads(line) for line in src.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    data=audit_pdf_sources(args.books,cases)
    target=args.out.expanduser().resolve()
    if target==ROOT or ROOT in target.parents or target.exists() or not target.parent.is_dir():
        raise ValueError("PRIVATE_NEW_REPORT_DESTINATION_REQUIRED")
    target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    target.chmod(0o600)
    print(json.dumps({"books":[{"edition":b["edition"],
       "extractable_text_pages":b["extractable_text_pages"],
       "pages":b["pages"]} for b in data["books"]],
       "citation_status":data["citation_status"]},ensure_ascii=False))

if __name__=="__main__":main()
