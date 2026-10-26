"""Private four-book page coverage and safe revision bookmarks.

Read-only SQLite, no OCR, no copied page text, no fictitious book mastery.
"""
from __future__ import annotations
from pathlib import Path
import sqlite3
from typing import Any

ED=("TC1","NTD","TC2","COURSE")


def page_coverage(private_index:Path, *, include_indexed:bool=False,
                  max_pages:int=200, registry_data:dict[str,Any]|None=None)->dict[str,Any]:
    if type(include_indexed) is not bool or type(max_pages) is not int or not 1<=max_pages<=10000:
        raise ValueError("BOUNDED_BOOK_COVERAGE_REQUEST_REQUIRED")
    from .book_index import _private_database
    dbpath=_private_database(private_index)
    if dbpath.is_symlink() or not dbpath.is_file():
        raise ValueError("LOCAL_PRIVATE_BOOK_INDEX_REQUIRED")
    if registry_data is None:
        from .book_rules import registry
        registry_data=registry()
    books=registry_data.get("books") if isinstance(registry_data,dict) else None
    cards=registry_data.get("rule_cards") if isinstance(registry_data,dict) else None
    if not isinstance(books,list) or not isinstance(cards,list):
        raise ValueError("FOUR_BOOK_REGISTRY_REQUIRED")
    reference={item["id"]:item for item in books}
    if set(reference)!=set(ED):
        raise ValueError("ALL_FOUR_EDITION_SOURCES_REQUIRED")
    with sqlite3.connect(dbpath.resolve().as_uri()+"?mode=ro",uri=True) as db:
        records={row[0]:row[1:] for row in db.execute(
            "SELECT book,sha256,total_pages,extracted_pages,missing_pages,source_type FROM volumes")}
        per_page={(ed,int(page)):n for ed,page,n in db.execute(
            "SELECT book,pdf_page,COUNT(*) FROM excerpts GROUP BY book,pdf_page")}
    report={}
    for ed in ED:
        meta=reference[ed]
        total=meta["pages"]
        row=records.get(ed)
        verified=bool(row and row[0]==meta["sha256"] and row[1]==total)
        pointers={}
        for card in cards:
            if card.get("source")!=ed or card.get("status")!="REVIEWED_PAGES":
                continue
            for page in card["pdf_pages"]:
                if type(page) is not int or not 1<=page<=total:
                    raise ValueError("REVIEWED_CARD_PAGE_OUT_OF_BOUNDS")
                pointers.setdefault(page,[]).append(card["id"])
        all_pages=[]
        empty=0
        for page in range(1,total+1):
            indexed=per_page.get((ed,page),0) if verified else 0
            empty+=int(indexed==0)
            if include_indexed or not indexed:
                all_pages.append({"pdf_page":page,"text_chunk_count":indexed,
                                  "reviewed_rule_ids":sorted(pointers.get(page,[]))})
        report[ed]={"verified_local_source_identity":verified,
            "declared_pdf_pages":total,"pages_with_text":total-empty,
            "pages_without_text":empty,"source_type":row[4] if verified else None,
            "indexed_text_is_not_image_review":True,
            "indexed_text_is_not_model_comprehension":True,
            "bookmarks":all_pages[:max_pages],
            "more_bookmarks":max(0,len(all_pages)-max_pages)}
    return {"mode":"PRIVATE_FOUR_BOOK_SOURCE_COVERAGE_NOT_MASTERED_PERCENT",
            "books":report,"book_mastery_percent":None,
            "private_source_text_exported":False,"execution_allowed":False}
