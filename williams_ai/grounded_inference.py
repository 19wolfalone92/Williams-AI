"""Give a REAL offline GGUF source pages for the CURRENT decision.

The model's static training weights are NOT changed. Private books stay only
in a user-controlled SQLite DB OUTSIDE GitHub. Per-case retrieval is checked
against the four pinned source fingerprints and is never exposed as a trade.
"""
from __future__ import annotations
from pathlib import Path
import sqlite3
from typing import Any

from .book_index import _private_database, _known_book, search_pages
from .book_rules import lookup_rules, registry
from .knowledge import query_terms

_TOPICS={
    "TC1":"волна Эллиотта AO фрактал дивергенция структура коррекция 4 5",
    "NTD":"Аллигатор фрактал Awesome Oscillator Acceleration Balance Line вход выход",
    "TC2":"Первый Второй Третий Мудрец ангуляция Super AO фрактал зубы стоп",
    "COURSE":"Profitunity Course Elliott fractal Hump Thumb Trading Partner",
}

def verified_private_book_pages(
    book_index:str|Path,edition:str, *,
    selected_setup:str|None=None,limit:int=2,
    rule_ids:tuple[str,...]=(), question:str|None=None,
)->list[dict[str,Any]]:
    """Return bounded PRIVATE source excerpts with a verified pinned edition ID.

    The excerpts are *not* assertions of book mastery, editorial accuracy or
    future trading results. Matching SHA only proves the indexed volume's
    identity record; genuine PDF index creation verified original PDF SHA.
    """
    known=_known_book(edition)
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=3:
        raise ValueError("At most three private excerpts")
    reviewed={c["id"]:c for c in registry()["rule_cards"]
              if c["source"]==edition and c["status"]=="REVIEWED_PAGES"}
    if (not isinstance(rule_ids,tuple) or len(rule_ids)>3
            or any(not isinstance(x,str) or x not in reviewed for x in rule_ids)
            or len(set(rule_ids))!=len(rule_ids)):
        raise ValueError("Only reviewed same-edition focus rules")
    if question is not None and (not isinstance(question,str) or not 1<=len(question)<=1200):
        raise ValueError("Invalid focused source question")
    db_path=_private_database(book_index)
    if not db_path.is_file():
        raise FileNotFoundError("Private book index is missing")
    conn=sqlite3.connect(f"file:{db_path}?mode=ro",uri=True)
    try:
        row=conn.execute(
            "SELECT sha256,total_pages FROM volumes WHERE book=?",
            (edition,)).fetchone()
        if row!=(known["sha256"],known["pages"]):
            raise ValueError("Private book edition/fingerprint does not match reference")
        if rule_ids:
            # A demonstrated gap must go to its actual reviewed pages,
            # rather than a broadly similar keyword anywhere in the book.
            tokens=query_terms(question or " ".join(reviewed[x]["summary"] for x in rule_ids))
            found=[]
            for priority,rule in enumerate(rule_ids):
                pages=reviewed[rule]["pdf_pages"]
                marks=",".join("?" for _ in pages)
                for page,text in conn.execute(
                    f"SELECT pdf_page,content FROM excerpts WHERE book=? AND pdf_page IN ({marks})",
                    (edition,*pages)):
                    if not isinstance(text,str) or not text.strip():
                        continue
                    lower=text.casefold()
                    score=sum(min(4,lower.count(token)) for token in tokens)
                    offsets=[lower.find(t) for t in tokens if t in lower]
                    start=max(0,min(offsets)-120) if offsets else 0
                    found.append((priority,-score,int(page),text[start:start+650]))
            found.sort(key=lambda x:(x[0],x[1],x[2]))
            selected=[]
            seen=set()
            for _,_,page,text in found:
                if (page,text) in seen:
                    continue
                seen.add((page,text))
                selected.append({"source":f"{edition}: PDF {page}","excerpt":text,
                    "status":"PRIVATE_INDEXED_PDF_TEXT_NOT_INDEPENDENTLY_INTERPRETED",
                    "pdf_page":page})
                if len(selected)==limit:
                    break
            return selected
    finally:
        conn.close()
    if selected_setup is not None and (
        not isinstance(selected_setup,str) or len(selected_setup)>80
    ):
        raise ValueError("Invalid setup name")
    # Always select from the requested edition; do not silently mix NTD
    # AO formations with TC2 Super AO or Course Thumb/Hump.
    terms=_TOPICS[edition]
    cards=lookup_rules(selected_setup or terms,edition=edition,max_cards=3)
    if selected_setup:
        terms+=" "+selected_setup
        for card in cards[:2]:
            terms+=" "+card["excerpt"][:200]
    results=search_pages(db_path,terms[:1400],edition=edition,limit=limit)
    return [{
        "source":result["source"],
        "excerpt":result["excerpt"][:650],
        "status":"PRIVATE_INDEXED_PDF_TEXT_NOT_INDEPENDENTLY_INTERPRETED",
        "pdf_page":result["page"],
    } for result in results]
