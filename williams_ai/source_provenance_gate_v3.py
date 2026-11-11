"""Strict PRIVATE book-source gate for expert QLoRA research lessons.

Original PDF file hash, page, extractable text and page-image review status
are checked from the user's private SQLite, never from model-generated claims.
A declared reviewer identity is NOT independently authenticated; this gate
cannot replace expert examination of author meaning.
"""
from __future__ import annotations
from pathlib import Path
import re
import sqlite3

from .book_index import _private_database
from .book_rules import registry
from .reasoning_research_v2 import verify_case, digest


def _normalise(value):
    return re.sub(r"\s+"," ",str(value)).strip().casefold()


def check_private_expert_source_cases(rows,private_book_index):
    path=_private_database(private_book_index)
    if not path.is_file():
        raise FileNotFoundError("PRIVATE_FOUR_BOOK_SOURCE_INDEX_REQUIRED")
    records=registry()
    registry_books={book["id"]:book for book in records["books"]}
    successful=[]
    with sqlite3.connect(f"file:{path}?mode=ro",uri=True) as db:
        for row in rows:
            verify_case(row)
            if "book_evidence" not in row:
                raise ValueError("VERIFIED_PRIVATE_SOURCE_EXCERPT_REQUIRED")
            book=row["edition"]
            known=registry_books[book]
            volume=db.execute(
                "SELECT sha256,total_pages,source_type FROM volumes WHERE book=?",
                (book,)).fetchone()
            if (not volume or volume[:2]!=(known["sha256"],known["pages"])
                    or volume[2] not in {"PDF_TEXT",
                                         "PDF_TEXT_WITH_UNREVIEWED_OCR"}):
                raise ValueError("WRONG_BOOK_VERSION_OR_UNKNOWN_SOURCE")
            excerpt=row["book_evidence"]["excerpt"]
            originals=[x[0] for x in db.execute(
                "SELECT content FROM excerpts WHERE book=? AND pdf_page=?",
                (book,row["pdf_page"]))]
            if (not originals or len(_normalise(excerpt))<24
                    or _normalise(excerpt) not in _normalise(" ".join(originals))):
                raise ValueError("SOURCE_EXCERPT_NOT_PRESENT_ON_PINNED_PAGE")
            # Even perfect OCR text matching cannot approve a scanned figure.
            if volume[2]=="PDF_TEXT_WITH_UNREVIEWED_OCR":
                # Lesson must be independently reviewed at the exact
                # page + case SHA, not just have checked_by_reviewers=true.
                if book!="COURSE":
                    raise ValueError("OCR_VERIFICATION_REQUIRED")
                tables={t[0] for t in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                if not {"course_pages","course_expert_case_reviews_v3"}<=tables:
                    raise ValueError("NO_INDEPENDENTLY_REVIEWED_COURSE_CASES")
                page=db.execute(
                    "SELECT page_image_sha256,semantic_review_status "
                    "FROM course_pages WHERE pdf_page=?",(row["pdf_page"],)).fetchone()
                if not page or page[1]!="HUMAN_CHECKED_TEXT_AND_DIAGRAMS":
                    raise ValueError("COURSE_PAGE_VISUAL_SEMANTICS_UNVERIFIED")
                case_sha=digest({k:v for k,v in row.items() if k!="reviewers"})
                reviewers={r["reviewer_id"] for r in row["reviewers"]}
                approvals=db.execute(
                    "SELECT reviewer_id FROM course_expert_case_reviews_v3 "
                    "WHERE case_sha256=? AND pdf_page=? AND image_sha256=? "
                    "AND source_sha256=? AND verdict='APPROVED'",
                    (case_sha,row["pdf_page"],page[0],volume[0]),
                ).fetchall()
                observed={x[0] for x in approvals}
                if not reviewers<=observed or len(observed)<2:
                    raise ValueError("UNATTESTED_VISUAL_CASE_APPROVAL")
            successful.append({"case_id":row["case_id"],
                               "edition":book,"pdf_page":row["pdf_page"],
                               "source_index_sha256":known["sha256"],
                               "case_sha256":digest({k:v for k,v in row.items()
                                                     if k!="reviewers"}),
                               "page_text_found":True,
                               "source_is_ocr":volume[2]!="PDF_TEXT"})
    return {"schema":"WILLIAMS_AI_PRIVATE_BOOK_CASE_SOURCE_GATE_V3",
            "verified_case_count":len(successful),
            "case_receipts":successful,
            "human_review_identity_cryptographically_authenticated":False,
            "independent_expert_judgement_proven_by_software":False,
            "execution_allowed":False}
