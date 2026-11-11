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
            verbatim_on_page=bool(
                originals and len(_normalise(excerpt))>=24 and
                _normalise(excerpt) in _normalise(" ".join(originals)))
            if volume[2]=="PDF_TEXT" and not verbatim_on_page:
                raise ValueError("SOURCE_EXCERPT_NOT_PRESENT_ON_PINNED_PAGE")
            # Visual-only Course page 56 has no OCR. Human-reviewed case-level
            # image attestations can authorize a paraphrased visual statement
            # without inventing missing PDF text.
            signed_visual_case=False
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
                from .signed_visual_reviews_v3 import verify_two_ed25519_course_approvals
                verify_two_ed25519_course_approvals(
                    db,case_sha256=case_sha,
                    pdf_page=row["pdf_page"],image_sha256=page[0],
                    source_sha256=volume[0],expected_reviewers=reviewers)
                signed_visual_case=True
            successful.append({"case_id":row["case_id"],
                               "edition":book,"pdf_page":row["pdf_page"],
                               "source_index_sha256":known["sha256"],
                               "case_sha256":digest({k:v for k,v in row.items()
                                                     if k!="reviewers"}),
                               "page_text_found":verbatim_on_page,
                               "signed_visual_interpretation":signed_visual_case,
                               "source_is_ocr":volume[2]!="PDF_TEXT"})
    return {"schema":"WILLIAMS_AI_PRIVATE_BOOK_CASE_SOURCE_GATE_V3",
            "verified_case_count":len(successful),
            "case_receipts":successful,
            "human_review_identity_cryptographically_authenticated":False,
            "independent_expert_judgement_proven_by_software":False,
            "execution_allowed":False}
