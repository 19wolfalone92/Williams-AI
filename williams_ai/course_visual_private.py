"""Private visual references for the user-provided 122-page Profitunity Course.

Reads only local user-owned SQLite + page images OUTSIDE public Git.
A page-image SHA binds illustration references to their PDF page. Automatic
OCR alignment is *not* an expert interpretation of Elliott/fractal diagrams.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from .book_index import _private_database, _known_book


def course_visual_refs(database: str | Path, pdf_page: int,
                       *, verify_images: bool = True, limit: int = 2
                       ) -> list[dict[str, Any]]:
    if (type(pdf_page) is not int or not 1 <= pdf_page <= 122 or
            type(limit) is not int or not 1 <= limit <= 3):
        raise ValueError("BOUNDED_COURSE_PAGE_AND_VISUAL_LIMIT_REQUIRED")
    path = _private_database(database)
    if not path.is_file():
        raise FileNotFoundError("PRIVATE_BOOK_INDEX_REQUIRED")
    known = _known_book("COURSE")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        row = db.execute(
            "SELECT sha256,total_pages,source_type FROM volumes WHERE book='COURSE'"
        ).fetchone()
        if (row is None or row[0] != known["sha256"] or row[1] != known["pages"]):
            raise ValueError("ORIGINAL_COURSE_BOOK_HASH_OR_PAGES_MISMATCH")
        has_tables = {x[0] for x in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if not {"course_pages", "course_visuals"} <= has_tables:
            return []  # Legacy indexed books are text-only.
        page = db.execute(
            "SELECT source_sha256,page_image_path,page_image_sha256,"
            "semantic_review_status FROM course_pages WHERE pdf_page=?",
            (pdf_page,),
        ).fetchone()
        if not page or page[0] != known["sha256"]:
            raise ValueError("UNPINNED_PRIVATE_VISUAL_PAGE")
        entries = db.execute(
            "SELECT figure_id,description,page_image_path,image_sha256,"
            "association_method,semantic_status FROM course_visuals "
            "WHERE pdf_page=? ORDER BY figure_id LIMIT ?",
            (pdf_page,limit),
        ).fetchall()
    refs=[]
    for figure_id, desc, rel, image_sha, method, semantic in entries:
        if not (isinstance(figure_id,str) and
                figure_id.startswith(f"COURSE:P{pdf_page:03d}:") and
                isinstance(desc,str) and 10<=len(desc)<=500 and
                rel==page[1] and image_sha==page[2] and
                semantic=="UNVERIFIED_VISUAL_DESCRIPTION" and
                method=="PAGE_PROVENANCE_AND_OCR_LEXICAL_OVERLAP"):
            raise ValueError("INVALID_OR_UNREVIEWED_FIGURE_PROVENANCE")
        relpath=Path(rel)
        if (relpath.is_absolute() or ".." in relpath.parts or
                not str(relpath).startswith("page_images/")):
            raise ValueError("UNSAFE_PRIVATE_VISUAL_IMAGE_PATH")
        image_file=path.parent/relpath
        if verify_images:
            if (not image_file.is_file() or image_file.is_symlink() or
                    hashlib.sha256(image_file.read_bytes()).hexdigest()!=image_sha):
                raise ValueError("VISUAL_PAGE_IMAGE_SHA256_MISMATCH_OR_MISSING")
        refs.append({
            "figure_id":figure_id,"pdf_page":pdf_page,
            "description":desc[:500],
            "image_sha256":image_sha,
            "image_relative_path":rel,
            "image_loaded_by_text_model":False,
            "source_status":"COURSE_PAGE_VISUAL_AUTO_LINKED_UNVERIFIED",
            "expert_figure_interpretation_verified":False,
        })
    return refs


def course_visual_audit(database: str | Path) -> dict[str, Any]:
    path=_private_database(database)
    known=_known_book("COURSE")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        volumes=db.execute("SELECT sha256,total_pages,source_type FROM volumes "
                           "WHERE book='COURSE'").fetchone()
        if volumes != (known["sha256"],122,"PDF_TEXT_WITH_UNREVIEWED_OCR"):
            raise ValueError("NOT_THE_PINNED_122_PAGE_OCR_COURSE")
        pages=db.execute("SELECT pdf_page,page_image_sha256 FROM course_pages "
                         "ORDER BY pdf_page").fetchall()
        if [r[0] for r in pages] != list(range(1,123)):
            raise ValueError("NOT_ALL_122_PAGE_IMAGES_MAPPED")
        annotated=db.execute("SELECT COUNT(*) FROM course_visuals").fetchone()[0]
        statuses=db.execute("SELECT DISTINCT semantic_review_status FROM course_pages").fetchall()
        if any(s[0] != "NOT_INDEPENDENTLY_VERIFIED" for s in statuses):
            raise ValueError("UNEXPECTED_UNVERIFIED_VISUAL_AUDIT_STATUS")
    return {"schema":"COURSE_VISUAL_PRIVATE_AUDIT_V1","edition":"COURSE",
            "source_sha256":known["sha256"],"pages_with_images":122,
            "illustrated_page_annotations":annotated,
            "all_semantic_links_expert_verified":False,
            "retrieval_allowed":True,"use_as_verified_training_gold":False}
