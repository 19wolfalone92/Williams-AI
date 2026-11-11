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
        pages=db.execute("SELECT pdf_page,page_image_path,page_image_sha256,"
                         "source_sha256 FROM course_pages "
                         "ORDER BY pdf_page").fetchall()
        if [r[0] for r in pages] != list(range(1,123)):
            raise ValueError("NOT_ALL_122_PAGE_IMAGES_MAPPED")
        for page_number,relative,checksum,source_hash in pages:
            rel=Path(relative)
            if (rel.is_absolute() or ".." in rel.parts or
                    not str(rel).startswith("page_images/") or
                    source_hash!=known["sha256"]):
                raise ValueError("INVALID_PRIVATE_PAGE_ORIGINAL_PROVENANCE")
            img=path.parent/rel
            if (not img.is_file() or img.is_symlink() or
                    hashlib.sha256(img.read_bytes()).hexdigest()!=checksum):
                raise ValueError("COURSE_PAGE_IMAGE_HASH_MISMATCH_OR_MISSING")
        annotated=db.execute("SELECT COUNT(*) FROM course_visuals").fetchone()[0]
        statuses=db.execute("SELECT DISTINCT semantic_review_status FROM course_pages").fetchall()
        if any(s[0] != "NOT_INDEPENDENTLY_VERIFIED" for s in statuses):
            raise ValueError("UNEXPECTED_UNVERIFIED_VISUAL_AUDIT_STATUS")
    return {"schema":"COURSE_VISUAL_PRIVATE_AUDIT_V1","edition":"COURSE",
            "source_sha256":known["sha256"],"pages_with_images":122,
            "illustrated_page_annotations":annotated,
            "all_semantic_links_expert_verified":False,
            "retrieval_allowed":True,"use_as_verified_training_gold":False}


def combine_with_existing_private_books(existing_database: str | Path,
                                        course_database: str | Path,
                                        *, destination: str | Path
                                       ) -> dict[str, Any]:
    """Create an atomic COPY of an existing multi-book index + scanned Course.

    NEVER edits the original books DB or commits copyrighted data to Git.
    Destination MUST live beside the private Course images, because visual
    references are pinned to page_images/ paths in that private directory.
    """
    import os
    from .book_rules import registry
    source = _private_database(existing_database)
    course = _private_database(course_database)
    target = _private_database(destination)
    if (not source.is_file() or not course.is_file()
            or source == course or target in (source,course)
            or target.exists() or target.is_symlink()
            or target.parent != course.parent):
        raise ValueError("NEW_COMBINED_INDEX_BESIDE_PRIVATE_IMAGES_REQUIRED")
    course_visual_audit(course)
    known = {b["id"]: b for b in registry()["books"]}
    with sqlite3.connect(f"file:{source}?mode=ro",uri=True) as original:
        source_rows = original.execute(
            "SELECT book,sha256,total_pages FROM volumes"
        ).fetchall()
        for edition,sha,pages in source_rows:
            if (edition not in known or sha!=known[edition]["sha256"]
                    or pages!=known[edition]["pages"]):
                raise ValueError("EXISTING_BOOK_SOURCE_IDENTITY_MISMATCH")
        fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
        os.close(fd)
        try:
            with sqlite3.connect(str(target)) as result:
                original.backup(result)
            with sqlite3.connect(str(target)) as result:
                result.execute("ATTACH DATABASE ? AS scanned",
                               (f"file:{course}?mode=ro",))
                # Copy only the pinned Course's source and page-visual tables.
                with result:
                    result.execute("DELETE FROM excerpts WHERE book='COURSE'")
                    result.execute("DELETE FROM volumes WHERE book='COURSE'")
                    result.execute("INSERT INTO volumes SELECT * FROM scanned.volumes "
                                   "WHERE book='COURSE'")
                    result.execute("INSERT INTO excerpts SELECT * FROM scanned.excerpts "
                                   "WHERE book='COURSE'")
                    for name in ("course_pages","course_visuals","course_paragraphs"):
                        if name not in {r[0] for r in result.execute(
                            "SELECT name FROM scanned.sqlite_master WHERE type='table'")}:
                            raise ValueError("MISSING_PRIVATE_VISUAL_TABLE")
                        # Compound copy only, not a destructive mutation of originals.
                        result.execute(f"DROP TABLE IF EXISTS main.{name}")
                        result.execute(f"CREATE TABLE main.{name} AS "
                                       f"SELECT * FROM scanned.{name}")
                result.execute("DETACH DATABASE scanned")
                if result.execute("PRAGMA quick_check").fetchone()[0]!="ok":
                    raise ValueError("COMBINED_INDEX_SQLITE_INTEGRITY_FAILURE")
            merged=course_visual_audit(target)
            with sqlite3.connect(f"file:{target}?mode=ro",uri=True) as result:
                output_editions={r[0] for r in result.execute(
                    "SELECT book FROM volumes")}
            return {"schema":"WILLIAMS_PRIVATE_COURSE_COMBINED_INDEX_V1",
                    "combined_database":str(target),
                    "editions":sorted(output_editions),
                    "existing_index_untouched":True,
                    "course_page_images_verified":merged["pages_with_images"],
                    "private_ocr_not_training_gold":True}
        except Exception:
            target.unlink(missing_ok=True)
            raise


def search_course_visual_descriptions(database: str | Path, question: str,
                                      *, limit: int = 2) -> list[dict[str, Any]]:
    """Find even OCR-empty illustrated pages (e.g. diagram-only PDF page 56).

    Descriptions are visually aligned annotations, NOT literal PDF text or
    independently verified technical truth. Search is local and read-only.
    """
    import re
    if (not isinstance(question,str) or not 1<=len(question)<=1200
            or type(limit) is not int or not 1<=limit<=3):
        raise ValueError("BOUNDED_PRIVATE_FIGURE_SEARCH_REQUIRED")
    tokens={t for t in re.findall(r"[a-z0-9]{4,}",question.casefold())}
    if not tokens:
        return []
    path=_private_database(database)
    with sqlite3.connect(f"file:{path}?mode=ro",uri=True) as db:
        tables={r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "course_visuals" not in tables:
            return []
        records=db.execute("SELECT pdf_page,description FROM course_visuals").fetchall()
    ranked=[]
    for page,description in records:
        words=set(re.findall(r"[a-z0-9]{4,}",description.casefold()))
        score=len(tokens & words)
        if score:
            ranked.append((-score,page,description))
    ranked.sort()
    found=[]
    for _,page,description in ranked[:limit]:
        refs=course_visual_refs(path,page)
        if not refs:
            continue
        found.append({
            "source":f"COURSE: PDF {page}",
            "excerpt":"[UNVERIFIED VISUAL DESCRIPTION, NOT OCR TEXT] "+description[:500],
            "pdf_page":page,
            "chunk_sha256":hashlib.sha256(description.encode("utf-8")).hexdigest(),
            "status":"PRIVATE_VISUAL_DESCRIPTION_UNVERIFIED_NEEDS_PAGE_IMAGE_REVIEW",
            "visual_refs":refs,
            "visual_meaning_not_authoritatively_checked":True,
        })
    return found
