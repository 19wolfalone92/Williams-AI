"""Resumable local book study. Memory is evidence, never execution authority.

Generated summaries remain hypotheses. Source fingerprints, page numbers and
supporting spans are recorded; no model can certify its own interpretation.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from .book_index import _private_database, _known_book
from .book_rules import lookup_rules


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plain(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


@contextmanager
def memory(path: str | Path):
    dest = _private_database(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        import os
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(fd)
    db = sqlite3.connect(dest, timeout=15)
    try:
        db.execute("""CREATE TABLE IF NOT EXISTS study_notes(
            edition TEXT NOT NULL, source_sha TEXT NOT NULL, page INTEGER NOT NULL,
            chunk_sha TEXT NOT NULL, model_id TEXT NOT NULL, result TEXT NOT NULL,
            learned_at TEXT NOT NULL,
            PRIMARY KEY(edition,source_sha,page,chunk_sha,model_id))""")
        # Old study_errors was keyed only by chunk+model and could let
        # failing identical text in TC1 block Course/NTD/TC2. Keep legacy
        # table for non-destructive upgrades; all new retries are scoped.
        db.execute("""CREATE TABLE IF NOT EXISTS study_errors(
            chunk_sha TEXT NOT NULL, model_id TEXT NOT NULL, attempts INTEGER NOT NULL,
            error_type TEXT NOT NULL, PRIMARY KEY(chunk_sha,model_id))""")
        db.execute("""CREATE TABLE IF NOT EXISTS scoped_study_errors(
            edition TEXT NOT NULL,source_sha TEXT NOT NULL,page INTEGER NOT NULL,
            chunk_sha TEXT NOT NULL,model_id TEXT NOT NULL,attempts INTEGER NOT NULL,
            error_type TEXT NOT NULL,
            PRIMARY KEY(edition,source_sha,page,chunk_sha,model_id))""")
        db.execute("""CREATE TABLE IF NOT EXISTS research_sources(
            source TEXT PRIMARY KEY, title TEXT NOT NULL, excerpt TEXT NOT NULL,
            retrieved_at TEXT NOT NULL, status TEXT NOT NULL)""")
        yield db
        db.commit()
    finally:
        db.close()


def _source(database: str | Path, edition: str) -> sqlite3.Connection:
    known = _known_book(edition)
    path = _private_database(database)
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    row = db.execute("SELECT sha256,total_pages FROM volumes WHERE book=?", (edition,)).fetchone()
    if row != (known["sha256"], known["pages"]):
        db.close()
        raise ValueError("Verified edition is missing or its fingerprint changed")
    return db


def validate_lesson(raw: str, passage: str) -> dict[str, Any]:
    """A quoted span proves provenance only, not that the conclusion is true."""
    if not isinstance(raw, str) or len(raw) > 12000:
        raise ValueError("Oversized lesson")
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) != {"summary", "claims", "questions"}:
        raise ValueError("Invalid lesson fields")
    if not isinstance(obj["summary"], str) or not 1 <= len(obj["summary"]) <= 1500:
        raise ValueError("Invalid summary")
    if not isinstance(obj["claims"], list) or len(obj["claims"]) > 6:
        raise ValueError("Too many claims")
    for claim in obj["claims"]:
        if not isinstance(claim, dict) or set(claim) != {"statement", "support"}:
            raise ValueError("Invalid claim")
        if not isinstance(claim["statement"], str) or not 1 <= len(claim["statement"]) <= 500:
            raise ValueError("Invalid statement")
        support = claim["support"]
        if not isinstance(support, str) or not 12 <= len(support) <= 240:
            raise ValueError("Invalid support span")
        if _plain(support).casefold() not in _plain(passage).casefold():
            raise ValueError("Invented supporting span")
    if (not isinstance(obj["questions"], list) or len(obj["questions"]) > 6
            or any(not isinstance(x, str) or not 1 <= len(x) <= 350 for x in obj["questions"])):
        raise ValueError("Invalid questions")
    return {**obj, "status": "SOURCE_DERIVED_UNVERIFIED", "trade_action": None}


def _lesson_error_code(error: Exception) -> str:
    """Keep actionable diagnostics without logging book text or model output."""
    if isinstance(error, json.JSONDecodeError):
        return "LESSON_INVALID_JSON"
    codes = {
        "Oversized lesson": "LESSON_OVERSIZED",
        "Invalid lesson fields": "LESSON_INVALID_FIELDS",
        "Invalid summary": "LESSON_INVALID_SUMMARY",
        "Too many claims": "LESSON_INVALID_CLAIM_COUNT",
        "Invalid claim": "LESSON_INVALID_CLAIM",
        "Invalid statement": "LESSON_INVALID_STATEMENT",
        "Invalid support span": "LESSON_INVALID_SUPPORT_SPAN",
        "Invented supporting span": "LESSON_SUPPORT_NOT_IN_SOURCE",
        "Invalid questions": "LESSON_INVALID_QUESTIONS",
        "LOCAL_STUDY_OUTPUT_TRUNCATED": "LOCAL_STUDY_OUTPUT_TRUNCATED",
    }
    return codes.get(str(error), type(error).__name__)


def study_batch(book_db: str | Path, memory_db: str | Path, edition: str,
                model: Any, *, model_id: str, budget: int = 4,
                pdf_pages: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Study unseen source chunks, checkpointing each success; bounded retry on failure."""
    if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,120}", model_id) or not 1 <= budget <= 100:
        raise ValueError("Invalid model version or batch budget")
    if _private_database(book_db) == _private_database(memory_db):
        raise ValueError("Books and study memory must be separate files")
    known = _known_book(edition)
    if pdf_pages is not None and (
            not isinstance(pdf_pages, tuple) or not 1 <= len(pdf_pages) <= 32
            or any(isinstance(p, bool) or not isinstance(p, int)
                   or not 1 <= p <= known["pages"] for p in pdf_pages)
            or len(set(pdf_pages)) != len(pdf_pages)):
        raise ValueError("Invalid source PDF page selection")
    source = _source(book_db, edition)
    done = failed = 0
    failure_codes: Counter[str] = Counter()
    try:
        with memory(memory_db) as dest:
            query = "SELECT pdf_page,content FROM excerpts WHERE book=?"
            args: list[Any] = [edition]
            if pdf_pages is not None:
                query += " AND CAST(pdf_page AS INTEGER) IN (" + ",".join("?" for _ in pdf_pages) + ")"
                args.extend(pdf_pages)
            rows = source.execute(query + " ORDER BY CAST(pdf_page AS INTEGER),rowid", args)
            for page, text in rows:
                sha = fingerprint(text)
                key = (edition, _known_book(edition)["sha256"], int(page), sha, model_id)
                if dest.execute("SELECT 1 FROM study_notes WHERE edition=? AND source_sha=? AND page=? AND chunk_sha=? AND model_id=?", key).fetchone():
                    continue
                error = dest.execute(
                    "SELECT attempts FROM scoped_study_errors "
                    "WHERE edition=? AND source_sha=? AND page=? AND chunk_sha=? AND model_id=?",
                    key).fetchone()
                if error and error[0] >= 3:
                    continue
                if done + failed >= budget:
                    break
                try:
                    cards = lookup_rules(text[:1400], edition=edition, max_cards=2)
                    raw = model.study_passage(text, edition=edition, page=int(page), rule_cards=cards)
                    lesson = validate_lesson(raw, text)
                    lesson["cross_check"] = "REQUIRES_INDEPENDENT_RULE_AND_GRAPH_REVIEW"
                    lesson["rule_references"] = [x["rule_id"] for x in cards]
                    dest.execute("INSERT INTO study_notes VALUES (?,?,?,?,?,?,?)", (*key, json.dumps(lesson, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
                    dest.execute(
                        "DELETE FROM scoped_study_errors "
                        "WHERE edition=? AND source_sha=? AND page=? AND chunk_sha=? AND model_id=?",
                        key)
                    done += 1
                except (ValueError, RuntimeError, TypeError, KeyError) as exc:
                    # Store only allowlisted codes: arbitrary exception text
                    # can contain private book passages or local paths.
                    code = _lesson_error_code(exc)
                    dest.execute(
                        "INSERT INTO scoped_study_errors VALUES (?,?,?,?,?,1,?) "
                        "ON CONFLICT(edition,source_sha,page,chunk_sha,model_id) "
                        "DO UPDATE SET attempts=attempts+1,error_type=excluded.error_type",
                        (*key,code))
                    failed += 1
                    failure_codes[code] += 1
                dest.commit()
    finally:
        source.close()
    return {"edition": edition, "studied_chunks": done, "failed_chunks": failed,
            "failure_codes": dict(sorted(failure_codes.items())),
            "model_id": model_id, "status": "RESEARCH_ONLY", "trade_action": None}


def study_report(book_db: str | Path, memory_db: str | Path,
                 *, model_id: str | None = None) -> dict[str, Any]:
    """Per-model page-by-page progress, not aggregated across unrelated GGUFs.

    A full indexed PDF is not the same as a model having actually processed
    its text; a processed chunk is NOT proof the student understood it.
    Previously counting DISTINCT chunk hashes across all model IDs falsely
    represented multiple partly trained students as one fully trained one.
    """
    if model_id is not None and not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,120}", model_id):
        raise ValueError("Invalid model ID for book-progress attribution")
    path = _private_database(book_db)
    source = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        with memory(memory_db) as dest:
            learners = [row[0] for row in dest.execute(
                "SELECT DISTINCT model_id FROM study_notes ORDER BY model_id")]
            books = []
            for edition, sha, total, extracted, missing in source.execute(
                    "SELECT book,sha256,total_pages,extracted_pages,missing_pages "
                    "FROM volumes ORDER BY book"):
                # An OCR correction or page re-index can replace passages
                # WITHOUT changing the original PDF SHA. Only count notes
                # matching the *current* page+chunk digest; stale knowledge
                # must never masquerade as full model reading.
                current={
                    (int(page),fingerprint(chunk))
                    for page,chunk in source.execute(
                        "SELECT pdf_page,content FROM excerpts WHERE book=?",
                        (edition,))
                }
                chunks=len(current)
                def student_count(student: str) -> dict[str, Any]:
                    historical={
                        (int(page),chunk_sha) for page,chunk_sha in dest.execute(
                            "SELECT page,chunk_sha FROM study_notes "
                            "WHERE edition=? AND source_sha=? AND model_id=?",
                            (edition,sha,student))
                    }
                    valid=current & historical
                    n=len(valid)
                    pages=len({page for page,_ in valid})
                    return {"studied_chunks":n,
                            "pages_started":pages,
                            "indexed_text_chunks_remaining":chunks-n,
                            "stale_or_superseded_chunks_ignored":len(historical-valid),
                            "all_pdf_pages_have_examined_text":(
                                n==chunks and pages==total and missing==0
                            )}
                per_model={student:student_count(student) for student in learners}
                if model_id:
                    selected=student_count(model_id)
                elif per_model:
                    # Backward-compatible unscoped display: the BEST single
                    # student's count, never SUM of different students.
                    selected=max(per_model.values(),
                                 key=lambda x:(x["studied_chunks"],x["pages_started"]))
                else:
                    selected={"studied_chunks":0,"pages_started":0,
                              "indexed_text_chunks_remaining":chunks,
                              "all_pdf_pages_have_examined_text":False}
                books.append({
                    "edition":edition,"total_pages":total,
                    "pages_with_text":extracted,"pages_without_text":missing,
                    "source_chunks":chunks,
                    "studied_chunks":selected["studied_chunks"] if selected else None,
                    "pages_started":selected["pages_started"] if selected else None,
                    "indexed_text_chunks_remaining":(
                        selected["indexed_text_chunks_remaining"] if selected else None),
                    "per_model_progress":per_model if model_id is None else None,
                    "model_has_reviewed_all_pdf_page_text":(
                        selected["all_pdf_pages_have_examined_text"] if selected else False
                    ),
                    "book_mastery_verified":False,
                })
            if model_id:
                errors=dest.execute(
                    "SELECT count(*) FROM scoped_study_errors WHERE model_id=?",
                    (model_id,)).fetchone()[0]
            else:
                errors=dest.execute("SELECT count(*) FROM scoped_study_errors").fetchone()[0]
        return {
            "books":books,
            "model_id":model_id,
            "tracked_model_ids":learners if model_id is None else [model_id],
            "unscoped_progress_not_summed_across_models":True,
            "unscoped_counts_represent_best_single_model_not_any_one_named_GGUF":(
                model_id is None),
            "chunks_needing_retry_or_review":errors,
            "book_comprehension_percent":None,
            "weights_retrained":False,
            "status":"SOURCE_GROUNDED_READ_ONLY_UNVERIFIED", "trade_action":None,
            "execution_allowed":False,
        }
    finally:
        source.close()


def recall(memory_db: str | Path, question: str, *, edition: str | None = None,
           limit: int = 3, model_id: str | None = None,
           book_index: str | Path | None = None) -> list[dict[str, str]]:
    """Recall model-scoped, source-identified study notes as UNVERIFIED hints.

    For decisions involving a specific GGUF, model_id is mandatory at the
    caller: unrelated students must never lend each other apparent mastery.
    An LLM-written summary is not a canonical trading rule.
    With a current source index, an OCR-corrected or removed passage must
    invalidate its old lesson at inference time as well as in progress reports.
    """
    if not isinstance(question,str) or len(question)>1500 or not 1<=limit<=5:
        raise ValueError("Invalid memory query")
    if edition:
        _known_book(edition)
    if model_id is not None and not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,120}",model_id):
        raise ValueError("Invalid GGUF model fingerprint")
    current = None
    if book_index is not None:
        if _private_database(book_index) == _private_database(memory_db):
            raise ValueError("Books and study memory must be separate files")
        from .book_rules import registry
        current = set()
        editions = [edition] if edition else [b["id"] for b in registry()["books"]]
        for book in editions:
            source = _source(book_index, book)
            try:
                current.update((book, int(page), fingerprint(text))
                               for page, text in source.execute(
                                   "SELECT pdf_page,content FROM excerpts WHERE book=?",
                                   (book,)))
            finally:
                source.close()
    terms=set(re.findall(r"[^\W_]{3,}",question.casefold()))
    ranked=[]
    with memory(memory_db) as db:
        query="SELECT edition,page,result,model_id,source_sha,chunk_sha FROM study_notes"
        conditions=[]
        args=[]
        if edition:
            conditions.append("edition=?")
            args.append(edition)
        if model_id is not None:
            conditions.append("model_id=?")
            args.append(model_id)
        if conditions:
            query+=" WHERE "+" AND ".join(conditions)
        for book,page,content,student,source_sha,chunk_sha in db.execute(query,args):
            # Reject retained notes referring to superseded source editions.
            if source_sha!=_known_book(book)["sha256"]:
                continue
            if current is not None and (book, int(page), chunk_sha) not in current:
                continue
            obj=json.loads(content)
            if not isinstance(obj,dict) or obj.get("status")!="SOURCE_DERIVED_UNVERIFIED":
                continue
            summary=obj.get("summary")
            if not isinstance(summary,str):
                continue
            score=len(terms & set(re.findall(r"[^\W_]{3,}",summary.casefold())))
            if score:
                ranked.append((score,{"source":f"STUDY:{book}: PDF {page}",
                                      "excerpt":"Непроверенный учебный вывод: "+summary[:430],
                                      "status":"MODEL_SPECIFIC_SOURCE_DERIVED_UNVERIFIED",
                                      "model_id":student}))
    ranked.sort(key=lambda x:(-x[0],x[1]["source"],x[1]["model_id"]))
    return [item for _,item in ranked[:limit]]


def remember_web(memory_db: str | Path, result: dict[str, Any]) -> int:
    from urllib.parse import urlsplit
    from .web_research import HOSTS
    count = 0
    with memory(memory_db) as db:
        for item in result.get("sources", [])[:6]:
            parsed = urlsplit(item["source"])
            if (parsed.scheme != "https" or parsed.hostname not in HOSTS
                    or parsed.port or parsed.username or parsed.password):
                raise ValueError("Unrecognized research source")
            at = datetime.fromisoformat(item["retrieved_at"])
            if at.tzinfo is None:
                raise ValueError("Research timestamp must have timezone")
            db.execute("INSERT OR REPLACE INTO research_sources VALUES (?,?,?,?,?)",
                       (item["source"], str(item["title"])[:150], str(item["excerpt"])[:1000],
                        at.isoformat(), "WEB_EVIDENCE_UNVERIFIED"))
            count += 1
    return count


def recall_web(memory_db: str | Path, question: str, *, max_age_hours: float = 24) -> list[dict[str, str]]:
    if not isinstance(question, str) or len(question) > 1500 or not 0 < max_age_hours <= 168:
        raise ValueError("Invalid research cache query")
    words = set(re.findall(r"[^\W_]{3,}", question.casefold()))
    ranked = []
    now = datetime.now(timezone.utc)
    with memory(memory_db) as db:
        for url, title, excerpt, at, status in db.execute("SELECT * FROM research_sources"):
            age = (now - datetime.fromisoformat(at)).total_seconds() / 3600
            if not 0 <= age <= max_age_hours:
                continue
            score = len(words & set(re.findall(r"[^\W_]{3,}", (title + " " + excerpt).casefold())))
            if score:
                ranked.append((score, {"source": url, "excerpt": excerpt,
                                       "retrieved_at": at, "status": status}))
    ranked.sort(key=lambda x: (-x[0], x[1]["source"]))
    return [x for _, x in ranked[:3]]
