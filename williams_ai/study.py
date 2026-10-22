"""Resumable local book study. Memory is evidence, never execution authority.

Generated summaries remain hypotheses. Source fingerprints, page numbers and
supporting spans are recorded; no model can certify its own interpretation.
"""
from __future__ import annotations

from contextlib import contextmanager
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
        db.execute("""CREATE TABLE IF NOT EXISTS study_errors(
            chunk_sha TEXT NOT NULL, model_id TEXT NOT NULL, attempts INTEGER NOT NULL,
            error_type TEXT NOT NULL, PRIMARY KEY(chunk_sha,model_id))""")
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


def study_batch(book_db: str | Path, memory_db: str | Path, edition: str,
                model: Any, *, model_id: str, budget: int = 4) -> dict[str, Any]:
    """Study unseen source chunks, checkpointing each success; bounded retry on failure."""
    if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,120}", model_id) or not 1 <= budget <= 100:
        raise ValueError("Invalid model version or batch budget")
    if _private_database(book_db) == _private_database(memory_db):
        raise ValueError("Books and study memory must be separate files")
    source = _source(book_db, edition)
    done = failed = 0
    try:
        with memory(memory_db) as dest:
            rows = source.execute("SELECT pdf_page,content FROM excerpts WHERE book=? ORDER BY CAST(pdf_page AS INTEGER),rowid", (edition,))
            for page, text in rows:
                sha = fingerprint(text)
                key = (edition, _known_book(edition)["sha256"], int(page), sha, model_id)
                if dest.execute("SELECT 1 FROM study_notes WHERE edition=? AND source_sha=? AND page=? AND chunk_sha=? AND model_id=?", key).fetchone():
                    continue
                error = dest.execute("SELECT attempts FROM study_errors WHERE chunk_sha=? AND model_id=?", (sha, model_id)).fetchone()
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
                    dest.execute("DELETE FROM study_errors WHERE chunk_sha=? AND model_id=?", (sha, model_id))
                    done += 1
                except (ValueError, RuntimeError, TypeError, KeyError) as exc:
                    # Store error class only: model exception text can contain private passages.
                    dest.execute("INSERT INTO study_errors VALUES (?,?,1,?) ON CONFLICT(chunk_sha,model_id) DO UPDATE SET attempts=attempts+1,error_type=excluded.error_type", (sha, model_id, type(exc).__name__))
                    failed += 1
                dest.commit()
    finally:
        source.close()
    return {"edition": edition, "studied_chunks": done, "failed_chunks": failed,
            "model_id": model_id, "status": "RESEARCH_ONLY", "trade_action": None}


def study_report(book_db: str | Path, memory_db: str | Path) -> dict[str, Any]:
    path = _private_database(book_db)
    source = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        with memory(memory_db) as dest:
            books = []
            for edition, sha, total, extracted, missing in source.execute("SELECT book,sha256,total_pages,extracted_pages,missing_pages FROM volumes ORDER BY book"):
                chunks = source.execute("SELECT count(*) FROM excerpts WHERE book=?", (edition,)).fetchone()[0]
                studied, pages = dest.execute("SELECT count(DISTINCT chunk_sha),count(DISTINCT page) FROM study_notes WHERE edition=? AND source_sha=?", (edition, sha)).fetchone()
                books.append({"edition": edition, "total_pages": total, "pages_with_text": extracted,
                              "pages_without_text": missing, "source_chunks": chunks,
                              "studied_chunks": studied, "pages_started": pages})
            errors = dest.execute("SELECT count(*) FROM study_errors").fetchone()[0]
        return {"books": books, "chunks_needing_retry_or_review": errors,
                "book_comprehension_percent": None, "weights_retrained": False,
                "status": "RESEARCH_ONLY", "trade_action": None}
    finally:
        source.close()


def recall(memory_db: str | Path, question: str, *, edition: str | None = None,
           limit: int = 3) -> list[dict[str, str]]:
    if not isinstance(question, str) or len(question) > 1500 or not 1 <= limit <= 5:
        raise ValueError("Invalid memory query")
    if edition:
        _known_book(edition)
    terms = set(re.findall(r"[^\W_]{3,}", question.casefold()))
    ranked = []
    with memory(memory_db) as db:
        query = "SELECT edition,page,result FROM study_notes"
        rows = db.execute(query + (" WHERE edition=?" if edition else ""), (edition,) if edition else ())
        for book, page, text in rows:
            obj = json.loads(text)
            score = len(terms & set(re.findall(r"[^\W_]{3,}", obj["summary"].casefold())))
            if score:
                ranked.append((score, {"source": f"STUDY:{book}: PDF {page}",
                                       "excerpt": "Непроверенный учебный вывод: " + obj["summary"][:700]}))
    ranked.sort(key=lambda x: (-x[0], x[1]["source"]))
    return [item for _, item in ranked[:limit]]


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
