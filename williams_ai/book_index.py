"""PRIVATE offline page-index of a user's licensed four Williams PDFs.

No copyrighted PDFs or extracted text is bundled with GitHub, APK or CI.
Index is SQLite stored *outside* the git checkout. Local GGUF RAG uses
short, page-cited excerpts. Course scans require separately reviewed OCR.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from .book_rules import registry

ROOT = Path(__file__).resolve().parents[1]
_WORD = re.compile(r"[^\W_]{2,}", re.UNICODE)


class SourceMismatch(ValueError):
    pass


def _private_database(path: str | Path) -> Path:
    dest = Path(path).expanduser().resolve()
    # Prevent shipping licensed book text accidentally in public git repo.
    if dest == ROOT or ROOT in dest.parents:
        raise ValueError("The books index MUST live outside the public repository")
    if dest.suffix not in {".db", ".sqlite3"}:
        raise ValueError("Use a private .db or .sqlite3 database")
    return dest


def _database(path: str | Path) -> sqlite3.Connection:
    path = _private_database(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(fd)
    db = sqlite3.connect(str(path))
    db.execute("""
        CREATE TABLE IF NOT EXISTS volumes(
            book TEXT PRIMARY KEY, sha256 TEXT NOT NULL, total_pages INTEGER NOT NULL,
            extracted_pages INTEGER NOT NULL, missing_pages INTEGER NOT NULL,
            source_type TEXT NOT NULL)
    """)
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS excerpts USING fts5(
            book UNINDEXED, pdf_page UNINDEXED, content,
            evidence_status UNINDEXED, tokenize='unicode61'
        )
    """)
    return db


def _known_book(book: str) -> dict[str, Any]:
    by_id = {row["id"]: row for row in registry()["books"]}
    if book not in by_id:
        raise ValueError("Unknown edition")
    return by_id[book]


def verify_original(book: str, pdf_path: str | Path) -> dict[str, Any]:
    source = _known_book(book)
    file = Path(pdf_path).expanduser().resolve()
    if not file.is_file() or file.suffix.lower() != ".pdf" or file.is_symlink():
        raise SourceMismatch("Expected a local PDF file")
    h = hashlib.sha256()
    with file.open("rb") as fd:
        for chunk in iter(lambda: fd.read(1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != source["sha256"]:
        raise SourceMismatch(
            f"{book}: checksum does not match the source copy reviewed for this project; "
            "no index was created"
        )
    return {"book": book, "verified_sha256": h.hexdigest(), "pdf": file}


def _chunks(text: str, length: int = 900) -> list[str]:
    plain = re.sub(r"\s+", " ", text).strip()
    return [plain[i:i + length] for i in range(0, min(len(plain), 35000), length)]


def index_pages(
    database: str | Path, book: str, verified_sha256: str, pages: Iterable[str],
    *, declared_pages: int, source_type: str = "PDF_TEXT",
) -> dict[str, int]:
    """Atomic offline index. Caller must verify PDF hash (see index_pdf)."""
    known = _known_book(book)
    if verified_sha256 != known["sha256"] or declared_pages != known["pages"]:
        raise SourceMismatch("Source ID/pages do not match the reviewed PDF")
    if source_type not in {"PDF_TEXT", "PDF_TEXT_WITH_UNREVIEWED_OCR"}:
        raise ValueError("Unknown input provenance")
    allpages = list(pages)
    if len(allpages) != declared_pages:
        raise SourceMismatch("Not all PDF pages are present")
    if any(not isinstance(x, str) for x in allpages):
        raise ValueError("Pages must be UTF-8 text")
    db = _database(database)
    try:
        with db:
            db.execute("DELETE FROM excerpts WHERE book=?", (book,))
            nonempty = 0
            for index, text in enumerate(allpages, 1):
                chunks = _chunks(text)
                if not chunks:
                    continue
                nonempty += 1
                for chunk in chunks:
                    db.execute(
                        "INSERT INTO excerpts(book,pdf_page,content,evidence_status) VALUES (?,?,?,?)",
                        (book, index, chunk, "UNVERIFIED_EXTRACTION"),
                    )
            db.execute(
                "INSERT OR REPLACE INTO volumes VALUES (?,?,?,?,?,?)",
                (book, verified_sha256, declared_pages, nonempty,
                 declared_pages - nonempty, source_type)
            )
        return {"total_pages": declared_pages, "nonempty_pages": nonempty,
                "missing_text_pages": declared_pages - nonempty}
    finally:
        db.close()


def index_pdf(database: str | Path, book: str, pdf_path: str | Path,
              *, optional_ocr_dir: str | Path | None = None) -> dict[str, int]:
    verified = verify_original(book, pdf_path)
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install optional PDF parser: python -m pip install pypdf") from exc
    reader = PdfReader(str(verified["pdf"]), strict=True)
    known = _known_book(book)
    if len(reader.pages) != known["pages"]:
        raise SourceMismatch("PDF has different number of pages than reviewed copy")
    ocr = Path(optional_ocr_dir).expanduser().resolve() if optional_ocr_dir else None
    pages = []
    used_ocr = False
    for index, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if not text and ocr is not None:
            file = ocr / f"page_{index:04d}.txt"
            if file.is_file() and not file.is_symlink() and file.stat().st_size <= 100000:
                text = file.read_text(encoding="utf-8").strip()
                used_ocr = used_ocr or bool(text)
        pages.append(text)
    return index_pages(database, book, verified["verified_sha256"], pages,
                       declared_pages=known["pages"],
                       source_type="PDF_TEXT_WITH_UNREVIEWED_OCR" if used_ocr else "PDF_TEXT")


def search_pages(database: str | Path, question: str, *,
                 edition: str | None = None, limit: int = 4) -> list[dict[str, Any]]:
    """Private excerpts, capped; never upload this DB to any web provider."""
    if edition:
        _known_book(edition)
    if not isinstance(question, str) or len(question) > 1500 or not 1 <= limit <= 8:
        raise ValueError("Invalid search request")
    tokens = list(dict.fromkeys(x.casefold() for x in _WORD.findall(question)))[:12]
    if not tokens:
        return []
    # fts expression built exclusively from alphanumeric tokens; no caller-supplied syntax.
    query = " OR ".join('"' + token + '"' for token in tokens)
    path = _private_database(database)
    if not path.is_file():
        raise FileNotFoundError("Private book index does not exist")
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        statement = """SELECT book,pdf_page,content,evidence_status
            FROM excerpts WHERE excerpts MATCH ?"""
        args: list[Any] = [query]
        if edition:
            statement += " AND book=?"
            args.append(edition)
        statement += " ORDER BY rank LIMIT ?"
        args.append(limit)
        rows = db.execute(statement, args).fetchall()
        return [{
            "source": f"{book}: PDF {page}", "excerpt": content[:650],
            "status": "PRIVATE_PAGE_TEXT_UNVERIFIED", "page": int(page),
        } for book, page, content, status in rows]
    finally:
        db.close()


def index_report(database: str | Path) -> dict[str, Any]:
    path = _private_database(database)
    if not path.exists():
        return {"volumes": [], "book_accuracy_percent": None}
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        records = db.execute(
            "SELECT book,total_pages,extracted_pages,missing_pages,source_type FROM volumes "
            "ORDER BY book"
        ).fetchall()
        return {
            "volumes": [dict(zip(
                ("book", "pages", "pages_with_text", "pages_without_text", "source_type"), row
            )) for row in records],
            "book_accuracy_percent": None,
            "note": "Indexing pages does not certify rules, screenshots or interpretation.",
        }
    finally:
        db.close()
