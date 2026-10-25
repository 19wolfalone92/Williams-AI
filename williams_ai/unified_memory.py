"""One private SQLite memory for study, mistakes, research and settled experience.

Original books retain their separate, immutable source index. Shared storage
never turns an unverified note into a book rule or a future outcome into a fact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from .book_index import _private_database, _known_book
from .decision_contract import _as_utc
from .knowledge import query_terms
from .study import memory

_TABLES = ("study_notes", "study_errors", "scoped_study_errors", "research_sources",
           "practice_attempts", "private_remediation", "outcomes", "policies",
           "decision_study_events", "actual_run_receipts")


class UnifiedMemory:
    def __init__(self, path: Path):
        original = Path(path).expanduser()
        if original.is_symlink():
            raise ValueError("PRIVATE_REGULAR_MEMORY_REQUIRED")
        self.path = _private_database(original)
        if self.path.exists() and (not self.path.is_file() or self.path.stat().st_mode & 0o077):
            raise ValueError("PRIVATE_0600_MEMORY_REQUIRED")
        from .adaptive_coach import notebook
        from .continual_trader import PrivateExperience
        with memory(self.path):
            pass
        with notebook(self.path):
            pass
        PrivateExperience(self.path)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS decision_study_events(
                event_id TEXT PRIMARY KEY,model_id TEXT NOT NULL,edition TEXT NOT NULL,
                case_sha256 TEXT NOT NULL,cutoff TEXT NOT NULL,recorded_at TEXT NOT NULL,
                diagnostics_json TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS retained_materials(
                document_sha256 TEXT PRIMARY KEY,edition TEXT NOT NULL,source TEXT NOT NULL,
                title TEXT NOT NULL,available_at TEXT NOT NULL,text TEXT NOT NULL,
                status TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS memory_import_rows(
                row_sha256 TEXT PRIMARY KEY,table_name TEXT NOT NULL,imported_at TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS actual_run_receipts(
                name TEXT PRIMARY KEY,payload TEXT NOT NULL,payload_sha256 TEXT NOT NULL)""")

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=15)
        db.execute("PRAGMA synchronous=FULL")
        return db

    def retain(self, text: str, *, source: str, title: str, edition: str,
               available_at: str) -> str:
        """Retain the complete bounded text, not just the excerpt in a prompt."""
        if edition != "GENERAL":
            _known_book(edition)
        at = _as_utc(available_at)
        if (at is None or not isinstance(text, str) or not 1 <= len(text) <= 500_000
                or not isinstance(source, str) or not 1 <= len(source) <= 180
                or not isinstance(title, str) or not 1 <= len(title) <= 160):
            raise ValueError("BOUNDED_DATED_RESEARCH_MATERIAL_REQUIRED")
        known = datetime.fromtimestamp(at, timezone.utc).isoformat()
        fields = [edition, source, title, known, text]
        digest = hashlib.sha256(json.dumps(fields, ensure_ascii=False,
            separators=(",", ":")).encode()).hexdigest()
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO retained_materials VALUES(?,?,?,?,?,?,?)",
                (digest, edition, source, title, known, text, "UNVERIFIED_RESEARCH_NOT_BOOK_AUTHORITY"))
        return digest

    def retrieve(self, question: str, *, edition: str, as_of: str,
                 limit: int = 2) -> list[dict[str, Any]]:
        _known_book(edition)
        cutoff = _as_utc(as_of)
        if (cutoff is None or not isinstance(question, str) or len(question) > 1500
                or type(limit) is not int or not 1 <= limit <= 3):
            raise ValueError("CAUSAL_RESEARCH_MEMORY_QUERY_REQUIRED")
        terms = query_terms(question)
        ranked = []
        with self.connect() as db:
            rows = db.execute("SELECT * FROM retained_materials WHERE edition IN (?,?) "
                "AND julianday(available_at)<=julianday(?) ORDER BY available_at,document_sha256",
                (edition, "GENERAL", as_of)).fetchall()
        for digest, book, source, title, at, text, status in rows:
            known = _as_utc(at)
            if known is None or known > cutoff or status != "UNVERIFIED_RESEARCH_NOT_BOOK_AUTHORITY":
                continue
            expected = hashlib.sha256(json.dumps([book, source, title, at, text],
                ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            if digest != expected:
                raise ValueError("RETAINED_MATERIAL_CHANGED")
            # Scan all chunks. Relevant material at a long document's end
            # must not disappear because only its introduction was searched.
            for start in range(0, len(text), 700):
                chunk = text[start:start + 900]
                score = sum(min(4, chunk.casefold().count(term)) for term in terms)
                if score:
                    offset = min(chunk.casefold().find(term) for term in terms if term in chunk.casefold())
                    excerpt = chunk[max(0, offset - 80):][:450]
                    ranked.append((score, digest, start, {
                        "source": "MATERIAL:" + digest, "excerpt": excerpt,
                        "edition": book, "available_at": at, "status": status}))
        ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        selected = []
        for _, digest, _, row in ranked:
            if any(item["source"] == "MATERIAL:" + digest for item in selected):
                continue
            selected.append(row)
            if len(selected) == limit:
                break
        return selected

    def import_checkpoint(self, source: Path) -> dict[str, Any]:
        """Atomic, repeatable import of known learning tables; no blind DB merge."""
        original = Path(source).expanduser()
        if original.is_symlink():
            raise ValueError("REGULAR_PRIVATE_CHECKPOINT_REQUIRED")
        path = _private_database(original)
        if not path.is_file() or path == self.path:
            raise ValueError("DIFFERENT_PRIVATE_CHECKPOINT_REQUIRED")
        counts = {}
        old = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            old.execute("PRAGMA query_only=ON")
            old.execute("BEGIN")
            tables = {name for name, sql in old.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='table'")
                if sql and "VIRTUAL TABLE" not in sql.upper()}
            with self.connect() as db:
                for table in _TABLES:
                    if table not in tables:
                        continue
                    schema = db.execute(f"PRAGMA table_info({table})").fetchall()
                    cols = [x[1] for x in schema]
                    if [x[1] for x in old.execute(f"PRAGMA table_info({table})")] != cols:
                        raise ValueError("LEGACY_MEMORY_SCHEMA_MISMATCH")
                    if old.execute(f"SELECT count(*) FROM {table}").fetchone()[0] > 200_000:
                        raise ValueError("BOUNDED_MEMORY_IMPORT_REQUIRED")
                    primary = [x[1] for x in sorted(schema, key=lambda x: x[5]) if x[5]]
                    inserted = 0
                    for row in old.execute(f"SELECT * FROM {table}"):
                        if table == "actual_run_receipts" and hashlib.sha256(row[1].encode()).hexdigest() != row[2]:
                            raise ValueError("LEGACY_RUN_RECEIPT_CHANGED")
                        row_digest = hashlib.sha256(json.dumps([table, list(row)],
                            sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                        if db.execute("SELECT 1 FROM memory_import_rows WHERE row_sha256=?", (row_digest,)).fetchone():
                            continue
                        if table == "practice_attempts":
                            # Local sequence numbers from two notebooks may
                            # collide; preserve every attempt with a new seq.
                            used_cols, values = cols[1:], row[1:]
                        else:
                            used_cols, values = cols, row
                            where = " AND ".join(k + "=?" for k in primary)
                            key = tuple(row[cols.index(k)] for k in primary)
                            existing = db.execute(f"SELECT * FROM {table} WHERE {where}", key).fetchone()
                            if existing is not None:
                                if tuple(existing) != tuple(row):
                                    raise ValueError("MEMORY_IMPORT_CONFLICT_NOT_OVERWRITTEN")
                                continue
                        marks = ",".join("?" for _ in values)
                        db.execute(f"INSERT INTO {table}({','.join(used_cols)}) VALUES({marks})", values)
                        db.execute("INSERT INTO memory_import_rows VALUES(?,?,?)",
                            (row_digest, table, datetime.now(timezone.utc).isoformat()))
                        inserted += 1
                    counts[table] = inserted
        finally:
            old.close()
        return {"imported_rows": counts, "unverified_notes_promoted": False,
                "source_checkpoints_modified": False, "execution_allowed": False}

    def inventory(self) -> dict[str, int]:
        with self.connect() as db:
            return {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                    for table in (*_TABLES, "retained_materials")}
