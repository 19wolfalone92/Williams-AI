"""Source-aware three-layer cognitive memory for LOCAL Williams AI research.

SQLite FTS5/BM25 lexical search + optional injected LOCAL embedding encoder,
immutable provenance, evidence links, model/run boundaries and real recorded_at.
No embeddings are downloaded, no book text leaves the machine, and no memory
entry can certify its own truth or become an approved trading signal.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Callable

from .book_index import _private_database, _known_book
from .decision_contract import _as_utc

KINDS = {"BOOK_REFERENCE", "MARKET_OBSERVATION", "HYPOTHESIS",
         "REVIEWED_ERROR", "RESEARCH_NOTE"}
STATUS = {"SOURCE_POINTER_ONLY", "MEASURED_OBSERVATION_UNVERIFIED",
          "MODEL_HYPOTHESIS_UNVERIFIED", "REVIEWED_ERROR_DECLARED",
          "UNVERIFIED_RESEARCH_NOT_BOOK_AUTHORITY"}
EDGES = {"SUPPORTS", "CONTRADICTS", "CORRECTS", "CITES", "PRECEDES"}
_WORD = re.compile(r"[^\W_]{2,}", re.UNICODE)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _utc(value):
    seconds = _as_utc(value)
    if seconds is None:
        raise ValueError("OFFSET_AWARE_TIMESTAMP_REQUIRED")
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                     allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _vector(value):
    if (not isinstance(value, (list, tuple)) or
            not 2 <= len(value) <= 1024 or
            any(type(v) not in (float, int) or not math.isfinite(v) for v in value)):
        raise ValueError("FINITE_LOCAL_EMBEDDING_VECTOR_REQUIRED")
    vec = [float(v) for v in value]
    norm = math.sqrt(sum(x*x for x in vec))
    if norm < 1e-8:
        raise ValueError("NONZERO_EMBEDDING_REQUIRED")
    return [round(x/norm, 9) for x in vec]


def _cos(a, b):
    if not a or not b or len(a) != len(b):
        return None
    return sum(x*y for x, y in zip(a, b))


class CognitiveMemory:
    """Private structured memory without time-travel, cross-model leakage or gold promotion."""

    def __init__(self, path):
        src = Path(path).expanduser()
        if src.is_symlink():
            raise ValueError("PRIVATE_MEMORY_SYMLINK_NOT_ALLOWED")
        self.path = _private_database(src)
        if self.path.exists() and (not self.path.is_file() or self.path.stat().st_mode & 0o077):
            raise ValueError("PRIVATE_0600_MEMORY_REQUIRED")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.close(fd)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS cognitive_entries_v4 (
                entry_id TEXT PRIMARY KEY, kind TEXT NOT NULL, edition TEXT NOT NULL,
                model_id TEXT, run_id TEXT, source_ref TEXT NOT NULL,
                text TEXT NOT NULL, observed_at TEXT NOT NULL,
                recorded_at TEXT NOT NULL, evidence_status TEXT NOT NULL,
                embedding_model_id TEXT, embedding_json TEXT,
                content_sha256 TEXT NOT NULL
            )""")
            db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS cognitive_fts_v4
                       USING fts5(entry_id UNINDEXED, text, tokenize='unicode61 remove_diacritics 2')""")
            db.execute("""CREATE TABLE IF NOT EXISTS cognitive_edges_v4 (
                edge_id TEXT PRIMARY KEY, from_id TEXT NOT NULL,
                to_id TEXT NOT NULL, relation TEXT NOT NULL,
                justification TEXT NOT NULL, recorded_at TEXT NOT NULL,
                FOREIGN KEY(from_id) REFERENCES cognitive_entries_v4(entry_id),
                FOREIGN KEY(to_id) REFERENCES cognitive_entries_v4(entry_id)
            )""")
            db.execute("""CREATE INDEX IF NOT EXISTS cognitive_scope_v4
                      ON cognitive_entries_v4(edition,model_id,recorded_at)""")

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def put(self, *, kind, edition, source_ref, text, observed_at,
            model_id=None, run_id=None, embedding_encoder=None, embedding_model_id=None):
        if kind not in KINDS:
            raise ValueError("UNKNOWN_COGNITIVE_MEMORY_KIND")
        if edition != "GENERAL":
            _known_book(edition)
        if (not isinstance(text, str) or not 12 <= len(text) <= 6000 or
            not isinstance(source_ref, str) or not 5 <= len(source_ref) <= 200):
            raise ValueError("BOUNDED_EXPLICIT_MEMORY_AND_SOURCE_REQUIRED")
        if kind in {"HYPOTHESIS", "REVIEWED_ERROR"} and (
            not isinstance(model_id, str) or not 6 <= len(model_id) <= 128 or
            not isinstance(run_id, str) or not 6 <= len(run_id) <= 128):
            raise ValueError("MODEL_AND_RUN_SCOPED_EPISODE_REQUIRED")
        if model_id is not None and (not isinstance(model_id, str) or not 6 <= len(model_id) <= 128):
            raise ValueError("MODEL_SCOPE_INVALID")
        if run_id is not None and (not isinstance(run_id, str) or not 6 <= len(run_id) <= 128):
            raise ValueError("RUN_SCOPE_INVALID")
        if kind == "BOOK_REFERENCE":
            # This is a page/reference pointer, NOT proof the model understood it.
            if edition == "GENERAL" or not source_ref.startswith(edition+":"):
                raise ValueError("BOOK_REFERENCE_MUST_CITE_EDITION")
        if (embedding_encoder is None) != (embedding_model_id is None):
            raise ValueError("LOCAL_EMBEDDER_AND_MODEL_FINGERPRINT_REQUIRED_TOGETHER")
        if embedding_model_id is not None and (
            not isinstance(embedding_model_id, str) or not 6 <= len(embedding_model_id) <= 120
            or not callable(embedding_encoder)):
            raise ValueError("VALID_LOCAL_EMBEDDING_MODEL_REQUIRED")
        at = _utc(observed_at)
        now = _now()
        if _as_utc(at) > _as_utc(now):
            raise ValueError("FUTURE_OBSERVATION_CANNOT_BE_RECORDED")
        statuses = {
            "BOOK_REFERENCE": "SOURCE_POINTER_ONLY",
            "MARKET_OBSERVATION": "MEASURED_OBSERVATION_UNVERIFIED",
            "HYPOTHESIS": "MODEL_HYPOTHESIS_UNVERIFIED",
            "REVIEWED_ERROR": "REVIEWED_ERROR_DECLARED",
            "RESEARCH_NOTE": "UNVERIFIED_RESEARCH_NOT_BOOK_AUTHORITY",
        }
        status = statuses[kind]
        embed = _vector(embedding_encoder(text)) if embedding_encoder is not None else None
        core = {"kind": kind, "edition": edition, "model_id": model_id,
                "run_id": run_id, "source_ref": source_ref, "text": text,
                "observed_at": at, "evidence_status": status,
                "embedding_model_id": embedding_model_id, "embedding": embed}
        ident = _sha(core)
        with self.connect() as db:
            existing = db.execute(
                "SELECT content_sha256 FROM cognitive_entries_v4 WHERE entry_id=?",
                (ident,)).fetchone()
            if existing and existing[0] != _sha(core):
                raise ValueError("RESEARCH_MEMORY_IMMUTABLE_HASH_COLLISION")
            if not existing:
                db.execute("""INSERT INTO cognitive_entries_v4 VALUES
                           (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                           (ident, kind, edition, model_id, run_id, source_ref,
                            text, at, now, status, embedding_model_id,
                            json.dumps(embed) if embed is not None else None,
                            _sha(core)))
                db.execute("INSERT INTO cognitive_fts_v4(entry_id,text) VALUES (?,?)",
                           (ident, text))
        return ident

    def link(self, from_id, to_id, *, relation, justification):
        if relation not in EDGES:
            raise ValueError("UNKNOWN_MEMORY_EVIDENCE_RELATION")
        if (not isinstance(justification, str) or not 12 <= len(justification) <= 700
            or from_id == to_id):
            raise ValueError("EXPLICIT_NONTRIVIAL_EVIDENCE_LINK_REQUIRED")
        with self.connect() as db:
            a, b = (db.execute(
                "SELECT model_id,run_id,recorded_at FROM cognitive_entries_v4 WHERE entry_id=?",
                (ident,)).fetchone() for ident in (from_id, to_id))
            if a is None or b is None:
                raise ValueError("UNKNOWN_MEMORY_GRAPH_VERTEX")
            # This prevents test/checkpoint memory from silently contaminating
            # a different trained model or run. Global book pointers are fine.
            if a[0] is not None and b[0] is not None and a[:2] != b[:2]:
                raise ValueError("MEMORY_GRAPH_CROSS_MODEL_LEAKAGE")
            timestamp = _now()
            edge_id = _sha([from_id, to_id, relation, justification])
            db.execute("""INSERT OR IGNORE INTO cognitive_edges_v4 VALUES(?,?,?,?,?,?)""",
                       (edge_id, from_id, to_id, relation, justification, timestamp))
        return edge_id

    def retrieve(self, question, *, edition, as_of, model_id=None, run_id=None,
                 limit=5, embedding_encoder=None, embedding_model_id=None):
        if edition != "GENERAL":
            _known_book(edition)
        cutoff = _utc(as_of)
        if (not isinstance(question, str) or not 2 <= len(question) <= 1500
            or type(limit) is not int or not 1 <= limit <= 12):
            raise ValueError("BOUNDED_COGNITIVE_RETRIEVAL_REQUIRED")
        if (embedding_encoder is None) != (embedding_model_id is None):
            raise ValueError("EMBEDDING_FINGERPRINT_REQUIRED")
        if embedding_encoder is not None and not callable(embedding_encoder):
            raise ValueError("LOCAL_EMBEDDING_ENCODER_REQUIRED")
        query_vector = _vector(embedding_encoder(question)) if embedding_encoder else None
        terms = list(dict.fromkeys(x.casefold() for x in _WORD.findall(question)))[:16]
        # Do not interpolate unsanitized user text into SQL or FTS MATCH.
        match = " OR ".join('"' + x.replace('"', '""') + '"' for x in terms)
        lexical_rank = {}
        with self.connect() as db:
            # Apply temporal and model/run isolation BEFORE limiting results.
            # Ranking all future/foreign-model FTS hits first can starve the
            # actual historical memories even though Python later filters them.
            allowed = """(e.edition=? OR e.edition='GENERAL')
                AND julianday(e.observed_at)<=julianday(?)
                AND julianday(e.recorded_at)<=julianday(?)
                AND ((e.model_id IS NULL AND (e.run_id IS NULL OR e.run_id=?))
                     OR (e.model_id=? AND e.run_id=?))"""
            scope = (edition,cutoff,cutoff,run_id,model_id,run_id)
            if match:
                for rank, (ident,) in enumerate(db.execute(
                    """SELECT f.entry_id FROM cognitive_fts_v4 AS f
                       JOIN cognitive_entries_v4 AS e ON e.entry_id=f.entry_id
                       WHERE cognitive_fts_v4 MATCH ? AND """+allowed+
                    """ ORDER BY bm25(cognitive_fts_v4) LIMIT 512""",
                    (match,)+scope)):
                    lexical_rank[ident] = 1.0 / (1 + rank)
            # Bounded local scan: no ANN dependency; after 5,000 eligible
            # episodes a dedicated vector index may be needed.
            rows = db.execute("""SELECT e.entry_id,e.kind,e.edition,e.model_id,e.run_id,
                         e.source_ref,e.text,e.observed_at,e.recorded_at,e.evidence_status,
                         e.embedding_model_id,e.embedding_json,e.content_sha256
                         FROM cognitive_entries_v4 AS e WHERE """+allowed+
                         """ ORDER BY e.recorded_at DESC,e.entry_id LIMIT 5000""",
                         scope).fetchall()
        results = []
        for (ident, kind, book, model, run, source, passage, seen, recorded,
             status, vec_model, vec_json, stamped_hash) in rows:
            if (_as_utc(seen) > _as_utc(cutoff) or
                _as_utc(recorded) > _as_utc(cutoff)):
                continue
            if model is not None and (model != model_id or run != run_id):
                continue
            if model is None and run is not None:
                # Mixed global/run records may be shared only by their run.
                if run != run_id:
                    continue
            core = {"kind": kind, "edition": book, "model_id": model,
                    "run_id": run, "source_ref": source, "text": passage,
                    "observed_at": seen, "evidence_status": status,
                    "embedding_model_id": vec_model,
                    "embedding": json.loads(vec_json) if vec_json else None}
            if stamped_hash != _sha(core) or ident != stamped_hash:
                raise ValueError("COGNITIVE_MEMORY_CONTENT_CHANGED")
            semantic = None
            if query_vector is not None and vec_model == embedding_model_id and vec_json:
                semantic = _cos(query_vector, json.loads(vec_json))
            lexical = lexical_rank.get(ident, 0.0)
            if lexical == 0 and (semantic is None or semantic <= 0):
                continue
            score = (0.35 * lexical + 0.65 * max(0.0, semantic)
                     if semantic is not None else lexical)
            if score <= 0:
                continue
            results.append((score, ident, {
                "entry_id": ident, "kind": kind, "edition": book,
                "source": source, "excerpt": passage[:700],
                "observed_at": seen, "recorded_at": recorded,
                "status": status, "model_scoped": model is not None,
                "lexical_match": lexical > 0,
                "semantic_match": semantic is not None and semantic > 0,
                "expert_gold": False, "execution_allowed": False}))
        results.sort(key=lambda x: (-x[0], x[1]))
        return [item for _, _, item in results[:limit]]

    def related(self, entry_id, *, as_of, model_id=None, run_id=None, limit=6):
        cutoff = _utc(as_of)
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("BOUNDED_MEMORY_GRAPH_QUERY_REQUIRED")
        with self.connect() as db:
            links = db.execute("""SELECT from_id,to_id,relation,justification,recorded_at
                     FROM cognitive_edges_v4 WHERE from_id=? OR to_id=?
                     ORDER BY recorded_at,edge_id""", (entry_id,entry_id)).fetchall()
        # Authorization must begin with the requested anchor itself; checking
        # only the neighbouring vertex would expose links of another model.
        output = []
        with self.connect() as db:
            anchor = db.execute(
                "SELECT model_id,run_id,observed_at,recorded_at FROM cognitive_entries_v4 "
                "WHERE entry_id=?", (entry_id,)).fetchone()
            if anchor is None:
                raise ValueError("UNKNOWN_MEMORY_GRAPH_VERTEX")
            owner, run, first_seen, ingested = anchor
            if (_as_utc(first_seen) > _as_utc(cutoff) or
                _as_utc(ingested) > _as_utc(cutoff) or
                (owner is not None and (owner != model_id or run != run_id)) or
                (owner is None and run is not None and run != run_id)):
                raise ValueError("MEMORY_GRAPH_ANCHOR_NOT_VISIBLE_FOR_MODEL")
            for from_id, to_id, relation, reason, recorded in links:
                if _as_utc(recorded) > _as_utc(cutoff):
                    continue
                other = to_id if from_id == entry_id else from_id
                r = db.execute("""SELECT edition,model_id,run_id,observed_at,recorded_at
                               FROM cognitive_entries_v4 WHERE entry_id=?""",
                               (other,)).fetchone()
                if not r:
                    raise ValueError("BROKEN_COGNITIVE_EVIDENCE_EDGE")
                book,model,run,seen,ingested = r
                if (_as_utc(seen)>_as_utc(cutoff) or
                    _as_utc(ingested)>_as_utc(cutoff) or
                    (model is not None and (model!=model_id or run!=run_id)) or
                    (model is None and run is not None and run!=run_id)):
                    continue
                output.append({"other_id":other,"relation":relation,
                               "justification":reason,
                               "expert_gold":False})
                if len(output)>=limit:
                    break
        return output

    def stats(self):
        with self.connect() as db:
            return {
                "entries": db.execute("SELECT count(*) FROM cognitive_entries_v4").fetchone()[0],
                "links": db.execute("SELECT count(*) FROM cognitive_edges_v4").fetchone()[0],
                "unverified_promoted_to_gold": False,
                "local_embedding_backend_required_for_semantic_recall": True,
                "trade_execution_allowed": False,
            }
