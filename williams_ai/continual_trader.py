"""Persistent, feedback-driven Williams AI research learner across markets.

Unlike a static indicator bot, this learner updates a SMALL NUMERIC MODEL
from outcomes of prior, resolved research scenarios and retains mistakes.
The real GGUF is the book-grounded reasoning component: this scorer is NOT
a fine-tuned GGUF or a substitute for verified WM1/WM2/WM3.

Strict chronological global train/validation/test embargo prevents the same
future outcome or correlated token from being leaked into past learning.
A frozen prospective test is reported, NEVER used to tune the policy.

No Binance keys, no network calls, no account orders, no live trade authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import sqlite3
import stat
from typing import Any, Iterable

from .paper_learning import FEATURES, ResearchSample, paper_samples, fit, _probability
from .book_index import _private_database

EDITION = "TC2"
ORIGIN = "COUNTERFACTUAL_FIXED_HORIZON_NOT_VERIFIED_WILLIAMS"
IDENTITY = "PAPER_RESEARCH_ONE_HOUR_MULTIMARKET_V1"
MODEL_SCHEMA = "WILLIAMS_AI_CONTINUAL_POLICY_RESEARCH_V1"


def _time(iso: str) -> datetime:
    if not isinstance(iso, str):
        raise ValueError("Timezone-aware market timestamp required")
    result = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Naive market timestamp forbidden")
    return result.astimezone(timezone.utc)


def _seconds(iso: str) -> float:
    return _time(iso).timestamp()


def _finite_vector(values: Any) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or len(values) != len(FEATURES):
        raise ValueError("Feature schema mismatch")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or abs(x) > 50 for x in values):
        raise ValueError("Invalid, nonfinite or unbounded model observation")
    if values[0] != 1.0:
        raise ValueError("Model bias must be 1")
    return tuple(float(x) for x in values)


@dataclass(frozen=True)
class Outcome:
    """Exactly one past paper counterfactual, no simulated futures order."""

    symbol: str
    interval: str
    side: str
    decision_at: str
    resolved_at: str
    features: tuple[float, ...]
    net_return: float
    stopped: bool
    source_hash: str

    def __post_init__(self):
        if (not isinstance(self.symbol, str)
                or not self.symbol.endswith("USDT")
                or len(self.symbol) > 24
                or not self.symbol.replace("USDT", "").isalnum()
                or self.side not in ("LONG", "SHORT")
                or self.interval not in ("H1", "H4", "D1", "M15")):
            raise ValueError("Unsupported or non-canonical research market")
        if _time(self.resolved_at) <= _time(self.decision_at):
            raise ValueError("Outcome must become known AFTER its decision")
        # Persist canonical UTC strings for SQLite's chronological keys:
        # timestamps with Z, +00:00 and arbitrary local offsets compare
        # identically and cannot sneak future labels through lexical order.
        object.__setattr__(self, "decision_at", _time(self.decision_at).isoformat())
        object.__setattr__(self, "resolved_at", _time(self.resolved_at).isoformat())
        object.__setattr__(self, "features", _finite_vector(self.features))
        if (isinstance(self.net_return, bool)
                or not isinstance(self.net_return, (int, float))
                or not math.isfinite(self.net_return)
                or not -5.0 <= self.net_return <= 5.0
                or not isinstance(self.stopped, bool)):
            raise ValueError("Untrusted paper outcome")
        if (not isinstance(self.source_hash, str)
                or len(self.source_hash) != 64
                or any(c not in "0123456789abcdef" for c in self.source_hash)):
            raise ValueError("Missing 256-bit local immutable source fingerprint")


def outcomes_from_closed_history(
    symbol: str, rows: list[dict[str, Any]], interval: str,
    *, horizon: int = 12, fee_bps: float = 4.0,
    slippage_bps: float = 2.0, stop_atr: float = 2.0,
) -> list[Outcome]:
    """Causal features at t; future is used ONLY for supervised labels t+1..t+H."""
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    examples = paper_samples(rows, interval, horizon=horizon,
                             fee_bps=fee_bps, slippage_bps=slippage_bps,
                             stop_atr=stop_atr)
    return [Outcome(symbol, interval, s.side, s.decision_at, s.resolved_at,
                    s.features, s.net_return, s.stopped, digest)
            for s in examples]


class PrivateExperience:
    """Private append-only experience ledger, no raw books/models/API keys."""

    def __init__(self, path: Path | str):
        original = Path(path).expanduser()
        if original.is_symlink():
            raise ValueError("Symlink to experience journal forbidden")
        self.path = _private_database(original)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink() or self.path.exists() and (
            not self.path.is_file() or self.path.stat().st_mode & 0o077):
            raise ValueError("Learning journal must be a private regular 0600 file")
        if not self.path.exists():
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.close(fd)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS outcomes (
                symbol TEXT NOT NULL, interval TEXT NOT NULL, side TEXT NOT NULL,
                decision_at TEXT NOT NULL, resolved_at TEXT NOT NULL,
                features_json TEXT NOT NULL, net_return REAL NOT NULL,
                stopped INTEGER NOT NULL, source_hash TEXT NOT NULL,
                origin TEXT NOT NULL,
                PRIMARY KEY (symbol,interval,side,decision_at,origin)
            )""")
            db.execute("""CREATE INDEX IF NOT EXISTS outcome_chronological
                          ON outcomes(resolved_at,decision_at)""")
            db.execute("""CREATE TABLE IF NOT EXISTS policies (
                fingerprint TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                trained_through TEXT NOT NULL, training_size INTEGER NOT NULL,
                state TEXT NOT NULL, body_json TEXT NOT NULL
            )""")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA busy_timeout=5000")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def append(self, experiences: Iterable[Outcome], *,
               knowledge_as_of: str) -> dict[str, Any]:
        """Never materialize a paper outcome before it has completely settled."""
        _time(knowledge_as_of)
        inserted = 0
        repeats = 0
        with self._connect() as db:
            for obj in experiences:
                if not isinstance(obj, Outcome):
                    raise TypeError("Outcome object mandatory")
                if _seconds(obj.resolved_at) > _seconds(knowledge_as_of):
                    raise ValueError("FUTURE_OUTCOME_LEAK: research label not known yet")
                f = json.dumps(_finite_vector(obj.features),
                               separators=(",", ":"))
                existing = db.execute(
                    """SELECT resolved_at,features_json,net_return,stopped,source_hash
                       FROM outcomes WHERE symbol=? AND interval=? AND side=?
                       AND decision_at=? AND origin=?""",
                    (obj.symbol, obj.interval, obj.side,
                     obj.decision_at, ORIGIN),
                ).fetchone()
                proposed = (obj.resolved_at, f, obj.net_return,
                            int(obj.stopped), obj.source_hash)
                if existing:
                    if tuple(existing) != proposed:
                        raise ValueError("CLOSED_OUTCOME_REVISION_REJECTED")
                    repeats += 1
                    continue
                db.execute(
                    """INSERT INTO outcomes VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (obj.symbol, obj.interval, obj.side,
                     obj.decision_at, obj.resolved_at, f, obj.net_return,
                     int(obj.stopped), obj.source_hash, ORIGIN))
                inserted += 1
        return {"mode": "PRIVATE_EXPERIENCE_LEDGER",
                "new_resolved_paper_outcomes": inserted,
                "identical_duplicate_outcomes": repeats,
                "model_gguf_weights_changed": False,
                "execution_allowed": False, "trade_action": None}

    def samples(self, *, resolved_before: str | None = None) -> list[Outcome]:
        max_time = resolved_before or "9999-12-31T00:00:00+00:00"
        _time(max_time)
        with self._connect() as db:
            rows = db.execute(
                """SELECT symbol,interval,side,decision_at,resolved_at,
                   features_json,net_return,stopped,source_hash
                   FROM outcomes WHERE resolved_at<=?
                   ORDER BY decision_at,symbol,side""",
                (max_time,),
            ).fetchall()
        return [Outcome(a, b, c, d, e, _finite_vector(json.loads(f)), g,
                        bool(h), i)
                for a, b, c, d, e, f, g, h, i in rows]

    def policies(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT body_json FROM policies ORDER BY trained_through"
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def save_policy(self, result: dict[str, Any]) -> bool:
        """Immutable content-addressed artifact. Heldout scores cannot authorize execution."""
        if result.get("schema") != MODEL_SCHEMA or result.get("execution_allowed") is not False:
            raise ValueError("Nonresearch model cannot enter private memory")
        digest = result["fingerprint"]
        raw = json.dumps(result, sort_keys=True, separators=(",", ":"))
        with self._connect() as db:
            existing = db.execute(
                "SELECT body_json FROM policies WHERE fingerprint=?", (digest,)
            ).fetchone()
            if existing:
                if existing[0] != raw:
                    raise ValueError("IMMUTABLE_POLICY_DIGEST_COLLISION")
                return False
            db.execute("INSERT INTO policies VALUES(?,?,?,?,?,?)",
                       (digest, datetime.now(timezone.utc).isoformat(),
                        result["trained_through"],
                        result["train_count"], result["validation_state"], raw))
            return True


def _research_sample(ex: Outcome, index: int) -> ResearchSample:
    return ResearchSample(
        index, index, ex.decision_at, ex.resolved_at, ex.side,
        ex.features, ex.net_return, ex.stopped
    )


def _brier(weights: list[float], rows: list[Outcome]) -> float:
    return sum((_probability(weights, s.features) -
                float(s.net_return > 0)) ** 2 for s in rows) / len(rows)


def _constant_brier(prior: float, rows: list[Outcome]) -> float:
    return sum((prior - float(s.net_return > 0)) ** 2 for s in rows) / len(rows)


def learn_from_private_experience(
    ledger: PrivateExperience, *, train_until: str,
    validate_until: str, observe_until: str, min_train: int = 60,
    min_validation: int = 16,
) -> dict[str, Any]:
    """One anchored chronological experiment using GLOBAL absolute timestamps.

    train decisions < train_until AND resolutions < train_until;
    validation starts train_until and resolves < validate_until;
    final TEST starts validate_until and resolved <= observe_until.
    No candidate selection on the heldout test. No shared outcome leakage.
    """
    a, b, c = map(_seconds, (train_until, validate_until, observe_until))
    if not a < b < c:
        raise ValueError("Chronological data cutoff order violated")
    all_outcomes = ledger.samples(resolved_before=observe_until)
    train = [x for x in all_outcomes
             if _seconds(x.decision_at) < a and _seconds(x.resolved_at) < a]
    validation = [x for x in all_outcomes
                  if a <= _seconds(x.decision_at) < b and
                  _seconds(x.resolved_at) < b]
    holdout = [x for x in all_outcomes
               if b <= _seconds(x.decision_at) < c and
               _seconds(x.resolved_at) <= c]
    if len(train) < min_train or len(validation) < min_validation or not holdout:
        raise ValueError("INSUFFICIENT_PURGED_MULTI_MARKET_SAMPLE")
    if len({(x.symbol, x.decision_at) for x in train}) < 20:
        raise ValueError("TOO_FEW_INDEPENDENT_DECISION_TIMES")
    # Missing/multiple contradictory sources are refused by journal append.
    model = fit([_research_sample(x, i) for i, x in enumerate(train)], epochs=100)
    prior = sum(x.net_return > 0 for x in train) / len(train)
    prior_brier = _constant_brier(prior, validation)
    candidate_brier = _brier(model["weights"], validation)
    # STRICTLY validation decides whether this experiment is worth further
    # research. Holdout outcomes are reported AFTER but never a selection gate.
    validation_state = (
        "VALIDATION_BEATS_PRIOR_RESEARCH_ONLY"
        if candidate_brier < prior_brier - 0.005
        else "NO_MEASURABLE_VALIDATION_GAIN"
    )
    def bucket(xs: list[Outcome]) -> dict[str, Any]:
        predicted = [_probability(model["weights"], s.features) for s in xs]
        selected = [s for s, p in zip(xs, predicted) if p >= 0.6]
        mean_return = sum(s.net_return for s in selected) / len(selected) if selected else None
        return {
            "examples": len(xs),
            "unique_decisions": len({(s.symbol, s.decision_at) for s in xs}),
            "symbols": sorted({s.symbol for s in xs}),
            "brier": round(_brier(model["weights"], xs), 8),
            "train_prior_brier": round(_constant_brier(prior, xs), 8),
            "fixed_threshold_0_6_selected": len(selected),
            "selected_mean_net_return": round(mean_return, 8) if mean_return is not None else None,
            "contains_paper_counterfactuals_not_book_proven_trades": True,
        }
    stable = {
        "algorithm": model["algorithm"],
        "features": list(FEATURES),
        "weights": [round(float(x), 12) for x in model["weights"]],
        "source_outcomes": sorted({x.source_hash for x in train}),
        "train_until": train_until, "validate_until": validate_until,
        "trained_through": max(x.resolved_at for x in train),
        "train_count": len(train),
    }
    fingerprint = hashlib.sha256(json.dumps(
        stable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema": MODEL_SCHEMA,
        "mode": "CONTINUAL_FEEDBACK_TRAINED_RESEARCH_SCORER",
        "fingerprint": fingerprint, **stable,
        "train_prior_positive_rate": prior,
        "validation_state": validation_state,
        "validation": bucket(validation),
        "frozen_test": bucket(holdout),
        "test_was_never_used_to_fit_or_select": True,
        "market_edition_rule_match_not_proven": True,
        "verified_WM1_WM2_WM3_signals": 0,
        "real_gguf_weights_updated": False,
        "source": ORIGIN,
        "research_provenance": IDENTITY,
        "trade_action": None, "execution_allowed": False,
    }


def research_prediction(
    model: dict[str, Any], features: tuple[float, ...],
    *, decision_at: str,
) -> dict[str, Any]:
    if model.get("schema") != MODEL_SCHEMA or model.get("execution_allowed") is not False:
        raise ValueError("Nonresearch candidate policy rejected")
    if _seconds(decision_at) <= _seconds(model["trained_through"]):
        raise ValueError("PAST_REPLAY_CANNOT_USE_LATER_LEARNED_MODEL")
    value = _probability(model["weights"], _finite_vector(features))
    return {
        "mode": "LEARNED_PAPER_OUTCOME_ESTIMATE_NOT_TRADE_PROBABILITY",
        "trained_through": model["trained_through"],
        "research_positive_outcome_estimate": round(value, 6),
        "scorer": model["fingerprint"],
        "current_book_signal_confirmed": False,
        "execution_allowed": False,
        "trade_action": None,
    }
