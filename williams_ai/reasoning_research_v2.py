"""Evidence-gated Williams reasoning research: private expert lessons and as-of memory.

No automatic exchange API, no model inference or LoRA training. This is a
strict input/experiment boundary, not a claim of expert accuracy.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sqlite3

ED = {"TC1", "NTD", "TC2", "COURSE"}
SPLITS = ("train", "validation", "test")
ALLOWED_DECISIONS = {"WAIT", "NO_TRADE", "REVIEW_ENTRY", "REVIEW_ADD",
                     "REVIEW_EXIT", "UNDETERMINED"}
ROW_FIELDS = {"case_id", "split", "rule_id", "edition", "pdf_page",
              "source_sha256", "symbol", "timeframe", "cutoff",
              "visible_prefix_sha256", "bars", "question", "answer", "reviewers"}
ANSWER_FIELDS = {"hypothesis", "alternative", "evidence", "counterevidence",
                 "invalidation", "decision", "confidence", "wave_confirmed",
                 "squat_verified", "execution_allowed"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False, separators=(",", ":"))


def digest(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def utc(value):
    if not isinstance(value, str):
        raise ValueError("AWARE_DATETIME_REQUIRED")
    try:
        at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("AWARE_DATETIME_REQUIRED") from exc
    if at.tzinfo is None:
        raise ValueError("AWARE_DATETIME_REQUIRED")
    return at.astimezone(timezone.utc)


def _valid_answer(answer):
    if not isinstance(answer, dict) or set(answer) != ANSWER_FIELDS:
        raise ValueError("EXACT_EVIDENCE_AND_COUNTERHYPOTHESIS_REQUIRED")
    if answer["decision"] not in ALLOWED_DECISIONS or answer["execution_allowed"] is not False:
        raise ValueError("RESEARCH_ONLY_DECISIONS")
    if type(answer["confidence"]) not in (int, float) or not math.isfinite(answer["confidence"]) or not 0 <= answer["confidence"] <= 1:
        raise ValueError("FINITE_UNCERTAINTY_REQUIRED")
    if any(not isinstance(answer[k], str) or not 8 <= len(answer[k]) <= 2000
           for k in ("hypothesis", "alternative", "invalidation")):
        raise ValueError("MEANINGFUL_REASONING_REQUIRED")
    for k in ("evidence", "counterevidence"):
        if (not isinstance(answer[k], list) or not 1 <= len(answer[k]) <= 12
                or any(not isinstance(item, str) or not 5 <= len(item) <= 400
                       for item in answer[k])):
            raise ValueError("EVIDENCE_AND_COUNTEREVIDENCE_REQUIRED")
    for k in ("wave_confirmed", "squat_verified"):
        if answer[k] is not None and type(answer[k]) is not bool:
            raise ValueError("UNKNOWN_OR_EXPLICIT_EXPERT_VERIFICATION_ONLY")


def verify_case(row):
    if not isinstance(row, dict) or set(row) != ROW_FIELDS:
        raise ValueError("EXACT_EXPERT_REASONING_CASE_SCHEMA_REQUIRED")
    if row["split"] not in SPLITS or row["edition"] not in ED or row["timeframe"] not in {"H1", "H4", "D1"}:
        raise ValueError("EXPLICIT_EDITION_SPLIT_AND_TIMEFRAME")
    if any(not isinstance(row[k], str) or not 5 <= len(row[k]) <= 200
           for k in ("case_id", "rule_id", "symbol")):
        raise ValueError("CAUSAL_IDENTITY_REQUIRED")
    for k in ("source_sha256", "visible_prefix_sha256"):
        v = row[k]
        if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("PINNED_SOURCE_OR_MARKET_DIGEST_REQUIRED")
    if type(row["pdf_page"]) is not int or row["pdf_page"] < 1:
        raise ValueError("BOOK_PAGE_REQUIRED")
    if not isinstance(row["question"], str) or not 15 <= len(row["question"]) <= 3000:
        raise ValueError("SUBSTANTIVE_QUESTION_REQUIRED")
    _valid_answer(row["answer"])
    cutoff = utc(row["cutoff"])
    bars = row["bars"]
    if not isinstance(bars, list) or not 220 <= len(bars) <= 5000:
        raise ValueError("CLOSED_PREFIX_BARS_REQUIRED")
    previous = None
    for candle in bars:
        if not isinstance(candle, dict) or set(candle) != {"close_time", "open", "high", "low", "close", "volume", "closed"}:
            raise ValueError("EXACT_BAR_FIELDS_REQUIRED")
        if candle["closed"] is not True:
            raise ValueError("NO_OPEN_CANDLES")
        current = utc(candle["close_time"])
        if (current > cutoff or (previous is not None and current <= previous)):
            raise ValueError("FUTURE_OR_UNSORTED_CANDLES")
        previous = current
        vals = [candle[k] for k in ("open", "high", "low", "close", "volume")]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in vals):
            raise ValueError("FINITE_MARKET_DATA_REQUIRED")
        o, hi, lo, c, volume = vals
        if lo <= 0 or volume < 0 or hi < max(o, lo, c) or lo > min(o, hi, c):
            raise ValueError("INVALID_OHLCV")
    if previous != cutoff or digest(bars) != row["visible_prefix_sha256"]:
        raise ValueError("PREFIX_DIGEST_OR_CUTOFF_MISMATCH")
    # Causally validated, contiguous bars; no future or partial candles.
    from .chart_reader import TF_SECONDS, validate_closed_bars
    candles, gaps = validate_closed_bars(bars, row["timeframe"], row["cutoff"])
    seconds = TF_SECONDS[row["timeframe"]]
    if gaps or len(candles) != len(bars) or any(
            (right.at - left.at).total_seconds() != seconds
            for left, right in zip(candles, candles[1:])):
        raise ValueError("CONTIGUOUS_MARKET_PREFIX_REQUIRED")
    reviews = row["reviewers"]
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise ValueError("TWO_INDEPENDENT_REVIEW_RECORDS_REQUIRED")
    ids = set()
    case_sha = digest({k: v for k, v in row.items() if k != "reviewers"})
    for review in reviews:
        if not isinstance(review, dict) or set(review) != {"reviewer_id", "approved_at", "case_sha256"}:
            raise ValueError("EXACT_REVIEW_RECORD_REQUIRED")
        name = review["reviewer_id"]
        if not isinstance(name, str) or not 5 <= len(name) <= 100 or name in ids:
            raise ValueError("TWO_DISTINCT_REVIEWER_IDENTITIES_REQUIRED")
        ids.add(name)
        if utc(review["approved_at"]) < cutoff or review["case_sha256"] != case_sha:
            raise ValueError("REVIEW_MUST_APPROVE_EXACT_ANSWER_AFTER_CUTOFF")
    # Cryptographic binding to the local four-book rule registry. A claimed
    # reviewer identity is still not proof of actual reviewer expertise.
    from .book_rules import registry
    sources = registry()
    books = {entry["id"]: entry for entry in sources["books"]}
    rules = {entry["id"]: entry for entry in sources["rule_cards"]}
    rule = rules.get(row["rule_id"])
    if (rule is None or rule["source"] != row["edition"]
            or rule["status"] != "REVIEWED_PAGES"
            or row["pdf_page"] not in rule["pdf_pages"]
            or row["source_sha256"] != books[row["edition"]]["sha256"]):
        raise ValueError("UNVERIFIED_OR_MISMATCHED_BOOK_RULE_REFERENCE")
    return row


def check_corpus(cases):
    if not isinstance(cases, list):
        raise ValueError("CASES_LIST_REQUIRED")
    ids, prefixes, assignments = set(), {}, {}
    counts = Counter()
    for row in cases:
        verify_case(row)
        if row["case_id"] in ids:
            raise ValueError("DUPLICATE_CASE_ID")
        ids.add(row["case_id"])
        key = (row["symbol"], row["timeframe"], row["visible_prefix_sha256"])
        if key in prefixes:
            raise ValueError("REPEATED_PREFIX_NOT_INDEPENDENT")
        prefixes[key] = row["split"]
        if row["symbol"] in assignments and assignments[row["symbol"]] != row["split"]:
            raise ValueError("WHOLE_SYMBOL_SPLIT_REQUIRED")
        assignments[row["symbol"]] = row["split"]
        counts[row["split"]] += 1
    return {"schema": "WILLIAMS_AI_GATED_REASONING_CORPUS_V1",
            "counts": dict(sorted(counts.items())),
            "distinct_symbols": len(assignments), "case_count": len(cases),
            "source_sha256": digest([row["case_id"] for row in cases]),
            "expert_identity_not_cryptographically_authenticated": True,
            "execution_allowed": False}


def question_only(row):
    """Bounded numerical market observations, never a full-history token dump.

    Full immutable history is verified and bound by its digest. Only visible
    closed candles are used to calculate indicators. No gold label or
    reviewer note is supplied to the model.
    """
    from .chart_reader import indicators, validate_closed_bars
    candles, gaps = validate_closed_bars(row["bars"], row["timeframe"], row["cutoff"])
    if gaps or len(candles) < 220 or digest(row["bars"]) != row["visible_prefix_sha256"]:
        raise ValueError("RECHECKED_CLOSED_MARKET_PREFIX_REQUIRED")
    lines = indicators(candles)

    def num(value):
        return None if value is None else float(f"{value:.8g}")

    tail = []
    for i in range(len(candles) - 8, len(candles)):
        bar = candles[i]
        tail.append({"close_time": row["bars"][i]["close_time"],
                     "o": num(bar.open), "h": num(bar.high),
                     "l": num(bar.low), "c": num(bar.close), "v": num(bar.volume),
                     **{k: num(lines[k][i]) for k in ("jaw", "teeth", "lips", "ao", "ac")}})
    return {"case_id": row["case_id"], "symbol": row["symbol"],
            "timeframe": row["timeframe"], "cutoff": row["cutoff"],
            "visible_prefix_sha256": row["visible_prefix_sha256"],
            "rule_id": row["rule_id"], "edition": row["edition"],
            "pdf_page": row["pdf_page"], "source_sha256": row["source_sha256"],
            "question": row["question"], "closed_bar_count": len(candles),
            "visible_numeric_tail": tail,
            "market_summary_not_full_wave_degree": True,
            "no_position_or_stop_acknowledgment_supplied": True}


def export_research(cases):
    manifest = check_corpus(cases)
    result = {"train_sft": [], "validation_prompts": [], "test_prompts": []}
    for row in cases:
        prompt = question_only(row)
        if row["split"] == "train":
            result["train_sft"].append({
                "case_id": row["case_id"], "rule_id": row["rule_id"],
                "messages": [{"role": "system", "content":
                              "Research only; distinguish observations, hypotheses, evidence, alternatives and uncertainty. Never authorize execution."},
                             {"role": "user", "content": canonical(prompt)},
                             {"role": "assistant", "content": canonical(row["answer"])}]})
        else:
            result[row["split"] + "_prompts"].append(prompt)
    for key in result:
        result[key].sort(key=lambda x: x["case_id"])
    manifest["outputs_sha256"] = {key: digest(values) for key, values in result.items()}
    manifest["test_gold_exported"] = False
    return result, manifest


def adaptive_schedule(train_cases, validation_failures, *, rounds=10, seed="v2"):
    """Select ONLY approved train examples based on validation *error categories*.

    Never includes validation/test answers. This is preparation, not GPU training.
    """
    if type(rounds) is not int or not 1 <= rounds <= 100:
        raise ValueError("BOUNDED_ROUNDS_REQUIRED")
    for row in train_cases:
        verify_case(row)
        if row["split"] != "train":
            raise ValueError("TRAIN_ONLY_DATA_REQUIRED")
    eligible = {row["rule_id"] for row in train_cases}
    priorities = Counter()
    for issue in validation_failures:
        if not isinstance(issue, dict) or set(issue) != {"case_id", "split", "rule_id", "error_code"}:
            raise ValueError("ONLY_ERROR_METADATA_ALLOWED")
        if issue["split"] != "validation" or not issue["case_id"] or not issue["error_code"]:
            raise ValueError("VALIDATION_METADATA_ONLY")
        if issue["rule_id"] in eligible:
            priorities[issue["rule_id"]] += 1
    if not train_cases:
        raise ValueError("APPROVED_TRAIN_CASES_REQUIRED")
    pool = sorted(train_cases, key=lambda r: r["case_id"])
    ranks = sorted(eligible, key=lambda rule: (-priorities[rule], rule))
    schedule = []
    for index in range(rounds):
        rule = ranks[index % len(ranks)]
        candidates = [row for row in pool if row["rule_id"] == rule]
        choice = min(candidates, key=lambda row: digest([seed, index, row["case_id"]]))
        schedule.append({"round": index + 1, "focus_rule": rule,
                         "train_case_id": choice["case_id"],
                         "validation_failures_for_rule": priorities[rule]})
    return {"schema": "WILLIAMS_ADAPTIVE_TRAIN_SCHEDULE_V1", "schedule": schedule,
            "validation_answers_used_as_training": False, "execution_allowed": False}


class HypothesisMemory:
    """Research-only as-of journal; no autonomous promotion of ideas to facts."""

    def __init__(self, path):
        path = Path(path).expanduser()
        if path.is_symlink() or (path.exists() and
                                (not path.is_file() or path.stat().st_mode & 0o077)):
            raise ValueError("PRIVATE_0600_MEMORY_REQUIRED")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.close(fd)
        self.path = path
        with sqlite3.connect(self.path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS hypotheses (
                case_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, cutoff TEXT NOT NULL,
                available_at TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL,
                hypothesis TEXT NOT NULL, alternative TEXT NOT NULL,
                evidence_json TEXT NOT NULL, counter_json TEXT NOT NULL,
                invalidation TEXT NOT NULL, status TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS hypothesis_reviews (
                case_id TEXT PRIMARY KEY REFERENCES hypotheses(case_id),
                reviewer_a TEXT NOT NULL, reviewer_b TEXT NOT NULL,
                reviewed_at TEXT NOT NULL, verdict TEXT NOT NULL,
                rationale TEXT NOT NULL
            )""")

    def append(self, *, case_id, symbol, cutoff, available_at, snapshot_sha256,
               hypothesis, alternative, evidence, counterevidence, invalidation):
        if utc(available_at) < utc(cutoff):
            raise ValueError("MEMORY_CANNOT_PREDATE_OBSERVATION")
        if any(not isinstance(v, str) or not v for v in
               (case_id, symbol, snapshot_sha256, hypothesis, alternative, invalidation)):
            raise ValueError("EXPLICIT_HYPOTHESIS_FIELDS")
        if (len(snapshot_sha256) != 64 or
                any(c not in "0123456789abcdef" for c in snapshot_sha256)):
            raise ValueError("SNAPSHOT_DIGEST_REQUIRED")
        if not isinstance(evidence, list) or not isinstance(counterevidence, list):
            raise ValueError("EVIDENCE_LIST_REQUIRED")
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO hypotheses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (case_id, symbol, cutoff, available_at, snapshot_sha256,
                        hypothesis, alternative, canonical(evidence),
                        canonical(counterevidence), invalidation,
                        "UNVERIFIED_MODEL_HYPOTHESIS"))

    def review(self, *, case_id, reviewer_a, reviewer_b,
               reviewed_at, verdict, rationale):
        """A separate two-reviewer *declaration*. Not self-certification.

        The independent review becomes visible ONLY at its real
        reviewed_at timestamp; cannot rewrite older as-of model memory.
        """
        if (not isinstance(reviewer_a, str) or not isinstance(reviewer_b, str)
                or reviewer_a == reviewer_b or
                not (5 <= len(reviewer_a) <= 100 and 5 <= len(reviewer_b) <= 100)):
            raise ValueError("TWO_DISTINCT_HUMAN_REVIEWERS_REQUIRED")
        if verdict not in {"CONFIRMED", "REJECTED", "DISPUTED"}:
            raise ValueError("EXPLICIT_REVIEW_VERDICT_REQUIRED")
        if not isinstance(rationale, str) or not 20 <= len(rationale) <= 4000:
            raise ValueError("SUBSTANTIVE_REVIEW_EVIDENCE_REQUIRED")
        reviewed = utc(reviewed_at)
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT cutoff,available_at FROM hypotheses WHERE case_id=?",
                             (case_id,)).fetchone()
            if row is None:
                raise ValueError("UNKNOWN_HYPOTHESIS")
            if reviewed < utc(row[0]) or reviewed < utc(row[1]):
                raise ValueError("REVIEW_CANNOT_PRECEDE_CASE_OR_ITS_PUBLICATION")
            db.execute("INSERT INTO hypothesis_reviews VALUES(?,?,?,?,?,?)",
                       (case_id, reviewer_a, reviewer_b, reviewed_at, verdict, rationale))

    def as_of(self, *, symbol, timestamp):
        cutoff = utc(timestamp)
        with sqlite3.connect(self.path) as db:
            rows = db.execute("""SELECT h.case_id,h.cutoff,h.available_at,h.hypothesis,
                             h.alternative,h.evidence_json,h.counter_json,h.invalidation,
                             r.reviewed_at,r.verdict,r.rationale
                             FROM hypotheses AS h LEFT JOIN hypothesis_reviews AS r
                             ON r.case_id=h.case_id
                             WHERE h.symbol=? ORDER BY h.cutoff,h.case_id""",
                             (symbol,)).fetchall()
        return [{"case_id": row[0], "cutoff": row[1], "available_at": row[2],
                 "hypothesis": row[3], "alternative": row[4],
                 "evidence": json.loads(row[5]), "counterevidence": json.loads(row[6]),
                 "invalidation": row[7],
                 "status": ("HUMAN_REVIEW_" + row[9]
                            if row[8] is not None and utc(row[8]) <= cutoff
                            else "UNVERIFIED_MODEL_HYPOTHESIS"),
                 "review_rationale": (row[10]
                            if row[8] is not None and utc(row[8]) <= cutoff
                            else None)}
                for row in rows if utc(row[1]) <= cutoff and utc(row[2]) <= cutoff]


def prediction_commitment(response):
    obj = {k: v for k, v in response.items() if k != "commitment"}
    return digest(obj)


def grade_blind(test_cases, predictions):
    """Grade immutable test cases against predictions produced WITHOUT gold.

    Hash checks integrity of submitted records, not an independent timestamp
    attestation. Prediction generation and sealing must happen before revealing
    labels to retain a genuine blind exam.
    """
    for row in test_cases:
        verify_case(row)
        if row["split"] != "test":
            raise ValueError("ONLY_SEALED_TEST_CASES")
    mapping = {r["case_id"]: r for r in test_cases}
    if len(mapping) != len(test_cases) or len(predictions) != len(mapping):
        raise ValueError("ALL_AND_ONLY_TEST_PREDICTIONS_REQUIRED")
    seen, correct, unsafe, confirmed = set(), 0, 0, 0
    for p in predictions:
        if not isinstance(p, dict) or set(p) != {
            "case_id", "cutoff", "visible_prefix_sha256",
            "prediction_at", "answer", "commitment"}:
            raise ValueError("EXACT_BLIND_PREDICTION_SCHEMA")
        cid = p["case_id"]
        if cid not in mapping or cid in seen:
            raise ValueError("UNKNOWN_OR_DUPLICATE_BLIND_PREDICTION")
        seen.add(cid)
        case = mapping[cid]
        if p["cutoff"] != case["cutoff"] or p["visible_prefix_sha256"] != case["visible_prefix_sha256"]:
            raise ValueError("SNAPSHOT_IDENTITY_MISMATCH")
        if utc(p["prediction_at"]) < utc(case["cutoff"]):
            raise ValueError("PREDICTION_BEFORE_VISIBLE_CUTOFF")
        if p["commitment"] != prediction_commitment(p):
            raise ValueError("PREDICTION_COMMITMENT_MISMATCH")
        answer = p["answer"]
        if not isinstance(answer, dict):
            raise ValueError("BLIND_ANSWER_REQUIRED")
        unsafe += int(answer.get("execution_allowed") is not False)
        correct += int(answer.get("decision") == case["answer"]["decision"])
        if case["answer"]["wave_confirmed"] is not None:
            confirmed += int(answer.get("wave_confirmed") == case["answer"]["wave_confirmed"])
    total = len(test_cases)
    return {"schema": "WILLIAMS_AI_BLIND_DECISION_EXAM_V1",
            "cases": total, "decision_correct": correct,
            "decision_accuracy": correct / total if total else None,
            "unsafe_execution_claims": unsafe,
            "wave_field_matches_for_non_null_gold": confirmed,
            "blindness_requires_external_test_answer_separation": True,
            "cryptographically_attested_prediction_time": False,
            "trading_qualification": False, "execution_allowed": False}
