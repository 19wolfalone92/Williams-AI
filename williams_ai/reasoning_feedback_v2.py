"""Causal error-memory -> adaptive SFT sampling, with no test gold.

Reviewer declarations are not proof of professional expertise. Every record
is derived from a validation prediction, bound to its source/cutoff/adapter.
Rejected test examples can NEVER appear in a training schedule.
"""
from __future__ import annotations
from pathlib import Path
import sqlite3

from .reasoning_research_v2 import HypothesisMemory, adaptive_schedule, utc
from .book_rules import registry

ERROR_CODES={
    "MISUNDERSTOOD_AC", "WRONG_AO_DIRECTION", "MISSED_ALTERNATIVE",
    "UNSUPPORTED_WAVE_CONFIRMATION", "WRONG_WAVE_STRUCTURE",
    "WRONG_RISK_BUDGET", "MISSING_STOP_PROTECTION", "MISSING_FRACTAL",
    "MISSING_CONTEXT_H4_D1", "UNSUPPORTED_SOURCE_CLAIM",
    "UNCONFIRMED_CAMPAIGN_STATE",
}

class ReviewedFailureJournal:
    """Private model/run-scoped validation-only observed failure journal."""
    def __init__(self, path, *, run_id, model_id):
        self._memory=HypothesisMemory(Path(path),run_id=run_id,model_id=model_id)
        self.path,self.run_id,self.model_id=(
            self._memory.path,run_id,model_id)
        with sqlite3.connect(self.path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS learning_feedback_v2 (
                run_id TEXT NOT NULL,model_id TEXT NOT NULL,
                case_id TEXT NOT NULL,rule_id TEXT NOT NULL,
                edition TEXT NOT NULL,source_sha256 TEXT NOT NULL,
                prediction_sha256 TEXT NOT NULL,decision_cutoff TEXT NOT NULL,
                reviewed_at TEXT NOT NULL,error_code TEXT NOT NULL,
                reviewer_a TEXT NOT NULL,reviewer_b TEXT NOT NULL,
                split TEXT NOT NULL CHECK(split='validation'),
                PRIMARY KEY (run_id,model_id,case_id,error_code)
            )""")

    def append_validation_failure(
        self, *,case_id,rule_id,edition,source_sha256,
        prediction_sha256,decision_cutoff,reviewed_at,
        error_code,reviewer_a,reviewer_b,split="validation",
    ):
        if split!="validation":
            raise ValueError("TEST_OR_TRAIN_GOLD_MUST_NOT_FEED_ADAPTATION")
        if error_code not in ERROR_CODES:
            raise ValueError("UNCLASSIFIED_REVIEWED_FAILURE_CODE")
        if (not isinstance(case_id,str) or not 5<=len(case_id)<=150 or
                not isinstance(prediction_sha256,str) or len(prediction_sha256)!=64 or
                any(x not in "0123456789abcdef" for x in prediction_sha256)):
            raise ValueError("FROZEN_PREDICTION_FINGERPRINT_REQUIRED")
        if (not isinstance(reviewer_a,str) or not isinstance(reviewer_b,str) or
                reviewer_a==reviewer_b or
                not 5<=len(reviewer_a)<=100 or not 5<=len(reviewer_b)<=100):
            raise ValueError("TWO_DISTINCT_REVIEWER_RECORDS_REQUIRED")
        if utc(reviewed_at)<=utc(decision_cutoff):
            raise ValueError("FEEDBACK_MUST_NOT_PREDATE_PREDICTION")
        rules={x["id"]:x for x in registry()["rule_cards"]}
        books={x["id"]:x["sha256"] for x in registry()["books"]}
        rule=rules.get(rule_id)
        if (edition not in books or rule is None or
                rule["source"]!=edition or rule["status"]!="REVIEWED_PAGES" or
                books[edition]!=source_sha256):
            raise ValueError("REVIEWED_ERROR_BOOK_OR_RULE_MISMATCH")
        with sqlite3.connect(self.path) as db:
            db.execute("""INSERT INTO learning_feedback_v2 VALUES
                       (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                       (self.run_id,self.model_id,case_id,rule_id,edition,
                        source_sha256,prediction_sha256,decision_cutoff,reviewed_at,
                        error_code,reviewer_a,reviewer_b,split))

    def validation_error_metadata_as_of(self, *,timestamp):
        cutoff=utc(timestamp)
        with sqlite3.connect(self.path) as db:
            rows=db.execute("""SELECT case_id,rule_id,error_code,reviewed_at,
                             decision_cutoff FROM learning_feedback_v2
                             WHERE run_id=? AND model_id=? AND split='validation'
                             ORDER BY reviewed_at,case_id,error_code""",
                            (self.run_id,self.model_id)).fetchall()
        return [{"case_id":case_id,"split":"validation",
                 "rule_id":rule_id,"error_code":error_code}
                for case_id,rule_id,error_code,reviewed_at,decision_at in rows
                if utc(reviewed_at)<cutoff and utc(decision_at)<utc(reviewed_at)]

    def schedule_from_previous_errors(self, train_cases, *,timestamp,rounds=10):
        failures=self.validation_error_metadata_as_of(timestamp=timestamp)
        if not failures:
            raise ValueError("NO_REVIEWED_HISTORIC_VALIDATION_FAILURES")
        result=adaptive_schedule(train_cases,failures,rounds=rounds)
        result["feedback_origin"]="MODEL_SCOPED_PREVIOUSLY_REVIEWED_VALIDATION_ERRORS"
        result["previously_reviewed_failure_count"]=len(failures)
        result["raw_validation_answers_retrieved"]=False
        return result
