"""Sequential as-of market reasoning exam for a caller-supplied model.

A model receives ONLY a visible closed market prefix, prior unverified
hypotheses available at that cutoff, and NO gold test answer.
Results are scored only after all predictions have been frozen.
"""
from __future__ import annotations

from collections import Counter

from .reasoning_research_v2 import (
    HypothesisMemory, _valid_answer, canonical, digest, grade_blind,
    prediction_commitment, question_only, utc, verify_case,
)


def run_blind_replay(test_cases, model_predict, memory: HypothesisMemory):
    if not callable(model_predict) or not isinstance(memory, HypothesisMemory):
        raise ValueError("EXPLICIT_MODEL_AND_PRIVATE_MEMORY_REQUIRED")
    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError("NONEMPTY_PRIVATE_TEST_CASES_REQUIRED")
    if any(row.get("split") != "test" for row in test_cases):
        raise ValueError("TEST_ONLY_REPLAY")
    cases = sorted(test_cases, key=lambda row: (utc(row["cutoff"]), row["symbol"], row["case_id"]))
    seen = set()
    output = []
    for row in cases:
        verify_case(row)
        if row["case_id"] in seen:
            raise ValueError("UNIQUE_BLIND_CASE_ID_REQUIRED")
        seen.add(row["case_id"])
        prompt = question_only(row)
        earlier = memory.as_of(symbol=row["symbol"], edition=row["edition"],
                               timestamp=row["cutoff"], strict_before=True)
        # A caller-provided inference adapter must not have access to private
        # gold labels. This interface never supplies them.
        answer = model_predict(prompt, earlier)
        _valid_answer(answer)
        memory.append(case_id=row["case_id"], symbol=row["symbol"],
                      cutoff=row["cutoff"], available_at=row["cutoff"],
                      snapshot_sha256=row["visible_prefix_sha256"],
                      hypothesis=answer["hypothesis"], alternative=answer["alternative"],
                      evidence=answer["evidence"], counterevidence=answer["counterevidence"],
                      invalidation=answer["invalidation"],
                      edition=row["edition"], rule_id=row["rule_id"],
                      book_sha256=row["source_sha256"])
        prediction = {"case_id": row["case_id"], "cutoff": row["cutoff"],
                      "visible_prefix_sha256": row["visible_prefix_sha256"],
                      "prediction_at": row["cutoff"], "answer": answer}
        prediction["commitment"] = prediction_commitment(prediction)
        output.append(prediction)
    return {"schema": "WILLIAMS_AI_ASOF_REPLAY_V1",
            "predictions": output, "grade": grade_blind(cases, output),
            "simulated_time_not_real_time_attestation": True,
            "gold_not_passed_to_predictor": True, "execution_allowed": False}


def compare_frozen_predictions(test_cases, parent_predictions, candidate_predictions):
    """Same private untouched test; cannot train or select on test results."""
    before = grade_blind(test_cases, parent_predictions)
    after = grade_blind(test_cases, candidate_predictions)
    return {"schema": "WILLIAMS_AI_PAIRED_BLIND_COMPARE_V1",
            "parent": before, "candidate": after,
            "accuracy_delta_points": 100 * (after["decision_accuracy"] - before["decision_accuracy"]),
            "risk_regression": after["unsafe_execution_claims"] > before["unsafe_execution_claims"],
            "test_used_for_checkpoint_selection": False,
            "execution_allowed": False}
