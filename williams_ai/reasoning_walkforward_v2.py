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


def run_prompt_only_replay(prompt_cases, model_predict, memory: HypothesisMemory):
    """Strict inference stage with NO test answers or human review records.

    Run this function in a separate worker that has ONLY test_prompts.jsonl,
    the frozen model and its private scoped memory. Freeze its returned
    predictions before a different process loads gold for grade_blind().
    This function never opens the book/exam gold reference.
    """
    if not isinstance(prompt_cases, list) or not prompt_cases or not callable(model_predict):
        raise ValueError("FROZEN_PROMPTS_AND_MODEL_REQUIRED")
    if not isinstance(memory, HypothesisMemory):
        raise ValueError("EXPLICIT_SCOPED_HYPOTHESIS_MEMORY_REQUIRED")
    required = {
        "case_id", "symbol", "timeframe", "cutoff",
        "visible_prefix_sha256", "rule_id", "edition", "pdf_page",
        "source_sha256", "question", "timeframe_observations",
        "complete_h1_h4_d1_context", "book_evidence",
        "human_source_expertise_not_cryptographically_proven",
        "no_position_or_stop_acknowledgment_supplied",
        "context_tail_bars_per_frame", "confirmed_pivots_per_frame_limit",
    }
    cases=sorted(prompt_cases, key=lambda p:(utc(p["cutoff"]),p["symbol"],p["case_id"]))
    seen=set()
    predictions=[]
    for row in cases:
        if not isinstance(row,dict) or set(row)!=required:
            raise ValueError("PROMPTS_ONLY_FIXED_SCHEMA_NO_GOLD_OR_REVIEWERS")
        if row["case_id"] in seen:
            raise ValueError("DUPLICATE_FROZEN_TEST_PROMPT")
        seen.add(row["case_id"])
        if row["complete_h1_h4_d1_context"] is not True:
            raise ValueError("INCOMPLETE_BLIND_CONTEXT")
        frames=row["timeframe_observations"]
        if (not isinstance(frames,dict) or
                not {"H1","H4","D1"}<=set(frames)):
            raise ValueError("THREE_CLOSED_TIMEFRAMES_REQUIRED")
        for frame in frames.values():
            if utc(frame["last_closed_at"])>utc(row["cutoff"]):
                raise ValueError("FUTURE_BAR_IN_FROZEN_PROMPT")
        prior=memory.as_of(symbol=row["symbol"],edition=row["edition"],
                           timestamp=row["cutoff"],strict_before=True)
        answer=model_predict(row,prior)
        _valid_answer(answer)
        memory.append(case_id=row["case_id"],symbol=row["symbol"],
                      cutoff=row["cutoff"],available_at=row["cutoff"],
                      snapshot_sha256=row["visible_prefix_sha256"],
                      hypothesis=answer["hypothesis"],
                      alternative=answer["alternative"],evidence=answer["evidence"],
                      counterevidence=answer["counterevidence"],
                      invalidation=answer["invalidation"],
                      edition=row["edition"],rule_id=row["rule_id"],
                      book_sha256=row["source_sha256"])
        prediction={"case_id":row["case_id"],"cutoff":row["cutoff"],
                    "visible_prefix_sha256":row["visible_prefix_sha256"],
                    "prediction_at":row["cutoff"],"answer":answer}
        prediction["commitment"]=prediction_commitment(prediction)
        predictions.append(prediction)
    return {"schema":"WILLIAMS_AI_TEST_PROMPTS_ONLY_REPLAY_V2",
            "predictions":predictions,
            "gold_answer_loaded_by_this_function":False,
            "prediction_at_is_simulated_asof_not_external_timestamp":True,
            "requires_separate_process_for_true_blindness":True,
            "execution_allowed":False}


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
