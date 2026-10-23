"""REAL small GGUF book tutoring rehearsal with private memory, public rule cards.

The test connects actual GGUF inference, 8 independently worded practices across
all 4 editions, immediate corrective lesson feedback, and the next chart
decision receiving the SAME model's private mistakes. Only short ORIGINAL
project rule summaries are involved -- NO user book text is transmitted.

This is in-context remediation, not neural-weight fine-tuning, real historical
P&L or proof of book understanding. An answered coached retry is NOT a blind
holdout. Model remains READ_ONLY; Binance orders unavailable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adaptive_coach import coach_practice
from .causal_paper_lab import frozen_markets
from .local_model import LocalWilliamsModel
from .real_gguf_acceptance import actual_gguf_sha256
from .reasoning_trial import evaluate_historical_thought


def actual_model_coaching(
    weights:Path, expected_sha256:str,private_notebook:Path,
    *, drills_per_edition:int=2,
)->dict:
    if not 1<=drills_per_edition<=5:
        raise ValueError("Limit coaching to five original questions per edition")
    weight_proof=actual_gguf_sha256(weights,expected_sha256)
    model_id="gguf-"+weight_proof["model_sha256"][:24]
    model=LocalWilliamsModel(weights,n_ctx=8192,gpu_layers=0,
                            use_decision_schema=True)
    sources=("TC1","NTD","TC2","COURSE")
    scores={}
    total=success=retry_correct=0
    for edition in sources:
        report=coach_practice(
            model,private_notebook,model_id=model_id,
            editions=(edition,),max_cases=drills_per_edition,
            repeat_after_feedback=True)
        scores[edition]={
            "initial_correct":report["initial_pass"],
            "total_cases":report["cases"],
            "coached_same_case_retries_correct":report["corrected_same_case_retries"],
            "private_misconception_topics":report["private_misconception_topics"],
        }
        total+=report["cases"]
        success+=report["initial_pass"]
        retry_correct+=report["corrected_same_case_retries"]
        print("ACTUAL_GGUF_TUTOR",edition,report["initial_pass"],
              "of",report["cases"],"uncoached first tries;", 
              report["corrected_same_case_retries"],"coached retries",flush=True)
    market=frozen_markets()["ETHUSDT"]
    try:
        case=evaluate_historical_thought(
            "ETHUSDT",market["bars"],tick_size=market["tick_size"],
            bar_index=899,edition="TC2",
            model=model,coach_notebook=private_notebook,model_id=model_id)
        audit=case["audit"]
        historical={
            "verdict":audit["verdict"],
            "setup":audit["setup"],
            "hard_failures":audit["hard_failures"],
            "model_saw_future_bars":case["model_saw_future_bars"],
            "previous_wrong_answers_context_received":
                case["private_prior_mistake_lessons_used"],
            "trade_action":None,
        }
    except (ValueError,TypeError,RuntimeError,KeyError,IndexError) as exc:
        historical={"verdict":"MALFORMED_MODEL_DECISION",
                    "hard_failures":[type(exc).__name__],
                    "trade_action":None}
    return {
        "mode":"REAL_GGUF_PRIVATE_EXTERNAL_MEMORY_COACH_NOT_WEIGHT_TRAINING",
        "actual_gguf_weights_verified":weight_proof,
        "editions":scores,
        "first_try_correct":success,
        "first_try_total":total,
        "coached_same_case_retries_correct":retry_correct,
        "next_historical_decision":historical,
        "real_book_pdf_pages_transmitted":False,
        "private_gguf_student_memory_persisted_for_future_sessions":True,
        "model_weights_updated":False,
        "unseen_out_of_sample_book_comprehension":None,
        "actual_trading_profitability":None,
        "trade_action":None,"execution_allowed":False,
        "caution":"Coached success on the SAME prompt does not prove transfer or 100% understanding.",
    }


def main()->None:
    p=argparse.ArgumentParser(description="Actual public GGUF self-correction, 4 editions, no trades")
    p.add_argument("--local-model",type=Path,required=True)
    p.add_argument("--expected-sha256",required=True)
    p.add_argument("--private-notebook",type=Path,required=True)
    p.add_argument("--drills-per-edition",type=int,default=2)
    a=p.parse_args()
    report=actual_model_coaching(
        a.local_model,a.expected_sha256,a.private_notebook,
        drills_per_edition=a.drills_per_edition)
    print("REAL_GGUF_COACH_REPORT",json.dumps(report,ensure_ascii=False,sort_keys=True),
          flush=True)


if __name__=="__main__":
    main()
