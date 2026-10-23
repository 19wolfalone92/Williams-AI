"""Expanded real GGUF historical chart exam: books AND next-day directions.

These cases are *predeclared* and historical public data in this repository
was examined during development. Therefore they ARE NOT an independent,
untouched blind cohort. We test (a) source-specific causal chart reasoning
across TC1 / NTD / TC2 / Course on 16 historical chart prefixes; (b)
a separate explicitly non-book-entry 24h direction/ABSTAIN study for 15
symbols vs matched same-sample baselines, without passing futures to GGUF.

Real pinned weights required, no fake model, no API keys, NO ORDERS. A safe
NO_SETUP and an ABSTAIN count as abstentions rather than book mastery.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Any

from .causal_paper_lab import frozen_markets
from .local_model import LocalWilliamsModel
from .real_gguf_acceptance import actual_gguf_sha256
from .reasoning_trial import evaluate_historical_thought
from .prospective_qualification import (
    freeze_prospective_call, settle_prospective_call, summarize_sealed_results,
)

# Pinned before inspecting this NEW code's actual model outputs.
SOURCE_CASES=(
    ("BTCUSDT",899,"TC2"),
    ("ETHUSDT",899,"TC2"),
    ("DOGEUSDT",923,"TC2"),
    ("AVAXUSDT",1109,"TC2"),
    ("SOLUSDT",1099,"TC1"),
    ("XRPUSDT",1151,"TC1"),
    ("LINKUSDT",1163,"TC1"),
    ("BCHUSDT",1107,"TC1"),
    ("BNBUSDT",1145,"NTD"),
    ("LTCUSDT",1121,"NTD"),
    ("SUIUSDT",1093,"NTD"),
    ("BTCUSDT",1119,"NTD"),
    ("ADAUSDT",1151,"COURSE"),
    ("DOTUSDT",1039,"COURSE"),
    ("TRXUSDT",1117,"COURSE"),
    ("NEARUSDT",957,"COURSE"),
)
DIRECTION_CUT=1047
MODEL_EDITION_COUNTS={"TC1":4,"NTD":4,"TC2":4,"COURSE":4}


def run_real_expanded_history(
    model: Any, *, verified_weights: dict[str,Any],
    code_id: str="fixed-research-source", 
) -> dict[str,Any]:
    if verified_weights.get("verified_model_magic")!="GGUF" or (
        not isinstance(verified_weights.get("model_sha256"),str) or
        len(verified_weights["model_sha256"])!=64
    ):
        raise ValueError("Genuine fingerprint-checked GGUF required")
    if len(SOURCE_CASES)!=16 or Counter(x[2] for x in SOURCE_CASES)!=MODEL_EDITION_COUNTS:
        raise ValueError("Source-cases protocol was mutated")
    markets=frozen_markets()
    if len(markets)!=15:
        raise ValueError("Previously reviewed frozen cohort unavailable")
    exam_results=[]
    for symbol,index,edition in SOURCE_CASES:
        m=markets[symbol]
        try:
            trial=evaluate_historical_thought(
                symbol,m["bars"],tick_size=m["tick_size"],
                bar_index=index,edition=edition,model=model,
                include_topdown=True,
            )
            audit=trial["audit"]
            exam={
                "symbol":symbol,"last_visible_index":index,
                "last_visible_time":trial["last_visible_at"],"edition":edition,
                "model_answered":True,
                "verdict":audit["verdict"],"setup":audit["setup"],
                "hard_failures":audit["hard_failures"],
                "uncertainties":audit["uncertainties"],
                "rule_evidence_count":len(audit.get("source_proofs",[])),
                "causal_multiscale_available":bool(trial["top_down_context"]),
                "model_saw_future_bars":trial["model_saw_future_bars"],
                "trade_action":None,"execution_allowed":False,
            }
        except (ValueError,RuntimeError,TypeError,KeyError,IndexError) as exc:
            # Genuine runtime failure is failure, NOT NO_SETUP or an abstention.
            exam={
                "symbol":symbol,"last_visible_index":index,
                "last_visible_time":m["bars"][index]["close_time"],
                "edition":edition,"model_answered":False,
                "verdict":"MODEL_INVALID_OR_RUNTIME_FAILURE",
                "setup":None,"hard_failures":[type(exc).__name__],
                "uncertainties":[],"rule_evidence_count":0,
                "causal_multiscale_available":False,
                "model_saw_future_bars":False,
                "trade_action":None,"execution_allowed":False,
            }
        exam_results.append(exam)
        print("ACTUAL_EXPANDED_SOURCE_CASE",
              json.dumps(exam,ensure_ascii=False,sort_keys=True),flush=True)

    # The 15-pair study explicitly compares direction to withheld future only
    # after sealing. It is intentionally NOT the Williams trade trigger exam.
    settled=[]
    inference_failures=[]
    for symbol in sorted(markets):
        m=markets[symbol]
        rows=m["bars"]
        try:
            sealed=freeze_prospective_call(
                symbol,rows[:DIRECTION_CUT+1],tick_size=m["tick_size"],
                model_id="gguf-"+verified_weights["model_sha256"],
                code_id=code_id,
                predictor=model.propose_24h_research_forecast,
            )
            outcome=settle_prospective_call(sealed,rows)
            settled.append(outcome)
            print("ACTUAL_EXPANDED_DIRECTION_CASE",
                  json.dumps({
                      "symbol":symbol,"cut":DIRECTION_CUT,
                      "model_direction":outcome["predicted_direction"],
                      "future_24h_direction":outcome["observed_24h_direction"],
                      "baseline":outcome["baseline_direction"],
                      "correct":outcome["correct"],
                      "execution_allowed":False,
                  },sort_keys=True),flush=True)
        except (ValueError,RuntimeError,TypeError,KeyError,IndexError) as exc:
            error={"symbol":symbol,"failure_code":type(exc).__name__,
                   "trade_action":None,"execution_allowed":False}
            inference_failures.append(error)
            print("ACTUAL_EXPANDED_DIRECTION_FAILED",
                  json.dumps(error,sort_keys=True),flush=True)
    summary=summarize_sealed_results(settled) if settled else None
    verdicts=Counter(x["verdict"] for x in exam_results)
    assert all(x["trade_action"] is None and x["execution_allowed"] is False
               and x["model_saw_future_bars"] is False for x in exam_results)
    return {
        "mode":"REVIEWED_HISTORY_REAL_GGUF_EXPANDED_BOOK_AND_DIRECTION_DIAGNOSTIC",
        "model_sha256":verified_weights["model_sha256"],
        "historical_cohort_already_examined_in_project":True,
        "independent_blind_holdout":False,
        "edited_model_weights":False,
        "historical_source_cases":len(SOURCE_CASES),
        "editions":MODEL_EDITION_COUNTS,
        "source_audit_verdicts":dict(verdicts),
        "source_consistent_cases":verdicts["PROVISIONAL_REASONING_ONLY"],
        "source_consistent_substantive_setups":sum(
            x["verdict"]=="PROVISIONAL_REASONING_ONLY"
            and x["setup"] not in (None,"NO_SETUP","UNKNOWN","WAVE_HYPOTHESIS")
            for x in exam_results),
        "safe_no_setup_is_not_understanding":True,
        "source_cases":exam_results,
        "directional_models_invoked":15,
        "directional_settled_cases":len(settled),
        "directional_inference_failures":inference_failures,
        "directional_report":summary,
        "direction_is_not_williams_signal_or_execution":True,
        "market_correlations_prevent_independent_15_trial_claim":True,
        "real_binance_order_fill_or_stop_ack_count":0,
        "wave_degree_truth_from_independent_humans":None,
        "book_image_comprehension_proved":False,
        "real_trading_qualification":"NOT_QUALIFIED",
        "trade_action":None,"execution_allowed":False,
    }


def main() -> None:
    p=argparse.ArgumentParser(description="Actual pinned GGUF expanded historical chart evaluation")
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--expected-sha256",required=True)
    p.add_argument("--code-id",required=True)
    p.add_argument("--ctx",type=int,default=8192)
    args=p.parse_args()
    weights=actual_gguf_sha256(args.model,args.expected_sha256)
    model=LocalWilliamsModel(args.model,n_ctx=args.ctx,gpu_layers=0,
                             use_decision_schema=True)
    t=monotonic()
    report=run_real_expanded_history(model,verified_weights=weights,code_id=args.code_id)
    report["elapsed_cpu_wall_seconds"]=round(monotonic()-t,2)
    print("ACTUAL_EXPANDED_HISTORICAL_SUMMARY",
          json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
