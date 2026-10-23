"""ACTUAL local Qwen GGUF historical reasoning exam, after adding Williams
nonlinear evidence, five TC1 bullets and trader psychology.

Eight fixed, PREDECLARED symbol/edition/cutoff cases, including the two
prior WM1 failures at BTC/ETH index 899. No test selection by results.
Model sees only OHLC <= decision timestamp, plus closed H4/D1 summaries,
short *public original rule cards*, no full PDFs, no future and no trades.

Unlike the 150-cutoff deterministic study, THIS component calls genuine
GGUF weights on a CPU. A valid structured decision alone is NOT proof of
profitable trading, deep psychology, book mastery or image comprehension.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import monotonic

from .causal_paper_lab import frozen_markets
from .real_gguf_acceptance import actual_gguf_sha256
from .local_model import LocalWilliamsModel
from .reasoning_trial import evaluate_historical_thought

# Chosen BEFORE results. Keep pinned when repeating to prevent cherry picking.
CASES=(
    ("BTCUSDT",899,"TC2"),
    ("ETHUSDT",899,"TC2"),
    ("SOLUSDT",1099,"TC1"),
    ("XRPUSDT",1151,"TC1"),
    ("BNBUSDT",1145,"NTD"),
    ("ADAUSDT",1151,"COURSE"),
    ("DOGEUSDT",923,"TC2"),
    ("LINKUSDT",1163,"TC1"),
)


def run_actual_history_exam(model, *,verified_weights:dict)->dict:
    if verified_weights.get("verified_model_magic")!="GGUF":
        raise ValueError("Only genuine fingerprint-checked GGUF weights qualify")
    frozen=frozen_markets()
    results=[]
    for symbol,bar_index,edition in CASES:
        x=frozen[symbol]
        try:
            trial=evaluate_historical_thought(
                symbol,x["bars"],tick_size=x["tick_size"],
                bar_index=bar_index,edition=edition,model=model,
                include_topdown=True)
            audit=trial["audit"]
            result={
                "symbol":symbol,"bar_index":bar_index,"edition":edition,
                "as_of":trial["last_visible_at"],
                "verdict":audit["verdict"],
                "setup":audit["setup"],
                "hard_failures":audit["hard_failures"],
                "uncertainties":audit["uncertainties"],
                "brief_explanation":trial["model_explanation"][:250],
                "future_candles_sent_to_model":trial["model_saw_future_bars"],
                "causal_h4_d1_present":bool(trial["top_down_context"]),
                "trade_action":None,
            }
        except (ValueError,TypeError,RuntimeError,KeyError,IndexError) as exc:
            # A bad GGUF JSON response is an ERROR, never rewritten to safe
            # NO_SETUP and never counted as a correct chart interpretation.
            result={
                "symbol":symbol,"bar_index":bar_index,"edition":edition,
                "as_of":x["bars"][bar_index]["close_time"],
                "verdict":"MALFORMED_OR_FAILED_REASONING",
                "setup":None,"hard_failures":[type(exc).__name__],
                "uncertainties":[],"brief_explanation":"",
                "future_candles_sent_to_model":False,
                "trade_action":None,
            }
        results.append(result)
        print("ACTUAL_GGUF_HISTORICAL_CASE",
              json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
    counts=Counter(x["verdict"] for x in results)
    failures=Counter(y for x in results for y in x["hard_failures"])
    supported=sum(x["verdict"]=="PROVISIONAL_REASONING_ONLY" for x in results)
    model_no_setup=sum(x["setup"]=="NO_SETUP" for x in results)
    return {
        "mode":"ACTUAL_GGUF_CLOSED_CHART_8_CASE_BLIND_TIMESTAMP_READ_ONLY",
        "weight_sha256":verified_weights["model_sha256"],
        "cases_selected_before_inference":len(CASES),
        "edition_counts":dict(Counter(edition for _,_,edition in CASES)),
        "results":results,
        "verdict_counts":dict(counts),"failure_reasons":dict(failures),
        "model_supported_reasoning_cases":supported,
        "model_no_setup_responses":model_no_setup,
        "strict_safety_qualification":"PASS_ALL" if supported==len(CASES)
                                      else "NOT_QUALIFIED",
        "not_concealed_future_prices_in_model_context":True,
        "full_private_books_used":False,
        "visual_image_understanding_tested":False,
        "weights_finetuned_in_this_experiment":False,
        "independent_book_accuracy_estimate":None,
        "out_of_sample_trade_PnL":None,
        "trade_action":None,"execution_allowed":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--expected-sha256",required=True)
    p.add_argument("--ctx",type=int,default=8192)
    p.add_argument("--fail-on-unsafe",action="store_true")
    args=p.parse_args()
    w=actual_gguf_sha256(args.model,args.expected_sha256)
    model=LocalWilliamsModel(args.model,n_ctx=args.ctx,gpu_layers=0,
                             use_decision_schema=True)
    start=monotonic()
    report=run_actual_history_exam(model,verified_weights=w)
    report["elapsed_cpu_wall_seconds"]=round(monotonic()-start,1)
    print("ACTUAL_GGUF_HISTORICAL_SUMMARY",
          json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)
    if args.fail_on_unsafe and report["strict_safety_qualification"]!="PASS_ALL":
        raise SystemExit(2)


if __name__=="__main__":
    main()
