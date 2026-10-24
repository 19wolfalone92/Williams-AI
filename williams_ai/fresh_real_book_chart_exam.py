"""Read-only REAL GGUF stress exam on newly retrieved, CLOSED Spot H1 charts.

Predeclared five source/timeframe tasks, BEFORE consulting the current market
feed. A real local model is checked against independent source / causal
contracts. Source consistency is weaker than independent expert wave labels.
No price direction accuracy, exchange signal, stop acknowledgement or
profitable execution can be established from one current closed candle.

Real Binance Futures public REST can return HTTP451 on GitHub runners: this
research is explicitly labeled SPOT and never silently relabeled Futures.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Any,Callable

from .public_forward_capture import capture_latest_closed_h1
from .reasoning_trial import evaluate_historical_thought
from .real_gguf_acceptance import actual_gguf_sha256

# Assignment chosen BEFORE market / model observations: covers each source.
CASES=(
    ("BTCUSDT","TC2"),
    ("ETHUSDT","TC1"),
    ("SOLUSDT","NTD"),
    ("XRPUSDT","COURSE"),
    ("BNBUSDT","TC2"),
)


def run_fresh_book_chart_exam(model: Any, *,
                              weights:dict[str,Any],
                              fetch_json:Callable[[str],Any]|None=None,
                              clock_ms:int|None=None) -> dict[str,Any]:
    if (weights.get("verified_model_magic")!="GGUF" or
            len(str(weights.get("model_sha256","")))!=64):
        raise ValueError("Actual pinned local GGUF mandatory")
    if len(CASES)!=5 or len({x[0] for x in CASES})!=5:
        raise ValueError("Predeclared source exam cases changed")
    results=[]
    for symbol,edition in CASES:
        args={"venue":"SPOT","clock_ms":clock_ms}
        if fetch_json is not None:
            args["fetch_json"]=fetch_json
        # Market endpoint and H1 cut known only NOW, not selected to fit
        # our theory or chosen by future-return statistics.
        public=capture_latest_closed_h1(symbol,**args)
        bars=public["market"]["bars"]
        as_of=bars[-1]["close_time"]
        details={
            "symbol":symbol,"edition":edition,
            "last_known_h1_close":as_of,
            "venue":"SPOT",
            "public_price_input_sha256":public["source"]["market_sha256"],
            "bars_available_at_inference":len(bars),
            "only_closed_h1":True,
        }
        try:
            trial=evaluate_historical_thought(
                symbol,bars,tick_size=public["market"]["tick_size"],
                bar_index=len(bars)-1,edition=edition,model=model,
                include_topdown=True,
            )
            audit=trial["audit"]
            details.update({
                "verdict":audit["verdict"],
                "setup":audit["setup"],
                "hard_failures":audit["hard_failures"],
                "uncertainties":audit["uncertainties"],
                "source_card_count":len(audit["source_proofs"]),
                "h4_d1_cross_scale_hypotheses_available":(
                    "cross_scale_wave_theses" in trial["top_down_context"]),
                "book_image_wave_truth_verified":False,
                "future_bars_sent_to_model":trial["model_saw_future_bars"],
            })
        except (ValueError,KeyError,TypeError,RuntimeError,IndexError) as exc:
            # Model JSON failures are nontrading failures; NEVER coerce
            # errors into fake NO_SETUP, source-consistent or abstentions.
            details.update({
                "verdict":"MODEL_MALFORMED_OR_RUNTIME_FAILED",
                "setup":None,
                "hard_failures":[type(exc).__name__],
                "uncertainties":[],
                "source_card_count":0,
                "h4_d1_cross_scale_hypotheses_available":False,
                "book_image_wave_truth_verified":False,
                "future_bars_sent_to_model":False,
            })
        details["trade_action"]=None
        details["execution_allowed"]=False
        results.append(details)
        print("NEW_REAL_GGUF_BOOK_CHART_CASE",
              json.dumps(details,sort_keys=True,ensure_ascii=False),flush=True)
    counts=Counter(x["verdict"] for x in results)
    substantive=sum(x["verdict"]=="PROVISIONAL_REASONING_ONLY" and
                    x["setup"] not in (None,"NO_SETUP","UNKNOWN","WAVE_HYPOTHESIS")
                    for x in results)
    no_setup=sum(x["setup"] in ("NO_SETUP","UNKNOWN") for x in results)
    return {
        "mode":"NEWLY_CAPTURED_PUBLIC_SPOT_H1_REAL_GGUF_BOOK_CONSISTENCY_EXAM",
        "venue":"SPOT_NOT_BINANCE_FUTURES",
        "model_sha256":weights["model_sha256"],
        "source_task_assignments_declared_before_market_observation":True,
        "case_count":len(results),
        "verdict_counts":dict(counts),
        "substantive_source_consistent_setups_not_fills":substantive,
        "safe_no_setup_cases_not_book_mastery":no_setup,
        "source_consistent_is_not_expert_williams_truth":True,
        "market_data_was_not_in_the_repo_before_run":True,
        "weights_pretraining_history_unknown":True,
        "out_of_sample_market_direction_accuracy":None,
        "expert_labeled_complex_wave_accuracy":None,
        "real_exchange_fills":0,
        "live_qualification":"NOT_QUALIFIED",
        "cases":results,
        "trade_action":None,"execution_allowed":False,
    }


def main()->None:
    parser=argparse.ArgumentParser(description="Real pinned GGUF tests its book claims on newly retrieved H1 Spot charts")
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--expected-sha256",required=True)
    parser.add_argument("--ctx",type=int,default=8192)
    a=parser.parse_args()
    weights=actual_gguf_sha256(a.model,a.expected_sha256)
    from .local_model import LocalWilliamsModel
    model=LocalWilliamsModel(a.model,n_ctx=a.ctx,gpu_layers=0,
                             use_decision_schema=True)
    start=monotonic()
    report=run_fresh_book_chart_exam(model,weights=weights)
    report["elapsed_cpu_seconds"]=round(monotonic()-start,3)
    print("NEW_REAL_GGUF_BOOK_CHART_SUMMARY",
          json.dumps(report,sort_keys=True,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
