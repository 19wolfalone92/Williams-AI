"""FIRST-LOOK independent 2025 Binance archive exam, frozen BEFORE outputs.

New 2025-03/04 USD-M Futures data is not the in-repository 2026 data.
The 4 book profiles × 2 chronological snapshots are predeclared without
seeing 2025 outcomes. Model is REAL, SHA-pinned local GGUF via llama.cpp.

What can be validated:
  - no later closed candle in earlier prompt;
  - source-consistent hypotheses vs independent Williams audit;
  - a new objective fact/structural invalidation acknowledged in GGUF input;
  - true uncoached first-try results, not a coached rerun;
  - five bullets unknown if tick volume absent;
  - five predeclared price-direction 24h outcomes vs naive baseline.
What cannot be validated without INDEPENDENT HUMAN WAVE LABELS:
  - correct nested Elliott degree and future profitability.
Once run the cohort is seen and can no longer be called blind again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Any

from .chart_reader import analyze_chart
from .causal_timeframes import aggregate_closed_h1
from .local_model import LocalWilliamsModel
from .reasoning_trial import evaluate_historical_thought
from .real_gguf_acceptance import actual_gguf_sha256
from .prospective_qualification import (
    freeze_prospective_call,settle_prospective_call,
    summarize_sealed_results,
)
from .temporal_wave_revision import temporal_revision
from .unseen_2025_cohort import SYMBOLS,MONTHS

# Before FIRST execution, 4 profiles x 2 different observations 168h apart.
PAIRED_BOOK_CASES=(
    ("BTCUSDT","TC2",(847,1039)),
    ("SOLUSDT","TC1",(847,1039)),
    ("ETHUSDT","NTD",(847,1039)),
    ("XRPUSDT","COURSE",(847,1039)),
)
DIRECTION_SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT")
DIRECTION_INDEX=1079
COHORT_SCHEMA="WILLIAMS_AI_EXPLICIT_2025_MAR_APR_UNSEEN_COHORT_V1"


def load_archives(path:Path)->dict[str,Any]:
    raw=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw,dict):
        raise ValueError("Malformed source archive file")
    seal=raw.get("source_manifest_sha256")
    payload={k:v for k,v in raw.items() if k!="source_manifest_sha256"}
    h=hashlib.sha256(json.dumps(payload,sort_keys=True,
                   separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if h!=seal or raw.get("schema")!=COHORT_SCHEMA:
        raise ValueError("Unpinned or modified historical archive")
    if sorted(raw["markets"])!=list(SYMBOLS) or (
        raw.get("intentionally_predeclared_dates")!=list(MONTHS)
    ):
        raise ValueError("Source identities or months differ from predeclared cohort")
    for symbol,entry in raw["markets"].items():
        if (entry.get("symbol")!=symbol or not isinstance(entry.get("bars"),list)
                or len(entry["bars"])<1200
                or any(x.get("close_time","")[:7] not in MONTHS for x in entry["bars"])):
            raise ValueError("Unexpected historical H1 identity")
    return raw


def _book_observation(market:dict[str,Any],index:int,
                      symbol:str)->dict[str,Any]:
    rows=market["bars"][:index+1]
    result=analyze_chart(symbol,"H1",rows,as_of=rows[-1]["close_time"],
                         tick_size=market.get("tick_size"))
    higher={}
    for tf in ("D1","H4"):
        closed=aggregate_closed_h1(rows,tf)
        if len(closed)<45:
            higher[tf]={"status":"INSUFFICIENT_CLOSED_HISTORY"}
        else:
            outcome=analyze_chart(
                symbol,tf,closed,as_of=rows[-1]["close_time"],
                tick_size=market.get("tick_size"))
            higher[tf]={
                "status":"OBSERVED_CLOSED_BARS",
                "closed_through":closed[-1]["close_time"],
                "market_context":outcome["evidence"]["market_context"],
                "wave_degree":"UNKNOWN",
            }
    result["higher_timeframe_context"]=higher
    return result


def audit_new_cohort(cohort:dict[str,Any], *,model:Any,
                     model_hash:str,code_id:str)->dict[str,Any]:
    if not isinstance(model_hash,str) or len(model_hash)!=64 or not code_id:
        raise ValueError("Real pinned weights and source code identity required")
    markets=cohort["markets"]
    book_results=[]
    revisions=[]
    for symbol,edition,(earlier_index,later_index) in PAIRED_BOOK_CASES:
        market=markets[symbol]
        earlier_chart=_book_observation(market,earlier_index,symbol)
        later_chart=_book_observation(market,later_index,symbol)
        temporal=temporal_revision(
            earlier_chart,later_chart,
            earlier_higher_context=earlier_chart["higher_timeframe_context"],
            later_higher_context=later_chart["higher_timeframe_context"],
        )
        revisions.append({
            "symbol":symbol,"edition":edition,
            "earlier_as_of":temporal["earlier_as_of"],
            "later_as_of":temporal["later_as_of"],
            "new_confirmed_facts":temporal["changed_confirmed_facts"],
            "earlier_hypothesis_invalidations":temporal["prior_scenarios_lose_support"],
            "unverified_wave_degrees_remain_unknown":not temporal["exact_wave_degree_verified"],
            "all_five_bullets_unverifiable_without_tick_count":(
                temporal["tick_volume_missing_prevents_TC1_squat_confirmation"]
            ),
            "trade_action":None,
        })
        for index,prior in ((earlier_index,None),(later_index,earlier_chart)):
            try:
                trial=evaluate_historical_thought(
                    symbol,market["bars"],tick_size=market.get("tick_size"),
                    bar_index=index,edition=edition,model=model,
                    include_topdown=True,previous_closed_chart=prior,
                )
                audit=trial["audit"]
                out={
                    "symbol":symbol,"edition":edition,"index":index,
                    "as_of":trial["last_visible_at"],
                    "source_verdict":audit["verdict"],
                    "setup":audit["setup"],
                    "source_rule_count":len(audit.get("source_proofs",[])),
                    "hard_failures":audit["hard_failures"],
                    "first_try_without_coaching":True,
                    "objective_temporal_revision_supplied":
                        bool(trial.get("temporal_revision")),
                    "future_closed_candles_sent":trial["model_saw_future_bars"],
                    "trade_action":None,
                }
            except (ValueError,TypeError,RuntimeError,KeyError,IndexError) as exc:
                out={
                    "symbol":symbol,"edition":edition,"index":index,
                    "as_of":market["bars"][index]["close_time"],
                    "source_verdict":"REAL_MODEL_FAILED",
                    "setup":None,"source_rule_count":0,
                    "hard_failures":[type(exc).__name__],
                    "first_try_without_coaching":True,
                    "objective_temporal_revision_supplied":prior is not None,
                    "future_closed_candles_sent":False,
                    "trade_action":None,
                }
            book_results.append(out)
            print("BLIND_2025_BOOK_CASE",
                  json.dumps(out,ensure_ascii=False,sort_keys=True),flush=True)

    outcomes=[]
    direction_errors=[]
    for symbol in DIRECTION_SYMBOLS:
        x=markets[symbol]
        try:
            sealed=freeze_prospective_call(
                symbol,x["bars"][:DIRECTION_INDEX+1],
                tick_size=x.get("tick_size") or 1e-8,
                model_id="real-gguf-"+model_hash,
                code_id=code_id,
                predictor=model.propose_24h_research_forecast,
            )
            settled=settle_prospective_call(sealed,x["bars"])
            outcomes.append(settled)
            print("BLIND_2025_DIRECTION",
                  json.dumps({
                      "symbol":symbol,
                      "model_direction":settled["predicted_direction"],
                      "future_24h_direction":settled["observed_24h_direction"],
                      "correct":settled["correct"],
                      "baseline":settled["baseline_direction"],
                  },sort_keys=True),flush=True)
        except (ValueError,RuntimeError,TypeError,KeyError,IndexError) as exc:
            direction_errors.append({"symbol":symbol,"failure":type(exc).__name__})
    verdicts=Counter(x["source_verdict"] for x in book_results)
    substantive=sum(x["source_verdict"]=="PROVISIONAL_REASONING_ONLY"
                    and x["setup"] not in ("NO_SETUP","UNKNOWN",None,"WAVE_HYPOTHESIS")
                    for x in book_results)
    return {
        "mode":"FIRST_RUN_PREDECLARED_2025_SOURCE_AND_TEMPORAL_GGUF_EXAM",
        "manifest_sha256":cohort["source_manifest_sha256"],
        "pinned_model_sha256":model_hash,
        "code_id":code_id,
        "source_months":list(MONTHS),
        "symbol_universe":len(SYMBOLS),
        "book_situations":len(book_results),
        "source_consistent_total_including_abstentions":
            verdicts["PROVISIONAL_REASONING_ONLY"],
        "source_consistent_substantive_setups":substantive,
        "model_error_count":verdicts["REAL_MODEL_FAILED"],
        "book_verdicts":dict(verdicts),
        "book_results":book_results,
        "temporal_evidence_revisions":revisions,
        "directional_outcomes":len(outcomes),
        "directional_failures":direction_errors,
        "directional_baselines":summarize_sealed_results(outcomes) if outcomes else None,
        "historical_2025_prices_are_new_to_this_project":True,
        "independent_human_elliott_wave_labels":None,
        "independent_book_image_exam_passed":False,
        "safe_abstention_is_not_wave_comprehension":True,
        "this_cohort_no_longer_blind_after_first_run":True,
        "GGUF_weights_modified":False,
        "trade_action":None,
        "execution_allowed":False,
        "trading_qualified":False,
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--local-model",type=Path,required=True)
    p.add_argument("--expected-sha256",required=True)
    p.add_argument("--code-id",required=True)
    args=p.parse_args()
    sources=load_archives(args.cohort)
    weights=actual_gguf_sha256(args.local_model,args.expected_sha256)
    t=monotonic()
    model=LocalWilliamsModel(args.local_model,n_ctx=8192,gpu_layers=0,
                             use_decision_schema=True)
    results=audit_new_cohort(
        sources,model=model,
        model_hash=weights["model_sha256"],
        code_id=args.code_id,
    )
    results["wall_seconds"]=round(monotonic()-t,2)
    print("BLIND_2025_GGUF_SUMMARY",
          json.dumps(results,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
