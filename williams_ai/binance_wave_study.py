"""Complete native Binance Futures timeframe matrix -> causal wave study.

15 separately sourced native timeframes, not synthetic 1h->D1 as the only
higher-TF history. Every TF has persisted exact exchange open/close sequence,
quote/taker volumes and source venue; incomplete TFs are transparently blocked.
A 100-140 bar "wave rhythm" is contextual, not a rigid Elliott proof.

Optional local Qwen GGUF sees one common causal set of independently CLOSED
timeframes. The model produces reviewed proposals, never autonomous orders.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .binance_wave_history import (
    ANALYSIS_TFS, INTERVAL_MS,VENUES,DurableWaveHistory,
    HistoryUnavailable,
)
from .chart_reader import analyze_chart,TF_SECONDS
from .market_hierarchy import reconcile_market_hierarchy
from .cross_scale_wave_theses import cross_scale_wave_theses

DEFAULT_WINDOW=1400


def _compact_chart(view:dict[str,Any])->dict[str,Any]:
    waves=view.get("wave_structure",{})
    market=view.get("market_position",{})
    return {
        "status":"OBSERVED_CLOSED_BARS",
        "bars":view["bars_used"],
        "closed_through":view["last_closed_at"],
        "market_context":view["evidence"]["market_context"],
        "ao":view["evidence"]["ao"],"ac":view["evidence"]["ac"],
        "ao_color":view["evidence"]["ao_color"],
        "ac_color":view["evidence"]["ac_color"],
        "zone":view["evidence"]["zone"],
        "alligator":{
            "jaw":view["evidence"]["jaw_visible"],
            "teeth":view["evidence"]["teeth_visible"],
            "lips":view["evidence"]["lips_visible"],
        },
        "confirmed_swings":[
            {k:p.get(k) for k in ("index","time","price","type","confirmed_at_index")}
            for p in waves.get("recent_confirmed_swings",[])[-5:]
            if isinstance(p,dict)],
        "impulse_alternatives":len(waves.get("impulse_alternatives",[])),
        "corrective_alternatives":len(waves.get("corrective_alternatives",[])),
        "wave_degree":"UNKNOWN",
        "recent_levels":market.get("recent_levels",{}),
        "observed_wise_men_by_type":{
            typ:sum(x.get("type")==typ for x in view.get("candidates",[]))
            for typ in ("WM1","WM2_SUPER_AO","WM3_FRACTAL")
        },
        "no_future_bar":True,
        "wave_degree_is_independently_confirmed":False,
    }


def analyze_all_cached(
    store:DurableWaveHistory,*,venue:str,symbol:str,
    intervals:tuple[str,...]=tuple(INTERVAL_MS),
    window:int=DEFAULT_WINDOW,model:Any=None,edition:str="TC2",
)->dict[str,Any]:
    if venue not in VENUES or edition not in {"TC1","NTD","TC2","COURSE"}:
        raise ValueError("UNKNOWN_MARKET_VENUE_OR_BOOK_EDITION")
    if (not isinstance(window,int) or isinstance(window,bool)
            or not 140<=window<=10000):
        raise ValueError("INVALID_WAVE_STUDY_WINDOW")
    if not intervals or len(set(intervals))!=len(intervals) or (
            any(tf not in INTERVAL_MS for tf in intervals)):
        raise ValueError("UNKNOWN_OR_DUPLICATE_BINANCE_TIMEFRAME")
    matrix={}
    views={}
    for native in intervals:
        analysis_tf=ANALYSIS_TFS[native]
        rows=store.recent(venue,symbol,native,limit=window)
        meta=store.status(venue,symbol,native)
        if len(rows)<45:
            matrix[native]={
                "study_status":"INSUFFICIENT_HISTORY",
                "analysis_tf":analysis_tf,
                "cached_total":meta["stored_closed_bars"],
                "closed_visible_bars":len(rows),
                "requires_at_least_45_closed_bars":True,
                "wave_degree":"UNKNOWN",
                "execution_allowed":False,
            }
            continue
        # Each TF is separately sourced and finalized on Binance UTC.
        # No H1 series is passed into the D1/H4 analyzer as if D1 was ready.
        view=analyze_chart(symbol,analysis_tf,rows,
                           as_of=rows[-1]["close_time"],
                           tick_size=float(meta["tick_size"]))
        if view["gaps"] or view["last_closed_at"]!=view["as_of"]:
            raise HistoryUnavailable("CACHED_MARKET_CAUSALITY_OR_GAPS_UNVERIFIED")
        passport=_compact_chart(view)
        passport.update({
            "study_status":"OBSERVED_CLOSED_BARS",
            "analysis_tf":analysis_tf,
            "cached_total":meta["stored_closed_bars"],
            "closed_visible_bars":len(rows),
            "first_cached_open_ms":meta["first_open_ms"],
            "last_cached_close_ms":meta["last_closed_ms"],
            "exchange_quote_volume_recent":rows[-1]["quote_volume"],
            "exchange_number_trades_recent":rows[-1]["number_of_trades"],
            "exchange_taker_buy_base_recent":rows[-1]["taker_buy_base_volume"],
            "100_140_bar_rhythm_is_heuristic_only":True,
            "trade_action":None,"execution_allowed":False,
        })
        views[native]=view
        matrix[native]=passport
    # Cross-TF parent/child summary is *only* a set of alternatives:
    # correlations, opposing phases and momentum do not certify Elliot waves.
    ordered=[q for q in reversed(INTERVAL_MS) if q in intervals]
    cross=[]
    for parent,child in zip(ordered,ordered[1:]):
        a,b=matrix[parent],matrix[child]
        aligned=(a.get("market_context")==b.get("market_context")
                 and a.get("market_context") in ("UP_CONTEXT","DOWN_CONTEXT"))
        cross.append({
            "parent":parent,"child":child,
            "status":(
                "INSUFFICIENT_CLOSED_BARS" if "market_context" not in a or "market_context" not in b
                else "ALIGNED_DIRECTION_NOT_WAVE_DEGREE" if aligned
                else "DIFFERENT_OR_UNRESOLVED_CONTEXT"
            ),
            "possible_interpretations":[
                "Continuation of larger movement",
                "Lower timeframe correction within higher movement",
                "Possible reversal if future structure confirms",
            ],
            "no_certified_degree":True,
            "execution_allowed":False,
        })
    core={}
    for tf,native in (("D1","1d"),("H4","4h"),("H1","1h")):
        if native in views:
            core[tf]={
                "status":"OBSERVED_CLOSED_BARS",
                "closed_through":views[native]["last_closed_at"],
                "market_context":views[native]["evidence"]["market_context"],
                "ao":views[native]["evidence"]["ao"],
                "ac":views[native]["evidence"]["ac"],
                "recent_levels":views[native]["market_position"]["recent_levels"],
                "wave_degree":"UNKNOWN",
            }
        else:
            core[tf]={"status":"INSUFFICIENT_CLOSED_HISTORY",
                      "market_context":"UNKNOWN","wave_degree":"UNKNOWN"}
    relationships=reconcile_market_hierarchy(core)
    # This is a hypothesis book, never a verified forecast or order.
    report={
        "mode":"ALL_15_NATIVE_BINANCE_FUTURES_WAVE_TIMEFRAME_STUDY",
        "venue":venue,
        "symbol":symbol,
        "coverage_native_intervals":len(intervals),
        "observed_intervals":sum(x.get("study_status")=="OBSERVED_CLOSED_BARS"
                                  for x in matrix.values()),
        "all_intervals_have_usable_wave_indicators":all(
            x.get("study_status")=="OBSERVED_CLOSED_BARS" for x in matrix.values()),
        "full_historical_backfill_completed_for_every_tf":False,
        "matrix":matrix,
        "adjacent_cross_scale_competing_hypotheses":cross,
        "h1_h4_d1_relationships":relationships["parent_child_relations"],
        "wave_degree_expert_verified":False,
        "alligator_displacement_no_future":True,
        "trading_signal_is_not_exchange_fill":True,
        "book_edition":edition,
        "gguf_independent_source_audit":None,
        "not_marked_as_deployable_ai_trading_engine":True,
        "trade_action":None,"execution_allowed":False,
    }
    # May be read without a model, and scores only a RESEARCH explanation.
    # The GGUF sees all 15 condensed native passports on one coherent as-of.
    if model is not None:
        if "1h" not in views:
            raise HistoryUnavailable("H1_REQUIRED_FOR_SHARED_BOOK_REASONING")
        from .algorithm_guard import audit_reasoning
        anchor=views["1h"]
        # Prevent passing FUTURE higher-TF bars to the H1 decision context.
        cutoff=datetime.fromisoformat(anchor["as_of"])
        causal_context={}
        for native,tf_view in views.items():
            if datetime.fromisoformat(tf_view["last_closed_at"])<=cutoff:
                causal_context[native]=matrix[native]
        anchor=dict(anchor)
        anchor["all_binance_timeframes_compact"]={
            k:{
                "status":v["study_status"],
                "close":v["closed_through"],
                "trend":v["market_context"],
                "ao":v["ao"],"ac":v["ac"],"zone":v["zone"],
                "wave_degree_verified":False,
                "recent_swings":v["confirmed_swings"][-2:],
            } for k,v in causal_context.items()
        }
        anchor["higher_timeframe_context"]={}
        for tf in ("H4","D1"):
            frame=core[tf]
            if (frame.get("status")=="OBSERVED_CLOSED_BARS"
                    and datetime.fromisoformat(frame["closed_through"])<=cutoff):
                anchor["higher_timeframe_context"][tf]=frame
            else:
                anchor["higher_timeframe_context"][tf]={
                    "status":"INSUFFICIENT_CLOSED_HISTORY_AS_OF_H1",
                    "wave_degree":"UNKNOWN",
                }
        submitted=model.propose_chart_reasoning(anchor,edition=edition)
        audit=audit_reasoning(anchor,submitted)
        report["gguf_independent_source_audit"]={
            "verdict":audit["verdict"],
            "hard_failures":audit["hard_failures"],
            "setup":audit["setup"],
            "source_rule_proofs":len(audit["source_proofs"]),
            "as_of":anchor["as_of"],
            "reviewed_native_intervals_visible_to_llm":sorted(causal_context),
            "not_trade_authorized":True,
        }
    return report


def _ms_from_date(s:str)->int:
    if s=="earliest":
        return 0
    try:
        dt=datetime.strptime(s,"%Y-%m-%d").replace(tzinfo=timezone.utc)
        if dt.year<1970:
            raise ValueError("before Unix epoch")
        return int(dt.timestamp()*1000)
    except ValueError as exc:
        raise ValueError("Use earliest or UTC date YYYY-MM-DD") from exc


def main()->None:
    p=argparse.ArgumentParser(description="Full-range, resumable Binance native-interval wave study")
    p.add_argument("action",choices=("sync","study","sync-study","status"))
    p.add_argument("--db",type=Path,required=True,
                   help="Private local SQLite with durable native exchange candles")
    p.add_argument("--venue",choices=tuple(VENUES),required=True,
                   help="PUBLIC_USDM history or DEMO_USDM; no implicit source switching")
    p.add_argument("--symbols",nargs="+",default=["ETHUSDT"],
                   help="Explicit contract list, verified against Binance exchangeInfo")
    p.add_argument("--intervals",nargs="+",default=["all"],
                   help="Native Binance: all or names like 1m 15m 1h 4h 1d 1w 1M")
    p.add_argument("--since",default="earliest",
                   help="earliest/UTC YYYY-MM-DD (only initial database backfill)")
    p.add_argument("--max-pages-per-interval",type=int,default=20)
    p.add_argument("--page-size",type=int,default=1000)
    p.add_argument("--window",type=int,default=DEFAULT_WINDOW)
    p.add_argument("--edition",choices=("TC1","NTD","TC2","COURSE"),default="TC2")
    p.add_argument("--model",type=Path,help="Local GGUF optional, read-only, no orders")
    p.add_argument("--expected-sha256",help="Actual GGUF SHA256 when model supplied")
    p.add_argument("--json",type=Path,help="Optional private report, create-only")
    args=p.parse_args()
    if args.model is None and args.expected_sha256 is not None or (
            args.model is not None and args.expected_sha256 is None):
        p.error("Provide BOTH --model and --expected-sha256")
    if (len(args.symbols)>20 or len(set(args.symbols))!=len(args.symbols)):
        p.error("No more than 20 unique Binance symbols per invocation")
    intervals=tuple(INTERVAL_MS) if args.intervals==["all"] else tuple(args.intervals)
    if not intervals or len(set(intervals))!=len(intervals) or any(
        x not in INTERVAL_MS for x in intervals
    ):
        p.error("Invalid or duplicated Binance intervals")
    store=DurableWaveHistory(args.db)
    model=None
    if args.model:
        from .real_gguf_acceptance import actual_gguf_sha256
        from .local_model import LocalWilliamsModel
        actual_gguf_sha256(args.model,args.expected_sha256)
        model=LocalWilliamsModel(args.model,n_ctx=8192,gpu_layers=0,
                                use_decision_schema=True)
    since=_ms_from_date(args.since)
    reports=[]
    for symbol in args.symbols:
        statuses=[]
        for tf in intervals:
            if args.action in ("sync","sync-study"):
                info=store.sync(venue=args.venue,symbol=symbol,interval=tf,
                    since_ms=since,max_pages=args.max_pages_per_interval,
                    page_size=args.page_size)
            else:
                info=store.status(args.venue,symbol,tf)
            statuses.append(info)
            print("WILLIAMS_AI_BINANCE_WAVE_CACHE",
                  json.dumps(info,sort_keys=True,ensure_ascii=False),flush=True)
        if args.action in ("study","sync-study"):
            report=analyze_all_cached(store,venue=args.venue,symbol=symbol,
                                      intervals=intervals,window=args.window,
                                      model=model,edition=args.edition)
            # Display source completeness honestly per TF. Old candles in
            # local cache are not a guarantee the remote market is current.
            state={s["interval"]:s.get("sync_state","NOT_CHECKED_THIS_RUN")
                   for s in statuses}
            report["per_tf_sync_state"]=state
            report["full_historical_backfill_completed_for_every_tf"]=all(
                state[tf]=="AT_LATEST_CLOSED_BAR" for tf in intervals)
            reports.append(report)
            print("WILLIAMS_AI_ALL_TF_WAVE_STUDY",
                  json.dumps(report,sort_keys=True,ensure_ascii=False),flush=True)
    if args.json is not None:
        import os
        out={"mode":"BINANCE_PUBLIC_READONLY_DURABLE_HISTORY",
             "venue":args.venue,"requested_intervals":intervals,
             "per_symbol_wave_study":reports,
             "execution_allowed":False,"trade_action":None}
        data=json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)
        fd=os.open(args.json,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"w",encoding="utf8") as f:
            f.write(data+"\n")


if __name__=="__main__":
    main()
