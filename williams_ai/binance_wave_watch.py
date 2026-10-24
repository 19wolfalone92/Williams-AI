"""Opt-in, foreground periodic public Binance Futures all-TF history refresh.

Continuously FETCHES MARKET DATA while this Python process is running; never
starts itself on Android, never runs inside GitHub, never accesses a trading
account and never sends a market order. Explicit venue and symbol allowlists.
Stop the foreground process with Ctrl+C. Failure pauses research (not risk
control on separately running trading systems).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable

from .binance_wave_history import INTERVAL_MS,VENUES,DurableWaveHistory,HistoryUnavailable
from .binance_wave_study import analyze_all_cached

MAX_SYMBOLS=5
MIN_PERIOD_SECONDS=300


def one_watch_cycle(
    store:DurableWaveHistory, *,
    venue:str,symbols:tuple[str,...],
    intervals:tuple[str,...]=tuple(INTERVAL_MS),
    max_pages:int=50,fetch:Callable[...,Any]|None=None,
    now_ms:int|None=None,
)->dict[str,Any]:
    if venue not in VENUES or not 1<=len(symbols)<=MAX_SYMBOLS or (
        len(set(symbols))!=len(symbols)) or any(
            not isinstance(s,str) or not s.endswith("USDT") for s in symbols
        ) or not intervals or len(set(intervals))!=len(intervals) or any(
            x not in INTERVAL_MS for x in intervals
        ):
        raise ValueError("INVALID_WATCH_VENUE_SYMBOLS_OR_TFS")
    out=[]
    for symbol in symbols:
        stages=[]
        for tf in intervals:
            kw={"venue":venue,"symbol":symbol,"interval":tf,"since_ms":0,
                "max_pages":max_pages}
            if fetch is not None:
                kw["fetch"]=fetch
            if now_ms is not None:
                kw["now_ms"]=now_ms
            try:
                result=store.sync(**kw)
            except (HistoryUnavailable,OSError,ValueError) as exc:
                # Mark actual error, not a fake market-empty NO_SETUP.
                stages.append({"interval":tf,"sync_state":"FAILED_UNVERIFIED_DATA",
                               "error_type":type(exc).__name__,
                               "execution_allowed":False})
                break
            stages.append({"interval":tf,"sync_state":result["sync_state"],
                           "bars":result["stored_closed_bars"],
                           "new_closed_bars":result["new_closed_bars_added"]})
        full=len(stages)==len(intervals) and all(
            p["sync_state"]=="AT_LATEST_CLOSED_BAR" for p in stages)
        status={
            "symbol":symbol,"venue":venue,
            "timeframes_checked":len(stages),
            "native_timeframes_expected":len(intervals),
            "history_usable_on_every_timeframe_this_cycle":full,
            "sources":stages,
            "trade_action":None,"execution_allowed":False,
        }
        # A missing or blocked candle feed prevents claiming a complete
        # all-TF wave study. Existing cache is never converted to live data.
        if full:
            research=analyze_all_cached(store,venue=venue,symbol=symbol,
                                        intervals=intervals,window=1400)
            status["wave_study"]={
                "observed_intervals":research["observed_intervals"],
                "wave_degree_expert_verified":False,
                "all_intervals_have_usable_wave_indicators":
                    research["all_intervals_have_usable_wave_indicators"],
                "requires_independent_book_and_chart_confirmation":True,
            }
        out.append(status)
    return {
        "mode":"FOREGROUND_BINANCE_PUBLIC_CLOSED_BAR_RESEARCH_WATCH",
        "symbols_checked":len(symbols),
        "per_symbol":out,
        "all_caught_up":all(x["history_usable_on_every_timeframe_this_cycle"]
                            for x in out),
        "account_keys_seen":False,
        "model_can_place_orders":False,
        "production_trading_enabled":False,
        "trade_action":None,"execution_allowed":False,
    }


def main()->None:
    p=argparse.ArgumentParser(description="Update all native Binance chart timeframes in foreground, READ-ONLY")
    p.add_argument("--db",required=True,type=Path)
    p.add_argument("--venue",required=True,choices=tuple(VENUES))
    p.add_argument("--symbols",nargs="+",default=["ETHUSDT"])
    p.add_argument("--intervals",nargs="+",default=["all"])
    p.add_argument("--pages-per-timeframe",type=int,default=50)
    p.add_argument("--poll-seconds",type=int,default=3600,
                   help=">=300; explicit foreground hourly update")
    p.add_argument("--max-cycles",type=int,default=1,
                   help="1 = one update. 0 = run until Ctrl+C; no hidden cloud background task")
    a=p.parse_args()
    if a.poll_seconds<MIN_PERIOD_SECONDS or a.max_cycles<0:
        p.error("poll interval >=300 seconds and max-cycles >=0 required")
    intervals=tuple(INTERVAL_MS) if a.intervals==["all"] else tuple(a.intervals)
    store=DurableWaveHistory(a.db)
    done=0
    try:
        while a.max_cycles==0 or done<a.max_cycles:
            summary=one_watch_cycle(
                store,venue=a.venue,symbols=tuple(a.symbols),
                intervals=intervals,max_pages=a.pages_per_timeframe)
            done+=1
            print("WILLIAMS_AI_BINANCE_ALL_TF_WATCH_CYCLE",
                  json.dumps(summary,ensure_ascii=False,sort_keys=True),flush=True)
            if a.max_cycles>0 and done>=a.max_cycles:
                break
            time.sleep(a.poll_seconds)
    except KeyboardInterrupt:
        print("WILLIAMS_AI_MARKET_READ_ONLY_WATCH_STOPPED")


if __name__=="__main__":
    main()
