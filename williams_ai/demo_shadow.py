"""One-pair REAL DEMO H1 -> READ-ONLY Williams chart observer.

No account API key required, no signed request, no broker execution path.
Models may discuss an audited TC2 candidate, but their answer is NEVER routed
into a market order. Only a subsequent separate human CLI can do a one-shot,
journaled, operator-approved trade; this shadow output is not that approval.

All fetched history is exactly Binance Futures DEMO, not main Futures or Spot.
No lookahead: reject partial or irregular H1 candles and stale market feeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from .demo_pilot_readiness import (
    PAIR, DemoSafetyError, check_public_demo, _read_only_demo_json,
)
from .chart_reader import analyze_chart, validate_closed_bars
from .causal_timeframes import aggregate_closed_h1
from .reasoning_trial import evaluate_historical_thought

HOUR_MS=3_600_000


def capture_demo_h1(
    *, fetch:Callable[[str,dict[str,str]|None],Any]=_read_only_demo_json,
    now_ms:int|None=None,
)->dict[str,Any]:
    status=check_public_demo(fetch=fetch,now_ms=now_ms)
    server=status.checked_server_ms
    raw=fetch("/fapi/v1/klines",{"symbol":PAIR,"interval":"1h","limit":"500"})
    if not isinstance(raw,list) or not 250<=len(raw)<=500:
        raise DemoSafetyError("DEMO_H1_NOT_ENOUGH_PUBLIC_BARS")
    rows=[]
    last_open=None
    for bar in raw:
        if not isinstance(bar,list) or len(bar)<7:
            raise DemoSafetyError("DEMO_H1_MALFORMED_PRICE_ROW")
        op,close_ms=bar[0],bar[6]
        if (not isinstance(op,int) or isinstance(op,bool)
             or not isinstance(close_ms,int) or isinstance(close_ms,bool)
             or op % HOUR_MS != 0 or close_ms-op != HOUR_MS-1):
            raise DemoSafetyError("DEMO_H1_MISALIGNED_TIME")
        if last_open is not None and op-last_open!=HOUR_MS:
            raise DemoSafetyError("DEMO_H1_MISSING_OR_DUPLICATE_CANDLE")
        last_open=op
        if close_ms>=server-5000:
            # Current unfinished H1 is NOT visible to chart reader/LLM.
            continue
        try:
            prices=[float(bar[i]) for i in (1,2,3,4,5)]
        except (ValueError,TypeError,OverflowError) as exc:
            raise DemoSafetyError("DEMO_H1_INVALID_OHLCV") from exc
        rows.append({
            "close_time":datetime.fromtimestamp(close_ms/1000,timezone.utc).isoformat(),
            "open":prices[0],"high":prices[1],"low":prices[2],
            "close":prices[3],"volume":prices[4],"closed":True,
        })
    if len(rows)<220:
        raise DemoSafetyError("DEMO_H1_INSUFFICIENT_CLOSED_HISTORY")
    observed_ms=round(datetime.fromisoformat(rows[-1]["close_time"]).timestamp()*1000)
    if not 5000<=server-observed_ms<2*HOUR_MS:
        raise DemoSafetyError("DEMO_LAST_COMPLETE_H1_IS_STALE")
    bars,gaps=validate_closed_bars(rows,"H1",rows[-1]["close_time"])
    if gaps or len(bars)!=len(rows):
        raise DemoSafetyError("DEMO_H1_INVALID_FINAL_HISTORY")
    return {"venue":"BINANCE_FUTURES_DEMO",
            "symbol":PAIR,"bars":rows,
            "tick_size":float(status.tick_size),
            "last_closed_at":rows[-1]["close_time"],
            "execution_allowed":False,"trade_action":None}


def analyze_demo_shadow(
    *,model:Any=None,
    fetch:Callable[[str,dict[str,str]|None],Any]=_read_only_demo_json,
    now_ms:int|None=None,
)->dict[str,Any]:
    data=capture_demo_h1(fetch=fetch,now_ms=now_ms)
    rows=data["bars"]
    as_of=data["last_closed_at"]
    # This data are DEMO Futures, not synthetic Spot, not live money.
    chart=analyze_chart(PAIR,"H1",rows,as_of=as_of,tick_size=data["tick_size"])
    h4=aggregate_closed_h1(rows,"H4")
    d1=aggregate_closed_h1(rows,"D1")
    report={
        "mode":"BINANCE_FUTURES_DEMO_WILLIAMS_READ_ONLY_SHADOW",
        "symbol":PAIR,"venue":"BINANCE_FUTURES_DEMO",
        "as_of":as_of,"closed_h1_bars":len(rows),
        "closed_h4_bars":len(h4),"closed_d1_bars":len(d1),
        "h1_market_context":chart["evidence"]["market_context"],
        "h1_candidate_types":[
            {"type":c.get("type"),"side":c.get("side"),
             "state":c.get("state")} for c in chart["candidates"][-8:]
             if isinstance(c,dict)],
        "independent_gguf_book_claim_audit":None,
        "independent_gguf_verified_wave_degree":False,
        "no_model_orders":True,
        "needs_real_demo_signed_stop_and_fill_acceptance":True,
        "real_trading_authorized":False,
        "manual_pilot_not_authorized_by_shadow":True,
        "execution_allowed":False,"trade_action":None,
    }
    if model is not None:
        trial=evaluate_historical_thought(
            PAIR,rows,tick_size=data["tick_size"],
            bar_index=len(rows)-1,edition="TC2",
            model=model,include_topdown=True,
        )
        audit=trial["audit"]
        report["independent_gguf_book_claim_audit"]={
            "verdict":audit["verdict"],
            "setup":audit["setup"],"hard_failures":audit["hard_failures"],
            "source_rule_proofs":audit["source_proofs"],
            "as_of":trial["last_visible_at"],
            "does_not_prove_market_price_breakout_fill":True,
        }
        if trial["model_saw_future_bars"] is not False or (
                report["independent_gguf_book_claim_audit"]["as_of"]!=as_of):
            raise DemoSafetyError("DEMO_MODEL_CAUSAL_CUTOFF_MISMATCH")
    return report


def main()->None:
    p=argparse.ArgumentParser(description="REAL Binance DEMO closed H1 research, no orders")
    p.add_argument("--model",type=Path,help="Optional actual local GGUF; NEVER used for order routing")
    p.add_argument("--expected-sha256",help="Required if actual GGUF supplied")
    p.add_argument("--ctx",type=int,default=8192)
    args=p.parse_args()
    if bool(args.model)!=bool(args.expected_sha256):
        p.error("Use both --model and --expected-sha256, or neither")
    model=None
    if args.model is not None:
        from .real_gguf_acceptance import actual_gguf_sha256
        from .local_model import LocalWilliamsModel
        actual_gguf_sha256(args.model,args.expected_sha256)
        model=LocalWilliamsModel(args.model,n_ctx=args.ctx,gpu_layers=0,
                                use_decision_schema=True)
    outcome=analyze_demo_shadow(model=model)
    print(json.dumps(outcome,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
