"""Prepare reproducible historical prefixes for reviewers, with no model labels.

Input captures are read-only public H1 observations. Each task contains the
actual earlier OHLC prefix, not just an image or a hash without its data.
Calendar selection is frozen before review, never selected by future profit.
This prepares a corpus; it cannot supply independent expert judgements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .chart_reader import validate_closed_bars
from .private_lora import _private_file
from .prospective_qualification import _utc


def _digest(value:Any)->str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def build_blind_review_packet(captures:list[dict[str,Any]],*,utc_days:int=8,
                            minimum_cases:int=30)->dict[str,Any]:
    """Use the last N shared completed UTC dates, excluding the capture date.

    Full warmup history is included. No future bar, inference output,
    retrospective wave label or engine-derived structure enters a task.
    """
    if (not isinstance(captures,list) or not 1<=len(captures)<=15
            or isinstance(utc_days,bool) or not isinstance(utc_days,int) or not 1<=utc_days<=30
            or isinstance(minimum_cases,bool) or not isinstance(minimum_cases,int)
            or not 1<=minimum_cases<=450):
        raise ValueError("INVALID_BLIND_REVIEW_BUDGET")
    checked=[];seen=set();venues=set();shared=None
    for capture in captures:
        if not isinstance(capture,dict):raise ValueError("PUBLIC_CAPTURE_REQUIRED")
        market=capture.get("market");source=capture.get("source")
        if not isinstance(market,dict) or not isinstance(source,dict):
            raise ValueError("PUBLIC_CAPTURE_REQUIRED")
        symbol=market.get("symbol");venue=source.get("venue");bars=market.get("bars")
        if (not isinstance(symbol,str) or not symbol.isascii() or not symbol.isalnum()
                or not symbol.endswith("USDT") or not symbol.isupper() or symbol in seen
                or venue not in {"SPOT","FUTURES"}
                or source.get("execution_allowed") is not False
                or source.get("trade_action","MISSING") is not None
                or source.get("clock_checked") is not True
                or not isinstance(bars,list) or len(bars)<220
                or source.get("market_sha256")!=_digest(market)):
            raise ValueError("TAMPERED_OR_UNTRUSTED_CAPTURE")
        tick=market.get("tick_size")
        if isinstance(tick,bool) or not isinstance(tick,(int,float)) or not math.isfinite(tick) or tick<=0:
            raise ValueError("INVALID_CAPTURE_TICK_SIZE")
        cutoff=_utc(source.get("as_of"))
        if _utc(source.get("observed_server_time"))<=cutoff:
            raise ValueError("CAPTURE_CLOCK_MUST_FOLLOW_CLOSED_BAR")
        candles,gaps=validate_closed_bars(bars,"H1",source["as_of"])
        if gaps or len(candles)!=len(bars) or _utc(bars[-1]["close_time"])!=cutoff:
            raise ValueError("ONLY_CONTIGUOUS_CLOSED_PREFIXES_ALLOWED")
        day_ends={}
        for i,bar in enumerate(bars):
            day=_utc(bar["close_time"]).date().isoformat()
            if i>=219 and day<cutoff.date().isoformat():day_ends[day]=i
        shared=set(day_ends) if shared is None else shared & set(day_ends)
        checked.append((symbol,venue,market,source,day_ends))
        seen.add(symbol);venues.add(venue)
    if len(venues)!=1:raise ValueError("SPOT_AND_FUTURES_MUST_NOT_SHARE_A_REVIEW_COHORT")
    if shared is None or len(shared)<utc_days:
        raise ValueError("NOT_ENOUGH_COMMON_UTC_REVIEW_DATES")
    selected_days=sorted(shared)[-utc_days:]
    if len(checked)*len(selected_days)<minimum_cases:
        raise ValueError("NOT_ENOUGH_UNIQUE_REVIEW_CASES")
    editions=(("TC1",53),("NTD",91),("TC2",148),("COURSE",56))
    tasks=[]
    for symbol,venue,market,source,ends in sorted(checked,key=lambda x:x[0]):
        for day in selected_days:
            # Copy ONLY known fields; don't propagate model outputs or arbitrary
            # extra capture fields into an independently viewed packet.
            prefix=[{k:b[k] for k in ("close_time","open","high","low","close","volume","closed")}
                    for b in market["bars"][:ends[day]+1]]
            prefix_hash=_digest(prefix)
            edition,page=editions[len(tasks)%len(editions)]
            tasks.append({
                "case_id":f"{symbol}-H1-{prefix_hash[:20]}",
                "symbol":symbol,"timeframe":"H1","venue":venue,
                "cutoff":prefix[-1]["close_time"],"visible_prefix_sha256":prefix_hash,
                "edition":edition,"source_page":page,
                "source_capture_market_sha256":source["market_sha256"],
                "market_prefix":{"symbol":symbol,"tick_size":market["tick_size"],"bars":prefix},
                "review_status":"UNLABELLED_REQUIRES_TWO_INDEPENDENT_EXPERTS",
            })
    packet={
        "schema":"WILLIAMS_AI_BLIND_HISTORICAL_PREFIX_PACKET_V1",
        "venue":next(iter(venues)),"utc_days":selected_days,"cases":len(tasks),
        "tasks":tasks,"calendar_selection_not_outcome_selection":True,
        "historical_not_a_future_prospective_cohort":True,
        "labels_completed":0,"reviewer_independence_not_programmatically_proven":True,
        "model_predictions_or_future_bars_in_tasks":False,
        "trade_action":None,"execution_allowed":False,
    }
    packet["packet_sha256"]=_digest(packet)
    return packet


def main()->None:
    parser=argparse.ArgumentParser(description="Private historical OHLC packet for two independent reviewers")
    parser.add_argument("--capture",type=Path,action="append",required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--utc-days",type=int,default=8)
    args=parser.parse_args()
    output=_private_file(args.out,must_exist=False)
    if output.exists():raise ValueError("REFUSE_OVERWRITE_FROZEN_REVIEW_PACKET")
    packet=build_blind_review_packet([json.loads(p.read_text()) for p in args.capture],utc_days=args.utc_days)
    # Exclusive creation preserves an existing packet on a race with another writer.
    with output.open("x",encoding="utf-8") as stream:
        output.chmod(0o600)
        json.dump(packet,stream,ensure_ascii=False)
    print(json.dumps({"cases":packet["cases"],"utc_days":len(packet["utc_days"]),
                      "packet_sha256":packet["packet_sha256"],"labels_completed":0}))


if __name__=="__main__":main()
