"""Deterministic closed H1/H4/D1 observation builder for cognitive research.

No language model gets to declare its own 'verified closed candle' flags.
This validates OHLCV ordering, completeness, warmup and correct indicator
lookback. It does NOT attest exchange provenance or actual order fills.
"""
from __future__ import annotations
from datetime import datetime, timezone
from .chart_reader import TF_SECONDS, indicators, validate_closed_bars
from .cognitive_memory_v4 import _sha
from .cognitive_timeframe_coherence_v5 import verify_closed_timeframe_coherence
from .decision_contract import _as_utc

_FRAMES=("H1","H4","D1")


def market_facts_from_closed_history(history, *, as_of):
    if not isinstance(history,dict) or set(history)!=set(_FRAMES):
        raise ValueError("EXACT_CLOSED_H1_H4_D1_HISTORY_REQUIRED")
    if _as_utc(as_of) is None:
        raise ValueError("OFFSET_AWARE_MARKET_CUTOFF_REQUIRED")
    seen={}
    facts=[]
    hashes={}
    closed_frames={}
    for tf in _FRAMES:
        bars=history[tf]
        if not isinstance(bars,list) or not 220<=len(bars)<=5000:
            raise ValueError("220_BAR_CLOSED_WARMUP_REQUIRED")
        closed,gaps=validate_closed_bars(bars,tf,as_of)
        if gaps or len(closed)!=len(bars) or any(
            (b.at-a.at).total_seconds()!=TF_SECONDS[tf]
            for a,b in zip(closed,closed[1:])):
            raise ValueError("CONTIGUOUS_CLOSED_TIMEFRAME_REQUIRED")
        closed_frames[tf]=closed
        last=closed[-1]
        staleness=_as_utc(as_of)-last.at.timestamp()
        if staleness<0 or staleness>=TF_SECONDS[tf]:
            raise ValueError("LAST_CLOSED_TIMEFRAME_STALE_OR_FUTURE")
        values=indicators(closed)
        if any(values[key][-1] is None for key in ("jaw","teeth","lips","ao","ac")):
            raise ValueError("TECHNICAL_INDICATOR_WARMUP_MISSING")
        hashes[tf]=_sha(bars)
        def fmt(v):
            return f"{float(v):.9g}"
        def add(code,observation):
            ident=tf+"_"+code
            facts.append({"fact_id":ident,
                "closed_at":bars[-1]["close_time"],
                "source":"DERIVED:"+tf+":"+hashes[tf][:20],
                "observation":observation,
                "closed_bar_verified":True})
        add("CLOSE", f"{tf} latest closed candle close = {fmt(last.close)}; volume = {fmt(last.volume)}")
        add("AO", f"{tf} AO = {fmt(values['ao'][-1])}; previous AO = {fmt(values['ao'][-2])}")
        add("AC", f"{tf} AC = {fmt(values['ac'][-1])}; previous AC = {fmt(values['ac'][-2])}")
        add("ALLIGATOR", f"{tf} visible jaw = {fmt(values['jaw'][-1])}, teeth = {fmt(values['teeth'][-1])}, lips = {fmt(values['lips'][-1])}")
        previous=closed[-2]
        if previous.volume>0 and last.volume>0:
            mfi_prev=(previous.high-previous.low)/previous.volume
            mfi_last=(last.high-last.low)/last.volume
            add("MFI_CONTEXT",f"{tf} raw MFI ratio (high-low)/volume = {fmt(mfi_last)}; previous = {fmt(mfi_prev)}; squat numeric condition = {last.volume>previous.volume and mfi_last<mfi_prev}; independent expert Squat confirmation = UNKNOWN")
        else:
            add("MFI_CONTEXT",f"{tf} raw MFI cannot be calculated reliably: zero volume; verified Squat status UNKNOWN")
        seen[tf]={"closed_at":bars[-1]["close_time"],
                  "source_prefix_sha256":hashes[tf],"closed_bars":len(closed)}
    coherence=verify_closed_timeframe_coherence(closed_frames)
    return {"schema":"WILLIAMS_CAUSAL_MARKET_FACTS_V4",
            "as_of":as_of,
            "frames":seen,
            "causal_timeframe_integrity":coherence,
            "facts":facts,
            "whole_snapshot_sha256":_sha(hashes),
            "input_exchange_authenticity_independently_verified":False,
            "wave_degree_expert_verified":False,
            "squat_expert_verified":False,
            "no_order_or_fill_verification":True,
            "execution_allowed":False}
