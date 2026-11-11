"""Discover possible POSITIVE and NEGATIVE market examples for expert review.

These are *silver observational candidates*, never approved trading labels.
Uses only closed H1 history and bar-index-confirmed fractals; no look-ahead.
A Bill Williams Squat candidate is a narrow numerical condition:
bar volume ↑ and (high-low)/volume ↓ versus previous *closed* bar.
It is NOT proof of a valid trading entry, wave count or market fill.
"""
from __future__ import annotations

from .chart_reader import validate_closed_bars
from .reasoning_research_v2 import digest, utc


def mine_review_candidates(bars,*,symbol,timeframe="H1",cutoff,limit=60):
    if (not isinstance(symbol,str) or not 3<=len(symbol)<=30
            or timeframe!="H1" or type(limit) is not int
            or not 1<=limit<=300):
        raise ValueError("EXPLICIT_H1_MARKET_REVIEW_UNIVERSE_REQUIRED")
    candles,gaps=validate_closed_bars(bars,timeframe,cutoff)
    if gaps or not 220<=len(candles)<=5000 or utc(bars[-1]["close_time"])!=utc(cutoff):
        raise ValueError("CONTIGUOUS_CLOSED_ASOF_BARS_REQUIRED")
    prefix_sha=digest(bars)
    candidates=[]
    last=max(2,len(candles)-limit)
    for i in range(last,len(candles)):
        previous,current=candles[i-1],candles[i]
        prev_mfi=((previous.high-previous.low)/previous.volume
                  if previous.volume>0 else None)
        current_mfi=((current.high-current.low)/current.volume
                     if current.volume>0 else None)
        if prev_mfi is None or current_mfi is None:
            continue
        proposed=current.volume>previous.volume and current_mfi<prev_mfi
        if proposed or (i%11==0):
            candidates.append({
                "candidate_id":f"{symbol}:SQUAT:{i}",
                "symbol":symbol,"timeframe":"H1",
                "cutoff":bars[i]["close_time"],
                "visible_prefix_sha256":digest(bars[:i+1]),
                "original_full_history_sha256":prefix_sha,
                "candidate_kind":"SQUAT_COLOR_NUMERIC",
                "candidate_positive":proposed,
                "observed_volume_increase":current.volume>previous.volume,
                "observed_mfi_decrease":current_mfi<prev_mfi,
                "mfi_formula":"(high-low)/volume",
                "volume_source_not_exchange_verified":True,
                "wave_confirmed":None,
                "squat_verified":None,
                "review_required":True,"expert_gold":False,
                "execution_allowed":False,
            })
    # A higher-order candidate means only five alternating confirmed
    # inflections, not canonical Elliott classification.
    pivots=[]
    for i in range(2,len(candles)-2):
        neighbors=[candles[k] for k in (i-2,i-1,i+1,i+2)]
        is_high=all(candles[i].high>x.high for x in neighbors)
        is_low=all(candles[i].low<x.low for x in neighbors)
        if is_high==is_low:continue
        kind="H" if is_high else "L"
        pivots.append({"i":i,"type":kind,
                       "price":candles[i].high if is_high else candles[i].low,
                       "pivot_at":bars[i]["close_time"],
                       "confirmed_at":bars[i+2]["close_time"]})
    for j in range(4,len(pivots)):
        seq=pivots[j-4:j+1]
        if seq[-1]["i"]<last-2:continue
        types="".join(p["type"] for p in seq)
        if types not in {"HLHLH","LHLHL"}:continue
        vals=[p["price"] for p in seq]
        if types=="HLHLH":
            # Condition is intentionally loose: creates a review candidate
            # without pretending to certify wave overlap/degree rules.
            plausible=vals[2]>vals[0] and vals[4]>vals[2]
        else:
            plausible=vals[2]<vals[0] and vals[4]<vals[2]
        if plausible or j%5==0:
            candidates.append({
                "candidate_id":f"{symbol}:WAVES:{seq[0]['i']}:{seq[-1]['i']}",
                "symbol":symbol,"timeframe":"H1",
                "cutoff":seq[-1]["confirmed_at"],
                "visible_prefix_sha256":digest(bars[:seq[-1]["i"]+3]),
                "original_full_history_sha256":prefix_sha,
                "candidate_kind":"FIVE_ALTERNATING_CONFIRMED_PIVOTS",
                "candidate_positive":plausible,
                "pivots":seq,
                "candidate_wave_degree":"UNKNOWN",
                "wave_confirmed":None,
                "squat_verified":None,
                "review_required":True,"expert_gold":False,
                "execution_allowed":False,
            })
    return {"schema":"WILLIAMS_MARKET_REVIEW_CANDIDATES_V3",
            "symbol":symbol,"cutoff":cutoff,
            "candidate_count":len(candidates),
            "positive_candidate_count":sum(x["candidate_positive"] for x in candidates),
            "negative_candidate_count":sum(not x["candidate_positive"] for x in candidates),
            "independent_expert_approval":False,
            "eligible_for_qlora_gold":False,
            "candidates":candidates}
