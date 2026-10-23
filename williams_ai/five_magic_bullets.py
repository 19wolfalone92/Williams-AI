"""TC1's FIVE MAGIC BULLETS, NOT NTD's five market dimensions.

Original Trading Chaos 1 PDF pp.53-54, 60-65, 82-84:
1) degree-relative price/Profitunity 5/34 divergence (not an unconditional AO),
2) W5 Fibonacci terminal target projection from W4 and 0->3,
3) a terminal high/low fractal,
4) Market Facilitation Index + rising *tick volume* = Squat bar near
   three terminal extremes,
5) reversal of Profitunity 5/34/5 momentum (oscillator vs its SMA5).

Signals work on M5/M15/H1/H4/D1 by *the same relative laws*, but valid
wave-degree/pivot resolution and exchange volume semantics still matter.
The presence of all five supports a possible ZERO POINT, not a certain
reversal and NOT an executable WM1/WM2/WM3 or Binance order. All historical
bar rights are confirmed and only data as-of the caller's cutoff is read.
"""
from __future__ import annotations
from math import isfinite
from typing import Any

from .wave_evidence import alternating_impulse_skeleton,confirmed_price_swings


def _sma5_available(ao:list[float|None])->list[float|None]:
    out=[None]*len(ao)
    for i in range(4,len(ao)):
        x=ao[i-4:i+1]
        if all(v is not None and isfinite(float(v)) for v in x):
            out[i]=sum(float(v) for v in x)/5
    return out


def _leg_extreme(ao:list[float|None],a:int,b:int,sign:int)->float|None:
    values=[float(x) for x in ao[a:b+1] if x is not None and isfinite(float(x))]
    if not values:
        return None
    return max(values) if sign>0 else min(values)


def _squat(
    bars:list[Any],data:list[dict[str,Any]]|None,
    *,extreme_index:int,sign:int
)->dict[str,Any]:
    """Tick counts and traded BASE ASSET VOLUME must NOT be conflated.

    A candle's Binance Futures base-asset volume is not 'tick volume'.
    It is deliberately refused as the classical TC1 Squat confirmation.
    Explicit nonnegative per-bar 'tick_volume' is accepted as a measured
    tick-count observation; no implicit alias/fill-in is allowed.
    """
    if data is None or len(data)!=len(bars):
        return {"status":"UNKNOWN_TICK_VOLUME_REQUIRED","observed_at":[]}
    indices=range(max(1,extreme_index-2),min(len(bars),extreme_index+2))
    ticks=[]
    for row in data:
        v=row.get("tick_volume")
        if v is None or isinstance(v,bool) or not isinstance(v,(int,float)) or not isfinite(v) or v<=0:
            return {
                "status":"UNKNOWN_TICK_VOLUME_REQUIRED",
                "observed_at":[],
                "note":"Binance base traded volume / number_of_trades is NOT silently treated as TC1 tick volume.",
            }
        ticks.append(float(v))
    found=[]
    for i in indices:
        if i<1:
            continue
        # MFI / tick count, not RSI-style money flow.
        mfi_now=(bars[i].high-bars[i].low)/ticks[i]
        mfi_prev=(bars[i-1].high-bars[i-1].low)/ticks[i-1]
        if ticks[i]>ticks[i-1] and mfi_now<mfi_prev:
            # Within extreme's last three HIGH/LOW bars (the candidate W5
            # pivot index or its two closest predecessor bars). In many
            # trends a measuring Squat may occur elsewhere too, so NOT
            # a claim that Squat alone completes the market swing.
            found.append({
                "index":i,
                "closed_at":bars[i].at.isoformat(),
                "volume_grew":True,
                "facilitation_fell":True,
            })
    return {
        "status":"OBSERVED_TC1_SQUAT_IN_TERMINAL_WINDOW" if found
                 else "NOT_OBSERVED_WITH_AVAILABLE_TICK_COUNTS",
        "observed_at":found[-3:],
        "volume_type":"EXPLICIT_TICK_COUNT_ONLY",
    }


def _bool_evidence(name:str,status:str,*,
                   note:str="",observed_at:Any=None)->dict[str,Any]:
    return {
        "id":name,
        "status":status,
        "observed_at":observed_at,
        "note":note,
    }


def five_magic_bullets(
    bars:list[Any],ao:list[float|None], *,
    data:list[dict[str,Any]]|None=None,
    timeframe:str="H1",
    last_n:int=16,
)->dict[str,Any]:
    if timeframe not in {"M5","M15","H1","H4","D1"}:
        raise ValueError("Unknown timeframe")
    if not bars or len(bars)!=len(ao):
        raise ValueError("Require aligned nonempty closed bars and oscillator")
    if data is not None and len(data)!=len(bars):
        raise ValueError("Tick count history must align with candles")
    pivots=confirmed_price_swings(bars)
    impulse=alternating_impulse_skeleton(pivots,ao,last_n=last_n)
    # NOTE: original skeleton uses last at most 40 pivots, and five
    # newly-confirmed pivot extremes; no future pivots are read here.
    scan=[]
    signal=_sma5_available(ao)
    for candidate in impulse:
        times=candidate["pivot_times"]
        matched=[]
        for stamp in times:
            pivot=next((p for p in pivots if p["time"]==stamp),None)
            if pivot is None:
                break
            matched.append(pivot)
        if len(matched)!=6:
            continue
        w0,w1,w2,w3,w4,w5=matched
        if w5["confirmed_at_index"]>=len(bars):
            continue
        sign=1 if candidate["direction"]=="BULL" else -1
        # First and fifth impulse legs have distinct AO extrema.
        peak3=_leg_extreme(ao,w2["index"],w3["index"],sign)
        peak5=_leg_extreme(ao,w4["index"],w5["index"],sign)
        price_extension=sign*(w5["price"]-w3["price"])>0
        divergence=(
            peak3 is not None and peak5 is not None and price_extension
            and sign*(peak5-peak3)<0
        )
        zero3=float(w3["price"]-w0["price"])
        projected=[w4["price"]+.62*zero3,
                   w4["price"]+zero3]
        lower,upper=sorted(projected)
        target=lower <= w5["price"] <= upper
        # w5 itself has already been confirmed as a two-right-bar
        # strict fractal in the underlying pivot stream.
        fractal_confirmed=w5["confirmed_at_index"]<len(bars)
        squat=_squat(bars,data,extreme_index=w5["index"],sign=sign)
        # A local trend-reversing Profitunity 5/34/5 shift:
        # oscillator (5/34) crosses its OWN 5-SMA signal, not AC,
        # and not a mere AO color change. Allow after terminal price
        # extreme until last currently closed bar (bounded to 8 bars).
        change_indices=[]
        for k in range(max(1,w5["index"]-2),
                       min(len(bars),w5["index"]+9)):
            if (signal[k] is None or signal[k-1] is None
                    or ao[k] is None or ao[k-1] is None):
                continue
            previous=sign*(ao[k-1]-signal[k-1])
            current=sign*(ao[k]-signal[k])
            if previous>=0 and current<0:
                change_indices.append(k)
        momentum=bool(change_indices)
        evidence={
            "DIVERGENCE_3_VS_5":_bool_evidence(
                "DIVERGENCE_3_VS_5",
                "OBSERVED" if divergence else "NOT_OBSERVED",
                note="5/34 oscillator extrema on completed W2→W3 and W4→W5 legs, not MFI/volume.",
                observed_at=[peak3,peak5] if peak3 is not None and peak5 is not None else None,
            ),
            "FIBONACCI_TARGET_ZONE":_bool_evidence(
                "FIBONACCI_TARGET_ZONE",
                "OBSERVED" if target else "NOT_OBSERVED",
                note="TC1 W5: W4+0.62..1.00*(W3-W0), geometry is still an unverified wave count.",
                observed_at={"low":lower,"high":upper,"w5_price":w5["price"]},
            ),
            "TERMINAL_FRACTAL":_bool_evidence(
                "TERMINAL_FRACTAL",
                "OBSERVED" if fractal_confirmed else "UNKNOWN_UNCONFIRMED",
                note="Price pivot has exactly two completed confirmation bars to the right, not an order trigger.",
                observed_at=w5["confirmed_at"],
            ),
            "SQUAT_MFI_VOLUME":_bool_evidence(
                "SQUAT_MFI_VOLUME",
                "OBSERVED" if squat["status"]=="OBSERVED_TC1_SQUAT_IN_TERMINAL_WINDOW"
                else "UNKNOWN" if squat["status"]=="UNKNOWN_TICK_VOLUME_REQUIRED"
                else "NOT_OBSERVED",
                note=squat["status"],
                observed_at=squat.get("observed_at",[]),
            ),
            "PROFITUNITY_5345_MOMENTUM_CHANGE":_bool_evidence(
                "PROFITUNITY_5345_MOMENTUM_CHANGE",
                "OBSERVED" if momentum else "NOT_OBSERVED",
                note="Oscillator 5/34 versus its SMA(5) signal; not the separate NTD AC.",
                observed_at=[bars[k].at.isoformat() for k in change_indices[-3:]],
            ),
        }
        yes=sum(x["status"]=="OBSERVED" for x in evidence.values())
        unk=sum(x["status"]=="UNKNOWN" or x["status"].startswith("UNKNOWN_")
                for x in evidence.values())
        scan.append({
            "timeframe":timeframe,
            "direction_of_prior_impulse":candidate["direction"],
            "potential_zero_point_after":"UP" if sign>0 else "DOWN",
            "wave_count_status":"HYPOTHESIS_NEVER_CERTIFIED_ELLIIOTT_W5",
            "terminal_extreme_at":w5["time"],
            "confirmed_through":bars[w5["confirmed_at_index"]].at.isoformat(),
            "observed_bullets":yes,
            "unknown_bullets":unk,
            "all_five_observed":yes==5,
            "classification":(
                "FIVE_OBSERVED_ZERO_POINT_HYPOTHESIS_NOT_REVERSAL"
                if yes==5 else
                "INCOMPLETE_EVIDENCE_NEEDS_MARKET_CONFIRMATION"
            ),
            "bullets":evidence,
            "alternative_explanations":[
                "This may be a smaller fifth inside an unfinished larger third.",
                "Squat can be a measuring bar within a continuing trend.",
                "A new structural low/high can invalidate the terminal hypothesis.",
            ],
            "trade_action":None,
        })
    return {
        "module":"TC1_FIVE_MAGIC_BULLETS_DISTINCT_FROM_NTD_FIVE_DIMENSIONS",
        "as_of":bars[-1].at.isoformat(),
        "timeframe":timeframe,
        "source":"Trading Chaos 1 PDF 53-54,60-65,82-84",
        "last_n_wave_candidates":scan[-5:],
        "wave_candidate_count":len(scan),
        "no_matching_picture_required":True,
        "works_on_any_supported_timeframe_with_sufficient_bars":True,
        "unverified_degree_not_an_entry_or_exit":True,
        "tick_volume_missing_does_not_fabricate_squat":True,
        "trade_action":None,
        "execution_allowed":False,
    }
