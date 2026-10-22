"""Auditable, causal Elliott/Williams wave hypotheses — NOT automatic wave truths.

Only confirmed five-bar price pivots (pivot index plus two right CLOSED bars)
are eligible. Searches alternating 6-pivot impulse-shaped structures,
maintains alternatives and explicitly marks all findings as hypotheses.
"""
from __future__ import annotations
from typing import Any


def confirmed_price_swings(bars: list[Any]) -> list[dict[str, Any]]:
    """Return locally confirmed extrema; do not use future candles to enter."""
    swings=[]
    for i in range(2,len(bars)-2):
        candidate=bars[i]
        neighbors=[bars[j] for j in (i-2,i-1,i+1,i+2)]
        is_high=all(candidate.high>x.high for x in neighbors)
        is_low=all(candidate.low<x.low for x in neighbors)
        if is_high:
            swings.append({"index":i,"confirmed_at_index":i+2,
                           "type":"H","price":candidate.high,
                           "time":candidate.at.isoformat()})
        if is_low:
            swings.append({"index":i,"confirmed_at_index":i+2,
                           "type":"L","price":candidate.low,
                           "time":candidate.at.isoformat()})
    # Same bar may theoretically satisfy both extrema (an outside bar).
    # It is ambiguous for ordering; do not use it as a unique wave point.
    counts={}
    for p in swings: counts[p["index"]]=counts.get(p["index"],0)+1
    points=[x for x in swings if counts[x["index"]]==1]
    result=[]
    for point in points:
        if result and result[-1]["type"]==point["type"]:
            extreme=(point["price"]>result[-1]["price"]
                     if point["type"]=="H" else
                     point["price"]<result[-1]["price"])
            if extreme: result[-1]=point
        else:
            result.append(point)
    return result


def alternating_impulse_skeleton(pivots: list[dict[str, Any]],
                                 ao: list[float | None] | None = None,
                                 *, last_n: int = 20) -> list[dict[str, Any]]:
    """Validate basic classical impulse shape; no diagonal/ABC certification."""
    results=[]
    bounded=pivots[-min(max(last_n,6),40):]
    for offset in range(max(0,len(bounded)-22),len(bounded)-5):
        points=bounded[offset:offset+6]
        types="".join(p["type"] for p in points)
        if types not in {"LHLHLH","HLHLHL"}:
            continue
        direction="BULL" if types=="LHLHLH" else "BEAR"
        vals=[p["price"] for p in points]
        sign=1 if direction=="BULL" else -1
        l1=sign*(vals[1]-vals[0])
        l3=sign*(vals[3]-vals[2])
        l5=sign*(vals[5]-vals[4])
        if not (l1>0 and l3>0 and l5>0):
            continue
        w2_valid=sign*(vals[2]-vals[0])>0
        new_high3=sign*(vals[3]-vals[1])>0
        new_high5=sign*(vals[5]-vals[3])>0
        third_not_shortest=l3>=min(l1,l5)
        w4_nonoverlap=sign*(vals[4]-vals[1])>0
        if not(w2_valid and new_high3 and third_not_shortest):
            continue
        diagnosis=[]
        if not w4_nonoverlap:
            diagnosis.append("W4 overlaps W1: classical impulse rejected; diagonal hypothesis needs separate proof")
        if not new_high5:
            diagnosis.append("W5 failed to exceed W3: possible truncation or incorrect labelling")
        bar_span=points[-1]["index"]-points[0]["index"]
        w4_ao_zero=None
        if ao:
            peak_ao=[abs(x) for x in ao[points[2]["index"]:points[3]["index"]+1]
                     if x is not None]
            during4=[abs(x) for x in ao[points[3]["index"]:points[4]["index"]+1]
                     if x is not None]
            if peak_ao and during4 and max(peak_ao)>0:
                w4_ao_zero=min(during4) <= 0.15*max(peak_ao)
        results.append({
            "pattern":"IMPULSE_SKELETON_POSSIBLE" if w4_nonoverlap else "DIAGONAL_OR_ALTERNATE_POSSIBLE",
            "direction":direction,
            "pivot_times":[p["time"] for p in points],
            "pivot_prices":vals,
            "last_pivot_confirmation_index":points[-1]["confirmed_at_index"],
            "scale_bar_span":bar_span,
            "rhythm_100_140_context":100<=bar_span<=140,
            "w2_within_origin":w2_valid,
            "w3_not_shortest":third_not_shortest,
            "w4_nonoverlap":w4_nonoverlap,
            "w5_extends_w3":new_high5,
            "w4_ao_near_zero_observed":w4_ao_zero,
            "issues":diagnosis,
            "status":"HYPOTHESIS_NOT_BOOK_VERIFIED",
            "source":"TC1 PDF 53-54, Course PDF 42-43; still requires subdivisions and degree identification"
        })
    return results[-5:]


def wave_context(bars: list[Any], ao: list[float | None]) -> dict[str, Any]:
    swings=confirmed_price_swings(bars)
    proposals=alternating_impulse_skeleton(swings,ao)
    return {
        "confirmed_swing_count":len(swings),
        "recent_confirmed_swings":swings[-16:],
        "impulse_alternatives":proposals,
        "wave_degree":"UNKNOWN",
        "unresolved_patterns":[
            "Zigzag/flat/triangle/combination subdivisions not verified",
            "Diagonals and extended fifth-wave internals not verified",
            "Multi-timeframe parent wave cannot be inferred from price-time containment alone",
        ],
        "can_execute":False,
    }
