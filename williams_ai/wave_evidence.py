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



def corrective_alternatives(pivots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tentative A-B-C / contracting triangle geometry, never a trade signal.

    Four pivots alone cannot establish Elliott subdivisions (5-3-5 versus
    3-3-5); these labels mean "compatible shape", NOT verified pattern.
    The initiating trend degree is unknown, so bear/bull refer only to A.
    """
    candidates: list[dict[str, Any]] = []
    for start in range(max(0, len(pivots) - 22), max(0, len(pivots) - 3)):
        p = pivots[start:start + 4]
        if len(p) != 4:
            continue
        types = "".join(item["type"] for item in p)
        if types not in {"LHLH", "HLHL"}:
            continue
        prices = [float(item["price"]) for item in p]
        direction = 1 if types == "LHLH" else -1
        leg_a = direction * (prices[1] - prices[0])
        leg_b = direction * (prices[1] - prices[2])
        leg_c = direction * (prices[3] - prices[2])
        if min(leg_a, leg_b, leg_c) <= 0:
            continue
        retracement = leg_b / leg_a
        c_vs_a = leg_c / leg_a
        if 0.25 <= retracement < 0.85 and 0.45 <= c_vs_a <= 2.6:
            kind = "ZIGZAG_SHAPE_CANDIDATE"
        elif 0.85 <= retracement <= 1.15 and 0.45 <= c_vs_a <= 2.6:
            kind = "FLAT_SHAPE_CANDIDATE"
        else:
            continue
        candidates.append({
            "type": kind,
            "a_direction": "UP" if direction == 1 else "DOWN",
            "pivot_times": [point["time"] for point in p],
            "confirmation_index": p[-1]["confirmed_at_index"],
            "a_retracement_ratio": round(retracement, 5),
            "c_to_a_ratio": round(c_vs_a, 5),
            "status": "SHAPE_ONLY_SUBWAVE_COUNT_MISSING",
            "source": "TC1: PDF 53-54 — corrective alternatives; not a canonical ratio filter",
        })
    # Contracting triangles: 5 alternating legs defined by 6 extrema.
    for start in range(max(0, len(pivots) - 22), max(0, len(pivots) - 5)):
        p = pivots[start:start + 6]
        if len(p) != 6:
            continue
        types = "".join(item["type"] for item in p)
        if types not in {"HLHLHL", "LHLHLH"}:
            continue
        highs = [float(item["price"]) for item in p if item["type"] == "H"]
        lows = [float(item["price"]) for item in p if item["type"] == "L"]
        if not (highs[0] > highs[1] > highs[2] and lows[0] < lows[1] < lows[2]):
            continue
        if highs[-1] <= lows[-1]:
            continue
        candidates.append({
            "type": "CONTRACTING_TRIANGLE_SHAPE_CANDIDATE",
            "pivot_times": [point["time"] for point in p],
            "confirmation_index": p[-1]["confirmed_at_index"],
            "status": "SHAPE_ONLY_FIVE_LEGS_UNVERIFIED",
            "source": "TC1: PDF 53-54 mentions triangles; geometry is engineering hypothesis",
        })
    return candidates[-8:]

def wave_context(bars: list[Any], ao: list[float | None]) -> dict[str, Any]:
    swings=confirmed_price_swings(bars)
    proposals=alternating_impulse_skeleton(swings,ao)
    return {
        "confirmed_swing_count":len(swings),
        "recent_confirmed_swings":swings[-16:],
        "impulse_alternatives":proposals,
        "corrective_alternatives":corrective_alternatives(swings),
        "wave_degree":"UNKNOWN",
        "unresolved_patterns":[
            "Zigzag/flat/triangle/combination subdivisions not verified",
            "Diagonals and extended fifth-wave internals not verified",
            "Multi-timeframe parent wave cannot be inferred from price-time containment alone",
        ],
        "can_execute":False,
    }
