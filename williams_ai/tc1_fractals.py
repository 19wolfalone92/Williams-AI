"""TC1 initiating fractal interpretation with equal-price bar exclusions.

Trading Chaos 1 PDF 68-70 says equal-high/low bars do NOT count toward
the required two strictly lower/higher bars. This differs from a simple
strict FIVE-adjacent-candles pivot. No future candles can be observed.
Return ambiguous plateau scenarios as visual review, NOT TC2 WM3 orders.
"""
from __future__ import annotations
from typing import Any


def _count_neighbors(
    bars:list[Any],pivot:int,side:str,step:int,
)->tuple[int|None,bool]:
    """Find 2 strictly lower highs/higher lows, skipping equals.

    A more extreme candidate before two qualifying bars invalidates pivot.
    Return rightmost confirmation index for right-side scan.
    """
    price=(bars[pivot].high if side=="LONG" else bars[pivot].low)
    equals=False
    count=0
    at=pivot+step
    while 0<=at<len(bars):
        value=(bars[at].high if side=="LONG" else bars[at].low)
        signed=(value-price if side=="LONG" else price-value)
        if signed>0:
            return None,equals
        if signed==0:
            equals=True
        else:
            count+=1
            if count==2:
                return at,equals
        at+=step
    return None,equals


def tc1_initiating_fractal_evidence(
    bars:list[Any], *,lookback:int=120,max_items:int=16,
)->dict[str,Any]:
    if (not isinstance(bars,list) or not 10<=lookback<=50000
            or not 1<=max_items<=100):
        raise ValueError("Invalid research window")
    findings=[]
    for pivot in range(max(2,len(bars)-lookback),max(2,len(bars)-2)):
        for side in ("LONG","SHORT"):
            left,left_equal=_count_neighbors(bars,pivot,side,-1)
            if left is None:
                continue
            right,right_equal=_count_neighbors(bars,pivot,side,1)
            if right is None:
                continue
            ambiguous=left_equal or right_equal
            findings.append({
                "type":"TC1_INITIATING_FRACTAL","side":side,
                "pivot_index":pivot,
                "confirmed_at_index":right,
                "pivot_time":bars[pivot].at.isoformat(),
                "confirmed_at":bars[right].at.isoformat(),
                "level":bars[pivot].high if side=="LONG" else bars[pivot].low,
                "same_level_bars_skipped":ambiguous,
                "state":("EQUAL_LEVEL_PLATEAU_REQUIRES_VISUAL_REVIEW" if ambiguous
                         else "CONFIRMED_CLOSED_PRICE_FORMATION_NOT_ENTRY"),
                "source":"TC1 PDF 68-70: equal-level bars do not count",
                "can_prove_price_fill":False,
                "trade_action":None,
            })
    return {
        "profile":"TC1_INITIATING_FRACTALS",
        "recent_formations":findings[-max_items:],
        "can_prove_executable_entry":False,
        "edition_boundary":"TC1 research shape is NOT the TC2/NTD initial-entry policy",
        "trade_action":None,
        "execution_allowed":False,
    }
