"""Cross-timeframe *conditional* interpretation, no fixed nested wave labels.

When H1 rises against H4 down, the market does not have to reproduce a
textbook fifth-wave picture. A lower-timeframe rebound can be a correction
inside H4 down OR the early phase of a genuine reversal. Only subsequent
completed structural and momentum evidence can distinguish these hypotheses.
"""
from __future__ import annotations

from typing import Any


def _dir(context: str) -> str:
    return "UP" if context == "UP_CONTEXT" else (
        "DOWN" if context == "DOWN_CONTEXT" else "UNRESOLVED"
    )


def reconcile_market_hierarchy(
    frames: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(frames,dict):
        raise ValueError("Expected independent timeframe passports")
    result=[]
    for parent,child in (("D1","H4"),("H4","H1")):
        older=frames.get(parent,{})
        newer=frames.get(child,{})
        if not isinstance(older,dict) or not isinstance(newer,dict):
            raise ValueError("Malformed timeframe evidence")
        a=_dir(older.get("market_context","UNKNOWN"))
        b=_dir(newer.get("market_context","UNKNOWN"))
        if (older.get("status")!="OBSERVED_CLOSED_BARS" or
                newer.get("status")!="OBSERVED_CLOSED_BARS" or
                a=="UNRESOLVED" or b=="UNRESOLVED"):
            phase="RELATION_UNRESOLVED"
            options=["INSUFFICIENT_CONFIRMED_MULTI_TIMEFRAME_CONTEXT"]
        elif a==b:
            phase="DIRECTIONALLY_ALIGNED_NOT_WAVE_DEGREE_PROVEN"
            options=[
                "POSSIBLE_TREND_CONTINUATION",
                "POSSIBLE_LATE_TREND_EXHAUSTION",
            ]
        else:
            phase="COUNTER_DIRECTION_UNVERIFIED_CORRECTION_OR_REVERSAL"
            options=[
                "LOWER_TIMEFRAME_CORRECTION_WITHIN_HIGHER_MOVEMENT",
                "LOWER_TIMEFRAME_EARLY_REVERSAL_OF_HIGHER_MOVEMENT",
                "RANGE_OR_COMPLEX_CORRECTION_IN_BOTH_TIMEFRAMES",
            ]
        result.append({
            "parent":parent,"child":child,
            "parent_context":a,"child_context":b,
            "relation":phase,"unconfirmed_alternatives":options,
            "evidence_needed":[
                "Price must confirm new parent-timeframe pivots after enough closed bars.",
                "AO/AC/Alligator momentum on each degree must be checked separately.",
                "A price trigger requires its own book-edition rule and actual fills.",
            ],
            "wave_degree_confirmed":False,
            "may_close_position_or_enter_market":False,
        })
    return {
        "mode":"CONDITIONAL_MULTISCALE_INTERPRETATION",
        "parent_child_relations":result,
        "parent_wave_degree_confirmed":False,
        "trade_action":None,
        "execution_allowed":False,
        "limitations":[
            "Different timeframe trends may disagree without error.",
            "Counterdirection H1 movement is not proof of H4 wave 4 or H4 reversal.",
            "Book illustrations are mechanisms, not templates requiring exact wave shapes.",
        ],
    }
