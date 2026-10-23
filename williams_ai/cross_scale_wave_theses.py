"""Falsifiable multi-timeframe wave *alternatives*, never asserted counts.

H1 can rise while H4 declines as a subordinate correction OR an emerging
reversal. A confirmed Elliott 4/5 wave degree cannot be inferred merely from
the sign of AO or a 100-140-bar resemblance. Every observation is timestamped
to completed candles, and structural levels are WATCHPOINTS, not stop orders.
This independent analysis is intended for evidence prompts, not trading.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

_PAIRS=(("D1","H4"),("H4","H1"))
_VALID={"UP_CONTEXT","DOWN_CONTEXT","MIXED_OR_TRANSITION"}


def _utc(s: Any):
    if not isinstance(s,str):
        return None
    try:
        value=datetime.fromisoformat(s.replace("Z","+00:00"))
    except ValueError:
        return None
    return value.astimezone(timezone.utc) if value.tzinfo is not None else None


def _level(frame: dict[str,Any],name:str):
    levels=frame.get("recent_levels",{})
    if not isinstance(levels,dict):
        return None
    n=levels.get(name)
    if isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or n<=0:
        return None
    return float(n)


def cross_scale_wave_theses(
    frames:dict[str,dict[str,Any]], *,
    as_of:str,
)->dict[str,Any]:
    """Evidence passports must be from fully CLOSED and temporally valid TFs."""
    decision=_utc(as_of)
    if decision is None or not isinstance(frames,dict):
        raise ValueError("Need timezone-aware cutoff and timeframe observations")
    result=[]
    for parent,child in _PAIRS:
        p=frames.get(parent,{})
        c=frames.get(child,{})
        if not isinstance(p,dict) or not isinstance(c,dict):
            raise ValueError("Invalid timeframe passport")
        p_at=_utc(p.get("closed_through"))
        c_at=_utc(c.get("closed_through"))
        ready=(p.get("status")=="OBSERVED_CLOSED_BARS"
               and c.get("status")=="OBSERVED_CLOSED_BARS"
               and p_at is not None and c_at is not None
               and p_at<=decision and c_at<=decision)
        if (p_at is not None and p_at>decision) or (c_at is not None and c_at>decision):
            raise ValueError("FUTURE_TIMEFRAME_CANDLE_FORBIDDEN")
        if not ready:
            result.append({
                "parent":parent,"child":child,"status":"INSUFFICIENT_CLOSED_TF_EVIDENCE",
                "alternatives":[],"verified_wave_degree":False,
                "trade_action":None,"execution_allowed":False,
            })
            continue
        parent_dir=p.get("market_context")
        child_dir=c.get("market_context")
        if parent_dir not in _VALID or child_dir not in _VALID:
            raise ValueError("Unknown market context")
        high=_level(p,"latest_confirmed_high")
        low=_level(p,"latest_confirmed_low")
        parent_close=_level(p,"close")
        child_close=_level(c,"close")
        # Levels can be used to falsify claims only when the level existed
        # before that TF's observation. They do NOT establish an Elliott count.
        watch={"parent_confirmed_high":high,"parent_confirmed_low":low}
        parent_ao=p.get("ao")
        child_ao=c.get("ao")
        def sign(v):
            return ("POSITIVE" if isinstance(v,(int,float)) and not isinstance(v,bool)
                    and math.isfinite(v) and v>0 else "NEGATIVE"
                    if isinstance(v,(int,float)) and not isinstance(v,bool)
                    and math.isfinite(v) and v<0 else "UNKNOWN")
        ao_signs={"parent":sign(parent_ao),"child":sign(child_ao)}
        opposite=(parent_dir=="UP_CONTEXT" and child_dir=="DOWN_CONTEXT") or (
                   parent_dir=="DOWN_CONTEXT" and child_dir=="UP_CONTEXT")
        aligned=(parent_dir==child_dir and parent_dir!="MIXED_OR_TRANSITION")
        alternatives=[]
        if opposite:
            supports=[
                "CLOSED_PARENT_ALLIGATOR_CONTEXT_"+parent_dir,
                "CLOSED_CHILD_ALLIGATOR_COUNTER_CONTEXT_"+child_dir,
            ]
            if ao_signs["parent"]!=ao_signs["child"] and "UNKNOWN" not in ao_signs.values():
                supports.append("PARENT_CHILD_AO_SIGNS_DISAGREE")
            correction_falsifier=(
                "Watch CLOSED parent break above its CONFIRMED prior swing high; "
                "if confirmed, the parent-down continuation hypothesis weakens."
                if parent_dir=="DOWN_CONTEXT" else
                "Watch CLOSED parent break below its CONFIRMED prior swing low; "
                "if confirmed, the parent-up continuation hypothesis weakens."
            )
            alternatives=[
                {
                    "hypothesis":"CHILD_CORRECTION_WITHIN_PARENT_MOVEMENT",
                    "support":supports,
                    "potential_falsifier":correction_falsifier,
                    "not_proved":"wave_4_ao_zero_or_wave_degree",
                },
                {
                    "hypothesis":"EARLY_REVERSAL_OF_PARENT_MOVEMENT",
                    "support":["CHILD_COUNTERDIRECTION_OBSERVED_ONLY"],
                    "potential_falsifier":"Failure to establish and maintain a new CONFIRMED parent-timeframe swing direction",
                    "not_proved":"parent_reversal_or_3rd_wave",
                },
                {
                    "hypothesis":"COMPLEX_CORRECTION_OR_TRANSITION",
                    "support":["CROSS_SCALE_CONFLICT_OBSERVED"],
                    "potential_falsifier":"Subsequent confirmed parent and child swing alignment",
                    "not_proved":"ABC_triangle_combination_subdivision",
                },
            ]
        elif aligned:
            alternatives=[
                {
                    "hypothesis":"PARENT_CHILD_TREND_CONTINUATION",
                    "support":["TWO_FULLY_CLOSED_ALLIGATOR_CONTEXTS_ALIGN"],
                    "potential_falsifier":"Future confirmed parent structural reversal / momentum breakdown",
                    "not_proved":"wave_3_or_extended_5",
                },
                {
                    "hypothesis":"POSSIBLE_LATE_PARENT_TREND_EXHAUSTION",
                    "support":["ALIGNMENT_CANNOT_EXCLUDE_EXHAUSTION"],
                    "potential_falsifier":"Sustained confirmed continuation without divergence/invalidating structure",
                    "not_proved":"5_of_5_or_magic_bullets",
                },
            ]
        else:
            alternatives=[{
                "hypothesis":"MIXED_PHASE_RANGE_OR_CORRECTION",
                "support":["AT_LEAST_ONE_NON_TRENDING_ALLIGATOR_CONTEXT"],
                "potential_falsifier":"Confirmed directional break and subsequent parent-child structural alignment",
                "not_proved":"complex_elliott_degree",
            }]
        for item in alternatives:
            item.update({
                "probability":None,"trade_action":None,
                "wave_degree_verified":False,
                "structural_levels_are_NOT_stop_orders":watch,
            })
        result.append({
            "parent":parent,"child":child,
            "status":"CONDITIONAL_CAUSAL_WAVE_ALTERNATIVES",
            "parent_closed_through":p_at.isoformat(),
            "child_closed_through":c_at.isoformat(),
            "market_contexts":{"parent":parent_dir,"child":child_dir},
            "ao_signs":ao_signs,
            "parent_reference_close":parent_close,
            "child_reference_close":child_close,
            "confirmed_structural_watch_levels":watch,
            "alternatives":alternatives,
            "verified_wave_degree":False,
            "trade_action":None,"execution_allowed":False,
        })
    return {
        "mode":"READ_ONLY_CROSS_SCALE_FALSIFIABLE_WAVE_THESIS",
        "as_of":decision.isoformat(),
        "parents_and_children":result,
        "fifth_wave_certified":False,
        "fourth_wave_ao_near_zero_certified":False,
        "note":"AO zero proximity and 100–140 bars are context, NOT wave-four proof; no future bars used.",
        "trade_action":None,"execution_allowed":False,
    }
