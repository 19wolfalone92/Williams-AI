"""Causal, *unconfirmed* wave nesting evidence across native Binance TFs.

Never infer Elliott wave numbers by simply nesting TF labels. Locate younger
confirmed child extrema inside time bounds of confirmed parent swing segment.
Both pivot confirmations must already exist by the SAME shared as-of time.
Reports candidate subdivisions and missing evidence, not W3/W5 certainties.
"""
from __future__ import annotations
from datetime import datetime,timezone
from typing import Any


def _utc(iso:Any)->datetime:
    if not isinstance(iso,str):
        raise ValueError("SWING_TIME_IS_REQUIRED")
    d=datetime.fromisoformat(iso.replace("Z","+00:00"))
    if d.tzinfo is None:
        raise ValueError("TIMEZONE_AWARE_SWING_REQUIRED")
    return d.astimezone(timezone.utc)


def cross_scale_pivot_evidence(
    parent:dict[str,Any]|None,child:dict[str,Any]|None,
    *,parent_tf:str,child_tf:str,common_as_of_ms:int,
)->dict[str,Any]:
    """Observe confirmed child pivots INSIDE a recent parent extremum leg."""
    if (not isinstance(common_as_of_ms,int) or isinstance(common_as_of_ms,bool)
            or common_as_of_ms<=0):
        raise ValueError("RESEARCH_CUTOFF_REQUIRED")
    if not isinstance(parent_tf,str) or not isinstance(child_tf,str) or parent_tf==child_tf:
        raise ValueError("DISTINCT_NATIVE_TIMEFRAMES_REQUIRED")
    base={
        "parent":parent_tf,"child":child_tf,
        "status":"INSUFFICIENT_MATCHED_CONFIRMED_SWINGS",
        "parent_leg_is_a_confirmed_elliott_degree":False,
        "child_waves_are_confirmed_elliott_subwaves":False,
        "possible_interpretations":[
            "A smaller impulse embedded in a larger observed move",
            "A lower timeframe correction or complex structure",
            "An unresolved or reversing price leg",
        ],
        "candidate_parent_leg":None,"confirmed_child_pivots_inside":0,
        "verified_nested_wave_label":None,
        "trade_action":None,"execution_allowed":False,
    }
    if not isinstance(parent,dict) or not isinstance(child,dict):
        return base
    asof=datetime.fromtimestamp(common_as_of_ms/1000,timezone.utc)
    def eligible(view):
        observed=_utc(view["last_closed_at"])
        if observed>asof:
            raise ValueError("NO_LOOKAHEAD_TF_SNAPSHOT_AFTER_SHARED_H1")
        count=view["bars_used"]
        points=[]
        for p in view["wave_structure"]["recent_confirmed_swings"]:
            if (not isinstance(p,dict) or
                    not isinstance(p.get("confirmed_at_index"),int) or
                    not 0<=p["confirmed_at_index"]<count):
                raise ValueError("UNCONFIRMED_EXTREMUM_IN_CROSS_SCALE_RELATION")
            stamp=_utc(p.get("time"))
            if stamp>observed or stamp>asof:
                raise ValueError("FUTURE_EXTREMUM_NOT_VISIBLE")
            points.append((stamp,p))
        return points
    parents=eligible(parent)
    children=eligible(child)
    if len(parents)<2:
        return base
    # Most recent chronological pair of confirmed, alternating pivots.
    # Neither the lagging parent pivot nor the child degree proves a W3.
    a,b=parents[-2],parents[-1]
    if a[0]>=b[0] or a[1].get("type")==b[1].get("type"):
        raise ValueError("PARENT_PRICE_SEGMENT_NOT_CHRONOLOGICAL")
    interior=[p for t,p in children if a[0]<t<b[0]]
    # Three interior child extrema (at least four alternating legs)
    # is only weak evidence for possible substructure, not an Elliott count.
    base["confirmed_child_pivots_inside"]=len(interior)
    base["candidate_parent_leg"]={
        "first_confirmed_swing_time":a[0].isoformat(),
        "last_confirmed_swing_time":b[0].isoformat(),
        "direction":"UP" if b[1]["type"]=="H" else "DOWN",
        "swing_extrema_proved_not_wave_labels":True,
    }
    base["status"]=(
        "POTENTIAL_CHILD_SUBDIVISION_HYPOTHESIS" if len(interior)>=3
        else "NO_SUFFICIENT_CHILD_SUBDIVISION_EVIDENCE"
    )
    return base


def nested_evidence_for_views(
    views:dict[str,dict[str,Any]], *,
    intervals:list[str],as_of_ms:int,
)->list[dict[str,Any]]:
    from .binance_wave_history import INTERVAL_MS
    # Explicit native scale order fixed by Binance, not by invented Elliott degrees.
    by_scale=[s for s in reversed(INTERVAL_MS) if s in intervals]
    return [cross_scale_pivot_evidence(
        views.get(parent),views.get(child),
        parent_tf=parent,child_tf=child,common_as_of_ms=as_of_ms)
        for parent,child in zip(by_scale,by_scale[1:])]
