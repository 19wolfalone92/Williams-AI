"""Closed-bar price structure and leg-anchored AO divergence evidence.

AO extremum may occur BEFORE the corresponding swing-price extreme. Comparing
only the AO histogram at the pivot candle falsely classifies/omits divergences.
We compare AO over each independently anchored completed price leg instead.
Pivot centers need two right-hand CLOSED bars, and no candidate is an entry.
"""
from __future__ import annotations
from typing import Any
from .wave_evidence import confirmed_price_swings


def _leg_momentum(
    pivots: list[dict[str, Any]], ao: list[float | None],
    target: dict[str, Any], *, side: str,
) -> dict[str, Any] | None:
    """Extremal AO on last opposite-swing -> target price swing.

    No guessed anchor: if there is no PRECEDING confirmed opposite pivot,
    the visual leg is unresolved and divergence must not be asserted.
    """
    opposite="L" if side=="BEARISH" else "H"
    anchors=[p for p in pivots if p["type"]==opposite and p["index"]<target["index"]]
    if not anchors:
        return None
    anchor=anchors[-1]
    begin=anchor["index"]
    end=target["index"]
    candidates=[(i,float(ao[i])) for i in range(begin,end+1) if ao[i] is not None]
    if not candidates:
        return None
    index,value=(max(candidates,key=lambda z:z[1]) if side=="BEARISH"
                 else min(candidates,key=lambda z:z[1]))
    return {
        "anchor_index":begin,
        "extremum_index":index,
        "extremum_value":value,
        "anchor_confirmation_index":anchor["confirmed_at_index"],
    }


def market_position(bars: list[Any], ao: list[float | None]) -> dict[str, Any]:
    if len(bars) != len(ao):
        raise ValueError("Candle and AO histories must align exactly")
    if not bars:
        return {
            "structure": "NO_DATA", "divergences": [], "recent_levels": {},
            "interpretation": "Нет свечей", "trade_action": None,
        }
    pivots = confirmed_price_swings(bars)
    highs = [x for x in pivots if x["type"] == "H"]
    lows = [x for x in pivots if x["type"] == "L"]
    structure = "UNRESOLVED"
    if len(highs) >= 2 and len(lows) >= 2:
        h1,h2 = highs[-2:]
        l1,l2 = lows[-2:]
        if h2["price"]>h1["price"] and l2["price"]>l1["price"]:
            structure = "HIGHER_HIGHS_HIGHER_LOWS"
        elif h2["price"]<h1["price"] and l2["price"]<l1["price"]:
            structure = "LOWER_HIGHS_LOWER_LOWS"
        else:
            structure = "MIXED_OR_RANGE"
    divergences = []
    for side, sequence in (("BEARISH", highs), ("BULLISH", lows)):
        if len(sequence) < 2:
            continue
        prior, latest = sequence[-2:]
        a=_leg_momentum(pivots,ao,prior,side=side)
        b=_leg_momentum(pivots,ao,latest,side=side)
        if a is None or b is None:
            continue
        # A confirmed price pivot may have originated several bars ago; AO
        # extremum is recorded only after the pivot is actually confirmed.
        price_div=(latest["price"]>prior["price"] if side=="BEARISH"
                   else latest["price"]<prior["price"])
        momentum_div=(b["extremum_value"]<a["extremum_value"]
                      if side=="BEARISH" else
                      b["extremum_value"]>a["extremum_value"])
        if price_div and momentum_div:
            divergences.append({
                "kind":side + "_AO_DIVERGENCE_CANDIDATE",
                "older_swing_confirmed_at":bars[prior["confirmed_at_index"]].at.isoformat(),
                "newer_swing_confirmed_at":bars[latest["confirmed_at_index"]].at.isoformat(),
                "price_pivots":[prior["price"],latest["price"]],
                "ao_leg_extrema":[a["extremum_value"],b["extremum_value"]],
                "ao_leg_extremum_times":[
                    bars[a["extremum_index"]].at.isoformat(),
                    bars[b["extremum_index"]].at.isoformat()
                ],
                "ao_leg_anchor_times":[
                    bars[a["anchor_index"]].at.isoformat(),
                    bars[b["anchor_index"]].at.isoformat()
                ],
                "pivot_ao_only_is_not_used":True,
                "status":"OBSERVED_LEG_DIVERGENCE_NOT_REVERSAL_CONFIRMATION",
                "source":"TC1 / visual wave handout: AO leg extrema versus confirmed price pivots",
            })
    latest_high = highs[-1] if highs else None
    latest_low = lows[-1] if lows else None
    levels = {
        "latest_confirmed_high": latest_high["price"] if latest_high else None,
        "latest_confirmed_low": latest_low["price"] if latest_low else None,
        "close": bars[-1].close,
        "relative_to_known_swing": (
            "ABOVE_CONFIRMED_HIGH" if latest_high and bars[-1].close>latest_high["price"]
            else "BELOW_CONFIRMED_LOW" if latest_low and bars[-1].close<latest_low["price"]
            else "INSIDE_CONFIRMED_SWING_RANGE"
            if latest_high and latest_low else "UNKNOWN"
        ),
    }
    return {
        "structure": structure,
        "recent_levels": levels,
        "divergences": divergences,
        "hypothesis_limitations": [
            "Each pivot requires two closed bars on its right.",
            "AO measured over swing legs, NOT the histogram exactly at swing pivot.",
            "Leg must have a preceding independently confirmed opposite pivot.",
            "Divergence is only momentum evidence, not an entry, exit or certain reversal.",
            "A higher-high structure does not uniquely define Elliott wave degree.",
        ],
        "trade_action": None,
    }
