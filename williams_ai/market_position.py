"""Explain *where* a closed price chart sits relative to confirmed structure.

This is observational context, not Elliott labelling, probability of
profit or authorization for any trading order.
"""
from __future__ import annotations
from typing import Any
from .wave_evidence import confirmed_price_swings


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
        a, b = ao[prior["index"]],ao[latest["index"]]
        if a is None or b is None:
            continue
        price_div = (latest["price"]>prior["price"] if side=="BEARISH"
                     else latest["price"]<prior["price"])
        momentum_div = (b<a if side=="BEARISH" else b>a)
        if price_div and momentum_div:
            divergences.append({
                "kind":side + "_AO_DIVERGENCE_CANDIDATE",
                "older_swing_confirmed_at":bars[prior["confirmed_at_index"]].at.isoformat(),
                "newer_swing_confirmed_at":bars[latest["confirmed_at_index"]].at.isoformat(),
                "price_pivots":[prior["price"],latest["price"]],
                "ao_at_pivots":[a,b],
                "status":"OBSERVED_DIVERGENCE_NOT_REVERSAL_CONFIRMATION",
                "source":"TC1: AO momentum versus price wave peaks; not a buy/sell trigger",
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
            "Confirmed price swings have at least two right-hand closed bars.",
            "Divergence is evidence of momentum change, not certain reversal.",
            "A higher-high structure does not uniquely define Elliott wave degree.",
        ],
        "trade_action": None,
    }
