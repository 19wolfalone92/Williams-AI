"""Principle-first, NONLINEAR, causal Williams market interpretation.

Books illustrate mechanisms, not images that all markets must imitate.
This code derives independently checkable *signs* from completed candles:
nested confirmed swings at several resolutions, displaced Alligator state,
AO/AC momentum and divergence, volatility regime and structural boundaries.

Do NOT infer "the market will go up" from wave-picture resemblance. Several
conflicting hypotheses are kept alive simultaneously; each has confirmation
requirements and potential structural invalidation. No probability here has
been calibrated on a truly unseen holdout. NO TRADING AUTHORITY.
"""
from __future__ import annotations

from math import isfinite
from statistics import median
from typing import Any

SCALES = (("LOCAL", 2), ("INTERMEDIATE", 5), ("BROAD", 8))


def _swings(bars: list[Any], radius: int) -> list[dict[str, Any]]:
    """Causal extremum: pivot needs 'radius' CLOSED bars to its right.

    Ambiguous both-high-and-low outside bars are never given a unique wave
    direction. Consecutive same-side extrema are reduced to the stronger
    extremum, but no future bar beyond caller's cutoff is accessed.
    """
    points = []
    for i in range(radius, len(bars)-radius):
        current = bars[i]
        before = bars[i-radius:i]
        after = bars[i+1:i+1+radius]
        is_h = all(current.high > x.high for x in before+after)
        is_l = all(current.low < x.low for x in before+after)
        if is_h == is_l:
            continue
        kind = "H" if is_h else "L"
        px = float(current.high if is_h else current.low)
        point = {
            "index": i, "type": kind, "price": px,
            "time": current.at.isoformat(),
            "confirmed_at_index": i+radius,
            "confirmed_at": bars[i+radius].at.isoformat(),
        }
        if points and points[-1]["type"] == kind:
            more_extreme = px > points[-1]["price"] if is_h else px < points[-1]["price"]
            if more_extreme:
                points[-1] = point
        else:
            points.append(point)
    return points


def _structure(pivots: list[dict[str, Any]]) -> str:
    highs = [p["price"] for p in pivots if p["type"] == "H"]
    lows = [p["price"] for p in pivots if p["type"] == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return "INSUFFICIENT_SWINGS"
    h_up = highs[-1] > highs[-2]
    l_up = lows[-1] > lows[-2]
    h_down = highs[-1] < highs[-2]
    l_down = lows[-1] < lows[-2]
    if h_up and l_up:
        return "UP_STRUCTURE"
    if h_down and l_down:
        return "DOWN_STRUCTURE"
    return "COMPLEX_OR_TRANSITION"


def _volatility_regime(bars: list[Any]) -> dict[str, Any]:
    ranges = [max(0.0, float(c.high-c.low)) for c in bars]
    short = median(ranges[-min(20,len(ranges)):])
    baseline = median(ranges[-min(120,len(ranges)):])
    ratio = short / baseline if baseline > 0 else None
    state = (
        "EXPANDING" if ratio is not None and ratio >= 1.55
        else "CONTRACTING" if ratio is not None and ratio <= .65
        else "BASELINE_OR_UNKNOWN"
    )
    return {
        "state": state,
        "relative_current_range": round(ratio, 6) if ratio is not None else None,
        "reference_bar_count": min(120,len(bars)),
        "warning": "Range expansion does not determine price direction.",
    }


def _momentum(ao: list[float | None], ac: list[float | None]) -> dict[str, Any]:
    usable = [float(v) for v in ao[-140:] if v is not None and isfinite(v)]
    last = ao[-1] if ao else None
    before = ao[-2] if len(ao) > 1 else None
    ac_last = ac[-1] if ac else None
    ac_before = ac[-2] if len(ac) > 1 else None
    ceiling = max((abs(x) for x in usable), default=0.0)
    near = bool(
        ceiling > 0 and last is not None and abs(last) <= .15*ceiling
    )
    sign = (
        "POSITIVE" if last is not None and last > 0 else
        "NEGATIVE" if last is not None and last < 0 else "ZERO_OR_UNKNOWN"
    )
    slope = (
        "RISING" if last is not None and before is not None and last > before else
        "FALLING" if last is not None and before is not None and last < before
        else "FLAT_OR_UNKNOWN"
    )
    ac_slope = (
        "RISING" if ac_last is not None and ac_before is not None and ac_last > ac_before
        else "FALLING" if ac_last is not None and ac_before is not None and ac_last < ac_before
        else "FLAT_OR_UNKNOWN"
    )
    return {
        "ao_sign": sign,
        "ao_change": slope,
        "ac_change": ac_slope,
        "ao_near_zero_relative_to_recent_amplitude": near,
        "ao_reference_bars": min(len(ao), 140),
        "wave_four_claim": "NOT_PROVEN_BY_AO_ZERO",
    }


def _scenario(
    kind: str,
    supports: list[str],
    contradictions: list[str],
    required: list[str],
    *,
    references: dict[str, float | None]
) -> dict[str, Any]:
    # NOT a pseudo-probability; avoids false certainty and overfitting scores.
    status = (
        "EVIDENCE_COMPATIBLE_NOT_PREDICTION" if len(supports) >= 2
        else "LOW_EVIDENCE_OR_CONFLICTING"
    )
    return {
        "hypothesis": kind,
        "status": status,
        "supporting_observations": supports,
        "contradicting_observations": contradictions,
        "checks_before_strengthening": required,
        "structural_reference_levels_not_stop_orders": references,
        "probability": None,
        "verified_elliott_degree": False,
        "trade_action": None,
    }


def nonlinear_market_principles(
    bars: list[Any],
    ao: list[float | None],
    ac: list[float | None],
    *,
    alligator_context: str = "UNKNOWN",
    diverging_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply general rules to *changing shapes*, not template-image similarity.

    Caller MUST supply contiguous FINALIZED bars and causal visible indicators;
    this helper is also safe to invoke on a shorter historical prefix. It
    never upgrades price shapes to certain Elliott wave numbers or orders.
    """
    if not isinstance(bars,list) or not bars or len(bars)!=len(ao) or len(bars)!=len(ac):
        raise ValueError("Completed bar/indicator arrays must align")
    if alligator_context not in {
        "UP_CONTEXT","DOWN_CONTEXT","MIXED_OR_TRANSITION","UNKNOWN"
    }:
        raise ValueError("Unknown Alligator context")
    scale_results: dict[str, dict[str, Any]] = {}
    for label, radius in SCALES:
        pivots = _swings(bars,radius)
        highs = [p for p in pivots if p["type"]=="H"]
        lows = [p for p in pivots if p["type"]=="L"]
        scale_results[label] = {
            "confirmation_right_closed_bars": radius,
            "structure": _structure(pivots),
            "latest_confirmed_high": highs[-1]["price"] if highs else None,
            "latest_confirmed_low": lows[-1]["price"] if lows else None,
            "confirmed_swings": pivots[-8:],
        }
    # A CLOSED structural price break is a useful new fact only if the
    # referenced pivot had been confirmed BEFORE this very candle. The
    # same bar could confirm a pivot and cross its level in unknown order.
    last= len(bars)-1
    local = _swings(bars,2)
    known_high=next((p for p in reversed(local)
                     if p["type"]=="H" and p["confirmed_at_index"]<last),None)
    known_low=next((p for p in reversed(local)
                    if p["type"]=="L" and p["confirmed_at_index"]<last),None)
    current_close=float(bars[-1].close)
    location=(
        "CLOSED_ABOVE_PRIOR_CONFIRMED_SWING_HIGH_NOT_ORDER"
        if known_high and current_close>known_high["price"] else
        "CLOSED_BELOW_PRIOR_CONFIRMED_SWING_LOW_NOT_ORDER"
        if known_low and current_close<known_low["price"] else
        "WITHIN_PRIOR_CONFIRMED_SWING_LEVELS_OR_UNRESOLVED"
    )
    mom = _momentum(ao,ac)
    vol = _volatility_regime(bars)
    structure = {label: v["structure"] for label,v in scale_results.items()}
    long_pivots = sum(x=="UP_STRUCTURE" for x in structure.values())
    short_pivots = sum(x=="DOWN_STRUCTURE" for x in structure.values())
    conflict = long_pivots>0 and short_pivots>0
    source_div = diverging_evidence.get("divergences",[]) if isinstance(diverging_evidence,dict) else []
    bullish_div = any(x.get("kind")=="BULLISH_AO_DIVERGENCE_CANDIDATE" for x in source_div if isinstance(x,dict))
    bearish_div = any(x.get("kind")=="BEARISH_AO_DIVERGENCE_CANDIDATE" for x in source_div if isinstance(x,dict))
    refs = {
        "last_confirmed_high": known_high["price"] if known_high else None,
        "last_confirmed_low": known_low["price"] if known_low else None,
    }
    up = []
    down = []
    up_against = []
    down_against = []
    if long_pivots:
        up.append("HIGHER_HIGHS_AND_LOWS_ON_AT_LEAST_ONE_CONFIRMED_SCALE")
        down_against.append("UP_SWING_STRUCTURE_EXISTS")
    if short_pivots:
        down.append("LOWER_HIGHS_AND_LOWS_ON_AT_LEAST_ONE_CONFIRMED_SCALE")
        up_against.append("DOWN_SWING_STRUCTURE_EXISTS")
    if alligator_context=="UP_CONTEXT":
        up.append("VISIBLE_ALLIGATOR_UP")
        down_against.append("VISIBLE_ALLIGATOR_UP")
    elif alligator_context=="DOWN_CONTEXT":
        down.append("VISIBLE_ALLIGATOR_DOWN")
        up_against.append("VISIBLE_ALLIGATOR_DOWN")
    if location=="CLOSED_ABOVE_PRIOR_CONFIRMED_SWING_HIGH_NOT_ORDER":
        up.append("CLOSED_PRICE_BREAKS_PRIOR_CONFIRMED_SWING_HIGH")
    elif location=="CLOSED_BELOW_PRIOR_CONFIRMED_SWING_LOW_NOT_ORDER":
        down.append("CLOSED_PRICE_BREAKS_PRIOR_CONFIRMED_SWING_LOW")
    if mom["ao_sign"]=="POSITIVE":
        up.append("AO_ABOVE_ZERO")
        down_against.append("AO_ABOVE_ZERO")
    elif mom["ao_sign"]=="NEGATIVE":
        down.append("AO_BELOW_ZERO")
        up_against.append("AO_BELOW_ZERO")
    if mom["ao_change"]=="RISING":
        up.append("AO_ACCELERATING_UP")
        down_against.append("AO_RISING")
    elif mom["ao_change"]=="FALLING":
        down.append("AO_ACCELERATING_DOWN")
        up_against.append("AO_FALLING")
    correction = []
    if mom["ao_near_zero_relative_to_recent_amplitude"]:
        correction.append("AO_NEAR_ZERO_RELATIVE_TO_RECENT_SWING")
    if conflict:
        correction.append("CONFIRMED_SWING_SCALES_DISAGREE")
    if "COMPLEX_OR_TRANSITION" in structure.values():
        correction.append("ALTERNATING_SWING_EXTREMA")
    if alligator_context=="MIXED_OR_TRANSITION":
        correction.append("ALLIGATOR_IN_MIXED_PHASE")
    if vol["state"]=="CONTRACTING":
        correction.append("PRICE_RANGE_CONTRACTING")
    reversal_up = ["AO_BULLISH_DIVERGENCE"] if bullish_div else []
    reversal_down = ["AO_BEARISH_DIVERGENCE"] if bearish_div else []
    if short_pivots and mom["ao_change"]=="RISING":
        reversal_up.append("AO_RECOVERS_WHILE_PRICE_STRUCTURE_DOWN")
    if long_pivots and mom["ao_change"]=="FALLING":
        reversal_down.append("AO_WEAKENS_WHILE_PRICE_STRUCTURE_UP")
    scenarios = [
        _scenario("CONTINUATION_UP",up,up_against,
                  ["Price trigger and relative Alligator phase require separate evidence.",
                   "Test whether the last confirmed higher low survives."],
                  references=refs),
        _scenario("CONTINUATION_DOWN",down,down_against,
                  ["Price trigger and relative Alligator phase require separate evidence.",
                   "Test whether the last confirmed lower high survives."],
                  references=refs),
        _scenario("CORRECTION_OR_COMPLEX_RANGE",correction,
                  ["ALIGNED_UP_TREND_EVIDENCE" if len(up)>=3 else "",
                   "ALIGNED_DOWN_TREND_EVIDENCE" if len(down)>=3 else ""],
                  ["Observe the next confirmed swing and AO response.",
                   "Do not label wave 4 from AO near zero alone."],
                  references=refs),
        _scenario("POSSIBLE_REVERSAL_UP",reversal_up,down,
                  ["Require a confirmed structural improvement and independent entry signal.",
                   "Divergence alone does not close SHORT."],
                  references=refs),
        _scenario("POSSIBLE_REVERSAL_DOWN",reversal_down,up,
                  ["Require a confirmed structural weakening and independent entry signal.",
                   "Divergence alone does not close LONG."],
                  references=refs),
    ]
    for s in scenarios:
        s["contradicting_observations"] = [x for x in s["contradicting_observations"] if x]
    return {
        "mode": "PRINCIPLES_NOT_PICTURE_TEMPLATES",
        "as_of": bars[-1].at.isoformat(),
        "observable_facts": {
            "multi_resolution_price_structure": structure,
            "scale_details": scale_results,
            "momentum": mom,
            "volatility": vol,
            "visible_alligator_context": alligator_context,
            "bullish_divergence_observed": bullish_div,
            "bearish_divergence_observed": bearish_div,
            "current_close": current_close,
            "closed_price_location_relative_to_previously_confirmed_levels": location,
        },
        "competing_market_scenarios": scenarios,
        "interpretation": "Nonlinear multi-scale hypotheses: not a guaranteed price forecast.",
        "hardcoded_wave_picture_matching": False,
        "wave_degree": "UNKNOWN",
        "no_fixed_100_140_bar_requirement": True,
        "automatic_entry_or_exit_allowed": False,
        "trade_action": None,
        "execution_allowed": False,
        "evidence_limitations": [
            "All pivots wait for their required closed right-hand bars.",
            "Different scales can legitimately give opposing trends.",
            "The 100–140-bar rhythm is contextual, not a hard wave shape.",
            "AO near zero can accompany a fourth wave but does NOT certify it.",
            "Opposite signals are not confirmed Binance stops or filled reversals.",
            "Trading requires book-edition rules, causal price trigger and verified exchange safety.",
        ],
    }
