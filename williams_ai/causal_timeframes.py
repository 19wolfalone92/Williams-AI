"""Causal top-down H1 -> H4 -> D1 market *context*, never an entry signal.

Aggregate only COMPLETE UTC-aligned higher-timeframe candles from closed H1
klines. No future candle, unfinished H4/D1 candle or synthetic higher-timeframe
bar may enter DecisionTrace. Wave degree remains a hypothesis.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

from .chart_reader import analyze_chart, validate_closed_bars

GROUPS = {"H4": 4, "D1": 24}
HOUR_MS = 3_600_000


def _utc_close_ms(row: dict[str, Any]) -> int:
    raw = row["close_time"]
    if not isinstance(raw, str):
        raise ValueError("Close timestamp must be ISO8601")
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timestamp requires timezone")
    return round(dt.timestamp() * 1000)


def aggregate_closed_h1(
    rows: list[dict[str, Any]], timeframe: str
) -> list[dict[str, Any]]:
    """Drop incomplete first/last UTC windows; reject holes instead of faking."""
    if timeframe not in GROUPS:
        raise ValueError("Only H4/D1 aggregation from H1 is supported")
    if not rows:
        return []
    _, gaps = validate_closed_bars(rows, "H1", rows[-1]["close_time"])
    if gaps:
        raise ValueError("Incomplete H1 archive: higher timeframe prohibited")
    hours = GROUPS[timeframe]
    slots: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        ms = _utc_close_ms(row)
        # Binance klines close at UTC hh:59:59.999.
        if (ms + 1) % HOUR_MS != 0:
            raise ValueError("H1 candle close must be aligned to UTC hour boundary")
        key = ms // (hours * HOUR_MS)
        slots.setdefault(key, []).append(row)
    out = []
    for key in sorted(slots):
        group = slots[key]
        expected_first = key * hours * HOUR_MS + HOUR_MS - 1
        expected_last = (key + 1) * hours * HOUR_MS - 1
        if (len(group) != hours or
                _utc_close_ms(group[0]) != expected_first or
                _utc_close_ms(group[-1]) != expected_last):
            # Do not pretend an incomplete current D1 bar is fully known.
            continue
        out.append({
            "close_time": group[-1]["close_time"],
            "open": group[0]["open"],
            "high": max(x["high"] for x in group),
            "low": min(x["low"] for x in group),
            "close": group[-1]["close"],
            "volume": sum(float(x["volume"]) for x in group),
            "closed": True,
        })
    return out


def top_down_market_evidence(
    symbol: str, closed_h1_rows: list[dict[str, Any]], *,
    tick_size: float | None = None,
) -> dict[str, Any]:
    """One symbol, one H1 close: snapshots include no later H1/H4/D1 bars.

    Explicitly distinguishes visually described wave alternatives from the
    observable Alligator / AO facts. Returns diagnostic evidence, not
    assumptions of a predictive Elliott hierarchy.
    """
    if not closed_h1_rows:
        raise ValueError("At least one H1 candle is required")
    bars, gaps = validate_closed_bars(closed_h1_rows, "H1", closed_h1_rows[-1]["close_time"])
    if gaps:
        raise ValueError("Missing hours prevent multi-scale inference")
    as_of = closed_h1_rows[-1]["close_time"]
    groups = {
        "H1": closed_h1_rows,
        "H4": aggregate_closed_h1(closed_h1_rows, "H4"),
        "D1": aggregate_closed_h1(closed_h1_rows, "D1"),
    }
    evidence = {}
    higher = []
    for tf in ("D1", "H4", "H1"):
        rows = groups[tf]
        # AO(34), AC and displaced Alligator need 45+ complete bars.
        if len(rows) < 45:
            evidence[tf] = {
                "status": "INSUFFICIENT_COMPLETE_BARS",
                "closed_bars": len(rows),
                "closed_through": rows[-1]["close_time"] if rows else None,
                "market_context": "UNKNOWN", "wave_degree": "UNKNOWN",
                "hypotheses": [], "trade_action": None,
            }
            continue
        result = analyze_chart(symbol, tf, rows, as_of=as_of, tick_size=tick_size)
        ctx = result["evidence"]
        evidence[tf] = {
            "status": "OBSERVED_CLOSED_BARS",
            "closed_bars": len(rows),
            "closed_through": rows[-1]["close_time"],
            "market_context": ctx["market_context"],
            "ao": ctx["ao"], "ac": ctx["ac"], "zone": ctx["zone"],
            "alligator": {
                "jaw_visible": ctx["jaw_visible"],
                "teeth_visible": ctx["teeth_visible"],
                "lips_visible": ctx["lips_visible"],
            },
            "confirmed_swings": result["wave_structure"]["recent_confirmed_swings"],
            "wave_alternatives": result["wave_structure"]["impulse_alternatives"],
            "corrective_alternatives": result["wave_structure"]["corrective_alternatives"],
            "wave_degree": result["wave_structure"]["wave_degree"],
            "principle_first_structure": (
                result["principle_evidence"]["observable_facts"]
                      ["multi_resolution_price_structure"]
            ),
            "principle_first_market_scenarios": [
                q["hypothesis"] for q in result["principle_evidence"]
                    ["competing_market_scenarios"]
                if q["status"]=="EVIDENCE_COMPATIBLE_NOT_PREDICTION"
            ],
            "ao_near_zero_not_w4_proof": result["principle_evidence"]
                ["observable_facts"]["momentum"]
                ["ao_near_zero_relative_to_recent_amplitude"],
            "trade_action": None,
        }
        higher.append((tf,ctx["market_context"]))
    confident=[x[1] for x in higher if x[0] in {"D1","H4"} and x[1] in {"UP_CONTEXT","DOWN_CONTEXT"}]
    if len(confident)==2 and confident[0]==confident[1]:
        trend="HIGHER_TIMEFRAMES_DIRECTIONALLY_ALIGNED"
    elif len(confident)==2:
        trend="HIGHER_TIMEFRAMES_CONFLICT"
    else:
        trend="HIGHER_DIRECTION_UNRESOLVED"
    return {
        "symbol": symbol, "as_of": as_of,
        "authority": "READ_ONLY_CONTEXT_NOT_EXECUTION",
        "timeframes": evidence,
        "higher_context": trend,
        "wave_hierarchy_status": "UNCONFIRMED_NESTED_DEGREES",
        "wave_rhythm": "100–140 bars is a heuristic for scale, NOT a proof or hard lookback",
        "trade_action": None, "execution_allowed": False,
        "limitations": [
            "No lookahead: H4/D1 require complete UTC-aligned candle windows.",
            "A correction on H1 might be a subwave of the H4/D1 movement: not labeled without separate evidence.",
            "Higher-timeframe conflict is context only, not a blanket ban on TC2 first Wise Man.",
            "This reads H1 OHLCV, not the images or actual Binance order/fill/stop confirmations.",
        ],
    }
