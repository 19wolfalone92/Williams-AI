"""Deterministic historical candle-by-candle evidence journal (NO ORDERS).

Each decision sees only bars whose closed timestamp <= that replay tick.
No model inference, trade execution or future bars are permitted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .chart_reader import analyze_chart, validate_closed_bars


def _key(candidate: dict[str, Any], observed_at: str) -> str:
    if candidate["type"] == "WM3_FRACTAL":
        marker = candidate["pivot_time"]
    elif candidate["type"] == "WM1":
        marker = candidate["bar_time"]
    else:
        marker = observed_at
    return ":".join((candidate["type"], candidate["side"], marker))


def replay_closed_chart(
    symbol: str, timeframe: str, bars: list[dict[str, Any]], *,
    tick_size: float | None = None, limit: int = 400,
) -> dict[str, Any]:
    """Return first observable signals and later lifecycle changes.

    Intentionally bounded: each step re-runs the full prefix to favor
    auditability over throughput. Never handles incomplete or future bars.
    """
    if not isinstance(bars, list) or not (45 <= len(bars) <= limit <= 500):
        raise ValueError("Replay requires 45..400 closed bars (max limit 500)")
    # Establish candle order and a causal timestamp for the entire run.
    canonical, gaps = validate_closed_bars(bars, timeframe, bars[-1]["close_time"])
    if gaps:
        raise ValueError("Incomplete history: replay forbidden until gaps are repaired")
    events: list[dict[str, Any]] = []
    state_by_signal: dict[str, str] = {}
    for count in range(45, len(canonical) + 1):
        now = canonical[count - 1].at.isoformat()
        analysis = analyze_chart(
            symbol, timeframe, bars[:count], as_of=now, tick_size=tick_size
        )
        if analysis["execution_allowed"] is not False:
            raise RuntimeError("Market analysis unexpectedly enabled execution")
        for candidate in analysis["candidates"]:
            key = _key(candidate, now)
            previous = state_by_signal.get(key)
            state = candidate["state"]
            if previous == state:
                continue
            occurred_at = (
                candidate["confirmed_at"]
                if candidate["type"] == "WM3_FRACTAL"
                else candidate.get("bar_time", now)
            )
            if datetime.fromisoformat(occurred_at) > datetime.fromisoformat(now):
                raise RuntimeError("Look-ahead candidate")
            events.append({
                "observed_at": now,
                "source_candle_count": count,
                "signal_key": key,
                "type": candidate["type"],
                "side": candidate["side"],
                "prior_state": previous,
                "state": state,
                "source": candidate["source"],
                "observable_earliest_at": occurred_at,
                "trade_action": None,
            })
            state_by_signal[key] = state
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "ticks_replayed": len(canonical) - 44,
        "events": events,
        "trade_action": None,
        "no_lookahead": True,
        "limitations": [
            "Observations are not validated book signals or trading orders.",
            "Historical replay does not simulate exchange fills, spreads or fees.",
            "DecisionTrace captures first visible candidate states, not hidden wave truth.",
        ],
    }
