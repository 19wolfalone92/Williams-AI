"""Pure causal H1→H4/D1 research summaries for the v6 expert exam.

Not a trading signal engine. AO=(SMA5-SMA34)(median_price), and AC=AO-
SMA5(AO), both from ONLY fully closed bars. A D1/H4 group is discarded
unless all underlying H1 candles were closed. No wave gold or future data.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import fsum
from typing import Any

from .independent_exam_v6 import _utc


def _bucket(start: datetime, hours: int) -> datetime:
    return start.replace(hour=(start.hour // hours) * hours,
                         minute=0, second=0, microsecond=0)


def aggregate_h1(bars: list[dict], *, group_hours: int,
                 cutoff_utc: str) -> list[dict]:
    if group_hours not in (1, 4, 24):
        raise ValueError("V6_MULTISCALE_UNSUPPORTED_TIMEFRAME")
    cutoff = _utc(cutoff_utc)
    groups = defaultdict(list)
    for bar in bars:
        a, b = _utc(bar["open_utc"]), _utc(bar["close_utc"])
        if not a < b <= cutoff:
            raise ValueError("V6_MULTISCALE_FUTURE_BAR")
        groups[_bucket(a, group_hours)].append(bar)
    output = []
    for start, bucket in sorted(groups.items()):
        # First/last bucket can be incomplete if history begins/ends mid TF.
        if len(bucket) != group_hours or _utc(bucket[0]["open_utc"]) != start:
            continue
        end = start.replace() + timedelta(hours=group_hours)
        if _utc(bucket[-1]["close_utc"]) != end or end > cutoff:
            continue
        for i, b in enumerate(bucket):
            expected_start = start + timedelta(hours=i)
            if _utc(b["open_utc"]) != expected_start or (
                _utc(b["close_utc"]) != expected_start + timedelta(hours=1)
            ):
                raise ValueError("V6_MULTISCALE_DISCONTIGUOUS_CANDLES")
        output.append({
            "closed_through": end.isoformat().replace("+00:00", "Z"),
            "open": bucket[0]["open"],
            "high": max(x["high"] for x in bucket),
            "low": min(x["low"] for x in bucket),
            "close": bucket[-1]["close"],
            "volume": fsum(x["volume"] for x in bucket),
        })
    return output


def _sma(values: list[float], n: int) -> float:
    return fsum(values[-n:]) / n


def _ao_ac(series: list[dict]) -> dict[str, float | None]:
    if len(series) < 34:
        return {"ao": None, "ao_previous": None,
                "ac": None, "ac_previous": None}
    medians = [(x["high"] + x["low"]) / 2 for x in series]
    aos = []
    for end in range(34, len(medians) + 1):
        subset = medians[:end]
        aos.append(_sma(subset, 5) - _sma(subset, 34))
    def near(value):
        return None if value is None else round(value, 10)
    latest, previous = aos[-1], aos[-2] if len(aos) >= 2 else None
    ac = latest - _sma(aos, 5) if len(aos) >= 5 else None
    ac_previous = (aos[-2] - _sma(aos[:-1], 5)
                   if len(aos) >= 6 else None)
    return {"ao": near(latest), "ao_previous": near(previous),
            "ac": near(ac), "ac_previous": near(ac_previous)}


def causal_context(chart: dict[str, Any], *, cutoff_utc: str) -> dict:
    if not isinstance(chart, dict) or not isinstance(chart.get("bars"), list):
        raise ValueError("V6_MULTISCALE_INVALID_CHART")
    bars = chart["bars"]
    cutoff = _utc(cutoff_utc)
    summary = {}
    for hours, tf in ((1, "H1"), (4, "H4"), (24, "D1")):
        series = aggregate_h1(bars, group_hours=hours, cutoff_utc=cutoff_utc)
        if not series:
            summary[tf] = {"closed_bars": 0, "latest_close_utc": None,
                           "ao_ac_median_price": _ao_ac([]),
                           "last_3_completed_ohlc": []}
            continue
        if _utc(series[-1]["closed_through"]) > cutoff:
            raise ValueError("V6_MULTISCALE_FUTURE_AGGREGATE")
        summary[tf] = {
            "closed_bars": len(series),
            "latest_close_utc": series[-1]["closed_through"],
            "ao_ac_median_price": _ao_ac(series),
            "last_3_completed_ohlc": [
                {"close_utc": x["closed_through"], "high": x["high"],
                 "low": x["low"], "close": x["close"]}
                for x in series[-3:]],
        }
    return {"source": "DETERMINISTIC_H1_H4_D1_NO_FUTURE",
            "only_completed_utc_buckets": True,
            "alligator_angulation_not_established": True,
            "wave_degree_expert_verified": False,
            "not_executable_signals": True, "timeframes": summary}
