"""Private, double-reviewed chart lessons for actual Qwen weight training.

Approvals bind the exact question, answer, book rule and closed market data.
They are declarations, not proof of a person's identity or expertise. Nothing
here generates an expert answer, routes an order or certifies profitability.
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta
import hashlib
import json
import math
import re
from typing import Any

from .chart_reader import TF_SECONDS, _time, indicators, validate_closed_bars

SCHEMA = "WILLIAMS_AI_REVIEWED_CLOSED_CHART_V1"
CHART_FIELDS = frozenset({"market_context", "independent_reviews"})
_MARKET_FIELDS = frozenset({
    "schema", "symbol", "venue", "timeframe", "tick_size", "cutoff",
    "bars", "market_sha256",
})
_BAR_FIELDS = frozenset({
    "close_time", "open", "high", "low", "close", "volume", "closed",
})
_REVIEW_FIELDS = frozenset({"reviewer_id", "reviewed_at", "lesson_sha256"})
_ID = re.compile(r"^[A-Za-z0-9_.:-]{6,100}$")
_FIXED_TFS = frozenset({"M5", "M15", "H1", "H4", "H6", "D1"})


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def market_digest(context: dict[str, Any]) -> str:
    return digest({k: v for k, v in context.items() if k != "market_sha256"})


def lesson_commitment(item: dict[str, Any]) -> str:
    """Changing even the partition or question invalidates both approvals."""
    return digest({k: v for k, v in item.items() if k not in {
        "independent_reviews", "reviewer_id", "review_status",
    }})


def checked_chart_context(context: Any) -> dict[str, Any]:
    if (not isinstance(context, dict) or set(context) != _MARKET_FIELDS
            or context.get("schema") != SCHEMA
            or not isinstance(context.get("symbol"), str)
            or re.fullmatch(r"[A-Z0-9]{3,22}", context["symbol"]) is None
            or context.get("venue") not in {"SPOT", "FUTURES"}
            or context.get("timeframe") not in _FIXED_TFS):
        raise ValueError("INVALID_REVIEWED_CHART_SCHEMA")
    tick = context["tick_size"]
    if (isinstance(tick, bool) or not isinstance(tick, (int, float))
            or not math.isfinite(tick) or tick <= 0):
        raise ValueError("INVALID_REVIEWED_CHART_TICK")
    bars = context["bars"]
    if (not isinstance(bars, list) or not 220 <= len(bars) <= 1500
            or any(not isinstance(b, dict) or set(b) != _BAR_FIELDS for b in bars)):
        raise ValueError("BOUNDED_CLOSED_OHLC_ONLY_NO_LABEL_FIELDS")
    if context["market_sha256"] != market_digest(context):
        raise ValueError("CHART_CHANGED_AFTER_REVIEW")
    candles, gaps = validate_closed_bars(bars, context["timeframe"], context["cutoff"])
    seconds = TF_SECONDS[context["timeframe"]]
    if (gaps or candles[-1].at != _time(context["cutoff"])
            or any((b.at - a.at).total_seconds() != seconds
                   for a, b in zip(candles, candles[1:]))
            or any(round(c.at.timestamp() * 1000 + 1) % (seconds * 1000)
                   for c in candles)):
        raise ValueError("REVIEW_CHART_MUST_BE_CONTIGUOUS_ALIGNED_AND_FINAL")
    return {"candles": candles, "start": candles[0].at - timedelta(seconds=seconds),
            "end": candles[-1].at, "symbol": context["symbol"],
            "venue": context["venue"], "timeframe": context["timeframe"]}


def checked_chart_lesson(item: dict[str, Any]) -> dict[str, Any]:
    context = checked_chart_context(item["market_context"])
    reviews = item["independent_reviews"]
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise ValueError("TWO_INDEPENDENT_CHART_REVIEWS_REQUIRED")
    expected = lesson_commitment(item)
    people = set()
    for review in reviews:
        if (not isinstance(review, dict) or set(review) != _REVIEW_FIELDS
                or not isinstance(review["reviewer_id"], str)
                or not _ID.fullmatch(review["reviewer_id"])
                or review["reviewer_id"] in people
                or review["lesson_sha256"] != expected):
            raise ValueError("MISSING_DISTINCT_OR_STALE_CHART_APPROVAL")
        if _time(review["reviewed_at"]) < context["end"]:
            raise ValueError("CHART_REVIEW_PREDATES_VISIBLE_MARKET")
        people.add(review["reviewer_id"])
    if item["reviewer_id"] not in people:
        raise ValueError("PRIMARY_REVIEWER_NOT_IN_CHART_APPROVALS")
    return {**context, "split": item["split"], "edition": item["edition"],
            "case_id": item["case_id"]}


def check_market_partitions(rows: list[dict[str, Any]], *, embargo_hours: int = 24
                            ) -> dict[str, Any]:
    """Purge shared visible history, also when two timeframes overlap.

    Checking only cutoffs or exact chart hashes leaks almost identical
    sliding windows into a supposedly untouched validation/test partition.
    """
    if (isinstance(embargo_hours, bool) or not isinstance(embargo_hours, int)
            or not 24 <= embargo_hours <= 24 * 30):
        raise ValueError("AT_LEAST_24_HOURS_MARKET_EMBARGO_REQUIRED")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["venue"], row["symbol"]), []).append(row)
    margin = timedelta(hours=embargo_hours)
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: r["start"])
        for i, current in enumerate(ordered):
            for other in ordered[i + 1:]:
                if other["start"] > current["end"] + margin:
                    break
                if other["split"] != current["split"]:
                    raise ValueError("OVERLAPPING_MARKET_HISTORY_ACROSS_TRAIN_HOLDOUT")
    return {
        "chart_examples_by_split": dict(Counter(x["split"] for x in rows)),
        "chart_editions_by_split": {
            split: dict(Counter(x["edition"] for x in rows if x["split"] == split))
            for split in ("train", "validation", "test")},
        "cross_split_visible_history_overlap_checked": True,
        "cross_timeframe_overlap_checked": True, "embargo_hours": embargo_hours,
        "human_identity_and_independence_not_programmatically_proven": True,
    }


def require_market_supervision(audit: dict[str, Any]) -> None:
    """Book prose alone must not qualify an adapter as trader training."""
    for split, minimum in (("train", 24), ("validation", 8), ("test", 8)):
        if (audit["chart_examples_by_split"].get(split, 0) < minimum
                or set(audit["chart_editions_by_split"][split]) != {
                    "TC1", "NTD", "TC2", "COURSE"}):
            raise ValueError("DOUBLE_REVIEWED_FOUR_BOOK_MARKET_SUPERVISION_REQUIRED")


def chart_prompt_observations(context: dict[str, Any]) -> dict[str, Any]:
    """Recompute a bounded observation view; no teacher's wave/setup answer.

    Alligator is displaced causally. Fractal extrema require their two
    right-hand CLOSED bars. Full raw history remains bound by its digest.
    This numeric summary is not image vision or a full Elliott genealogy.
    """
    checked = checked_chart_context(context)
    candles = checked["candles"]
    lines = indicators(candles)
    def number(v):
        return None if v is None else float(f"{v:.8g}")
    tail = []
    for i in range(len(candles) - 16, len(candles)):
        b = candles[i]
        tail.append([i, number(b.open), number(b.high), number(b.low), number(b.close), number(b.volume),
                     *[number(lines[k][i]) for k in ("jaw", "teeth", "lips", "ao", "ac")]])
    pivots = []
    for i in range(2, len(candles) - 2):
        b = candles[i]
        others = [candles[k] for k in (i - 2, i - 1, i + 1, i + 2)]
        if all(b.high > c.high for c in others):
            pivots.append([i, "HIGH", number(b.high), i + 2])
        if all(b.low < c.low for c in others):
            pivots.append([i, "LOW", number(b.low), i + 2])
    ranges = []
    for size in (34, 100, 140, 220):
        ranges.append({"bars": size,
                       "high": number(max(b.high for b in candles[-size:])),
                       "low": number(min(b.low for b in candles[-size:])),
                       "first_close": number(candles[-size].close)})
    return {"symbol": context["symbol"], "venue": context["venue"],
            "timeframe": context["timeframe"], "as_of": context["cutoff"],
            "market_sha256": context["market_sha256"], "closed_bars": len(candles),
            "tail_columns": ["index", "open", "high", "low", "close", "volume", "jaw",
                             "teeth", "lips", "ao", "ac"], "tail": tail,
            "strict_fractal_columns": ["pivot_index", "type", "price", "confirmed_index"],
            "last_confirmed_strict_fractals": pivots[-12:], "prior_ranges": ranges,
            "numeric_summary_not_full_visual_chart": True,
            "volume_not_verified_as_tick_count": True,
            "broker_fill_or_live_position_not_supplied": True}
