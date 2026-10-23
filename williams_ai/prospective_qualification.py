"""Two-stage falsifiable prospective Williams-AI research protocol.

STAGE 1: Seal a real model's (or identified baseline's) direction/ABSTAIN
against ONLY closed candles available when the decision is made.
STAGE 2: Much later, attach newly observed future candles and score without
ever asking the predictor to revise the sealed answer.

This is an audit of directional forecasts, NOT Williams trade entries, risk-
adjusted PnL, proven book understanding, or real exchange fills. A published
frozen old data set is NOT an untouched independent prospective holdout.
No market APIs, exchange code, automatic orders, or positions.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone, timedelta
from statistics import median
from typing import Any, Callable

from .chart_reader import analyze_chart, validate_closed_bars

HORIZON = 24
SCHEMA = "WILLIAMS_AI_PROSPECTIVE_DIRECTIONAL_V1"
_ALLOWED = {"UP", "DOWN", "ABSTAIN"}


def _utc(s: str) -> datetime:
    if not isinstance(s, str):
        raise ValueError("Expected timezone-aware ISO date")
    try:
        value = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid date") from exc
    if value.tzinfo is None:
        raise ValueError("Must specify time zone")
    return value.astimezone(timezone.utc)


def _canonical(v: Any) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _validated_identity(model_id: str, code_id: str) -> None:
    for value in (model_id, code_id):
        if not isinstance(value, str) or not 8 <= len(value) <= 200:
            raise ValueError("Pinned model and source-code identity required")
        if any(ord(x) < 32 for x in value):
            raise ValueError("Invalid identity characters")


def freeze_prospective_call(
    symbol: str,
    closed_history: list[dict[str, Any]], *,
    tick_size: float,
    model_id: str,
    code_id: str,
    predictor: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Only pass a sealed AS-OF analysis to the caller's model.

    Caller must guarantee that the predictor does not separately access
    future data (e.g. through hidden external state). This interface never
    passes withheld futures to it. Publish/hash the returned commitment
    BEFORE any future bars are seen, not retroactively.
    """
    _validated_identity(model_id, code_id)
    if not callable(predictor):
        raise ValueError("Predictor callback required; cannot invent AI answers")
    if not isinstance(closed_history, list) or len(closed_history) < 220:
        raise ValueError("Minimum 220 H1 closed bars for prospective study")
    as_of = closed_history[-1].get("close_time")
    candles, gaps = validate_closed_bars(closed_history, "H1", as_of)
    if gaps:
        raise ValueError("Incomplete closed-bar history")
    # Generic OHLC validator catches missing hours but tolerates a shorter
    # interval (e.g. 30m). A 24-H1-bar experiment requires EXACT H1 cadence.
    if any(b.at-a.at != timedelta(hours=1)
           for a,b in zip(candles,candles[1:])):
        raise ValueError("EXACT_H1_CADENCE_REQUIRED")
    if len(candles) < 25:
        raise ValueError("Insufficient momentum baseline history")
    evidence = analyze_chart(symbol, "H1", closed_history,
                             as_of=as_of, tick_size=tick_size)
    if evidence["execution_allowed"] is not False or evidence["gaps"]:
        raise ValueError("Causal chart gate failed")
    decision = predictor(evidence)
    if not isinstance(decision, dict):
        raise ValueError("Predictor must return an explicit structured answer")
    # Never repair an LLM's malformed answer or silently reinterpret
    # an unqualified speculative wave number as a trade.
    if (set(decision) != {"symbol", "as_of", "direction",
                          "trade_action", "execution_allowed"} or
            decision.get("symbol") != symbol or
            _utc(decision.get("as_of")) != candles[-1].at or
            decision.get("direction") not in _ALLOWED or
            decision.get("trade_action", "INVALID") is not None or
            decision.get("execution_allowed") is not False):
        raise ValueError("Malformed, mistimed or executable model decision")
    direction = decision["direction"]
    # One clean, SHA-256-sealed factual record, no future label.
    past24 = candles[-25].close
    close = candles[-1].close
    past_direction = ("UP" if close > past24 else
                      "DOWN" if close < past24 else "ABSTAIN")
    threshold = 0.25 * median(c.high-c.low for c in candles[-24:])
    if threshold <= 0 or not math.isfinite(threshold):
        raise ValueError("Invalid predeclared neutrality threshold")
    core = {
        "schema": SCHEMA, "symbol": symbol, "model_id": model_id,
        "code_id": code_id, "as_of": candles[-1].at.isoformat(),
        "horizon_hours": HORIZON, "direction": direction,
        "past24_direction_baseline": past_direction,
        "neutrality_absolute_price_threshold": threshold,
        "price_at_commitment": close,
        "history_sha256": _fingerprint(closed_history),
        "closed_history_count": len(closed_history),
        "tick_size": tick_size,
        "no_trade": True,
        "future_seen_by_this_function": False,
    }
    return {**core, "commitment_sha256": _fingerprint(core),
            "trade_action": None, "execution_allowed": False}


def settle_prospective_call(
    sealed: dict[str, Any],
    full_history_after_future: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score an IMMUTABLE prior commitment; do not call/tune the model.

    Requires the original *exact* prefix and next 24 full hourly bars.
    Any mutated OHLC candle or post-hoc revised prediction is rejected.
    """
    if not isinstance(sealed, dict):
        raise ValueError("Commitment must be a structured record")
    core = {k: v for k, v in sealed.items()
            if k not in {"commitment_sha256", "trade_action",
                         "execution_allowed"}}
    if (sealed.get("commitment_sha256") != _fingerprint(core) or
            sealed.get("trade_action", "INVALID") is not None or
            sealed.get("execution_allowed") is not False or
            core.get("schema") != SCHEMA or
            core.get("horizon_hours") != HORIZON or
            core.get("direction") not in _ALLOWED or
            core.get("no_trade") is not True):
        raise ValueError("Sealed decision was modified or unsupported")
    _validated_identity(core.get("model_id"), core.get("code_id"))
    length = core.get("closed_history_count")
    if not isinstance(length, int) or isinstance(length,bool) or length < 220:
        raise ValueError("Invalid sealed prefix length")
    if not isinstance(full_history_after_future, list) or (
            len(full_history_after_future) < length + HORIZON):
        raise ValueError("Future observation not mature; do not score early")
    observed_cutoff = full_history_after_future[length-1].get("close_time")
    if _utc(observed_cutoff) != _utc(core["as_of"]):
        raise ValueError("Cutoff no longer agrees with original observation")
    # Recheck every bar on the commitment prefix and exactly 24 FUTURE H1
    # closes, including chronological gaps and bar-finality.
    truncated = full_history_after_future[:length+HORIZON]
    candles, gaps = validate_closed_bars(
        truncated, "H1", truncated[-1]["close_time"])
    if gaps or len(candles) != length+HORIZON:
        raise ValueError("Gapped or malformed observed future")
    for i in range(length, length+HORIZON):
        if candles[i].at-candles[i-1].at != timedelta(hours=1):
            raise ValueError("Future hours must be contiguous and closed")
    if candles[length+HORIZON-1].at != _utc(core["as_of"]) + timedelta(hours=HORIZON):
        raise ValueError("Maturity exactly 24 H1 closes after sealed cutoff required")
    if _fingerprint(full_history_after_future[:length]) != core["history_sha256"]:
        raise ValueError("Historical prefix changed since model commitment")
    if float(core["price_at_commitment"]) != candles[length-1].close:
        raise ValueError("Close at commitment altered")
    new_close = candles[length+HORIZON-1].close
    delta = new_close - candles[length-1].close
    threshold = float(core["neutrality_absolute_price_threshold"])
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid sealed threshold")
    realized = ("UP" if delta > threshold else
                "DOWN" if delta < -threshold else "NEUTRAL")
    prediction = core["direction"]
    baseline = core["past24_direction_baseline"]
    outcome = {
        "mode": "SEALED_24H_DIRECTIONAL_EXPERIMENT_NOT_TRADING",
        "symbol": core["symbol"], "as_of": core["as_of"],
        "model_id": core["model_id"], "code_id": core["code_id"],
        "commitment_sha256": sealed["commitment_sha256"],
        "future_confirmed_at": candles[-1].at.isoformat(),
        "predicted_direction": prediction,
        "observed_24h_direction": realized,
        "baseline_direction": baseline,
        "called": prediction != "ABSTAIN",
        "non_neutral": realized != "NEUTRAL",
        "correct": prediction == realized if prediction != "ABSTAIN" and realized != "NEUTRAL" else None,
        "baseline_correct_on_selected": baseline == realized if prediction != "ABSTAIN" and realized != "NEUTRAL" else None,
        "trade_action": None, "execution_allowed": False,
        "actual_exchange_fills": None,
        "risk_adjusted_profitability": None,
    }
    return outcome


def summarize_sealed_results(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Give both selective accuracy AND coverage; don't game it by abstention.

    Correlated crypto pairs and repeated H1 observations cannot be treated
    as independent statistical trials. No one-shot qualification of trading.
    """
    if not isinstance(outcomes, list) or not outcomes:
        raise ValueError("Need settled future observations")
    seen = set()
    called = 0
    scored = 0
    hit = 0
    base = 0
    always_up = 0
    always_down = 0
    nonneutral = 0
    days = set()
    model_id = None
    code_id = None
    for x in outcomes:
        if (not isinstance(x, dict) or
                x.get("mode") != "SEALED_24H_DIRECTIONAL_EXPERIMENT_NOT_TRADING" or
                x.get("execution_allowed") is not False or
                x.get("trade_action", "INVALID") is not None or
                x.get("observed_24h_direction") not in {"UP","DOWN","NEUTRAL"} or
                x.get("predicted_direction") not in _ALLOWED):
            raise ValueError("Malformed outcome")
        if model_id is None:
            model_id=x.get("model_id")
            code_id=x.get("code_id")
            _validated_identity(model_id,code_id)
        elif x.get("model_id")!=model_id or x.get("code_id")!=code_id:
            raise ValueError("Cannot mix model or source versions in one cohort")
        key=(x["symbol"], x["as_of"])
        if key in seen:
            raise ValueError("Duplicate historical observation in evaluation")
        seen.add(key)
        days.add(_utc(x["as_of"]).date().isoformat())
        if x["observed_24h_direction"]!="NEUTRAL":
            nonneutral+=1
        if x["called"]:
            called+=1
            if x["non_neutral"]:
                scored+=1
                hit+=bool(x["correct"])
                base+=bool(x["baseline_correct_on_selected"])
                always_up+=x["observed_24h_direction"]=="UP"
                always_down+=x["observed_24h_direction"]=="DOWN"
    return {
        "mode": "PROSPECTIVE_EVIDENCE_ONLY_NOT_PROFITABILITY",
        "total_cases": len(outcomes),
        "model_id": model_id,"code_id":code_id,
        "unique_symbols": len({x["symbol"] for x in outcomes}),
        "unique_decision_calendar_dates_not_independent": len(days),
        "directional_calls": called, "abstentions": len(outcomes)-called,
        "non_neutral_future_cases": nonneutral,
        "called_non_neutral_cases": scored,
        "directional_coverage_fraction": round(called/len(outcomes), 6),
        "called_accuracy": round(hit/scored, 6) if scored else None,
        "matched_baseline_accuracy": round(base/scored, 6) if scored else None,
        "always_up_baseline_on_selected":round(always_up/scored,6) if scored else None,
        "always_down_baseline_on_selected":round(always_down/scored,6) if scored else None,
        "best_matched_baseline_accuracy":(
            round(max(base,always_up,always_down)/scored,6) if scored else None),
        "beats_lagged_momentum_on_selected_cases": hit>base if scored else None,
        "beats_all_declared_baselines_on_selected":(
            hit>max(base,always_up,always_down) if scored else None),
        "further_independent_prospective_holdout_needed": True,
        "cross_market_dependence_not_adjusted": True,
        "real_trade_profitability_verified": False,
        "live_trade_qualification": "NOT_QUALIFIED",
        "trade_action": None, "execution_allowed": False,
    }
