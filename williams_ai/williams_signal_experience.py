"""Outcome feedback on causal TC2 *approximate* WM1/WM2/WM3 candidates.

This is a genuinely signal-conditioned learning dataset, not a generic
predict-next-close bot. At closed candle t form a candidate using only past
indicators. At t+1 (unseen at decision) score whether its trigger was reached,
whether the structural SL was hit, and net hypothetical return after costs.
No signal -> NO positive trading examples; no retrospective market fill claims.

Historical OHLC bar touches are not authenticated Binance trades.
WM1 angulation and WM2/WM3 stop conventions are approximations and require
book-chart expert review before anything could be called TC2 exact trading.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from typing import Any

from .causal_paper_lab import _candidate
from .chart_reader import validate_closed_bars, indicators
from .continual_trader import Outcome


def _clip(z: float) -> float:
    return max(-8.0, min(8.0, z))


def _vector(bars: list[Any], lines: dict[str, list[Any]], i: int, sign: int
            ) -> tuple[float, ...] | None:
    prev = bars[i - 1]
    if i < 45 or bars[i].close <= 0:
        return None
    atr = 0.0
    for j in range(i - 13, i + 1):
        price = bars[j]
        prior = bars[j - 1].close
        atr += max(price.high - price.low,
                   abs(price.high - prior), abs(price.low - prior))
    atr /= 14.0
    if atr <= 0 or not math.isfinite(atr):
        return None
    ao, ac, jaw, lips = [lines[k][i] for k in ("ao", "ac", "jaw", "lips")]
    if any(v is None or not math.isfinite(v) for v in (ao, ac, jaw, lips)):
        return None
    return (1.0, _clip(sign * ao / atr), _clip(sign * ac / atr),
            _clip(abs(jaw - lips) / atr),
            _clip(sign * (bars[i].close - lips) / atr),
            _clip(100 * atr / bars[i].close))


def signal_conditioned_outcomes(
    symbol: str, rows: list[dict[str, Any]], interval: str, *,
    tick_size: float, horizon: int = 12,
    fee_bps: float = 4.0, slippage_bps: float = 2.0,
    start_index: int = 80, max_cases: int = 200_000,
) -> dict[str, Any]:
    if interval not in {"H1","H4","D1","M15"}:
        raise ValueError("No trained book signal on unsupported TF")
    if (not isinstance(horizon,int) or isinstance(horizon,bool)
            or not 1 <= horizon <= 48
            or not isinstance(start_index,int) or isinstance(start_index,bool)
            or start_index < 45
            or not isinstance(max_cases,int) or isinstance(max_cases,bool)
            or not 1 <= max_cases <= 2_000_000):
        raise ValueError("Invalid causal trial budget")
    for x in (tick_size, fee_bps, slippage_bps):
        if (isinstance(x,bool) or not isinstance(x,(float,int))
                or not math.isfinite(x) or x < 0):
            raise ValueError("Nonfinite exchange risk or friction")
    if tick_size <= 0 or fee_bps > 100 or slippage_bps > 100:
        raise ValueError("Risk/friction out of fixed research bounds")
    if not rows:
        raise ValueError("Empty real closed price history")
    bars, gaps = validate_closed_bars(rows, interval, rows[-1]["close_time"])
    if gaps:
        raise ValueError("GAPPED_HISTORY_BLOCKS_TRADE_EXPERIENCES")
    ind = indicators(bars)
    # A growing live market archive must NOT invalidate past already
    # closed/settled research records. Prefix SHA-256 at each resolution
    # excludes any bars that are added later after an earlier outcome.
    config = json.dumps({
        "horizon":horizon,"tick_size":tick_size,"fee_bps":fee_bps,
        "slippage_bps":slippage_bps,"assumption":"NEXT_BAR_TRIGGER_STOP_FIRST",
    },sort_keys=True,separators=(",", ":"))
    chain = hashlib.sha256(config.encode("utf8"))
    prefix_hashes = []
    for row in rows:
        chain.update(json.dumps(row,sort_keys=True,
                                separators=(",", ":")).encode("utf8"))
        chain.update(b"\\n")
        prefix_hashes.append(chain.hexdigest())
    source_hash = prefix_hashes[-1]
    result: list[Outcome] = []
    fired: dict[str, int] = {"WM1":0,"WM2":0,"WM3":0}
    missed = 0
    invalid = 0
    candidate_count = 0
    for i in range(start_index, len(bars) - horizon - 1):
        for setup in ("WM1","WM2","WM3"):
            # Existing legacy deterministic approximation; NOT local GGUF.
            signals = _candidate(bars, ind, setup, i, tick_size)
            for sig in signals:
                candidate_count += 1
                side = sig["side"]
                sign = 1 if side == "LONG" else -1
                feat = _vector(bars, ind, i, sign)
                if feat is None:
                    invalid += 1
                    continue
                following = bars[i + 1]
                trigger = sig["trigger"]
                reached = (following.high >= trigger if sign == 1
                           else following.low <= trigger)
                if not reached:
                    missed += 1
                    continue
                # Teeth at signal close, rather than the following candle,
                # can be known at t and cannot leak t+1 future price data.
                if setup == "WM3":
                    teeth = ind["teeth"][i]
                    if teeth is None or (trigger <= teeth if sign == 1
                                        else trigger >= teeth):
                        invalid += 1
                        continue
                entry = (max(following.open, trigger) if sign == 1
                         else min(following.open, trigger))
                entry *= 1 + sign * slippage_bps / 10_000.0
                stop = sig["stop"]
                if sign * (entry - stop) <= tick_size or stop <= 0:
                    invalid += 1
                    continue
                # Intrabar both stop+entry is ambiguous -> stop first,
                # pessimistically. No same-bar future sequence inference.
                stopped = False
                end_idx = i + horizon
                exit_price = bars[end_idx].close
                for k in range(i + 1, i + horizon + 1):
                    b = bars[k]
                    if (b.low <= stop if sign == 1 else b.high >= stop):
                        exit_price = (min(stop, b.open) if sign == 1
                                      else max(stop, b.open))
                        stopped = True
                        end_idx = k
                        break
                exit_price *= 1 - sign * slippage_bps / 10_000.0
                net = sign * (exit_price - entry) / entry
                net -= (fee_bps / 10_000.0) * (1 + exit_price / entry)
                outcome = Outcome(
                    symbol, interval, side,
                    bars[i].at.isoformat(), bars[end_idx].at.isoformat(),
                    feat, float(net), stopped, prefix_hashes[end_idx],
                    setup=setup, rule_id=sig["source_rule"])
                result.append(outcome)
                fired[setup] += 1
                if len(result) >= max_cases:
                    break
            if len(result) >= max_cases:
                break
        if len(result) >= max_cases:
            break
    return {
        "mode": "CAUSAL_TC2_PROXY_OUTCOME_LEARNING",
        "symbol": symbol, "timeframe": interval,
        "source_sha256": source_hash,
        "research_approximated_wm_candidates": candidate_count,
        "valid_hypothetical_next_bar_triggers": len(result),
        "candidates_no_next_bar_trigger": missed,
        "other_rejections": invalid,
        "setup_counts": fired,
        "outcomes": result,
        "all_outcomes_resolved_at_or_before": rows[-1]["close_time"],
        "exchange_fills_observed": 0,
        "canonical_4book_wave_interpretations_verified": False,
        "real_gguf_decisions_made": 0,
        "execution_allowed": False, "trade_action": None,
    }
