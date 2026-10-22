"""A small trainable local research scorer, NOT a Williams execution strategy.

Features use closed prefixes only. Labels simulate a next-open entry and fixed
protective stop, then mark-to-market at a research horizon. Horizon valuation
is NOT a trading exit rule. Chronological splits purge overlapping labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

from .chart_reader import Candle, indicators, validate_closed_bars

FEATURES = ("bias", "side_ao_atr", "side_ac_atr", "mouth_width_atr", "side_price_lips_atr", "atr_fraction")


@dataclass(frozen=True)
class ResearchSample:
    decision_index: int
    resolved_index: int
    decision_at: str
    resolved_at: str
    side: str
    features: tuple[float, ...]
    net_return: float
    stopped: bool


def _finite(value: float, low: float, high: float) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and low <= value <= high


def paper_samples(rows: list[dict[str, Any]], timeframe: str, *, horizon: int = 12,
                  fee_bps: float = 4, slippage_bps: float = 2,
                  stop_atr: float = 2) -> list[ResearchSample]:
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= 100:
        raise ValueError("Invalid horizon")
    if not (_finite(fee_bps, 0, 100) and _finite(slippage_bps, 0, 100) and _finite(stop_atr, 0.1, 10)):
        raise ValueError("Invalid costs or stop distance")
    if not rows:
        raise ValueError("History is empty")
    bars, gaps = validate_closed_bars(rows, timeframe, rows[-1]["close_time"])
    if gaps or len(bars) < 100:
        raise ValueError("Need at least 100 consecutive closed candles")
    lines = indicators(bars)  # All constituent rolling formulas are causal.
    out = []
    slip, fee = slippage_bps / 10000, fee_bps / 10000
    for index in range(45, len(bars) - horizon, horizon):
        recent = bars[index - 13:index + 1]
        atr = sum(max(b.high - b.low, abs(b.high - bars[j - 1].close), abs(b.low - bars[j - 1].close))
                  for j, b in zip(range(index - 13, index + 1), recent)) / 14
        if atr <= 0:
            continue
        # Research examples in both directions; not a claim of a valid WM setup.
        for side, sign in (("LONG", 1), ("SHORT", -1)):
            op = bars[index + 1].open
            entry = op * (1 + sign * slip)
            stop = entry - sign * stop_atr * atr
            if stop <= 0:
                continue
            resolved = index + horizon
            exit_price = bars[resolved].close
            stopped = False
            for j in range(index + 1, resolved + 1):
                bar = bars[j]
                if (sign == 1 and bar.low <= stop) or (sign == -1 and bar.high >= stop):
                    # A gap through the stop receives the worse open fill.
                    exit_price = min(stop, bar.open) if sign == 1 else max(stop, bar.open)
                    resolved, stopped = j, True
                    break
            exit_price *= 1 - sign * slip
            net = sign * (exit_price - entry) / entry - fee * (1 + exit_price / entry)
            ao, ac, jaw, lips = (lines[k][index] for k in ("ao", "ac", "jaw", "lips"))
            clip = lambda v: max(-8.0, min(8.0, v))
            values = (1.0, clip(sign * ao / atr), clip(sign * ac / atr),
                      clip(abs(jaw - lips) / atr), clip(sign * (bars[index].close - lips) / atr),
                      clip(atr / bars[index].close * 100))
            out.append(ResearchSample(index, resolved, bars[index].at.isoformat(),
                       bars[resolved].at.isoformat(), side, values, net, stopped))
    return out


def chronological_split(samples: list[ResearchSample]) -> tuple[list[ResearchSample], ...]:
    indices = sorted({s.decision_index for s in samples})
    if len(indices) < 15:
        raise ValueError("Insufficient independent decision times: need at least 15")
    validation_start = indices[int(len(indices) * 0.6)]
    test_start = indices[int(len(indices) * 0.8)]
    train = [s for s in samples if s.decision_index < validation_start and s.resolved_index < validation_start]
    valid = [s for s in samples if validation_start <= s.decision_index < test_start and s.resolved_index < test_start]
    test = [s for s in samples if s.decision_index >= test_start]
    if not all((train, valid, test)):
        raise ValueError("Empty purged split")
    return train, valid, test


def _probability(weights: list[float], features: tuple[float, ...]) -> float:
    z = max(-35.0, min(35.0, sum(a * b for a, b in zip(weights, features))))
    return 1 / (1 + math.exp(-z))


def fit(samples: list[ResearchSample], *, epochs: int = 120) -> dict[str, Any]:
    """Fit only on the training prefix. No threshold/epoch search on holdouts."""
    if not isinstance(epochs, int) or not 1 <= epochs <= 500 or not samples:
        raise ValueError("Invalid training request")
    weights = [0.0] * len(FEATURES)
    for _ in range(epochs):
        gradient = [0.0] * len(weights)
        for sample in samples:
            error = _probability(weights, sample.features) - (sample.net_return > 0)
            for j, x in enumerate(sample.features):
                gradient[j] += error * x
        for j in range(len(weights)):
            weights[j] -= 0.04 * (gradient[j] / len(samples) + (0.01 * weights[j] if j else 0))
    return {"algorithm": "logistic_regression_v1", "features": list(FEATURES),
            "weights": weights, "train_count": len(samples),
            "trained_through": max(s.resolved_at for s in samples)}


def _metrics(model: dict[str, Any], samples: list[ResearchSample]) -> dict[str, Any]:
    probabilities = [_probability(model["weights"], s.features) for s in samples]
    selected = [s for s, p in zip(samples, probabilities) if p >= 0.6]
    return {"samples": len(samples), "decision_times": len({s.decision_index for s in samples}),
            "brier_score": sum((p - (s.net_return > 0)) ** 2 for s, p in zip(samples, probabilities)) / len(samples),
            "selected_at_fixed_0_6": len(selected),
            "mean_selected_net_return": sum(s.net_return for s in selected) / len(selected) if selected else None,
            "constant_prior_brier": sum((model["training_positive_rate"] - (s.net_return > 0)) ** 2 for s in samples) / len(samples)}


def learn_from_history(rows: list[dict[str, Any]], timeframe: str, **kwargs) -> dict[str, Any]:
    samples = paper_samples(rows, timeframe, **kwargs)
    train, validation, test = chronological_split(samples)
    model = fit(train)
    model["training_positive_rate"] = sum(s.net_return > 0 for s in train) / len(train)
    valid_metrics = _metrics(model, validation)
    # Never publish an execution permission, even with favorable validation.
    return {"status": "EXPERIMENTAL_RESEARCH_ONLY", "model": model,
            "validation": valid_metrics, "test": _metrics(model, test),
            "dataset_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            "configuration": {"timeframe": timeframe, "horizon": 12, "fee_bps": 4,
                              "slippage_bps": 2, "stop_atr": 2, **kwargs},
            "trade_action": None, "execution_allowed": False,
            "limitations": ["Research examples are not verified Williams entries.",
              "Horizon mark-to-market is a label, not an authorized campaign exit.",
              "Funding, order book, partial fills and portfolio exposure are not simulated.",
              "Repeated use of the test set invalidates it as an untouched holdout.",
              "A favorable score is not evidence of future profitability."]}
