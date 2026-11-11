"""Source-independent, strictly causal OHLCV cross-timeframe integrity checks.

A chart labeled CLOSED is not enough: H1/H4/D1 must close on the same UTC
exchange candle grid and agree on all fully observable higher timeframe
candles. Without this, a Qwen analyst may learn mutually contradictory
multi-timeframe states even though each separate price history looks valid.

This detects corrupted/misaligned datasets; it does not authenticate exchange
origin, independently verify expert wave labels, or authorize trading.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math

from .chart_reader import TF_SECONDS, Candle

_TIMEFRAMES=("H1","H4","D1")
_EPOCH=datetime(1970,1,1,tzinfo=timezone.utc)


def _boundary_milliseconds(close: datetime,timeframe: str) -> int:
    if close.tzinfo is None or timeframe not in _TIMEFRAMES:
        raise ValueError("UTC_CLOSED_BAR_GRID_REQUIRED")
    instant=close.astimezone(timezone.utc)
    diff=instant-_EPOCH
    millis=(diff.days*86400000+diff.seconds*1000+
            diff.microseconds//1000)
    if diff.microseconds%1000:
        raise ValueError("SUBMILLISECOND_CLOSE_GRID_NOT_SUPPORTED")
    period=TF_SECONDS[timeframe]*1000
    # Both normalized end-boundary timestamps and native Binance closeTime
    # (exactly one millisecond before the *next* candle opens) are valid.
    remainder=millis%period
    if remainder==0:
        return millis
    if remainder==period-1:
        return millis+1
    raise ValueError("MISALIGNED_EXCHANGE_CANDLE_UTC_CLOSE_GRID")


def _close_enough(actual: float,expected: float) -> bool:
    return math.isclose(actual,expected,rel_tol=1e-7,abs_tol=1e-8)


def _matches_aggregation(observed:Candle,parts:list[Candle]) -> bool:
    expected={"open":parts[0].open,"high":max(c.high for c in parts),
              "low":min(c.low for c in parts),"close":parts[-1].close,
              "volume":sum(c.volume for c in parts)}
    for key in ("open","high","low","close"):
        if not _close_enough(float(getattr(observed,key)),expected[key]):
            return False
    # Binance OHLCV historical data may be parsed with float precision;
    # tolerance is only for that precision, not a market discrepancy.
    return math.isclose(observed.volume,expected["volume"],
                        rel_tol=1e-6,abs_tol=1e-6)


def verify_closed_timeframe_coherence(closed:dict[str,list[Candle]]) -> dict:
    """Verify UTC-aligned and mutually consistent fully covered OHLCV windows.

    Returns counts only; no trusted gold labels. Recomputes parent bars from
    lower-TF closed bars using timestamps already visible at the cutoff.
    """
    if not isinstance(closed,dict) or set(closed)!=set(_TIMEFRAMES):
        raise ValueError("EXACT_THREE_CLOSED_TIMEFRAMES_REQUIRED")
    grids={}
    for tf in _TIMEFRAMES:
        bars=closed[tf]
        if not isinstance(bars,list) or len(bars)<220 or any(
                not isinstance(bar,Candle) for bar in bars):
            raise ValueError("AT_LEAST_220_VALID_CLOSED_BARS_REQUIRED")
        aligned=[_boundary_milliseconds(c.at,tf) for c in bars]
        period=TF_SECONDS[tf]*1000
        if any(b-a!=period for a,b in zip(aligned,aligned[1:])):
            raise ValueError("NONCONTIGUOUS_EXCHANGE_TIMEFRAME_GRID")
        grids[tf]=dict(zip(aligned,bars))
    counts={}
    for child,parent,minimum in (("H1","H4",10),("H4","D1",5)):
        ratio=TF_SECONDS[parent]//TF_SECONDS[child]
        step=TF_SECONDS[child]*1000
        exact=0
        for parent_end,parent_candle in grids[parent].items():
            stamps=[parent_end-step*(ratio-1-i) for i in range(ratio)]
            if not all(t in grids[child] for t in stamps):
                # Incomplete coverage at the beginning of a 220-bar sample
                # cannot be treated as verified evidence.
                continue
            lower=[grids[child][stamp] for stamp in stamps]
            if not _matches_aggregation(parent_candle,lower):
                raise ValueError(
                    "CROSS_TIMEFRAME_OHLCV_CONTRADICTION_"+child+"_TO_"+parent)
            exact+=1
        if exact<minimum:
            raise ValueError("INSUFFICIENT_OVERLAPPING_CLOSED_TIMEFRAME_EVIDENCE")
        counts[child+"_TO_"+parent]=exact
    return {"status":"INTERNAL_OHLCV_TIMEFRAME_CONSISTENCY_ONLY",
            "verified_complete_windows":counts,
            "utc_candle_closes_valid":True,
            "original_exchange_bytes_attested":False,
            "wave_labels_expert_verified":False,
            "execution_allowed":False}
