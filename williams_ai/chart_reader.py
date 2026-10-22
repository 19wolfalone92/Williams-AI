"""Causal OHLCV 'chart eyes' for Williams AI, *not* discretionary visual intelligence.

The analyzer sees market price structures as a trader would inspect them:
closed price bars, visible displaced Alligator, AO/AC, zone and fractals.
Visual WM1 angulation and Elliott labels remain provisional hypotheses.
It cannot send orders. No future candle can confirm a past decision.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any

TF_SECONDS = {"M5":300, "M15":900, "H1":3600, "H4":14400, "D1":86400}


@dataclass(frozen=True)
class Candle:
    at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("close_time must be ISO-8601 with timezone")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid close_time") from exc
    if result.tzinfo is None:
        raise ValueError("Every bar and as_of must have a timezone")
    return result.astimezone(timezone.utc)


def _price(value: Any, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("OHLCV values must be finite numbers")
    n = float(value)
    if not isfinite(n) or (n < 0 if allow_zero else n <= 0):
        raise ValueError("OHLCV contains non-finite or nonpositive price")
    return n


def validate_closed_bars(bars: list[dict[str, Any]], timeframe: str,
                         as_of: str) -> tuple[list[Candle], int]:
    if timeframe not in TF_SECONDS:
        raise ValueError("Unsupported timeframe")
    if not isinstance(bars, list) or len(bars) > 50000:
        raise ValueError("Provide an array of at most 50,000 closed candles")
    cutoff = _time(as_of)
    out: list[Candle] = []
    gaps = 0
    prev: datetime | None = None
    for row in bars:
        if not isinstance(row, dict) or row.get("closed") is not True:
            raise ValueError("Only explicitly finalized candles are eligible")
        at = _time(row.get("close_time"))
        if at > cutoff:
            raise ValueError("NO_LOOKAHEAD: future or unfinished candle rejected")
        if prev is not None:
            delta = (at - prev).total_seconds()
            if delta <= 0:
                raise ValueError("Duplicate or out-of-order candles")
            if delta > TF_SECONDS[timeframe] * 1.5:
                gaps += 1
        op, hi, lo, cl = (_price(row.get(k)) for k in ("open","high","low","close"))
        vol = _price(row.get("volume", 0), allow_zero=True)
        if lo > min(op, cl) or hi < max(op, cl) or lo > hi:
            raise ValueError("Invalid OHLC range")
        out.append(Candle(at, op, hi, lo, cl, vol))
        prev = at
    return out, gaps


def _sma(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= period:
            total -= values[i - period]
        if i >= period - 1:
            result[i] = total / period
    return result


def _smma(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if len(values) < period:
        return result
    smoothed = sum(values[:period]) / period  # engineering SMA seed
    result[period - 1] = smoothed
    for i in range(period, len(values)):
        smoothed = (smoothed * (period - 1) + values[i]) / period
        result[i] = smoothed
    return result


def _visible_shift(values: list[float | None], shift: int) -> list[float | None]:
    # Plot offset pushes prior smoothed values FORWARD on the graph.
    # At candle index i the visible line is computed from index i - shift,
    # never from index i + shift (forbidden look-ahead).
    return [values[i - shift] if i >= shift else None
            for i in range(len(values))]


def indicators(bars: list[Candle]) -> dict[str, list[float | None]]:
    med = [(x.high + x.low) / 2 for x in bars]
    a5 = _sma(med, 5)
    a34 = _sma(med, 34)
    ao = [a - b if a is not None and b is not None else None
          for a,b in zip(a5, a34)]
    ac_window = _sma([v or 0.0 for v in ao], 5)
    ac = [v - ac_window[i] if v is not None and i >= 37
          else None for i,v in enumerate(ao)]
    return {
        "jaw": _visible_shift(_smma(med,13),8),
        "teeth": _visible_shift(_smma(med,8),5),
        "lips": _visible_shift(_smma(med,5),3),
        "ao": ao,
        "ac": ac,
    }


def _angle_candidate(bars: list[Candle], lines: dict[str, list[float | None]],
                     direction: str) -> dict[str, Any] | None:
    i = len(bars) - 1
    if i < 42: return None
    now, before = bars[-1], bars[-2]
    jaw, teeth, lips = (lines[k][-1] for k in ("jaw","teeth","lips"))
    oldjaw = lines["jaw"][-6]
    if any(x is None for x in (jaw, teeth, lips, oldjaw)):
        return None
    midpoint = (now.low + now.high) / 2
    if direction == "LONG":
        bar_pattern = now.low < before.low and now.close > midpoint
        beyond = now.high < min(jaw,teeth,lips)
        edge_slope = (now.low - bars[-6].low) / 5
        previous_context = bars[-6].close > bars[-2].close
        mouth_distance_expanding = abs(now.low-jaw) > abs(bars[-5].low-lines["jaw"][-5])
    else:
        bar_pattern = now.high > before.high and now.close < midpoint
        beyond = now.low > max(jaw,teeth,lips)
        edge_slope = (now.high - bars[-6].high) / 5
        previous_context = bars[-6].close < bars[-2].close
        mouth_distance_expanding = abs(now.high-jaw) > abs(bars[-5].high-lines["jaw"][-5])
    mouth_slope = (jaw - oldjaw) / 5
    relative_angle = abs(edge_slope) > abs(mouth_slope) * 1.05
    if not (bar_pattern and beyond):
        return None
    return {
        "type": "WM1", "side": direction,
        "state": "REVERSAL_BAR_VISIBLE_ANGULATION_REVIEW_REQUIRED",
        "bar_time": now.at.isoformat(),
        "bar_extreme": now.high if direction=="LONG" else now.low,
        "initial_stop_reference": now.low if direction=="LONG" else now.high,
        "angle_heuristic": bool(relative_angle and mouth_distance_expanding and previous_context),
        "checks": {"reversal_bar":bar_pattern,"outside_mouth":beyond,
                   "previous_direction":previous_context,
                   "diverging_jaw_distance":mouth_distance_expanding,
                   "price_vs_mouth_slope":relative_angle},
        "source": "TC2: PDF 121-125, 131",
        "warning": "Line-slope approximation is NOT canonical visual angulation; human/chart verification required",
    }


def _confirmed_fractals(bars: list[Candle], lines: dict[str, list[float | None]],
                        *, tick_size: float | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pivot in range(2, len(bars)-2):
        window = bars[pivot-2:pivot+3]
        if bars[pivot].high > max(c.high for j,c in enumerate(window) if j != 2):
            side, level = "LONG", bars[pivot].high
        elif bars[pivot].low < min(c.low for j,c in enumerate(window) if j != 2):
            side, level = "SHORT", bars[pivot].low
        else:
            continue
        confirmed_at = pivot + 2
        # No triggering before two *right* closed candles.
        broken_after = any(
            (c.high > level if side=="LONG" else c.low < level)
            for c in bars[confirmed_at + 1:]
        )
        teeth_at_signal = lines["teeth"][confirmed_at]
        tooth_ok = teeth_at_signal is not None and (
            level > teeth_at_signal if side=="LONG" else level < teeth_at_signal
        )
        out.append({
            "type":"WM3_FRACTAL", "side":side,
            "pivot_time":bars[pivot].at.isoformat(),
            "confirmed_at":bars[confirmed_at].at.isoformat(),
            "level":level, "teeth_filter_at_confirmation":tooth_ok,
            "already_broken_on_closed_bars":broken_after,
            "entry_trigger":round(
                level + tick_size if side=="LONG" else level - tick_size, 10
            ) if tick_size else None,
            "state":"WAITING_REVALIDATION" if tooth_ok and not broken_after else "REJECTED_OR_STALE",
            "source":"TC2 WM3, NTD fractals (edition rules differ)",
        })
    return out


def analyze_chart(symbol: str, timeframe: str, bars: list[dict[str, Any]],
                  *, as_of: str, tick_size: float | None = None) -> dict[str, Any]:
    candles, gaps = validate_closed_bars(bars,timeframe,as_of)
    if not isinstance(symbol,str) or not symbol.endswith("USDT") or len(symbol)>20:
        raise ValueError("Unsupported market symbol")
    if tick_size is not None:
        if not isinstance(tick_size, (int,float)) or isinstance(tick_size,bool):
            raise ValueError("Tick size invalid")
        if not isfinite(tick_size) or tick_size <= 0:
            raise ValueError("Tick size invalid")
    result: dict[str,Any] = {
        "symbol":symbol,"timeframe":timeframe,"as_of":_time(as_of).isoformat(),
        "bars_used":len(candles),"last_closed_at":candles[-1].at.isoformat() if candles else None,
        "gaps":gaps,"no_lookahead":True,"execution_allowed":False,
        "evidence":{},"warnings":[],"candidates":[],
        "hypotheses":[]
    }
    if gaps:
        result["warnings"].append("Дыры в истории: кандидатам запрещено доверять без восстановления свечей.")
    if len(candles)<45:
        result["warnings"].append("Не хватает 45 закрытых свечей для всех индикаторов.")
        return result
    lines=indicators(candles)
    last={key:line[-1] for key,line in lines.items()}
    ao_prev=lines["ao"][-2]
    ac_prev=lines["ac"][-2]
    zone = ("GREEN" if last["ao"] > ao_prev and last["ac"] > ac_prev else
            "RED" if last["ao"] < ao_prev and last["ac"] < ac_prev else "GREY")
    lips,teeth,jaw=(last[k] for k in ("lips","teeth","jaw"))
    phase = ("UP_CONTEXT" if lips>teeth>jaw and candles[-1].close>lips else
             "DOWN_CONTEXT" if lips<teeth<jaw and candles[-1].close<lips else
             "MIXED_OR_TRANSITION")
    result["evidence"]={
        "ao":last["ao"],"ac":last["ac"],"zone":zone,
        "jaw_visible":jaw,"teeth_visible":teeth,"lips_visible":lips,
        "market_context":phase,
        "median_price":(candles[-1].high+candles[-1].low)/2,
        "ao_color":"GREEN" if last["ao"]>ao_prev else "RED" if last["ao"]<ao_prev else "FLAT",
        "ac_color":"GREEN" if last["ac"]>ac_prev else "RED" if last["ac"]<ac_prev else "FLAT",
        "sources":{"alligator":"NTD / TC2; visible displaced lines",
                   "ao_ac":"NTD; closed median-price candles","zone":"NTD PDF 91-93"},
    }
    if not gaps:
        result["candidates"]+=_confirmed_fractals(candles,lines,tick_size=tick_size)[-12:]
        for side in ("LONG","SHORT"):
            candidate=_angle_candidate(candles,lines,side)
            if candidate: result["candidates"].append(candidate)
        for side in ("LONG","SHORT"):
            ao=lines["ao"]
            follow = all(
                (ao[-k] > ao[-k-1] if side=="LONG" else ao[-k] < ao[-k-1])
                for k in (1,2,3)
            )
            if follow:
                result["candidates"].append({
                    "type":"WM2_SUPER_AO","side":side,
                    "state":"MOMENTUM_CANDIDATE_NOT_ORDER",
                    "price_trigger_reference":candles[-1].high if side=="LONG" else candles[-1].low,
                    "source":"TC2 WM2; independent of NTD Saucer/Twin Peaks",
                })
    else:
        result["warnings"].append("Пропускаем сигналы из-за пропущенных свечей.")
    last_ao=last["ao"]
    ao_window=[x for x in lines["ao"][-140:] if x is not None]
    if len(ao_window)>=20:
        peak=max(abs(v) for v in ao_window)
        if peak>0 and abs(last_ao)/peak<0.15:
            result["hypotheses"].append({
                "hypothesis":"Momentum returned near zero relative to visible 140-bar amplitude.",
                "possible_interpretation":"Potential correction / W4 only IF impulse degree and W3 peak are verified.",
                "status":"UNCONFIRMED_WAVE_DEGREE",
                "source":"TC1 PDF 53-54 / Course PDF 42-43",
            })
    from .wave_evidence import wave_context
    from .ntd_evidence import ntd_closed_bar_evidence
    waves=wave_context(candles,lines["ao"])
    result["wave_structure"]=waves
    # NTD remains an independent, observation-only source profile. A NTD
    # entry requires a separate proven initial outside-mouth fractal breakout
    # and must never piggyback TC2's Wise Men first-entry policy.
    result["edition_evidence"]={
        "NTD": ntd_closed_bar_evidence(
            lines["ao"],lines["ac"],initial_fractal_triggered=False
        ),
        "TC2": {"profile":"TC2_THREE_WISE_MEN",
                "candidate_count":len(result["candidates"]),
                "trade_action":None},
    }
    if waves["impulse_alternatives"]:
        result["hypotheses"].append({
            "hypothesis":"Confirmed swing pivots admit tentative impulse-shaped alternatives.",
            "alternatives": waves["impulse_alternatives"],
            "status":"UNCONFIRMED_WAVE_DEGREE",
            "source":"TC1 PDF 53-54 / Course PDF 42-43",
        })
    if waves["corrective_alternatives"]:
        result["hypotheses"].append({
            "hypothesis":"Price extrema permit unverified zigzag/flat/triangle shapes.",
            "alternatives":waves["corrective_alternatives"],
            "status":"SHAPE_ONLY_SUBWAVE_COUNTS_NOT_CONFIRMED",
            "source":"TC1 PDF 53-54; manual graph validation required",
        })
    result["warnings"].append(
        "WM1 angulation numerical check is an approximation; no verified nested Elliott wave labelling.")
    return result


def analyze_timeframes(chart: dict[str,Any]) -> dict[str,Any]:
    if not isinstance(chart,dict) or not isinstance(chart.get("timeframes"),dict):
        raise ValueError("Expected chart JSON with timeframes")
    symbol=chart.get("symbol")
    as_of=chart.get("as_of")
    analyses={}
    for tf,data in chart["timeframes"].items():
        if not isinstance(data,dict) or not isinstance(data.get("bars"),list):
            raise ValueError("Each timeframe must have a bars array")
        analyses[tf]=analyze_chart(
            symbol,tf,data["bars"],as_of=as_of,tick_size=chart.get("tick_size"))
    context=[f'{tf}: {a["evidence"].get("market_context","NO_DATA")}'
             for tf,a in analyses.items()]
    return {
        "symbol":symbol,"as_of":_time(as_of).isoformat(),
        "timeframes":analyses,"context_summary":"; ".join(context),
        "wave_hierarchy":"UNRESOLVED: requires pivot-degree alternatives and historical graphic validation",
        "trade_action":None,
        "limitations":["Timeframe alignment is context, not proof of nested Elliott degree.",
                       "All results are based only on explicitly closed bars.",
                       "Manual/screenshot chart review and verified source examples remain required."]
    }
