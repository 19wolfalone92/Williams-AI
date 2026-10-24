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

# The 15 native Binance futures candle intervals are available to the
# same CLOSED-BAR research engine. Month length varies 28-31 UTC days;
# source ingestion separately verifies exact calendar open/close boundaries.
TF_SECONDS = {
    "M1":60,"M3":180,"M5":300,"M15":900,"M30":1800,
    "H1":3600,"H2":7200,"H4":14400,"H6":21600,"H8":28800,
    "H12":43200,"D1":86400,"D3":259200,"W1":604800,"MN1":2678400,
}


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
    """Audit the WM3 fractal lifecycle using ONLY closed historical bars.

    TC2 ch.11 (PDF 147-151) distinguishes formation from the price trigger:
    Teeth must be checked when price crosses the trigger, not only when the
    pivot or confirming bar formed. An OHLC touch is NOT an exchange fill.
    The most recent confirmed same-side fractal supersedes an older signal.
    """
    formations: list[tuple[int, str, float, bool]] = []
    for pivot in range(2, len(bars) - 2):
        window = bars[pivot - 2:pivot + 3]
        neighbors = [c for j, c in enumerate(window) if j != 2]
        high = bars[pivot].high > max(c.high for c in neighbors)
        low = bars[pivot].low < min(c.low for c in neighbors)
        # Outside bars can be both high and low fractals; do not silently
        # convert an ambiguous two-direction formation into a LONG signal.
        if high:
            formations.append((pivot, "LONG", bars[pivot].high, low))
        if low:
            formations.append((pivot, "SHORT", bars[pivot].low, high))

    out: list[dict[str, Any]] = []
    for pivot, side, level, ambiguous in formations:
        sign = 1 if side == "LONG" else -1
        confirmed = pivot + 2
        trigger = (round(level + sign * tick_size, 10) if tick_size else None)
        threshold = trigger if trigger is not None else level
        teeth_at_confirmation = lines["teeth"][confirmed]
        at_confirmation = (
            teeth_at_confirmation is not None and
            sign * (level - teeth_at_confirmation) > 0
        )
        state = "AMBIGUOUS_OUTSIDE_BAR" if ambiguous else "AWAITING_PRICE_TRIGGER"
        price_touch_at = None
        teeth_at_trigger = None
        trigger_filter = None
        newer_confirmed_at = None

        if not ambiguous:
            for index in range(confirmed + 1, len(bars)):
                bar = bars[index]
                touched = (bar.high >= threshold if trigger is not None else bar.high > level) if sign == 1 else (
                    bar.low <= threshold if trigger is not None else bar.low < level
                )
                # Another confirmed fractal of the SAME direction replaces
                # the older pending setup. A tie on the SAME OHLC candle has
                # unknowable intrabar ordering: refuse to classify it as a fill.
                newer = [
                    new_pivot + 2 for new_pivot, new_side, _, _ in formations
                    if new_side == side and new_pivot > pivot and new_pivot + 2 == index
                ]
                if touched and newer:
                    state = "INTRABAR_ORDER_UNKNOWN_TOUCH_AND_SUPERSESSION"
                    newer_confirmed_at = bar.at.isoformat()
                    price_touch_at = bar.at.isoformat()
                    break
                if newer:
                    state = "SUPERSEDED_BY_NEWER_CONFIRMED_FRACTAL"
                    newer_confirmed_at = bar.at.isoformat()
                    break
                if touched:
                    price_touch_at = bar.at.isoformat()
                    teeth_at_trigger = lines["teeth"][index]
                    trigger_filter = (
                        teeth_at_trigger is not None and
                        sign * (threshold - teeth_at_trigger) > 0
                    )
                    state = (
                        "CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL"
                        if trigger_filter else "CLOSED_BAR_TOUCH_REJECTED_BY_TEETH"
                    )
                    break

        out.append({
            "type": "WM3_FRACTAL", "side": side,
            "pivot_time": bars[pivot].at.isoformat(),
            "confirmed_at": bars[confirmed].at.isoformat(),
            "level": level, "entry_trigger": trigger,
            "state": state,
            "teeth_filter_at_confirmation": at_confirmation,
            "teeth_filter_at_trigger": trigger_filter,
            "teeth_visible_at_trigger": teeth_at_trigger,
            "price_touch_observed_at": price_touch_at,
            "newer_fractal_confirmed_at": newer_confirmed_at,
            "already_broken_on_closed_bars": price_touch_at is not None,
            "can_prove_exchange_fill": False,
            "source": "TC2 ch.11 PDF 147-151; NTD ch.4 has a separate first-entry gate",
            "warning": (
                "OHLC evidence cannot prove order acceptance or intrabar ordering; "
                "a pending setup is not a validated order."
            ),
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
    from .market_position import market_position
    waves=wave_context(candles,lines["ao"])
    result["wave_structure"]=waves
    result["market_position"]=market_position(candles,lines["ao"])
    # Principle-first interpretation MUST come from closed-price / indicator
    # relationships at several scales, not the shape of example book images.
    # This does not certify the Elliott count or authorize an order.
    from .nonlinear_market_principles import nonlinear_market_principles
    result["principle_evidence"]=nonlinear_market_principles(
        candles,lines["ao"],lines["ac"],alligator_context=phase,
        diverging_evidence=result["market_position"])
    from .five_magic_bullets import five_magic_bullets
    # Source: TC1 pp.53-54,82-84, *distinct* from NTD five dimensions.
    # Binance Futures kline base volume is not the same as TC1 tick volume.
    result["tc1_five_magic_bullets"]=five_magic_bullets(
        candles,lines["ao"],data=bars,timeframe=timeframe)
    # NTD remains an independent, observation-only source profile. A NTD
    # entry requires a separate proven initial outside-mouth fractal breakout
    # and must never piggyback TC2's Wise Men first-entry policy.
    from .tc1_fractals import tc1_initiating_fractal_evidence
    result["edition_evidence"]={
        "TC1": (
            tc1_initiating_fractal_evidence(candles)
            if not gaps else
            {"profile":"TC1_INITIATING_FRACTALS","recent_formations":[],
             "trade_action":None,"execution_allowed":False,
             "status":"GAPPED_HISTORY_NOT_ELIGIBLE"}
        ),
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
    from .signal_proofs import current_signal_proofs
    # Verify currently observable signals independently of pictures or LLM.
    result["signal_proofs"]=current_signal_proofs(result)
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
