"""Deterministic, read-only market evidence validation and explanation.

No trading, network, API credentials, or direct imports from execution modules.
All inputs are treated as untrusted evidence rather than trading instructions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import re

TIMEFRAME_MINUTES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}
SIGNAL_NAMES = {"WM1", "WM2", "WM3", "FRACTAL", "AO", "AC"}
DIRECTIONS = {"LONG", "SHORT"}
PHASES = {"UNKNOWN", "SLEEPING", "AWAKENING", "EATING", "SATISFIED", "TREND", "CORRECTION", "RANGE"}


def _utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field}: expected ISO-8601 timestamp")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field}: invalid ISO-8601 timestamp") from exc
    if dt.tzinfo is None:
        raise ValueError(f"{field}: timezone is required")
    return dt.astimezone(timezone.utc)


def _text(value: Any, maximum: int = 180) -> str:
    if not isinstance(value, str):
        return "нет данных"
    return " ".join(value.split())[:maximum] or "нет данных"


def _number(value: Any) -> float | None:
    # Bool is a subclass of int, but not a valid indicator value.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    if not (-1e100 < val < 1e100):
        return None
    return val


def build_report(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build explainable observations without creating any trade decision.

    Only explicitly whitelisted fields are copied into the output, so raw
    exchange credentials, account IDs and arbitrary metadata cannot enter
    the report sent to an optional language model.
    """
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a JSON object")

    symbol = str(snapshot.get("symbol", "")).upper()
    if not re.fullmatch(r"[A-Z0-9]{5,24}", symbol):
        raise ValueError("symbol must be an uppercase exchange symbol")
    as_of = _utc(snapshot.get("as_of"), "as_of")
    warnings: list[str] = []
    observations: list[str] = []
    evidence: dict[str, Any] = {"timeframes": {}, "signals": [], "position": {}}

    tfs = snapshot.get("timeframes", {})
    if not isinstance(tfs, dict):
        raise ValueError("timeframes must be an object")
    for tf in TIMEFRAME_MINUTES:
        data = tfs.get(tf)
        if data is None:
            continue
        if not isinstance(data, dict):
            raise ValueError(f"timeframes.{tf} must be an object")
        candle_at = _utc(data.get("last_closed_at"), f"{tf}.last_closed_at")
        if candle_at > as_of:
            raise ValueError(f"{tf}: future candle rejected (no look-ahead)")
        age_min = (as_of - candle_at).total_seconds() / 60.0
        if age_min > 2 * TIMEFRAME_MINUTES[tf]:
            warnings.append(f"{tf}: данные устарели ({int(age_min)} мин).")
        phase = str(data.get("phase", "UNKNOWN")).upper()
        if phase not in PHASES:
            phase = "UNKNOWN"
        wave = _text(data.get("wave_hypothesis"))
        indicators = {key: _number(data.get(key)) for key in ("ao", "ac")}
        evidence["timeframes"][tf] = {
            "last_closed_at": candle_at.isoformat(),
            "phase": phase,
            "wave_hypothesis": wave,
            "ao": indicators["ao"],
            "ac": indicators["ac"],
        }
        observations.append(
            f"{tf}: фаза {phase}; AO={indicators['ao']}; AC={indicators['ac']}; "
            f"волновая гипотеза: {wave} (не подтверждённый факт)."
        )

    if not evidence["timeframes"]:
        warnings.append("Нет подтверждённых закрытых свечей для анализа.")

    signals = snapshot.get("signals", [])
    if not isinstance(signals, list):
        raise ValueError("signals must be an array")
    for index, signal in enumerate(signals[:100]):
        if not isinstance(signal, dict):
            raise ValueError(f"signals[{index}] must be an object")
        name = str(signal.get("name", "")).upper()
        direction = str(signal.get("direction", "")).upper()
        tf = str(signal.get("timeframe", "")).upper()
        if name not in SIGNAL_NAMES or direction not in DIRECTIONS or tf not in TIMEFRAME_MINUTES:
            warnings.append(f"Сигнал #{index + 1}: неизвестный тип/направление/ТФ; пропущен.")
            continue
        bar_at = _utc(signal.get("confirmed_at"), f"signals[{index}].confirmed_at")
        if bar_at > as_of:
            raise ValueError(f"signals[{index}]: future signal rejected (no look-ahead)")
        is_confirmed = signal.get("confirmed") is True
        ev = {
            "name": name,
            "direction": direction,
            "timeframe": tf,
            "confirmed": is_confirmed,
            "confirmed_at": bar_at.isoformat(),
            "reason": _text(signal.get("reason")),
        }
        evidence["signals"].append(ev)
        state = "заявлен как подтверждённый" if is_confirmed else "НЕ подтверждён"
        observations.append(f"{tf} {name} {direction}: {state}; {ev['reason']}.")
        if not is_confirmed:
            warnings.append(f"{tf} {name} {direction}: нельзя считать подтверждённым сигналом.")

    position = snapshot.get("position")
    if position is not None:
        if not isinstance(position, dict):
            raise ValueError("position must be an object")
        status = str(position.get("status", "UNKNOWN")).upper()
        side = str(position.get("side", "UNKNOWN")).upper()
        stop_ok = position.get("stop_confirmed_on_exchange") is True
        evidence["position"] = {
            "status": status if status in {"OPEN", "CLOSED", "NONE"} else "UNKNOWN",
            "side": side if side in DIRECTIONS else "UNKNOWN",
            "stop_confirmed_on_exchange": stop_ok,
        }
        if status == "OPEN" and not stop_ok:
            warnings.insert(
                0,
                "КРИТИЧНО: открытая позиция без подтверждённого биржей SL. "
                "Нужна немедленная проверка человеком; AI не исполняет ордера."
            )
        elif status == "OPEN":
            observations.append("Открытая позиция: наличие SL заявлено, но AI не проверял биржу.")

    return {
        "mode": "READ_ONLY",
        "symbol": symbol,
        "as_of": as_of.isoformat(),
        "warnings": warnings,
        "observations": observations,
        "evidence": evidence,
        "trade_action": None,
        "limitations": [
            "AI не подключён к Binance и не подтверждает состояния биржи.",
            "Волновые гипотезы и индикаторы получены из входных данных, а не вычислены здесь.",
            "Подтверждённый сигнал требует отдельной верификации по первичным свечам и правилам книги.",
        ],
    }


def render_russian_report(report: dict[str, Any]) -> str:
    lines = [f"Williams AI — наблюдение {report['symbol']} ({report['as_of']})",
             "Режим: только чтение; торговые команды отсутствуют."]
    lines.extend("⚠ " + item for item in report["warnings"])
    lines.extend("• " + item for item in report["observations"])
    lines.extend("Ограничение: " + item for item in report["limitations"])
    return "\n".join(lines)
