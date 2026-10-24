"""Read-only audit of exchange execution evidence for a single TC2 campaign.

This is a verifier of a PROVIDED event journal, not a trade simulator.
A claim in the journal is not independent confirmation from Binance.
No network, credentials or trading-module imports, and never any orders.
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any


_ALLOWED = {
    "OPEN_FILL", "STOP_ACK", "STOP_REPLACE_ACK", "STOP_CANCEL_ACK",
    "OPPOSITE_SIGNAL", "CLOSE_FILL", "RESTART",
}
_EXIT_REASONS = {"SL_TRIGGERED", "OPPOSITE_TRIGGER", "MANUAL", "RISK_EMERGENCY", "UNKNOWN"}


def _utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Event timestamps must be ISO-8601 strings")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid event timestamp") from exc
    if dt.tzinfo is None:
        raise ValueError("Timezone required")
    return dt.astimezone(timezone.utc)


def _positive(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(float(value)) or float(value) <= 0):
        raise ValueError(f"{name} must be a finite positive price")
    return float(value)


def audit_campaign(journal: dict[str, Any]) -> dict[str, Any]:
    """Validate observable chronology, SL protection and exit evidence.

    A contrary signal alone is NOT a stop fill, reversal fill or instruction
    to close. TC2 can reverse at an actually TRIGGERED opposite entry, but
    a claimed fill needs independently checked exchange history.
    """
    if not isinstance(journal, dict):
        raise ValueError("Expected JSON journal")
    if journal.get("edition") != "TC2":
        raise ValueError("Audit is specific to TC2; never infer rules for NTD/COURSE/TC1")
    symbol = journal.get("symbol")
    if not isinstance(symbol, str) or not symbol.endswith("USDT") or len(symbol) > 20:
        raise ValueError("Invalid market symbol")
    as_of = _utc(journal.get("as_of"))
    events = journal.get("events")
    if not isinstance(events, list) or len(events) > 5000:
        raise ValueError("Expected up to 5000 journal events")
    previous = None
    position = None
    stop = None
    findings: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    closed = 0

    def report(code: str, severity: str, at: datetime) -> None:
        findings.append({"code": code, "severity": severity, "at": at.isoformat()})

    for i, event in enumerate(events):
        if not isinstance(event, dict) or event.get("type") not in _ALLOWED:
            raise ValueError(f"Unsupported journal event at index {i}")
        at = _utc(event.get("at"))
        if at > as_of:
            raise ValueError("NO_LOOKAHEAD: future event rejected")
        if previous is not None and at < previous:
            raise ValueError("Events must be chronologically ordered")
        previous = at
        kind = event["type"]
        ack = event.get("exchange_confirmed") is True
        timeline.append({"at": at.isoformat(), "event": kind, "trade_action": None})

        if kind == "OPEN_FILL":
            if position is not None:
                report("DUPLICATE_OR_OPPOSITE_OPEN_WHILE_POSITION_ACTIVE", "CRITICAL", at)
                continue
            side = event.get("side")
            if side not in {"LONG", "SHORT"}:
                raise ValueError("OPEN_FILL requires LONG or SHORT")
            entry = _positive(event.get("price"), "open price")
            position = {"side": side, "entry_price": entry, "opened_at": at.isoformat()}
            stop = None
            if not ack:
                report("OPEN_NOT_VERIFIED_BY_EXCHANGE", "CRITICAL", at)
            # A protective stop is separately acknowledged. A same-timestamp
            # ACK does not prove an interval of unprotected exposure.

        elif kind in {"STOP_ACK", "STOP_REPLACE_ACK"}:
            level = _positive(event.get("stop_price"), "stop price")
            if position is None:
                report("STOP_EVENT_WITHOUT_OPEN_POSITION", "WARNING", at)
            elif not ack:
                report("STOP_UNCONFIRMED_BY_EXCHANGE", "CRITICAL", at)
            elif stop is None and at > _utc(position["opened_at"]):
                report("EXPOSURE_INTERVAL_BEFORE_FIRST_STOP_ACK", "CRITICAL", at)
                stop = level
            elif stop is not None and (
                (position["side"] == "LONG" and level < stop)
                or (position["side"] == "SHORT" and level > stop)
            ):
                report("PROTECTIVE_STOP_WIDENED", "CRITICAL", at)
            else:
                stop = level

        elif kind == "STOP_CANCEL_ACK":
            if position is not None and ack:
                stop = None
                report("PROTECTIVE_STOP_CANCELLED_WHILE_OPEN", "CRITICAL", at)
            elif position is not None:
                report("STOP_CANCEL_STATUS_UNCONFIRMED", "CRITICAL", at)

        elif kind == "CLOSE_FILL":
            if position is None:
                report("CLOSE_WITHOUT_OPEN_POSITION", "CRITICAL", at)
                continue
            reason = event.get("reason", "UNKNOWN")
            if reason not in _EXIT_REASONS:
                reason = "UNKNOWN"
            if not ack:
                report("CLOSE_NOT_VERIFIED_BY_EXCHANGE", "CRITICAL", at)
                continue
            # An opposite Wise Man observation is not an executed reversal.
            if reason == "OPPOSITE_TRIGGER" and event.get("opposite_trigger_fill_confirmed") is not True:
                report("OPPOSITE_SIGNAL_MISTAKEN_FOR_EXECUTED_REVERSAL", "CRITICAL", at)
            if reason == "UNKNOWN":
                report("CLOSE_REASON_UNKNOWN", "CRITICAL", at)
            if stop is None:
                report("CLOSE_FILL_WITHOUT_PRIOR_STOP_ACK", "CRITICAL", at)
            if reason in {"MANUAL", "RISK_EMERGENCY"}:
                report("NON_STRATEGY_EXIT_REQUIRES_SEPARATE_EVIDENCE", "INFO", at)
            position = None
            stop = None
            closed += 1

        # OPPOSITE_SIGNAL and RESTART intentionally do not change position or SL.

    if position is not None and stop is None:
        report("OPEN_POSITION_WITHOUT_CONFIRMED_PROTECTIVE_SL", "CRITICAL", as_of)
    # A previous one-time uncovered interval is not erased by a later STOP_ACK.
    # An investigator can distinguish historical gaps from current state.
    return {
        "symbol": symbol, "edition": "TC2", "as_of": as_of.isoformat(),
        "mode": "READ_ONLY", "trade_action": None, "execution_allowed": False,
        "events_processed": len(events), "campaigns_closed": closed,
        "position": (
            {"status": "OPEN", "side": position["side"],
             "stop_confirmed_in_journal": stop is not None,
             "stop_price": stop}
            if position else {"status": "NONE"}
        ),
        "findings": findings, "timeline": timeline,
        "limitations": [
            "Journal ACK/FILL flags are untrusted until independently reconciled with Binance.",
            "OHLC alone cannot determine intrabar order or actual limit/stop fills.",
            "This is not proof of book conformity, slippage, PnL or financial safety.",
        ],
    }
