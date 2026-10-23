"""Conservative *paper-only* risk budget for Williams AI hypotheses.

This module is NOT connected to an exchange and NEVER creates orders.
Inputs are research evidence, not authenticated exchange fills or SL receipts.
Portfolio limits must be checked on *shared* equity across correlated crypto
symbols; summing one-pair backtests is NOT a portfolio stress test.

A paper allocation only shows the maximum hypothetical quantity permitted by
the declared constraints. It does not upgrade a book pattern to a valid
execution signal or imply a profitable strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any
import math


@dataclass(frozen=True)
class PaperRiskLimits:
    max_single_idea_risk: float = 0.005
    max_portfolio_open_risk: float = 0.015
    max_crypto_cluster_risk: float = 0.01
    max_total_gross_notional: float = 1.5
    max_single_symbol_notional: float = 0.5
    max_daily_realized_loss: float = 0.02
    fee_bps_each_side: float = 4.0
    slippage_bps_each_side: float = 5.0
    min_paper_notional: float = 5.0

    def __post_init__(self) -> None:
        fractions = ("max_single_idea_risk", "max_portfolio_open_risk",
                     "max_crypto_cluster_risk", "max_total_gross_notional",
                     "max_single_symbol_notional", "max_daily_realized_loss")
        for key in fractions:
            val = getattr(self, key)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or (
                not math.isfinite(val) or not 0 < val <= 5
            ):
                raise ValueError("Invalid paper risk limit: " + key)
        for key in ("fee_bps_each_side", "slippage_bps_each_side"):
            val = getattr(self, key)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or (
                not math.isfinite(val) or not 0 <= val <= 500
            ):
                raise ValueError("Invalid friction assumption: " + key)
        if not 0 < self.min_paper_notional < 100000:
            raise ValueError("Invalid minimum paper notional")


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected numeric " + name)
    val = float(value)
    if not math.isfinite(val) or val <= 0:
        raise ValueError("Expected finite positive " + name)
    return val


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected numeric " + name)
    val = float(value)
    if not math.isfinite(val):
        raise ValueError("Expected finite " + name)
    return val


def _utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Timezone-aware ISO timestamp required")
    try:
        t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid timestamp") from exc
    if t.tzinfo is None:
        raise ValueError("Timezone offset required")
    return t.astimezone(timezone.utc)


def _paper_base(symbol: Any, as_of: Any, reasons: list[str]) -> dict[str, Any]:
    return {
        "mode": "PORTFOLIO_PAPER_RISK_STUDY_ONLY",
        "symbol": symbol, "as_of": as_of,
        "study_status": "BLOCKED" if reasons else "PAPER_SIZED_NOT_TRADABLE",
        "reason_codes": list(dict.fromkeys(reasons)),
        "hypothetical_quantity": 0.0,
        "hypothetical_notional": 0.0,
        "hypothetical_risk_including_friction": 0.0,
        "trade_action": None, "execution_allowed": False,
        "exchange_stop_acknowledged": False,
        "limitations": [
            "Synthetic portfolio state is not a Binance exchange ledger.",
            "OHLC trigger and candidate source proof are not authenticated order fills.",
            "Risk can exceed a stop budget due to gaps, liquidity and liquidation.",
            "No crypto pair is assumed independent of the broader market.",
            "No amount of paper sizing authorizes automatic or live trading.",
        ],
    }


def size_paper_hypothesis(
    signal: dict[str, Any],
    portfolio: dict[str, Any],
    *,
    limits: PaperRiskLimits = PaperRiskLimits(),
) -> dict[str, Any]:
    """Estimate PAPER size using closed evidence and a single shared risk pool.

    signal: symbol, side, entry, stop, tick_size, lot_step, as_of,
            confirmed_at, source_audit (independent algorithm_guard result).
    portfolio: as_of, equity, session_start_equity, daily_realized_pnl,
               positions: [{symbol,side,entry,stop,qty,stop_confirmed}],
               where stop_confirmed indicates PAPER ledger validation only.

    No broker order or real protective stop can be confirmed by this function.
    A rejected/weak/future model claim is not sized, even in paper research.
    """
    if not isinstance(signal, dict) or not isinstance(portfolio, dict):
        raise ValueError("Expected independent paper signal and portfolio objects")
    if not isinstance(limits, PaperRiskLimits):
        raise ValueError("PaperRiskLimits required")
    symbol = signal.get("symbol")
    as_of = signal.get("as_of")
    reasons: list[str] = []
    result = _paper_base(symbol, as_of, reasons)
    if not isinstance(symbol, str) or not symbol.endswith("USDT") or len(symbol) > 20:
        raise ValueError("Expected USDT crypto pair")
    if signal.get("side") not in {"LONG", "SHORT"}:
        raise ValueError("Expected LONG or SHORT paper direction")
    now = _utc(as_of)
    portfolio_time = _utc(portfolio.get("as_of"))
    confirmation = _utc(signal.get("confirmed_at"))
    if confirmation > now or portfolio_time != now:
        reasons.append("FUTURE_CONFIRMATION_OR_STALE_PORTFOLIO")
    entry = _positive(signal.get("entry"), "hypothetical entry")
    stop = _positive(signal.get("stop"), "hypothetical structural stop")
    tick = _positive(signal.get("tick_size"), "tick size")
    step = _positive(signal.get("lot_step"), "lot step")
    sign = 1 if signal["side"] == "LONG" else -1
    distance = sign * (entry - stop)
    if distance <= 2 * tick:
        reasons.append("NO_VALID_STRUCTURAL_STOP_DISTANCE")

    audit = signal.get("source_audit")
    if (not isinstance(audit, dict) or audit.get("symbol") != symbol or
            audit.get("as_of") != now.isoformat() or
            audit.get("verdict") != "PROVISIONAL_REASONING_ONLY" or
            audit.get("hard_failures") != [] or
            audit.get("execution_allowed") is not False or
            audit.get("trade_action", "UNSAFE") is not None or
            audit.get("edition") != "TC2" or
            audit.get("setup") != "WM3"):
        reasons.append("SOURCE_PROOF_NOT_INDEPENDENTLY_ACCEPTED")
    # Deliberately restrictive: other Wise Men / editions can be studied,
    # but this paper sizing cannot mistake a momentum observation for a
    # confirmed price breakout. Even WM3 requires independently checked
    # current touch evidence, not an awaiting fractal.
    if signal.get("entry_state") != "CLOSED_BAR_TOUCH_ONLY":
        reasons.append("NO_CURRENT_VERIFIED_PRICE_TOUCH_FOR_PAPER_SIZING")
    if audit is not None and isinstance(audit, dict):
        if "WM3_CLOSED_BAR_TOUCH_NOT_AN_EXCHANGE_FILL" not in audit.get("uncertainties", []):
            reasons.append("MISSING_CURRENT_WM3_TOUCH_CAUTION")

    equity = _positive(portfolio.get("equity"), "equity")
    session_equity = _positive(portfolio.get("session_start_equity"), "session equity")
    daily = _finite(portfolio.get("daily_realized_pnl"), "daily realized PnL")
    if daily <= -session_equity * limits.max_daily_realized_loss:
        reasons.append("DAILY_LOSS_LOCKED")
    existing = portfolio.get("positions")
    if not isinstance(existing, list) or len(existing) > 100:
        raise ValueError("Invalid positions array")
    invested = 0.0
    open_risk = 0.0
    cluster_risk = 0.0
    seen: set[str] = set()
    for pos in existing:
        if not isinstance(pos, dict):
            raise ValueError("Invalid paper position")
        p_symbol = pos.get("symbol")
        if (not isinstance(p_symbol, str) or not p_symbol.endswith("USDT")
                or p_symbol in seen):
            reasons.append("UNKNOWN_OR_DUPLICATE_POSITION")
            continue
        seen.add(p_symbol)
        if p_symbol == symbol:
            reasons.append("CAMPAIGN_ALREADY_OWNS_SYMBOL")
        p_side = pos.get("side")
        if p_side not in {"LONG", "SHORT"}:
            reasons.append("INVALID_OPEN_POSITION")
            continue
        p_entry = _positive(pos.get("entry"), "existing entry")
        p_stop = _positive(pos.get("stop"), "existing stop")
        qty = _positive(pos.get("qty"), "existing quantity")
        if pos.get("stop_confirmed") is not True:
            reasons.append("UNPROTECTED_OR_UNVERIFIED_PAPER_POSITION")
        invested += p_entry * qty
        p_sign = 1 if p_side == "LONG" else -1
        # A tightened profitable stop still incurs exit costs. Never let
        # locked-in paper gains silently increase the risk budget.
        p_risk = max(0.0, p_sign * (p_entry - p_stop))
        friction = ((p_entry + p_stop) *
                    (limits.fee_bps_each_side + limits.slippage_bps_each_side) / 10000)
        open_risk += qty * (p_risk + friction)
        cluster_risk += qty * (p_risk + friction)

    result["existing_paper_open_risk"] = round(open_risk, 8)
    result["existing_paper_gross_notional"] = round(invested, 8)
    if open_risk >= equity * limits.max_portfolio_open_risk:
        reasons.append("PORTFOLIO_RISK_BUDGET_EXHAUSTED")
    if cluster_risk >= equity * limits.max_crypto_cluster_risk:
        reasons.append("CRYPTO_CORRELATION_BUDGET_EXHAUSTED")
    if invested >= equity * limits.max_total_gross_notional:
        reasons.append("TOTAL_GROSS_NOTIONAL_BUDGET_EXHAUSTED")
    if reasons:
        result["reason_codes"] = list(dict.fromkeys(reasons))
        return result

    per_unit_worst = (distance + (entry + stop) *
                      (limits.fee_bps_each_side + limits.slippage_bps_each_side) / 10000)
    if per_unit_worst <= 0:
        result["reason_codes"] = ["INVALID_WORST_CASE_UNIT_RISK"]
        return result
    budget = min(
        equity * limits.max_single_idea_risk,
        equity * limits.max_portfolio_open_risk - open_risk,
        equity * limits.max_crypto_cluster_risk - cluster_risk,
    )
    notional_room = min(
        equity * limits.max_total_gross_notional - invested,
        equity * limits.max_single_symbol_notional,
    )
    units = min(budget / per_unit_worst, notional_room / entry)
    # Round DOWN to exchange-like lot_step for simulation only. Decimal
    # prevents floating rounding from exceeding a risk cap.
    try:
        size_dec = ((Decimal(str(units)) / Decimal(str(step))).to_integral_value(
            rounding=ROUND_DOWN) * Decimal(str(step)))
    except (InvalidOperation, OverflowError) as exc:
        raise ValueError("Cannot quantize hypothetical quantity") from exc
    qty = float(size_dec)
    if not math.isfinite(qty) or qty <= 0 or qty * entry < limits.min_paper_notional:
        result["reason_codes"] = ["PAPER_QUANTITY_BELOW_MINIMUM"]
        return result
    if qty * per_unit_worst > budget + 1e-7 or qty * entry > notional_room + 1e-7:
        result["reason_codes"] = ["ROUNDING_EXCEEDED_PAPER_RISK_BUDGET"]
        return result
    result.update({
        "study_status": "PAPER_SIZED_NOT_TRADABLE",
        "reason_codes": [],
        "hypothetical_quantity": qty,
        "hypothetical_notional": round(qty * entry, 8),
        "hypothetical_risk_including_friction": round(qty * per_unit_worst, 8),
        "paper_risk_budget": round(budget, 8),
        "paper_notional_room": round(notional_room, 8),
        "source_explanation": (
            "A valid closed-bar WM3 claim is STILL not an exchange order or fill."
        ),
    })
    return result
