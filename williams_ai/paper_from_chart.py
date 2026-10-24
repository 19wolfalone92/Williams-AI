"""Read-only H1 closed-bar -> independently audited WM3 -> paper risk bridge.

Unlike a model-provided summary, this wrapper reconstructs the chart and WM3
lifecycle from the supplied complete chronological bars, then invokes the
source-aware auditor and shared-capital paper budget. It is never an order.
"""
from __future__ import annotations

from typing import Any

from .algorithm_guard import audit_reasoning
from .chart_reader import analyze_chart
from .portfolio_risk_research import (
    PaperRiskLimits, size_paper_hypothesis, _paper_base,
)


def paper_wm3_from_closed_h1(
    symbol: str, bars: list[dict[str, Any]], portfolio: dict[str, Any], *,
    side: str, tick_size: float, lot_step: float,
    limits: PaperRiskLimits = PaperRiskLimits(),
) -> dict[str, Any]:
    """Research a closed-bar WM3 *past* touch; no inference of actual fills.

    Five preceding completed candles supply a conservative LAB stop proxy,
    not the canonical Williams campaign stop. This only assesses a historical
    hypothetical risk budget, it cannot order a trade at a past trigger.
    """
    if side not in ("LONG", "SHORT"):
        raise ValueError("Side must be LONG or SHORT")
    if not isinstance(bars,list) or len(bars)<45:
        raise ValueError("At least 45 closed H1 bars required")
    as_of=bars[-1].get("close_time")
    evidence=analyze_chart(symbol,"H1",bars,as_of=as_of,tick_size=tick_size)
    if evidence["gaps"] or evidence["last_closed_at"]!=evidence["as_of"]:
        return _paper_base(symbol,evidence["as_of"],["GAPS_OR_STALE_H1_CANDLE"])
    candidates=[
        c for c in evidence["candidates"]
        if c.get("type")=="WM3_FRACTAL" and c.get("side")==side
        and c.get("state")=="CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL"
        and c.get("price_touch_observed_at")==evidence["as_of"]
        and c.get("teeth_filter_at_trigger") is True
    ]
    if not candidates:
        return _paper_base(symbol,evidence["as_of"],
                           ["NO_CURRENT_CLOSED_WM3_TOUCH"])
    # Time-stamped chart candidates were derived with t+2 confirmation.
    # Do not replay an older fractal when a more recent same-side pivot exists.
    latest=max(candidates,key=lambda c:c["confirmed_at"])
    trigger=latest.get("entry_trigger")
    if not isinstance(trigger,(int,float)) or trigger<=0:
        return _paper_base(symbol,evidence["as_of"],
                           ["WM3_PRICE_TRIGGER_NOT_QUANTIFIED"])
    audit=audit_reasoning(evidence,{
        "symbol":symbol, "as_of":evidence["as_of"],"edition":"TC2",
        "setup":"WM3","side":side,"entry_state":"CLOSED_BAR_TOUCH_ONLY",
        "rule_ids":["TC2.WM3"], "exit_basis":"NONE",
        "trade_action":None, "execution_allowed":False,
    })
    if audit["hard_failures"]:
        result=_paper_base(symbol,evidence["as_of"],
                           ["INDEPENDENT_WM3_AUDIT_FAILED"])
        result["source_audit_failure_codes"]=audit["hard_failures"]
        return result
    # Historical entry assumption is deliberately no better than BOTH the
    # established trigger and an opening gap through it. Only the five known
    # candles BEFORE this touch can be used to hypothesize an initial stop.
    recent=bars[-6:-1]
    entry=(max(trigger,float(bars[-1]["open"])) if side=="LONG"
           else min(trigger,float(bars[-1]["open"])))
    stop=(min(float(c["low"]) for c in recent) if side=="LONG"
          else max(float(c["high"]) for c in recent))
    report=size_paper_hypothesis({
        "symbol":symbol,"as_of":evidence["as_of"],
        "confirmed_at":latest["confirmed_at"],"side":side,
        "entry":entry,"stop":stop,"tick_size":tick_size,
        "lot_step":lot_step, "entry_state":"CLOSED_BAR_TOUCH_ONLY",
        "source_audit":audit,
    },portfolio,limits=limits)
    report["paper_method"]="CURRENT_CLOSED_H1_WM3_TOUCH_AUDITED_AS_OF_CLOSE"
    report["paper_structural_stop_basis"]="FIVE_PREVIOUS_CLOSED_H1_BAR_PROXY"
    report["source_confirmation_time"]=latest["confirmed_at"]
    report["observed_touch_time"]=latest["price_touch_observed_at"]
    report["hypothetical_entry_reference"]=entry
    report["hypothetical_structural_stop_reference"]=stop
    report["not_available_at_intrabar_price_touch"]=True
    report["requires_new_future_signal_for_actual_execution"]=True
    return report
