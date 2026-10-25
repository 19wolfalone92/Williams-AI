"""Evidence-bound TC2 WM3 state/side guidance, independent of GGUF prose.

The full set of possible setup/state combinations is derived entirely from
CONFIRMED historical observations. It is a HINT for GGUF JSON consistency,
not proof of a trade, future price direction, or authority to execute.
Independent algorithm_guard MUST re-check every model response.
"""
from __future__ import annotations

from datetime import datetime
import json
from typing import Any


def _as_utc(s:Any):
    if not isinstance(s,str):
        return None
    try:
        t=datetime.fromisoformat(s.replace("Z","+00:00"))
        if t.tzinfo is None:
            return None
        return t.timestamp()
    except (ValueError,OverflowError):
        return None


def legal_wm3_discussion_states(chart:dict[str,Any])->list[dict[str,str]]:
    if chart.get("execution_allowed") is not False or chart.get("gaps"):
        raise ValueError("Only contiguous read-only chart observations accepted")
    as_of=_as_utc(chart.get("as_of"))
    if as_of is None:
        raise ValueError("Chart cutoff timestamp missing")
    allowed={}
    for c in chart.get("candidates",[]):
        if not isinstance(c,dict) or c.get("type")!="WM3_FRACTAL":
            continue
        side=c.get("side")
        if side not in ("LONG","SHORT"):
            continue
        confirmed=_as_utc(c.get("confirmed_at"))
        if confirmed is None or confirmed>as_of:
            continue
        state=c.get("state")
        entry_state=None
        if state=="AWAITING_PRICE_TRIGGER":
            # A pending signal must not already have a historical touch.
            if c.get("price_touch_observed_at") is None:
                entry_state="PENDING_TRIGGER"
        elif state=="CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL":
            # No candle after the original breakout is entitled to the
            # fresh-touch label. The exact timestamp is verified.
            if _as_utc(c.get("price_touch_observed_at"))==as_of:
                entry_state="CLOSED_BAR_TOUCH_ONLY"
        if entry_state:
            allowed[(side,entry_state)]={
                "setup":"WM3","side":side,"entry_state":entry_state,
                "source_rule":"TC2.WM3",
                "price_trigger_status":(
                    "AWAITING_BREAKOUT_NOT_FILLED"
                    if entry_state=="PENDING_TRIGGER"
                    else "CLOSED_BAR_TOUCH_NOT_EXCHANGE_FILL"
                ),
            }
    return [allowed[k] for k in sorted(allowed)]


def contract_for_model(chart:dict[str,Any],edition:str)->dict[str,Any]:
    """Bounded as-of decision hints, NEVER a recommendation to execute."""
    result={
        "market_as_of":chart["as_of"],
        "edition":edition,
        "no_trade_is_always_permitted":True,
        "narrative_claim_must_match_observed_setup":True,
        "exchange_fill_and_SL_must_not_be_inferred":True,
        "wm3_valid_setup_side_entry_state_combinations":[],
        "trade_action":None,
        "execution_allowed":False,
    }
    if edition=="TC2":
        result["wm3_valid_setup_side_entry_state_combinations"]=legal_wm3_discussion_states(chart)
    return result


def previous_decision_review(chart:dict[str,Any], proposal:dict[str,Any],
                             edition:str)->dict[str,Any]:
    """Return critic diagnostics, never a replacement answer or future label.

    The proposal is untrusted prior model output, not source knowledge. The
    critic is rerun here instead of accepting caller-supplied failure codes.
    """
    if (not isinstance(proposal,dict) or proposal.get("symbol")!=chart.get("symbol")
            or proposal.get("edition")!=edition
            or _as_utc(proposal.get("as_of")) is None
            or _as_utc(proposal.get("as_of"))!=_as_utc(chart.get("as_of"))):
        raise ValueError("PREVIOUS_DECISION_MUST_MATCH_CURRENT_CLOSED_CHART")
    try:
        raw=json.dumps(proposal,ensure_ascii=False,allow_nan=False)
    except (TypeError,ValueError) as exc:
        raise ValueError("BOUNDED_PREVIOUS_DECISION_REQUIRED") from exc
    if len(raw)>6000:
        raise ValueError("BOUNDED_PREVIOUS_DECISION_REQUIRED")
    for field,limit in (("reason",600),("learning_question",400)):
        if field in proposal and (not isinstance(proposal[field],str) or len(proposal[field])>limit):
            raise ValueError("BOUNDED_PREVIOUS_DECISION_TEXT_REQUIRED")
    from .algorithm_guard import audit_reasoning
    review=audit_reasoning(chart,proposal)
    fields=("symbol","as_of","edition","setup","side","entry_state","rule_ids",
            "exit_basis","reason","learning_need","learning_question",
            "exchange_fill_confirmed","exchange_sl_confirmed","wave_degree_verified")
    return {"previous_model_answer_unverified":{k:proposal[k] for k in fields if k in proposal},
            "independent_critic_failures":review["hard_failures"],
            "independent_critic_uncertainties":review["uncertainties"],
            "same_closed_market_cutoff":True,"correct_answer_supplied":False,
            "previous_answer_is_not_market_or_book_evidence":True,
            "execution_allowed":False,"trade_action":None}
