"""Time-consistent, multi-scale Williams interpretation ledger.

A five-bullet "zero point" is not a confirmed reversal: at t+N, the first
hypothesis must be RECHECKED against newly closed evidence, not permanently
assigned an Elliott wave number. Checks use only confirmations available
at each time; earlier entries are immutable after a later candle arrives.

This is deterministic *evidence bookkeeping*, not market forecasting, real
GGUF weight training, or permission to open, close or flip a position.
"""
from __future__ import annotations

from datetime import datetime,timezone
from typing import Any


def _parse(x:str)->datetime:
    if not isinstance(x,str):
        raise ValueError("Time must be explicit ISO timestamp")
    out=datetime.fromisoformat(x.replace("Z","+00:00"))
    if out.tzinfo is None:
        raise ValueError("Unzoned market timestamp")
    return out.astimezone(timezone.utc)


def _snapshot(chart:dict[str,Any])->dict[str,Any]:
    if not isinstance(chart,dict) or chart.get("no_lookahead") is not True or (
        chart.get("execution_allowed") is not False
    ):
        raise ValueError("Require closed, causal read-only evidence")
    facts=chart.get("principle_evidence",{}).get("observable_facts",{})
    structural=facts.get("multi_resolution_price_structure",{})
    if not isinstance(structural,dict) or "LOCAL" not in structural:
        raise ValueError("Need a full three-scale structure")
    signals=chart.get("tc1_five_magic_bullets",{})
    levels=facts.get("scale_details",{}).get("LOCAL",{})
    all_scenarios=chart.get("principle_evidence",{}).get("competing_market_scenarios",[])
    return {
        "symbol":chart["symbol"],
        "as_of":chart["as_of"],
        "timeframe":chart["timeframe"],
        "bar_count":chart["bars_used"],
        "last_close":float(facts["current_close"]),
        "structure":dict(structural),
        "ao_sign":facts["momentum"]["ao_sign"],
        "ao_change":facts["momentum"]["ao_change"],
        "ac_change":facts["momentum"]["ac_change"],
        "ao_near_zero_not_w4_proof":
            facts["momentum"]["ao_near_zero_relative_to_recent_amplitude"],
        "alligator":facts["visible_alligator_context"],
        "last_local_high":levels.get("latest_confirmed_high"),
        "last_local_low":levels.get("latest_confirmed_low"),
        "evidence_compatible_scenarios":[x["hypothesis"] for x in all_scenarios
            if x.get("status")=="EVIDENCE_COMPATIBLE_NOT_PREDICTION"],
        "five_bullet_candidates":signals.get("wave_candidate_count",0),
        "five_bullet_full_confirmations":sum(
            x.get("all_five_observed") is True for x in signals.get("last_n_wave_candidates",[])
        ),
        "wave_degree":"UNKNOWN",
        "trade_action":None,"execution_allowed":False,
    }


def temporal_revision(
    earlier:dict[str,Any],later:dict[str,Any], *,
    earlier_higher_context:dict[str,Any]|None=None,
    later_higher_context:dict[str,Any]|None=None,
)->dict[str,Any]:
    """Independently compare evidence at two strictly ordered H1 closed times."""
    a,b=_snapshot(earlier),_snapshot(later)
    if a["symbol"]!=b["symbol"] or a["timeframe"]!=b["timeframe"]:
        raise ValueError("Cannot compare unrelated markets or timeframes")
    if not _parse(a["as_of"])<_parse(b["as_of"]) or a["bar_count"]>=b["bar_count"]:
        raise ValueError("Need genuine forward movement of completed candles")
    changes=[]
    for key in ("structure","ao_sign","ao_change","ac_change",
                "ao_near_zero_not_w4_proof","alligator",
                "evidence_compatible_scenarios"):
        if a[key]!=b[key]:
            changes.append(key)
    contradicted=[]
    # Structural levels are from BEFORE the new movement. Use only
    # earlier reference prices to test whether previous continuation
    # thesis has LOST an essential structural support (not proof of reversal).
    if (a["last_local_low"] is not None
            and b["last_close"]<a["last_local_low"]
            and "CONTINUATION_UP" in a["evidence_compatible_scenarios"]):
        contradicted.append("EARLIER_CONTINUATION_UP_WEAKENED_BY_LOWER_CLOSE")
    if (a["last_local_high"] is not None
            and b["last_close"]>a["last_local_high"]
            and "CONTINUATION_DOWN" in a["evidence_compatible_scenarios"]):
        contradicted.append("EARLIER_CONTINUATION_DOWN_WEAKENED_BY_HIGHER_CLOSE")
    # A confirmed changed AO sign does not certify wave 4 or fifth termination.
    ah=earlier_higher_context or {}
    bh=later_higher_context or {}
    higher={}
    for tf in ("H4","D1"):
        x=ah.get(tf,{});y=bh.get(tf,{})
        if not isinstance(x,dict) or not isinstance(y,dict):
            raise ValueError("Malformed source-specific higher TF passport")
        older_time=x.get("closed_through")
        newer_time=y.get("closed_through")
        if older_time and _parse(older_time)>_parse(a["as_of"]):
            raise ValueError("Earlier higher TF used future closed bar")
        if newer_time and _parse(newer_time)>_parse(b["as_of"]):
            raise ValueError("Later higher TF used future closed bar")
        higher[tf]={
            "previous":x.get("market_context","UNKNOWN"),
            "now":y.get("market_context","UNKNOWN"),
            "changed":x.get("market_context")!=y.get("market_context"),
            "degree_of_elliott_wave":"UNKNOWN",
        }
    return {
        "mode":"CAUSAL_TEMPORAL_REVISION_NOT_PICTURE_MATCHING",
        "symbol":a["symbol"],"timeframe":a["timeframe"],
        "earlier_as_of":a["as_of"],"later_as_of":b["as_of"],
        "earlier_state":a,"later_state":b,
        "changed_confirmed_facts":changes,
        "prior_scenarios_lose_support":contradicted,
        "new_scenarios":sorted(
            set(b["evidence_compatible_scenarios"])
            -set(a["evidence_compatible_scenarios"])
        ),
        "previous_scenarios_no_longer_supported":sorted(
            set(a["evidence_compatible_scenarios"])
            -set(b["evidence_compatible_scenarios"])
        ),
        "higher_tf_market_context_revision":higher,
        "smaller_wave_completion_not_equal_to_larger_wave_completion":True,
        "ao_near_zero_not_proof_of_W4":True,
        "tick_volume_missing_prevents_TC1_squat_confirmation":(
            a["five_bullet_full_confirmations"]==0
            and b["five_bullet_full_confirmations"]==0
        ),
        "was_prior_hypothesis_invalidated_by_price":bool(contradicted),
        "exact_wave_degree_verified":False,
        "trade_action":None,"execution_allowed":False,
        "limitation":"Hypotheses can lose support but independently expert-labeled Elliott degrees are absent.",
    }
