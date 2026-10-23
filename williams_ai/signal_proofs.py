"""Source-bound proof passports: PRICE FACT != SIGNAL != EXCHANGE EXECUTION.

No image template or impossible 'wave 5 is certain' shortcut. Only evidence
visible at the exact closed cutoff can support current TC1/TC2/NTD signals.
Five magic bullets, reversal-bar angulation and pending fractal entry remain
research observations, never order authorization.
"""
from __future__ import annotations

from typing import Any


def current_signal_proofs(observation:dict[str,Any])->dict[str,Any]:
    if not isinstance(observation,dict) or observation.get("execution_allowed") is not False:
        raise ValueError("Only causal read-only observed chart facts accepted")
    now=observation.get("as_of")
    if not isinstance(now,str):
        raise ValueError("Missing closed chart cutoff")
    observed=observation.get("candidates",[])
    if not isinstance(observed,list):
        raise ValueError("Malformed candidates")
    proofs=[]
    for c in observed:
        if not isinstance(c,dict):
            continue
        kind,side=c.get("type"),c.get("side")
        if side not in ("LONG","SHORT"):
            continue
        state=c.get("state")
        claim=None;status=None;trigger=None;rule=None
        if kind=="WM1":
            claim="WM1";rule="TC2.WM1.BAR"
            if c.get("bar_time")==now:
                status=(
                    "REVERSAL_BAR_CURRENT_ANGLULATION_UNCONFIRMED"
                    if c.get("angle_heuristic") else
                    "REVERSAL_BAR_CURRENT_ANGLE_NOT_ESTABLISHED"
                )
            elif c.get("bar_time") is None:
                status="REVERSAL_BAR_TIME_UNVERIFIED_VISUAL_REVIEW_ONLY"
            else:status="HISTORICAL_REVERSAL_BAR_NOT_FRESH_SIGNAL"
        elif kind=="WM2_SUPER_AO":
            claim="WM2";rule="TC2.WM2"
            status=(
                "CURRENT_CLOSED_THREE_AO_MOMENTUM_BARS_NOT_ORDER"
                if state=="MOMENTUM_CANDIDATE_NOT_ORDER"
                else "UNRECOGNIZED_OR_STALE_AO"
            )
            trigger=c.get("price_trigger_reference")
        elif kind=="WM3_FRACTAL":
            claim="WM3";rule="TC2.WM3"
            trigger=c.get("entry_trigger")
            # Share the *same timezone-aware confirmation contract* as
            # the model prompt and independent audit. A claimed pending WM3
            # with no confirmation, future confirmation, or an old price
            # touch is NEVER a currently eligible signal.
            from .decision_contract import legal_wm3_discussion_states
            legal=legal_wm3_discussion_states({
                "as_of":now,"candidates":[c],
                "execution_allowed":False,"gaps":0,
            })
            stages={x["entry_state"] for x in legal}
            if "PENDING_TRIGGER" in stages:
                status="CONFIRMED_FRACTAL_PENDING_PRICE_TRIGGER"
            elif "CLOSED_BAR_TOUCH_ONLY" in stages:
                status="CURRENT_CLOSED_BAR_TOUCH_TEETH_PASS_NOT_FILL"
            elif state=="CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL":
                status="OLD_OR_UNVERIFIED_FRACTAL_TOUCH_NOT_CURRENT_SIGNAL"
            elif state=="CLOSED_BAR_TOUCH_REJECTED_BY_TEETH":
                status="PRICE_TOUCH_FAILED_TEETH"
            else:
                status="UNCONFIRMED_OR_SUPERSEDED_FRACTAL"
        if claim:
            proofs.append({
                "setup":claim,"side":side,"status":status,
                "source_rule_id":rule,"as_of":now,
                "observed_candidate_type":kind,
                "price_trigger_reference_not_an_exchange_order":trigger,
                "is_current_eligible_observation":status in {
                    "REVERSAL_BAR_CURRENT_ANGLULATION_UNCONFIRMED",
                    "CURRENT_CLOSED_THREE_AO_MOMENTUM_BARS_NOT_ORDER",
                    "CONFIRMED_FRACTAL_PENDING_PRICE_TRIGGER",
                    "CURRENT_CLOSED_BAR_TOUCH_TEETH_PASS_NOT_FILL",
                },
                "execution_allowed":False,
            })
    ed=observation.get("edition_evidence",{})
    if not isinstance(ed,dict):
        ed={}
    tc1=ed.get("TC1",{})
    tc1=tc1 if isinstance(tc1,dict) else {}
    for c in tc1.get("recent_formations",[]) or []:
        if isinstance(c,dict) and c.get("type")=="TC1_INITIATING_FRACTAL":
            proofs.append({
                "setup":"TC1_FRACTAL","side":c.get("side"),
                "status":"TC1_PATTERN_VISIBLE_NOT_EXECUTED",
                "source_rule_id":"TC1.FRACTAL","as_of":now,
                "is_current_eligible_observation":True,
                "execution_allowed":False,
            })
    magic=observation.get("tc1_five_magic_bullets",{})
    magic=magic if isinstance(magic,dict) else {}
    for x in (magic.get("last_n_wave_candidates",[]) or [])[-2:]:
        if not isinstance(x,dict):continue
        proofs.append({
            "setup":"TC1_FIVE_MAGIC_BULLETS","side":"NONE",
            "status":("ALL_FIVE_SIGNS_OBSERVED_ZERO_POINT_HYPOTHESIS"
                      if x.get("all_five_observed") is True
                      else "INCOMPLETE_FIVE_SIGNS_NOT_ZERO_POINT_CONFIRMATION"),
            "source_rule_id":"TC1.MAGIC.DIVERGENCE","as_of":now,
            "observed_bullets":x.get("observed_bullets"),
            "unknown_bullets":x.get("unknown_bullets"),
            "is_current_eligible_observation":x.get("all_five_observed") is True,
            "execution_allowed":False,
        })
    ntd=ed.get("NTD",{})
    ntd=ntd if isinstance(ntd,dict) else {}
    mapping={
        "AO_ZERO_CROSS":"AO_ZERO_CROSS",
        "AO_SAUCER_THREE_BAR":"AO_SAUCER",
        "AO_TWIN_PEAKS_POSSIBLE":"AO_TWIN_PEAKS",
        "AC_ACCELERATION_CANDIDATE":"AC",
    }
    for c in ntd.get("signals",[]) or []:
        if isinstance(c,dict) and c.get("name") in mapping:
            proofs.append({
                "setup":mapping[c["name"]],"side":c.get("side"),
                "status":(
                    "INDICATOR_SIGN_PRESENT_AFTER_INITIAL_FRACTAL_ADMISSION"
                    if ntd.get("entry_admitted") is True else
                    "INDICATOR_SIGN_PRESENT_BUT_INITIAL_NTD_FRACTAL_GATE_MISSING"
                ),
                "source_rule_id":(
                    "NTD.AO" if c["name"].startswith("AO_") else "NTD.AC"
                ),
                "as_of":now,
                "is_current_eligible_observation":ntd.get("entry_admitted") is True,
                "execution_allowed":False,
            })
    unique=[]
    seen=set()
    for item in proofs:
        # Preserve distinct historical and current stages; do NOT conflate
        # a valid formation with the actual current-bar price touch.
        key=(item.get("setup"),item.get("side"),item.get("status"),
             item.get("price_trigger_reference_not_an_exchange_order"))
        if key not in seen:
            seen.add(key);unique.append(item)
    return {
        "mode":"CAUSE_AND_SOURCE_GROUNDED_EVIDENCE_PROOFS",
        "as_of":now,
        "observed_proofs":unique,
        "current_proof_count":sum(bool(x["is_current_eligible_observation"])
                                  for x in unique),
        "not_observed_is_not_a_book_violation":True,
        "conditional_wave_degree":"UNKNOWN",
        "price_trigger_is_not_order_or_fill":True,
        "trade_action":None,
        "execution_allowed":False,
    }


def proof_supports_current_claim(proofbook:dict[str,Any],setup:str,
                                 side:str,entry_state:str)->bool:
    """Second independent fail-closed current-observation check for model claims."""
    items=[
        x for x in proofbook.get("observed_proofs",[])
        if x.get("setup")==setup and x.get("side")==side
    ]
    if setup=="WM3":
        stage=(
            {"CURRENT_CLOSED_BAR_TOUCH_TEETH_PASS_NOT_FILL"}
            if entry_state=="CLOSED_BAR_TOUCH_ONLY"
            else {"CONFIRMED_FRACTAL_PENDING_PRICE_TRIGGER"}
            if entry_state=="PENDING_TRIGGER"
            else {
                "CONFIRMED_FRACTAL_PENDING_PRICE_TRIGGER",
                "CURRENT_CLOSED_BAR_TOUCH_TEETH_PASS_NOT_FILL",
                "TEETH_PASS_BUT_TOUCH_DATE_UNKNOWN_NOT_ENTRY",
            }
        )
        return any(x["status"] in stage for x in items)
    if setup=="WM1":
        allowed={"REVERSAL_BAR_CURRENT_ANGLULATION_UNCONFIRMED",
                 "REVERSAL_BAR_CURRENT_ANGLE_NOT_ESTABLISHED",
                 "REVERSAL_BAR_TIME_UNVERIFIED_VISUAL_REVIEW_ONLY"}
        if entry_state in ("CLOSED_BAR_TOUCH_ONLY","EXCHANGE_FILLED"):
            return any(x["status"]=="REVERSAL_BAR_CURRENT_ANGLULATION_UNCONFIRMED"
                       for x in items)
        # A tentative WM1 hypothesis without numeric angulation evidence
        # can be discussed with a human chart-review warning, never
        # described as a filled order or a proved price touch.
        return any(x["status"] in allowed for x in items)
    if setup=="WM2":
        return any(x["status"]=="CURRENT_CLOSED_THREE_AO_MOMENTUM_BARS_NOT_ORDER"
                   for x in items)
    if setup=="TC1_FIVE_MAGIC_BULLETS":
        return any(x["status"]=="ALL_FIVE_SIGNS_OBSERVED_ZERO_POINT_HYPOTHESIS"
                   for x in proofbook.get("observed_proofs",[]))
    if setup in {"AO_ZERO_CROSS","AO_SAUCER","AO_TWIN_PEAKS","AC"}:
        if entry_state in ("PENDING_TRIGGER","CLOSED_BAR_TOUCH_ONLY","EXCHANGE_FILLED"):
            return any(x["is_current_eligible_observation"] for x in items)
        # Observation of an NTD indicator pattern is legitimate even
        # before the FIRST outside-mouth fractal admits an NTD campaign.
        return bool(items)
    return True
