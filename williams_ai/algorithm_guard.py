"""Deterministic, source-aware critic for proposed Williams AI explanations.

This audits whether a reasoning claim is *supported by already closed charts*,
not whether an LLM is intelligent or whether the signal has an edge. Every
outcome is read-only; no order, close or stop placement is permitted.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .book_rules import registry

_ALLOWED_ED = {"TC1","NTD","TC2","COURSE"}
_TRADE_SETUPS = {"WM1","WM2","WM3"}
_NTD_SETUPS={"AO_ZERO_CROSS","AO_SAUCER","AO_TWIN_PEAKS","AC","ZONE","BALANCE_LINE"}
_NONENTRY_SETUPS={"WAVE_HYPOTHESIS","TC1_FIVE_MAGIC_BULLETS","NO_SETUP","UNKNOWN"}
_TC1_SETUPS={"TC1_FRACTAL"}
_ALL_SETUP=_TRADE_SETUPS|_NTD_SETUPS|_NONENTRY_SETUPS|_TC1_SETUPS
_ALLOWED_EXIT={"NONE","SL_CONFIRMED","OPPOSITE_SIGNAL_ONLY","EXECUTED_REVERSAL",
               "STRUCTURAL_INVALIDATION","UNKNOWN"}


def _timestamp(value: Any)->datetime:
    if not isinstance(value,str):
        raise ValueError("Expected ISO timestamp")
    try:
        result=datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid timestamp") from exc
    if result.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return result.astimezone(timezone.utc)


def audit_reasoning(
    observation:dict[str,Any],
    proposal:dict[str,Any],
)->dict[str,Any]:
    """Grade a candidate thought against chart and explicitly bound sources.

    The AI output is never allowed to certify actual exchange fills, a real
    protective stop, a specific future wave label or automatic orders.
    """
    if not isinstance(observation,dict) or not isinstance(proposal,dict):
        raise ValueError("Both chart observation and proposal must be objects")
    if observation.get("execution_allowed") is not False:
        raise ValueError("Expected READ_ONLY causal observation")
    as_of=_timestamp(observation.get("as_of"))
    decision_time=_timestamp(proposal.get("as_of"))
    source=proposal.get("edition")
    setup=proposal.get("setup")
    side=proposal.get("side")
    if source not in _ALLOWED_ED:
        raise ValueError("Expected TC1, NTD, TC2 or COURSE")
    if setup not in _ALL_SETUP:
        raise ValueError("Unknown setup")
    if side not in {"LONG","SHORT","NONE"}:
        raise ValueError("Direction must be LONG, SHORT or NONE")
    if proposal.get("exit_basis","NONE") not in _ALLOWED_EXIT:
        raise ValueError("Unknown exit claim")
    evidence=observation.get("candidates",[])
    if not isinstance(evidence,list):
        raise ValueError("Malformed candidate evidence")
    rules=registry()["rule_cards"]
    rule_map={x["id"]:x for x in rules}
    refs=proposal.get("rule_ids",[])
    if not isinstance(refs,list) or len(refs)>12 or len(refs)!=len(set(str(x) for x in refs)):
        raise ValueError("Invalid or duplicated source references")
    if any(not isinstance(r,str) for r in refs):
        raise ValueError("Rule identifiers must be strings")
    failures=[]
    cautions=[]
    supports=[]
    def fail(code:str)->None:
        if code not in failures:failures.append(code)
    def note(code:str)->None:
        if code not in cautions:cautions.append(code)

    if decision_time!=as_of:
        fail("DECISION_CHART_CUTOFF_MISMATCH")
    if proposal.get("symbol")!=observation.get("symbol"):
        fail("SYMBOL_MISMATCH")
    if observation.get("gaps",0):
        fail("GAPPED_MARKET_HISTORY")
    if proposal.get("trade_action") is not None or proposal.get("execution_allowed") is True:
        fail("EXECUTION_REQUEST_FORBIDDEN")
    entry=proposal.get("entry_state","UNKNOWN")
    if entry not in {"UNKNOWN","NO_SETUP","CLOSED_CANDIDATE",
                     "PENDING_TRIGGER","CLOSED_BAR_TOUCH_ONLY","EXCHANGE_FILLED"}:
        fail("UNKNOWN_ENTRY_EVIDENCE_CLAIM")
    elif entry=="EXCHANGE_FILLED":
        fail("NO_EXCHANGE_FILL_LEDGER")
    if side!="NONE" and setup in _NONENTRY_SETUPS:
        fail("DIRECTION_CLAIM_WITHOUT_ENTRY_SETUP")
    if proposal.get("exchange_fill_confirmed") is True:
        fail("CANNOT_CONFIRM_EXCHANGE_FILL_FROM_OHLC")
    if proposal.get("exchange_sl_confirmed") is True:
        fail("CANNOT_CONFIRM_EXCHANGE_SL_FROM_OHLC")
    if proposal.get("wave_degree_verified") is True:
        fail("WAVE_DEGREE_UNVERIFIABLE_FROM_CURRENT_EVIDENCE")
    if proposal.get("future_candle_used") is True:
        fail("FUTURE_CANDLE_CLAIM")
    # In the REAL Qwen3 1.7B historical exam, a structured WM2 / WM3
    # was sometimes accompanied by the literal explanation "NO_SETUP" or
    # "No setup confirmed". A structurally valid JSON must not be graded as
    # reasoned knowledge when the answer itself denies its selected signal.
    reason=proposal.get("reason")
    if isinstance(reason,str) and setup not in {"NO_SETUP","UNKNOWN","WAVE_HYPOTHESIS"}:
        import re
        negation=reason.strip().lower()
        if (re.match(r"^no_setup(?:\b|[\s.:;,])",negation)
                or re.match(r"^no (?:valid |confirmed |clear )?setup (?:found|confirmed|observed|detected)(?:\b|[.:;,])",negation)
                or re.match(r"^нет подтвержд[её]нн(?:ого|ых) (?:сигнала|сетапа|паттерна)(?:\b|[.:;,])",negation)):
            fail("EXPLANATION_CONTRADICTS_SELECTED_SETUP")
    if proposal.get("exit_basis")=="OPPOSITE_SIGNAL_ONLY":
        fail("UNEXECUTED_OPPOSITE_WISE_MAN_NOT_A_TRADE_EXIT")
    if proposal.get("exit_basis") in {"SL_CONFIRMED","EXECUTED_REVERSAL"}:
        fail("NO_EXCHANGE_CLOSE_LEDGER_WAS_PROVIDED")
    for ref in refs:
        c=rule_map.get(ref)
        if c is None:
            fail("HALLUCINATED_SOURCE_RULE")
        elif c["source"]!=source:
            fail("CROSS_EDITION_RULE_MIXING")
        else:
            supports.append({
                "rule_id":ref, "source":source,
                "pdf_pages":c["pdf_pages"],
                "source_review_status":c["status"],
            })
            if c["status"]!="REVIEWED_PAGES":
                note("SOURCE_CARD_STILL_AWAITING_PAGE_RECHECK")
    expected_rule_groups={
        "WM1":{"TC2.WM1.BAR","TC2.WM1.ANG","TC2.WM1.TRIGGER","TC2.WM1.STOP"},
        "WM2":{"TC2.WM2"},
        "WM3":{"TC2.WM3"},
        "TC1_FRACTAL":{"TC1.FRACTAL"},
        "AO_ZERO_CROSS":{"NTD.AO"},
        "AO_SAUCER":{"NTD.AO"},
        "AO_TWIN_PEAKS":{"NTD.AO"},
        "AC":{"NTD.AC"},
        "ZONE":{"NTD.ZONE","NTD.ZONE.ADDED"},
        "BALANCE_LINE":{"NTD.BALANCE"},
        "TC1_FIVE_MAGIC_BULLETS":{
            "TC1.MAGIC.DIVERGENCE","TC1.MAGIC.TARGET",
            "TC1.MAGIC.FRACTAL","TC1.MAGIC.SQUAT",
            "TC1.MAGIC.MOMENTUM",
        },
        "WAVE_HYPOTHESIS":{
            "TC1.W3","TC1.W4","TC1.W5","TC1.RHYTHM",
            "COURSE.W4.ZERO","COURSE.RHYTHM","COURSE.HUMP",
        },
    }
    if setup in expected_rule_groups:
        if not refs:
            fail("MISSING_SOURCE_RULE_FOR_SUBSTANTIVE_CLAIM")
        elif not any(ref in expected_rule_groups[setup] for ref in refs):
            fail("CITED_RULE_DOES_NOT_SUPPORT_THIS_SETUP")
    if setup in _TRADE_SETUPS and source!="TC2":
        fail("TC2_WISE_MAN_IN_OTHER_EDITION")
    if setup in _NTD_SETUPS and source!="NTD":
        fail("NTD_PATTERN_IS_NOT_TC2_WISE_MAN")
    if setup in _TRADE_SETUPS and side=="NONE":
        fail("SETUP_SIDE_REQUIRED")
    if setup=="TC1_FIVE_MAGIC_BULLETS":
        if source!="TC1":
            fail("MAGIC_BULLETS_ARE_TC1_NOT_NTD_FIVE_DIMENSIONS")
        found=observation.get("tc1_five_magic_bullets",{})
        sample=found.get("last_n_wave_candidates",[]) if isinstance(found,dict) else []
        if not sample:
            fail("NO_CLOSED_FIVE_BULLET_WAVE_CANDIDATE")
        else:
            note("ZERO_POINT_EVIDENCE_NOT_AUTOMATIC_EXECUTED_REVERSAL")
            if not any(x.get("all_five_observed") for x in sample if isinstance(x,dict)):
                # Announcing FIVE complete bullets when Squat tick volume is
                # missing is a FALSE claim, not merely a soft uncertainty.
                fail("FIVE_MAGIC_BULLETS_NOT_ALL_OBSERVED")
    if setup in _TC1_SETUPS and source!="TC1":
        fail("TC1_INITIATING_FRACTAL_CANNOT_USE_OTHER_EDITION")
    if setup in _TC1_SETUPS and side=="NONE":
        fail("TC1_FRACTAL_SIDE_REQUIRED")
    if setup=="WM1":
        matches=[c for c in evidence if c.get("type")=="WM1"
                 and c.get("side")==side]
        if not matches:
            fail("WM1_REVERSAL_BAR_NOT_OBSERVED")
        else:
            if any(x.get("bar_time") is not None
                   and x.get("bar_time")!=observation.get("as_of")
                   for x in matches):
                fail("WM1_REVERSAL_BAR_FROM_PAST_NOT_CURRENT_CANDLE")
            note("WM1_ANGLULATION_HUMAN_IMAGE_REVIEW_REQUIRED")
            if not any(x.get("angle_heuristic") is True for x in matches):
                note("WM1_NUMERICAL_ANGULATION_NOT_EVEN_POSITIVE")
                if entry in {"CLOSED_BAR_TOUCH_ONLY","EXCHANGE_FILLED"}:
                    fail("WM1_ENTRY_CLAIM_WITHOUT_EVEN_ANGLE_HEURISTIC")
    elif setup=="WM2":
        matches=[c for c in evidence if c.get("type")=="WM2_SUPER_AO"
                 and c.get("side")==side]
        if not matches:
            fail("WM2_SUPER_AO_NOT_OBSERVED")
        else:
            note("WM2_MOMENTUM_IS_NOT_A_CONFIRMED_EXECUTED_TRIGGER")
    elif setup=="WM3":
        matches=[c for c in evidence if c.get("type")=="WM3_FRACTAL"
                 and c.get("side")==side]
        valid_pending=any(
            x.get("state")=="AWAITING_PRICE_TRIGGER"
            and x.get("confirmed_at")<=observation.get("as_of","")
            for x in matches if x.get("confirmed_at") is not None
        )
        valid_current_touch=any(
            x.get("state")=="CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL"
            and x.get("price_touch_observed_at")==observation.get("as_of")
            for x in matches
        )
        # Book signal and exchange execution are different. In particular
        # yesterday's touch is NOT a breakout on the current candle, even
        # if the fractal is still visible on screen. An unspecified
        # entry_state cannot be silently upgraded to 'filled'.
        if not matches:
            fail("WM3_NO_CONFIRMED_FRACTAL_IN_OBSERVATION")
        if entry=="PENDING_TRIGGER":
            if valid_pending:
                note("WM3_AWAITING_BREAKOUT_NOT_AN_EXCHANGE_FILL")
            else:
                fail("WM3_NO_CURRENT_PENDING_VALID_FRACTAL")
        elif entry=="CLOSED_BAR_TOUCH_ONLY":
            if valid_current_touch:
                note("WM3_CLOSED_BAR_TOUCH_NOT_AN_EXCHANGE_FILL")
            else:
                fail("WM3_NO_PRICE_TOUCH_ON_THIS_CLOSED_BAR")
        else:
            fail("WM3_ENTRY_STATE_MUST_BE_PENDING_OR_FRESH_CLOSED_TOUCH")
    elif setup=="TC1_FRACTAL":
        shapes=observation.get("edition_evidence",{}).get("TC1",{}).get("recent_formations",[])
        matched=[x for x in shapes if x.get("side")==side
                 and x.get("type")=="TC1_INITIATING_FRACTAL"]
        if not matched:
            fail("TC1_FRACTAL_NOT_OBSERVED_ON_CLOSED_BARS")
        else:
            note("TC1_FRACTAL_SHAPE_IS_NOT_TRADE_EXECUTION")
            if any(x.get("same_level_bars_skipped") for x in matched):
                note("EQUAL_LEVEL_FRACTAL_REQUIRES_VISUAL_INSPECTION")
    elif setup in _NTD_SETUPS:
        ntd=observation.get("edition_evidence",{}).get("NTD",{})
        if not ntd.get("entry_admitted"):
            note("NTD_FIRST_OUTSIDE_FRACTAL_BREAKOUT_NOT_ESTABLISHED")
            if entry in {"PENDING_TRIGGER","CLOSED_BAR_TOUCH_ONLY","EXCHANGE_FILLED"}:
                fail("NTD_ENTRY_CLAIM_WITHOUT_FIRST_OUTSIDE_FRACTAL")
        named={
            "AO_ZERO_CROSS":"AO_ZERO_CROSS","AO_SAUCER":"AO_SAUCER_THREE_BAR",
            "AO_TWIN_PEAKS":"AO_TWIN_PEAKS_POSSIBLE",
            "AC":"AC_ACCELERATION_CANDIDATE",
        }
        if setup in named and not any(
            x.get("name")==named[setup] and x.get("side")==side
            for x in ntd.get("signals",[])
        ):
            fail("NTD_SIGNAL_NOT_OBSERVED")
        if setup in {"ZONE","BALANCE_LINE"}:
            note("NTD_ZONE_OR_BALANCE_LINE_NOT_A_STANDALONE_FIRST_ENTRY")
    else:
        note("STRUCTURAL_INTERPRETATION_ONLY_NOT_ENTRY")
    return {
        "mode":"READ_ONLY_REASONING_AUDIT",
        "symbol":observation.get("symbol"),
        "edition":source,
        "setup":setup,
        "as_of":as_of.isoformat(),
        "verdict":"UNSUPPORTED_CLAIM" if failures else "PROVISIONAL_REASONING_ONLY",
        "hard_failures":failures,"uncertainties":cautions,
        "source_proofs":supports,
        "trade_action":None,"execution_allowed":False,
        "actual_book_mastery_percent":None,
        "limitations":[
            "This is deterministic safety/source consistency, NOT book mastery certification.",
            "A correct historical explanation does not prove profitable future execution.",
            "No filled orders, stops, execution ACK, partial fills or exchange balances were observed.",
        ],
    }
