"""Operational analogue of Williams trader psychology: disciplined behavior.

TC1 PDF pp.129-137 discusses fear/self-preservation, greed, confidence,
enjoyment/flow, beliefs and willingness to abandon a wrong market thesis.
It is NOT validated neuroscience and NOT a claim that a GGUF experiences
human fear, greed or euphoria. We enforce OBSERVABLE decisions instead:
no FOMO chasing, averaging down, revenge scaling, unprotected positions,
hallucinated certainty or discretionary closures because AI 'feels' like it.

This is a read-only pretrade PLAN AUDIT, not real Binance order placement.
"""
from __future__ import annotations
from typing import Any


def psychology_discipline_audit(
    plan:dict[str,Any], *, position:dict[str,Any]|None=None,
)->dict[str,Any]:
    if not isinstance(plan,dict):
        raise ValueError("Trading plan must be a JSON object")
    if position is not None and not isinstance(position,dict):
        raise ValueError("Position state must be an object")
    risks=[]
    def block(code:str):
        if code not in risks:risks.append(code)

    source=plan.get("edition")
    if source not in {"TC1","NTD","TC2","COURSE"}:
        block("UNIDENTIFIED_STRATEGY_AND_BOOK_RULE")
    if not plan.get("rule_ids") or not isinstance(plan.get("rule_ids"),list):
        block("NO_CITED_SETUP_OR_STUDIED_RULE")
    if plan.get("risk_limit_verified") is not True:
        block("RISK_LIMIT_NOT_VERIFIED")
    if plan.get("account_identity_verified") is not True:
        block("ACCOUNT_IDENTITY_NOT_VERIFIED")
    if plan.get("data_fresh") is not True:
        block("STALE_OR_UNVERIFIED_MARKET_DATA")
    if plan.get("price_trigger_confirmed") is not True:
        block("FOMO_PRICE_CHASE_NO_CONFIRMED_TRIGGER")
    if plan.get("exchange_protective_stop_ack") is not True:
        block("PROTECTIVE_STOP_NOT_ACKNOWLEDGED")
    if plan.get("daily_loss_limit_reached") is True:
        block("LOSS_LIMIT_REACHED_NO_REVENGE_TRADING")
    if plan.get("increase_risk_because_recent_loss") is True:
        block("REVENGE_SIZE_INCREASE_AFTER_LOSS")
    if plan.get("average_losing_position") is True:
        block("UNPLANNED_AVERAGING_DOWN")
    if plan.get("cancel_or_widen_stop_to_avoid_loss") is True:
        block("FEAR_BASED_STOP_CANCELLATION_OR_WIDENING")
    if plan.get("opposite_signal_only_requests_exit") is True:
        block("OPPOSITE_UNEXECUTED_SIGNAL_NOT_A_FILLED_EXIT")
    if plan.get("market_direction_certain") is True or plan.get("wave_degree_guaranteed") is True:
        block("OVERCONFIDENT_CERTAINTY_ABOUT_CHAOTIC_MARKET")
    if plan.get("execution_policy_override_requested") is True:
        block("MODEL_CANNOT_OVERRIDE_EXECUTION_DOOR")
    if position:
        # No new campaign on an already occupied pair (owner binding).
        if position.get("active_campaign") and plan.get("new_campaign"):
            block("ONE_CAMPAIGN_PER_SYMBOL_ALREADY_ACTIVE")
        if position.get("unprotected_position") is True:
            block("UNPROTECTED_POSITION_REQUIRES_EXCHANGE_RECOVERY_FIRST")
        owner=position.get("owning_edition")
        if owner and owner!=source and plan.get("manage_existing_campaign"):
            block("NON_OWNER_ATTEMPTS_TO_MANAGE_EXISTING_CAMPAIGN")
    return {
        "source":"TC1 PDF 129-137: beliefs, fear/greed/confidence vs planned discipline; TC1 84 structural invalidation",
        "psychology_means_observable_policy_not_claimed_AI_EMOTIONS":True,
        "pretrade_gate":"BLOCK" if risks else "ELIGIBLE_FOR_SEPARATE_DETERMINISTIC_REVIEW",
        "blocking_reasons":risks,
        "required_next_step":(
            "Reconstruct missing facts / recover protective stop before proposing."
            if risks else
            "Independently prove canonical edition rule, Binance risk and execution ACK."
        ),
        "uncertainty_accepted":True,
        "fear_or_greed_in_AI_diagnosed":False,
        "risk_increased_after_loss_by_this_module":False,
        "trade_action":None,
        "execution_allowed":False,
    }
