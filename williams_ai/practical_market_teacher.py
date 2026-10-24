"""Causal practical trader classroom: learn to USE, not recite Williams.

Teaching modules see actual contiguous CLOSED market bars, never future data
or chart answer keys. They construct independent, falsifiable hypotheses,
explicit "wait" conditions, source-specific entry gates and risk questions.
The output is a lesson, NOT a signal or a trading order. No profits, weight
training or expert-approved Elliott counts are claimed.

Each chart is independently reanalysed from source OHLCV; any supplied LLM
prose or chart claims cannot change the canonical result. Books are distinct:
TC1 != NTD != TC2 != Course.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from typing import Any

from .chart_reader import analyze_chart
from .four_book_curriculum import single_book_model_lesson

_EDITIONS=frozenset(("TC1","NTD","TC2","COURSE"))
_STUDY_SCENARIOS=frozenset((
    "CONTINUATION_UP","CONTINUATION_DOWN","CORRECTION_OR_COMPLEX_RANGE",
    "POSSIBLE_REVERSAL_UP","POSSIBLE_REVERSAL_DOWN",
))


def _level(x: Any)->float | None:
    if x is None:return None
    if (isinstance(x,bool) or not isinstance(x,(int,float))
            or not isfinite(x) or x<=0):
        raise ValueError("UNSAFE_STRUCTURAL_PRICE_REFERENCE")
    return float(x)


def rehearse_trade_reasoning(
    symbol: str, timeframe: str, bars: list[dict[str,Any]], *,
    edition: str="TC2", tick_size: float,
) -> dict[str,Any]:
    """Produce a reproducible teacher's decision sheet from closed bars.

    No future candle is given to the teacher, hypothesis or local Qwen.
    The last visible as_of must equal the last FINALIZED candle. Scenario
    falsifiers are context levels, not stop orders, and probabilities are
    deliberately unavailable until separately calibrated.
    """
    if edition not in _EDITIONS:raise ValueError("UNKNOWN_BOOK_EDITION")
    if (not isinstance(bars,list) or len(bars)<60 or len(bars)>50000
            or not isinstance(bars[-1],dict)):
        raise ValueError("INSUFFICIENT_OR_MALFORMED_CLOSED_HISTORY")
    as_of=bars[-1].get("close_time")
    if not isinstance(as_of,str):
        raise ValueError("MISSING_CLOSED_MARKET_CUTOFF")
    current=analyze_chart(symbol,timeframe,bars,as_of=as_of,tick_size=tick_size)
    if (current.get("gaps") != 0 or current.get("execution_allowed") is not False
            or current.get("last_closed_at")!=current.get("as_of")):
        raise ValueError("GAPPED_OR_NONCAUSAL_MARKET_REJECTED")
    fact=current["principle_evidence"]["observable_facts"]
    levels=current["market_position"]["recent_levels"]
    lo=_level(levels.get("latest_confirmed_low"))
    hi=_level(levels.get("latest_confirmed_high"))
    if lo is not None and hi is not None and lo>=hi:
        raise ValueError("CONTRADICTORY_CONFIRMED_SWING_BOUNDS")
    last=_level(levels["close"])
    context=current["evidence"]["market_context"]
    momentum=fact["momentum"]
    alternatives=[]
    for s in current["principle_evidence"]["competing_market_scenarios"]:
        if (s.get("hypothesis") not in _STUDY_SCENARIOS
                or s.get("trade_action") is not None
                or s.get("probability") is not None):
            raise ValueError("UNVERIFIED_PROBABILISTIC_SCENARIO")
        kind=s["hypothesis"]
        # A known structurally confirmed boundary is a measurable
        # falsification context, not a promise of actual future reversal.
        invalidate=(
            {"if_closed_below_prior_confirmed_low":lo}
            if kind in ("CONTINUATION_UP","POSSIBLE_REVERSAL_UP") else
            {"if_closed_above_prior_confirmed_high":hi}
            if kind in ("CONTINUATION_DOWN","POSSIBLE_REVERSAL_DOWN") else
            {"needs_new_confirmed_pivots":True}
        )
        alternatives.append({
            "hypothesis":kind,"state":s["status"],
            "supporting_observation_codes":s["supporting_observations"][:5],
            "contradiction_codes":s["contradicting_observations"][:5],
            "next_research_only_falsification_reference":invalidate,
            "wave_degree_confirmed":False,
            "prediction_probability":None,"price_action_issued":False,
        })
    # Never use a fallback "one wave count is guaranteed". Include
    # hypotheses on both sides and the correction/range alternative.
    if len(alternatives)!=5:
        raise ValueError("INCOMPLETE_COMPETING_MARKET_SCENARIOS")
    eligible=[
        {"setup":p["setup"],"side":p["side"],"state":p["status"],
         "source_rule":p["source_rule_id"]}
        for p in current["signal_proofs"]["observed_proofs"]
        if p.get("is_current_eligible_observation") is True
        and p.get("as_of")==current["as_of"]
        and isinstance(p.get("source_rule_id"),str)
        and p["source_rule_id"].startswith(edition+".")
    ][:8]
    if edition=="TC2":
        from .tc2_trade_lifecycle_coach import teach_tc2_trade_lifecycle
        tc2=teach_tc2_trade_lifecycle(current,bars,tick_size=tick_size)
        # A teacher must not cite WO candidate if it lacks a price trigger.
        setup_gate={
            "book":"TC2","status":"AWAIT_FUTURE_VERIFIED_PRICE_TRIGGER",
            "earliest_signal_can_be":"WM1_OR_WM2_OR_WM3",
            "entry_stage_is_not_exchange_fill":True,
            "current_entry_observations":len(tc2["entry_observations"]),
            "risk_notes":"First SL structural; inverse pyramid only after profit and protection.",
            "exit_notes":"An opposite Wise Man without filled reversal cannot close by itself.",
        }
    elif edition=="NTD":
        # Chart_reader intentionally uses a CLOSED, read-only *unadmitted*
        # NTD first-entry candidate. Do not upgrade it from AO, AC or zone.
        ntd=current["edition_evidence"]["NTD"]
        if ntd.get("entry_admitted") is not False:
            raise ValueError("NTD_INITIAL_FRACTAL_ADMISSION_NOT_VERIFIED")
        setup_gate={
            "book":"NTD","status":"INITIAL_OUTSIDE_ALLIGATOR_FRACTAL_TRIGGER_NOT_VERIFIED",
            "no_independent_initial_AO_AC_ZONE_BALANCE_ENTRY":True,
            "exit_notes":"Teeth stop-on-close and real exchange SL are different mechanisms.",
        }
    elif edition=="TC1":
        setup_gate={
            "book":"TC1","status":"WAVE_AND_MAGIC_BULLETS_OBSERVATIONS_ONLY",
            "rhythm_100_140_bars_is_context_not_trigger":True,
            "missing_tick_volume_must_not_be_imputed":True,
        }
    else:
        setup_gate={
            "book":"COURSE","status":"HUMP_THUMB_TRADING_PARTNER_NOT_VISUALLY_APPROVED",
            "nested_waves_remain_alternative_hypotheses":True,
            "scanned_diagrams_need_independent_visual_review":True,
        }
    # Exact timestamp + closed-prefix digest uniquely identifies a
    # prospective lesson. NO future returns or fill status in this digest.
    safe_bars=[{key:r[key] for key in (
        "close_time","open","high","low","close","volume","closed")}
        for r in bars]
    digest=sha256(json.dumps(safe_bars,sort_keys=True,
                 separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    return {
        "mode":"BOOK_SCOPED_CAUSAL_PRACTICAL_TRADER_REHEARSAL",
        "symbol":symbol,"timeframe":timeframe,"as_of":current["as_of"],
        "book":edition,"closed_bar_count":len(bars),
        "review_prefix_sha256":digest,
        "market_context":context,
        "ao_sign":momentum["ao_sign"],
        "ao_direction":momentum["ao_change"],
        "ac_direction":momentum["ac_change"],
        "confirmed_swing_levels_not_orders":{
            "high":hi,"low":lo,"last_close":last,
        },
        "five_competing_falsifiable_scenarios":alternatives,
        "same_edition_current_signal_observations_not_fills":eligible,
        "edition_specific_entry_gate":setup_gate,
        "source_learning_objectives":single_book_model_lesson(edition)["lessons"],
        "next_closed_bar_review_questions":[
            "Which hypothesis gained or lost evidence AFTER the next real candle?",
            "Was the setup price trigger actually seen, or just an indicator signal?",
            "Did a verified exchange fill and protective stop exist? NEVER infer from OHLC.",
            "What new structural pivot would falsify the current scenario?",
            "Did we observe trend, a nested correction, or unresolved mixed waves?",
            "Would fees, slippage, funding and portfolio cap still permit a paper entry?",
        ],
        "autonomous_execution_allowed":False,
        "future_bars_read":False,"claimed_win_rate":None,
        "elliott_wave_degree_certified":False,
        "model_weights_modified_here":False,
        "independently_expert_labelled_chart":False,
        "trade_action":None,"execution_allowed":False,
    }
