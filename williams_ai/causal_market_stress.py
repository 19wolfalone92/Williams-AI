"""Adversarial closed-chart research evaluation on 15 known public H1 markets.

Not a real GGUF evaluation or subjective Elliott chart label. It verifies that
every analyzed H1 cut, derived H4/D1 view, putative wave hypothesis and source
claim can be independently reconstructed WITHOUT future bars. A negative
control deliberately invents future knowledge, fill/stop evidence and verified
wave degree: the independent source critic MUST reject all four false claims.

This benchmark is a software-causality stress test on ALREADY REVIEWED data,
NOT a new untouched prospective holdout, profitable strategy or book mastery.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from typing import Any

from .algorithm_guard import audit_reasoning
from .causal_paper_lab import frozen_markets
from .chart_reader import analyze_chart
from .causal_timeframes import aggregate_closed_h1
from .cross_scale_wave_theses import cross_scale_wave_theses
from .market_hierarchy import reconcile_market_hierarchy

# These times are fixed before ANY stress study outcomes are observed.
CUTOFFS = (699,899,1099)
FUTURE_HOURS = 24
BOOKS = ("TC1","NTD","TC2","COURSE")
EXPECTED_HARD_FAILURES = {
    "NO_EXCHANGE_FILL_LEDGER",
    "CANNOT_CONFIRM_EXCHANGE_FILL_FROM_OHLC",
    "CANNOT_CONFIRM_EXCHANGE_SL_FROM_OHLC",
    "WAVE_DEGREE_UNVERIFIABLE_FROM_CURRENT_EVIDENCE",
    "FUTURE_CANDLE_CLAIM",
}


def _hash(obj: Any) -> str:
    return sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,
             separators=(",",":"),allow_nan=False).encode("utf8")).hexdigest()


def _candidate_summary(view: dict[str,Any]) -> dict[str,Any]:
    """Observable evidence only. Do not convert a candidate into a fill."""
    from .decision_contract import contract_for_model
    contract=contract_for_model({
        "as_of":view["as_of"],"candidates":view["candidates"],
        "execution_allowed":False,"gaps":view["gaps"],
    },"TC2")
    return {
        "current_candidates_by_type":dict(Counter(
            x.get("type","UNKNOWN") for x in view["candidates"]
            if isinstance(x,dict))),
        "market_context":view["evidence"]["market_context"],
        "zone":view["evidence"]["zone"],
        "visible_teeth":view["evidence"]["teeth_visible"],
        "open_wm3_evidence_accepted_only_as_observation":contract.get(
            "wm3_valid_setup_side_entry_state_combinations",[]),
        "wave_degree_verified":False,
        "automatic_market_action":None,
    }


def _top_down(prefix: list[dict[str,Any]], symbol:str,tick:float) -> dict[str,Any]:
    as_of=prefix[-1]["close_time"]
    h1=analyze_chart(symbol,"H1",prefix,as_of=as_of,tick_size=tick)
    frames={"H1":{
        "status":"OBSERVED_CLOSED_BARS",
        "closed_through":h1["as_of"],
        "market_context":h1["evidence"]["market_context"],
        "ao":h1["evidence"]["ao"],
        "recent_levels":h1["market_position"]["recent_levels"],
    }}
    for tf in ("H4","D1"):
        whole=aggregate_closed_h1(prefix,tf)
        if len(whole)<45:
            frames[tf]={"status":"INSUFFICIENT_CLOSED_HISTORY",
                         "wave_degree":"UNKNOWN"}
        else:
            view=analyze_chart(symbol,tf,whole,
                               as_of=as_of,tick_size=tick)
            frames[tf]={
                "status":"OBSERVED_CLOSED_BARS",
                "closed_through":view["last_closed_at"],
                "market_context":view["evidence"]["market_context"],
                "ao":view["evidence"]["ao"],
                "recent_levels":view["market_position"]["recent_levels"],
                "wave_degree":"UNKNOWN",
            }
    relations=reconcile_market_hierarchy(frames)
    waves=cross_scale_wave_theses(frames,as_of=as_of)
    return {
        "h1":h1,"frames":frames,
        "relationships":relations,
        "wave_alternatives":waves,
    }


def _false_model_claim(symbol: str, as_of: str, edition: str) -> dict[str,Any]:
    # A malicious hallucination should fail even when there IS a real
    # fractal or momentum event visible in the last candle.
    return {
        "symbol":symbol,"as_of":as_of,"edition":edition,
        "setup":"NO_SETUP","side":"NONE",
        "entry_state":"EXCHANGE_FILLED",
        "exit_basis":"SL_CONFIRMED",
        "rule_ids":[],
        "wave_degree_verified":True,
        "exchange_fill_confirmed":True,
        "exchange_sl_confirmed":True,
        "future_candle_used":True,
        "trade_action":None,"execution_allowed":False,
        "reason":"No trading permission can derive from a historical H1 chart.",
    }


def stress_one(symbol:str,rows:list[dict[str,Any]],tick:float,index:int)->dict[str,Any]:
    """Score causal integrity at an old closed bar; all futures withheld."""
    if not isinstance(index,int) or isinstance(index,bool) or index<220:
        raise ValueError("Invalid fixed H1 cutoff")
    if index+FUTURE_HOURS>=len(rows):
        raise ValueError("Not enough future bars for adversarial contamination")
    prefix=rows[:index+1]
    as_of=prefix[-1]["close_time"]
    observed=_top_down(prefix,symbol,tick)
    view=observed["h1"]
    if (view["gaps"] or view["as_of"]!=view["last_closed_at"]
            or view["execution_allowed"] is not False
            or view["wave_structure"].get("wave_degree")!="UNKNOWN"
            or observed["wave_alternatives"]["execution_allowed"] is not False
            or observed["relationships"]["execution_allowed"] is not False):
        raise ValueError("Causal or wave certification invariant failed")
    future=rows[index+1:index+1+FUTURE_HOURS]
    assert len(future)==FUTURE_HOURS
    # POISON every withheld future candle. There must be NO way to get a
    # different old observation from this modified future: only the exact
    # closed prefix is ever allowed to reach analyzers.
    tampered=copy.deepcopy(rows[:index+FUTURE_HOURS+1])
    for i in range(index+1,len(tampered)):
        c=tampered[i]
        for f in ("open","high","low","close"):
            c[f]=float(c[f])*(1.4 if (i-index)%2 else 0.6)
    unchanged=(
        _hash(_top_down(tampered[:index+1],symbol,tick))
        ==_hash(observed)
    )
    if not unchanged:
        raise ValueError("Historical chart changed when only FUTURE bars changed")
    future_pass_rejected=False
    try:
        analyze_chart(symbol,"H1",tampered,as_of=as_of,tick_size=tick)
    except ValueError as exc:
        future_pass_rejected="NO_LOOKAHEAD" in str(exc)
    if not future_pass_rejected:
        raise ValueError("Analyzer accepted futures as already closed")
    book_guard={}
    for edition in BOOKS:
        rejected=audit_reasoning(view,_false_model_claim(symbol,as_of,edition))
        failures=set(rejected["hard_failures"])
        if rejected["verdict"]!="UNSUPPORTED_CLAIM" or not EXPECTED_HARD_FAILURES.issubset(failures):
            raise ValueError("Unjustified model certainty slipped past independent critic")
        book_guard[edition]=sorted(EXPECTED_HARD_FAILURES.intersection(failures))
    observed_summary=_candidate_summary(view)
    relationships=observed["wave_alternatives"]["parents_and_children"]
    for relation in relationships:
        if relation.get("verified_wave_degree") is not False:
            raise ValueError("Elliott wave level was incorrectly certified")
        for alternative in relation.get("alternatives",[]):
            if alternative.get("trade_action","INVALID") is not None:
                raise ValueError("Historical wave hypothesis became trade execution")
            if alternative.get("probability","NOT_EXPLICIT") is not None:
                raise ValueError("Uncalibrated wave probability was invented")
    return {
        "symbol":symbol,
        "at_index":index,"as_of":as_of,
        "prefix_hash":_hash(prefix),
        "current_h1_facts":observed_summary,
        "h4_status":observed["frames"]["H4"]["status"],
        "d1_status":observed["frames"]["D1"]["status"],
        "top_down_hypotheses": [{
            "parent":q["parent"],"child":q["child"],
            "status":q["status"],
            "not_confirmed":not q["verified_wave_degree"],
            "unproven_alternative_count":len(q.get("alternatives",[])),
        } for q in relationships],
        "future_24h_withheld":True,
        "future_poison_has_no_effect":unchanged,
        "passing_future_to_analyzer_hard_rejected":future_pass_rejected,
        "all_four_book_fake_executions_rejected":len(book_guard)==len(BOOKS),
        "book_guard_falsification_codes":book_guard,
        "real_GGUF_was_queried":False,
        "wave_truth_expert_label_available":False,
        "trade_action":None,"execution_allowed":False,
    }


def run_closed_chart_stress() -> dict[str,Any]:
    markets=frozen_markets()
    if len(markets)!=15:
        raise ValueError("Expected exactly reviewed 15-pair historical corpus")
    cases=[]
    for symbol in sorted(markets):
        history=markets[symbol]
        for index in CUTOFFS:
            cases.append(stress_one(symbol,history["bars"],
                                    history["tick_size"],index))
    return {
        "mode":"REVIEWED_HISTORICAL_CAUSAL_CHART_STRUCTURE_ADVERSARIAL_AUDIT",
        "historical_cases":len(cases),
        "symbols":len(markets),"fixed_cutoffs":list(CUTOFFS),
        "future_injection_rejected":sum(x["passing_future_to_analyzer_hard_rejected"] for x in cases),
        "changed_future_never_changed_old_decision":all(x["future_poison_has_no_effect"] for x in cases),
        "fake_fills_wave_certainty_and_future_claims_rejected_by_four_books":sum(
            x["all_four_book_fake_executions_rejected"] for x in cases),
        "h4_frames_with_enough_history":sum(
            x["h4_status"]=="OBSERVED_CLOSED_BARS" for x in cases),
        "d1_frames_with_enough_history":sum(
            x["d1_status"]=="OBSERVED_CLOSED_BARS" for x in cases),
        "real_GGUF_accuracy_proved":False,
        "independent_expert_wave_labels_provided":False,
        "independent_unseen_data":False,
        "live_trade_qualification":"NOT_QUALIFIED",
        "cases":cases,
        "trade_action":None,"execution_allowed":False,
    }


def main() -> None:
    parser=argparse.ArgumentParser(description="Verify falsifiable closed-chart structure without future leakage")
    parser.add_argument("--summary",action="store_true")
    args=parser.parse_args()
    report=run_closed_chart_stress()
    if args.summary:
        report.pop("cases")
    print("CAUSAL_CHART_STRESS_RESULT",
          json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
