"""Evaluate REAL local GGUF book reasoning on one HISTORICAL CLOSED prefix.

Usage:
python -m williams_ai.reasoning_trial --symbol ETHUSDT --bar-index 900 \
  --edition TC2 --local-model /private/real-model.gguf

Or inspect a saved proposal without a GGUF:
python -m williams_ai.reasoning_trial --symbol ETHUSDT --bar-index 900 \
  --edition TC2 --proposal-file /private/hypothesis.json

The next candle is NEVER exposed to the GGUF or the rule critic. No orders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .algorithm_guard import audit_reasoning
from .causal_paper_lab import FROZEN_DATA,frozen_markets
from .chart_reader import analyze_chart
from .causal_timeframes import aggregate_closed_h1


def evaluate_historical_thought(
    symbol:str, rows:list[dict[str,Any]], *,
    tick_size:float,bar_index:int,edition:str="TC2",
    model:Any=None,proposal:dict[str,Any]|None=None,
    include_topdown:bool=True,
    book_index:Path|None=None,
    coach_notebook:Path|None=None,
    model_id:str|None=None,
)->dict[str,Any]:
    """bar_index is a 0-based last closed bar, not next bar to trade.

    OHLCV that follows bar_index is excluded before ANY analysis call.
    """
    if (not isinstance(bar_index,int) or isinstance(bar_index,bool)
            or not 60<=bar_index<len(rows)):
        raise ValueError("Need an index of a closed bar after warmup")
    if (model is None)==(proposal is None):
        raise ValueError("Provide exactly one of real local GGUF or saved proposal")
    prefix=rows[:bar_index+1]
    as_of=prefix[-1]["close_time"]
    evidence=analyze_chart(symbol,"H1",prefix,as_of=as_of,tick_size=tick_size)
    if evidence["gaps"] or evidence["execution_allowed"] is not False:
        raise ValueError("No causal and contiguous market history")
    if include_topdown:
        # Efficiently reconstruct only completely CLOSED higher bars. Reuse
        # already calculated H1 observation, avoiding an expensive duplicate
        # full H1 chart analysis on every historical checkpoint.
        higher={}
        for tf in ("D1","H4"):
            closed=aggregate_closed_h1(prefix,tf)
            if len(closed)<45:
                higher[tf]={"status":"INSUFFICIENT_CLOSED_HISTORY",
                            "bars":len(closed),"wave_degree":"UNKNOWN"}
            else:
                tf_view=analyze_chart(
                    symbol,tf,closed,as_of=as_of,tick_size=tick_size)
                higher[tf]={
                    "status":"OBSERVED_CLOSED_BARS",
                    "bars":len(closed),
                    "closed_through":closed[-1]["close_time"],
                    "market_context":tf_view["evidence"]["market_context"],
                    "ao":tf_view["evidence"]["ao"],
                    "ac":tf_view["evidence"]["ac"],
                    "wave_degree":tf_view["wave_structure"]["wave_degree"],
                    "wave_hypotheses_unverified":True,
                    "recent_levels":tf_view["market_position"]["recent_levels"],
                    "nonlinear_structure":tf_view["principle_evidence"]
                        ["observable_facts"]["multi_resolution_price_structure"],
                    "supported_not_proven_market_scenarios":[
                        q["hypothesis"] for q in tf_view["principle_evidence"]
                            ["competing_market_scenarios"]
                        if q["status"]=="EVIDENCE_COMPATIBLE_NOT_PREDICTION"
                    ],
                }
        from .market_hierarchy import reconcile_market_hierarchy
        frames_for_relations={**higher,"H1":{
            "status":"OBSERVED_CLOSED_BARS",
            "closed_through":evidence["last_closed_at"],
            "market_context":evidence.get("evidence",{}).get("market_context","UNKNOWN"),
            "ao":evidence.get("evidence",{}).get("ao"),
            "ac":evidence.get("evidence",{}).get("ac"),
            "recent_levels":evidence["market_position"]["recent_levels"],
        }}
        higher["conditional_parent_child_relations"] = (
            reconcile_market_hierarchy(frames_for_relations)["parent_child_relations"]
        )
        from .cross_scale_wave_theses import cross_scale_wave_theses
        # These hypotheses remain competing and falsifiable. Never claim
        # that AO near zero proves wave four or that H1 correction is a
        # certified higher degree. All TF passports are fully CLOSED.
        higher["cross_scale_wave_theses"] = cross_scale_wave_theses(
            frames_for_relations,as_of=evidence["as_of"])
        evidence["higher_timeframe_context"]=higher
    sources=[]
    if book_index is not None:
        if model is None:
            raise ValueError("Private source pages require a locally running model")
        from .grounded_inference import verified_private_book_pages
        current=evidence.get("candidates",[])
        latest=current[-1]["type"] if current else None
        sources=verified_private_book_pages(book_index,edition,selected_setup=latest)
    prior=[]
    if coach_notebook is not None:
        if model is None or model_id is None:
            raise ValueError("Private lessons require an actual model fingerprint")
        from .adaptive_coach import recall_model_corrections
        related=[]
        names={"WM1":("TC2.WM1.BAR","TC2.WM1.ANG","TC2.WM1.TRIGGER"),
               "WM2_SUPER_AO":("TC2.WM2",),
               "WM3_FRACTAL":("TC2.WM3",)}
        for candidate in evidence.get("candidates",[])[-12:]:
            if isinstance(candidate,dict):
                related.extend(names.get(candidate.get("type"),()))
        if edition=="TC1":
            related.append("TC1.FRACTAL")
        if edition=="NTD":
            related.extend(("NTD.AO","NTD.AC","NTD.FRACTAL"))
        focus=tuple(sorted(set(
            x for x in related if x.startswith(edition+".")
        )))
        prior=recall_model_corrections(coach_notebook,model_id,edition,
                                       focus_rule_ids=focus,limit=2)
        # If the active signals have no matching corrected rules, still
        # recall a single general lesson in the SAME edition. Otherwise a
        # model that studied TC1 wave degree but sees only TC1 fractals
        # would never receive its prior book knowledge.
        if not prior and focus:
            prior=recall_model_corrections(coach_notebook,model_id,edition,
                                           limit=1)
    if model is not None:
        args={"edition":edition}
        if book_index is not None:
            args["primary_pages"]=sources
        if coach_notebook is not None:
            args["prior_lessons"]=prior
        submitted=model.propose_chart_reasoning(evidence,**args)
    else:
        submitted=proposal
    result=audit_reasoning(evidence,submitted)
    reason=submitted.get("reason","") if isinstance(submitted,dict) else ""
    return {
        "mode":"LOCAL_BLINDED_REASONING_TRIAL",
        "symbol":symbol,"edition":edition,
        "last_visible_bar_index":bar_index,
        "last_visible_at":as_of,
        "private_book_pages_used":[page["source"] for page in sources],
        "private_prior_mistake_lessons_used":[note["source"] for note in prior],
        "private_book_passages_exported":False,
        "private_coaching_text_exported":False,
        "top_down_context":evidence.get("higher_timeframe_context",{}),
        "model_saw_future_bars":False,
        "model_weights_trained_here":False,
        "book_mastery_percent":None,
        "model_explanation":reason[:600] if isinstance(reason,str) else "",
        "audit":result,
        "trade_action":None,"execution_allowed":False,
    }


def main()->None:
    p=argparse.ArgumentParser(description="Blind by timestamp, source-scoped AI reasoning trial")
    p.add_argument("--dataset",type=Path,default=FROZEN_DATA)
    p.add_argument("--symbol",required=True)
    p.add_argument("--bar-index",required=True,type=int)
    p.add_argument("--edition",choices=["TC1","NTD","TC2","COURSE"],default="TC2")
    p.add_argument("--no-topdown",action="store_true",help="Disable complete H4/D1 context")
    p.add_argument("--book-index",type=Path,help="Private indexed four-book SQLite outside repository")
    p.add_argument("--coach-notebook",type=Path,
                   help="Private GGUF-specific lesson notebook generated by adaptive_coach")
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--local-model",type=Path)
    group.add_argument("--proposal-file",type=Path)
    a=p.parse_args()
    markets=frozen_markets(a.dataset)
    if a.symbol not in markets:
        p.error("Unknown symbol in selected frozen dataset")
    model=None
    proposal=None
    if a.local_model:
        from .local_model import LocalWilliamsModel
        model=LocalWilliamsModel(a.local_model)
    else:
        proposal=json.loads(a.proposal_file.read_text(encoding="utf-8"))
    x=markets[a.symbol]
    if a.coach_notebook and a.local_model is None:
        p.error("--coach-notebook requires --local-model")
    model_id=None
    if a.coach_notebook:
        from .study_cli import _model_identity
        model_id=_model_identity(a.local_model)
    output=evaluate_historical_thought(
        a.symbol,x["bars"],tick_size=x["tick_size"],bar_index=a.bar_index,
        edition=a.edition,model=model,proposal=proposal,
        include_topdown=not a.no_topdown,book_index=a.book_index,
        coach_notebook=a.coach_notebook,model_id=model_id)
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
