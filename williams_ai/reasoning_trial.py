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
                }
        evidence["higher_timeframe_context"]=higher
    sources=[]
    if book_index is not None:
        if model is None:
            raise ValueError("Private source pages require a locally running model")
        from .grounded_inference import verified_private_book_pages
        current=evidence.get("candidates",[])
        latest=current[-1]["type"] if current else None
        sources=verified_private_book_pages(book_index,edition,selected_setup=latest)
    if model is not None:
        submitted=(model.propose_chart_reasoning(evidence,edition=edition,
                    primary_pages=sources) if book_index is not None
                   else model.propose_chart_reasoning(evidence,edition=edition))
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
        "private_book_passages_exported":False,
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
    output=evaluate_historical_thought(
        a.symbol,x["bars"],tick_size=x["tick_size"],bar_index=a.bar_index,
        edition=a.edition,model=model,proposal=proposal,
        include_topdown=not a.no_topdown,book_index=a.book_index)
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
