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


def evaluate_historical_thought(
    symbol:str, rows:list[dict[str,Any]], *,
    tick_size:float,bar_index:int,edition:str="TC2",
    model:Any=None,proposal:dict[str,Any]|None=None,
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
    submitted=model.propose_chart_reasoning(evidence,edition=edition) if model is not None else proposal
    result=audit_reasoning(evidence,submitted)
    reason=submitted.get("reason","") if isinstance(submitted,dict) else ""
    return {
        "mode":"LOCAL_BLINDED_REASONING_TRIAL",
        "symbol":symbol,"edition":edition,
        "last_visible_bar_index":bar_index,
        "last_visible_at":as_of,
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
        edition=a.edition,model=model,proposal=proposal)
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
