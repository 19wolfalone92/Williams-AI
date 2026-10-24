"""Prequential learner of AFTER-COST paper returns, not just win probability.

High accuracy or high P(win) can coexist with NEGATIVE expectancy. This
experiment fits a small regularized, Huber-gradient numerical regressor,
shared across cryptocurrency markets with a source-scoped setup/side bias.

Only a settled result strictly before the next decision and before the
global frozen training cutoff may update weights. Validation and final
holdout remain unseen during training; all model choice is validation-only.

Risk limits and executable entries are not inferred from this model; labels
are TC2 proxy fixed-horizon paper outcomes, not book-valid campaigns.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime
import hashlib
import heapq
import json
import math
from typing import Any

from .continual_trader import Outcome,PrivateExperience,_finite_vector,_time
from .paper_learning import FEATURES

VERSION="WILLIAMS_AI_PREQUENTIAL_HUBER_AFTER_FRICTIONS_PAPER_V1"
SETUPS=("WM1","WM2","WM3")
SIDES=("LONG","SHORT")
COHORTS=tuple((setup,side) for setup in SETUPS for side in SIDES)
N=len(FEATURES)+len(COHORTS)
MIN_PREDICTED_NET=0.003
# Predeclared research constants, NOT selected on frozen test.
ETA=0.004
L2=0.0005
HUBER_DELTA=0.04


def _cohort(ex:Outcome)->int:
    return COHORTS.index((ex.setup,ex.side))


def _predict(w:list[float],ex:Outcome)->float:
    x=_finite_vector(ex.features)
    raw=sum(a*b for a,b in zip(w[:len(FEATURES)],x))
    raw+=w[len(FEATURES)+_cohort(ex)]
    return max(-0.5,min(0.5,raw))


def _train(w:list[float],ex:Outcome)->None:
    # Regression targets are REAL simulated net returns (including fees).
    # Huber bounds the gradient on a one-off liquidation/gap tail; the
    # ACTUAL un-clipped tail still counts in evaluation metrics.
    delta=_predict(w,ex)-ex.net_return
    g=max(-HUBER_DELTA,min(HUBER_DELTA,delta))
    x=_finite_vector(ex.features)
    for i,val in enumerate(x):
        w[i]-=ETA*(g*val+(L2*w[i] if i else 0))
    cohort_i=len(FEATURES)+_cohort(ex)
    w[cohort_i]-=ETA*(g+L2*w[cohort_i])


def _scoring(rows:list[tuple[Outcome,float]],base:dict[tuple[str,str],float]
            )->dict[str,Any]:
    if not rows:
        return {"examples":0,"status":"EMPTY_PURGED_PERIOD"}
    n=len(rows)
    mse=sum((pred-x.net_return)**2 for x,pred in rows)/n
    bmse=sum((base[(x.setup,x.side)]-x.net_return)**2 for x,_ in rows)/n
    selected=[x for x,p in rows if p>=MIN_PREDICTED_NET]
    gains=sum(x.net_return for x in selected if x.net_return>0)
    losses=-sum(x.net_return for x in selected if x.net_return<0)
    return {
        "examples":n,
        "distinct_utc_dates":len({_time(x.decision_at).date() for x,_ in rows}),
        "assets":len({x.symbol for x,_ in rows}),
        "paper_net_mse":round(mse,10),
        "cohort_constant_training_mean_mse":round(bmse,10),
        "paper_selected_ideas":len(selected),
        "selected_different_utc_days":len({_time(x.decision_at).date() for x in selected}),
        "selected_assets":len({x.symbol for x in selected}),
        "selected_positive_fraction":(
            round(sum(x.net_return>0 for x in selected)/len(selected),6)
            if selected else None),
        "selected_mean_net_after_proxy_costs":(
            round(sum(x.net_return for x in selected)/len(selected),9)
            if selected else None),
        "selected_proxy_profit_factor":(
            round(gains/losses,6) if losses>0 else None),
        "fixed_paper_expected_return_threshold":MIN_PREDICTED_NET,
        "no_same_symbol_or_cluster_portfolio_bookkeeping":True,
        "outcomes_are_not_actual_trades":True,
    }


def causal_expected_net_study(
    ledger:PrivateExperience,*,train_until:str,validation_until:str,
    observe_until:str,min_training_decisions:int=40,
    min_validation_decisions:int=12,
)->dict[str,Any]:
    a,b,c=(_time(z) for z in (train_until,validation_until,observe_until))
    if not a<b<c:raise ValueError("CHRONOLOGICAL_HOLDOUT_CUTOFF_INVALID")
    if (not isinstance(min_training_decisions,int)
            or isinstance(min_training_decisions,bool)
            or min_training_decisions<1
            or not isinstance(min_validation_decisions,int)
            or isinstance(min_validation_decisions,bool)
            or min_validation_decisions<1):
        raise ValueError("INVALID_PREDECLARED_MINIMUMS")
    cases=[
        x for x in ledger.samples(resolved_before=observe_until)
        if x.setup in SETUPS and _time(x.decision_at)<c
    ]
    grouped:dict[datetime,list[Outcome]]=defaultdict(list)
    for x in cases:grouped[_time(x.decision_at)].append(x)
    weights=[0.0]*N
    pending=[]
    seq=0
    updates=0
    actually_settled=[]
    rows={"train":[],"validation":[],"test":[]}
    purged={"train":0,"validation":0,"test":0}
    for t in sorted(grouped):
        # Training is ON-LINE but ONLY when it becomes known. Futures from
        # BTC cannot enter earlier ETH decisions even on the same timestamp.
        while pending and pending[0][0]<t and pending[0][0]<a:
            _,_,item=heapq.heappop(pending)
            _train(weights,item)
            actually_settled.append(item)
            updates+=1
        partition="train" if t<a else ("validation" if t<b else "test")
        cutoff={"train":a,"validation":b,"test":c}[partition]
        for x in grouped[t]:
            resolution=_time(x.resolved_at)
            known_at_cutoff=(resolution<=cutoff if partition=="test" else
                             resolution<cutoff)
            if not known_at_cutoff:
                purged[partition]+=1
                continue
            pred=_predict(weights,x)
            rows[partition].append((x,pred))
            if partition=="train":
                heapq.heappush(pending,(resolution,seq,x))
                seq+=1
    while pending and pending[0][0]<a:
        _,_,item=heapq.heappop(pending)
        _train(weights,item)
        actually_settled.append(item)
        updates+=1
    train_days={_time(x.decision_at) for x in actually_settled}
    validation_days={_time(x.decision_at) for x,_ in rows["validation"]}
    if len(train_days)<min_training_decisions or updates<min_training_decisions:
        raise ValueError("INSUFFICIENT_SETTLED_TRAINING_EPISODES")
    if len(validation_days)<min_validation_decisions or not rows["test"]:
        raise ValueError("INSUFFICIENT_PURGED_UNTOUCHED_HOLDOUT_EPISODES")
    global_mean=sum(x.net_return for x in actually_settled)/len(actually_settled)
    cohorts:dict[tuple[str,str],list[float]]=defaultdict(list)
    for x in actually_settled:cohorts[(x.setup,x.side)].append(x.net_return)
    base={key:(sum(cohorts[key])/len(cohorts[key])
               if cohorts[key] else global_mean)
          for key in COHORTS}
    valid=_scoring(rows["validation"],base)
    final=_scoring(rows["test"],base)
    # Both error AND historically selected net must support any positive
    # research statement. Multiple symbols in one day are not many epochs.
    viable=(
        valid["cohort_constant_training_mean_mse"]-
            valid["paper_net_mse"]>0
        and valid["paper_selected_ideas"]>=12
        and valid["selected_different_utc_days"]>=6
        and valid["selected_assets"]>=3
        and valid["selected_mean_net_after_proxy_costs"] is not None
        and valid["selected_mean_net_after_proxy_costs"]>0
        and valid["selected_proxy_profit_factor"] is not None
        and valid["selected_proxy_profit_factor"]>1.2
    )
    # The historical test is reported for falsification ONLY.
    state={"mode":VERSION,
           "frozen_train_until":a.isoformat(),
           "frozen_validation_until":b.isoformat(),
           "frozen_observe_until":c.isoformat(),
           "learned_weights":[round(q,12) for q in weights],
           "trained_after_prior_settlement":updates,
           "truly_distinct_training_decision_times":len(train_days),
           "train_cohort_average_net":{
               ":".join(k):round(v,9) for k,v in sorted(base.items())},
           "train_cutoff_purged_late_outcomes":purged,
           "validation_frozen":valid,"test_frozen":final,
           "validation_qualified_for_additional_research":bool(viable),
           "qualification":(
                "PROMISING_PAPER_ONLY_NEEDS_NEW_PROSPECTIVE_DATA" if viable
                else "ABSTAIN_EXPECTED_RETURN_NOT_VALIDATED"),
           "validation_and_test_never_update_model_weights":True,
           "paper_fixed_horizon_is_not_book_exit":True,
           "model_is_small_numeric_not_GGUF_finetuning":True,
           "current_execution_not_authorized":True,
           "trade_action":None,"execution_allowed":False}
    state["fingerprint"]=hashlib.sha256(
        json.dumps(state,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return state
