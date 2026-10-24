"""Profit-factor and expectancy research from RESOLVED TC2 paper episodes.

The classifier's probability of making money is NOT expected profitability.
Even 90% wins can lose money after 10% oversized losses and commissions.
This deliberately conservative, edition-specific study uses ONLY past
settled outcomes, freezes its learned evidence at the training cutoff,
and evaluates on disjoint historical validation / final periods. Cross-token
dates are clustered rather than counted as independent experiments.

No execution permissions, model promotion, price forecasts, Binance API,
GGUF weight updates, or claims of statistically independent days.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import heapq
import math
from typing import Any

from .continual_trader import PrivateExperience, Outcome, _time

SOURCE="TC2_PROVISIONAL_PAPER_WM1_WM2_WM3"
THRESHOLD_DAYS=12
THRESHOLD_ASSETS=3
MIN_GAIN_TO_PAIN=1.25
# Research tolerance, not calibrated probability or live-trading criterion.
LOWER_BOUND_Z=1.96


def _past_edge(rows: list[Outcome]) -> dict[str, Any]:
    if not rows:
        return {"status":"INSUFFICIENT_SETTLED_PAPER_EVIDENCE",
                "episodes":0,"distinct_utc_dates":0,"assets":0,
                "lower_bound_mean_net":None,"paper_profit_factor":None,
                "qualified":False}
    # All same-day crypto outcomes share substantial risk. First average
    # within the day; never pretend 15 simultaneous coins are 15 days.
    by_date:dict[str,list[float]]=defaultdict(list)
    for row in rows:
        by_date[_time(row.decision_at).date().isoformat()].append(row.net_return)
    day_means=[sum(v)/len(v) for v in by_date.values()]
    days=len(day_means)
    average=sum(day_means)/days
    daily_var=(sum((v-average)**2 for v in day_means)/(days-1)
               if days>1 else 0.0)
    lower=average-LOWER_BOUND_Z*math.sqrt(daily_var/days) if days>1 else None
    gain=sum(max(0.0,row.net_return) for row in rows)
    loss=-sum(min(0.0,row.net_return) for row in rows)
    ratio=(gain/loss if loss>0 else None)
    markets=len({row.symbol for row in rows})
    sufficient=days>=THRESHOLD_DAYS and markets>=THRESHOLD_ASSETS and loss>0
    supported=(sufficient and lower is not None and lower>0
               and ratio is not None and ratio>=MIN_GAIN_TO_PAIN)
    return {
        "status":("POSITIVE_HISTORICAL_PAPER_EXPECTANCY_NOT_PREDICTIVE"
                  if supported else "NO_ROBUST_PAPER_EXPECTANCY_PROVEN"),
        "episodes":len(rows),"distinct_utc_dates":days,"assets":markets,
        "daily_cluster_mean_net":round(average,9),
        "lower_bound_mean_net":round(lower,9) if lower is not None else None,
        "paper_profit_factor":round(ratio,6) if ratio is not None else None,
        "observed_win_fraction":round(sum(x.net_return>0 for x in rows)/len(rows),6),
        "positive_net_after_proxy_fees_not_fill_confirmed":True,
        "qualified":bool(supported),
    }


def _portfolio_diagnostic(selected:list[Outcome])->dict[str,Any]:
    """Settled paper returns, with no claims about true intrabar equity."""
    if not selected:
        return {"paper_positions":0,"positive_positions":0,
                "positive_fraction":None,"mean_net_return":None,
                "gross_profit_factor":None,"max_settled_mark_drawdown":None,
                "distinct_utc_entry_dates":0,"not_realized_portfolio_pnl":True}
    total=sum(x.net_return for x in selected)
    wins=sum(x.net_return for x in selected if x.net_return>0)
    losses=-sum(x.net_return for x in selected if x.net_return<0)
    # Equal PAPER allocation settled-order diagnostic, not true portfolio
    # mark-to-market (outcomes have no path, funding, or capital allocation).
    equity=1.0
    peak=1.0
    drawdown=0.0
    for x in sorted(selected,key=lambda z:(_time(z.resolved_at),z.symbol,z.decision_at)):
        equity+=x.net_return/max(1,len({r.symbol for r in selected}))
        peak=max(peak,equity)
        if peak>0:drawdown=max(drawdown,1-equity/peak)
    return {
        "paper_positions":len(selected),
        "positive_positions":sum(x.net_return>0 for x in selected),
        "positive_fraction":round(sum(x.net_return>0 for x in selected)/len(selected),6),
        "mean_net_return":round(total/len(selected),9),
        "gross_profit_factor":round(wins/losses,6) if losses else None,
        "max_settled_mark_drawdown":round(drawdown,7),
        "distinct_utc_entry_dates":len({_time(x.decision_at).date() for x in selected}),
        "not_realized_portfolio_pnl":True,
    }


def causal_profit_quality_study(
    ledger:PrivateExperience,*,train_until:str,validation_until:str,
    observe_until:str,
)->dict[str,Any]:
    """Fixed training-only edge map; withheld periods NEVER revise gates.

    Decisions in validation and test can be paper-evaluated only if their
    full outcome resolved by THAT partition's own cutoff.
    """
    a,b,c=(_time(s) for s in (train_until,validation_until,observe_until))
    if not a<b<c: raise ValueError("INVALID_FIXED_TEMPORAL_BOUNDS")
    samples=[
        s for s in ledger.samples(resolved_before=observe_until)
        if s.setup in {"WM1","WM2","WM3"} and _time(s.decision_at)<c
    ]
    train=[s for s in samples
           if _time(s.decision_at)<a and _time(s.resolved_at)<a]
    validation=[s for s in samples
                if a<=_time(s.decision_at)<b and _time(s.resolved_at)<b]
    test=[s for s in samples
          if b<=_time(s.decision_at)<c and _time(s.resolved_at)<=c]
    if not train or not validation or not test:
        raise ValueError("NO_COMPLETE_PURGED_TRAIN_VALIDATION_AND_TEST")

    # One frozen set of *past* realized hypotheses by wise-man and direction.
    # Pre-resolution and boundary-crossing outcomes never enter this map.
    cohorts:dict[tuple[str,str],list[Outcome]]=defaultdict(list)
    for x in train:
        cohorts[(x.setup,x.side)].append(x)
    gates={f"{k[0]}:{k[1]}":_past_edge(v)
           for k,v in sorted(cohorts.items())}

    def paper_review(rows:list[Outcome])->dict[str,Any]:
        selected=[]
        excluded=0
        active_to:dict[str,datetime]={}
        # Same-symbol campaigns cannot overlap: one prospective paper
        # hypothesis at a time. Can use resolution times only to know
        # whether a preceding selected case has ALREADY settled.
        for x in sorted(rows,key=lambda z:(_time(z.decision_at),z.symbol,z.setup)):
            key=f"{x.setup}:{x.side}"
            gate=gates.get(key)
            if not gate or not gate["qualified"]:
                excluded+=1
                continue
            at=_time(x.decision_at)
            if at<active_to.get(x.symbol,datetime.min.replace(tzinfo=at.tzinfo)):
                excluded+=1
                continue
            selected.append(x)
            active_to[x.symbol]=_time(x.resolved_at)
        return {
            "eligible_settled_hypotheses":len(rows),
            "blocked_or_overlapping_hypotheses":excluded,
            **_portfolio_diagnostic(selected),
        }

    # Only training prefix decides learned evidence and qualification.
    return {
        "mode":"FROZEN_CAUSAL_PROFIT_EXPECTANCY_RESEARCH_V1",
        "edition_provenance":SOURCE,
        "train_until":a.isoformat(),
        "validation_until":b.isoformat(),
        "observe_until":c.isoformat(),
        "training_examples":len(train),
        "training_only_cohort_gates":gates,
        "validation_frozen":paper_review(validation),
        "test_frozen":paper_review(test),
        "cutoff_purged_not_counted_in_each_partition":{
            "train":sum(_time(s.decision_at)<a and _time(s.resolved_at)>=a
                        for s in samples),
            "validation":sum(a<=_time(s.decision_at)<b and _time(s.resolved_at)>=b
                             for s in samples),
            "test":sum(b<=_time(s.decision_at)<c and _time(s.resolved_at)>c
                       for s in samples),
        },
        "is_price_prediction_or_exchange_profit_proof":False,
        "positive_win_rate_alone_never_authorizes_trading":True,
        "market_day_clusters_are_not_independent_trials":True,
        "no_test_selection_or_weight_training":True,
        "true_cross_market_portfolio_mark_to_market_available":False,
        "paper_exit_is_fixed_horizon_not_book_exit":True,
        "trade_action":None,
        "execution_allowed":False,
    }
