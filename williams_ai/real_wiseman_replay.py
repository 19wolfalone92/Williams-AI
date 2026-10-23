"""REAL historical William(s) Wise Men EVENT replay, not synthetic stress.

Every bar makes an as-of causal decision from already CLOSED H1 OHLC.
The future is opened only by a separate evaluator after event facts are sealed.
Unlike 10m synthetic fractal transformations, these examples come from
distinct (symbol, closed timestamp, observed signal family) events.

WM1 is still a *heuristic angulation candidate*; WM2 an observed AO
momentum candidate; WM3 a confirmed fractal price break with current Teeth
filter but NO exchange fill. None is proof of a canonical book entry.
The directional 12-bar endpoint is a RESEARCH label, not a trade exit.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any

from .chart_reader import indicators, validate_closed_bars
from .paper_learning import ResearchSample,fit,_metrics

FAMILIES=("WM1","WM2","WM3")
HORIZON=12
MIN_WARMUP=65


def _context_features(bars,lines,index:int,side:str)->tuple[float,...]:
    direction=1 if side=="LONG" else -1
    high=bars[index].high
    low=bars[index].low
    atr=sum(max(
        bars[j].high-bars[j].low,
        abs(bars[j].high-bars[j-1].close),
        abs(bars[j].low-bars[j-1].close))
        for j in range(index-13,index+1))/14
    atr=max(atr,1e-10)
    clamp=lambda v:max(-8.0,min(8.0,v))
    return (
        1.,
        clamp(direction*lines["ao"][index]/atr),
        clamp(direction*lines["ac"][index]/atr),
        clamp(abs(lines["jaw"][index]-lines["lips"][index])/atr),
        clamp(direction*(bars[index].close-lines["lips"][index])/atr),
        clamp(100*atr/max(bars[index].close,1e-10)),
    )


def replay_closed_history(rows:list[dict[str,Any]], *,
                          symbol:str,tf:str="H1",
                          horizon:int=HORIZON)->list[dict[str,Any]]:
    if tf!="H1":
        raise ValueError("Research teacher currently supports REAL H1, never fake M5")
    if not isinstance(horizon,int) or isinstance(horizon,bool) or horizon<1 or horizon>48:
        raise ValueError("Horizon must be 1..48 completed H1 bars")
    candles,gaps=validate_closed_bars(rows,tf,rows[-1]["close_time"])
    if gaps or len(candles)<MIN_WARMUP+horizon:
        raise ValueError("Need contiguous real closed H1 candles")
    lines=indicators(candles)
    out=[]
    pending={"LONG":None,"SHORT":None}
    seen=set()
    for t in range(MIN_WARMUP,len(candles)-horizon):
        bar=candles[t]
        before=candles[t-1]
        found=[]
        # Confirm pivot at t-2 only when t, the SECOND right candle, is
        # actually closed. No breakout can use a newly confirmed pivot's
        # level on its confirmation candle because intrabar order is unknown.
        pivot=t-2
        now_high=candles[pivot].high
        now_low=candles[pivot].low
        neighbors=(candles[pivot-2],candles[pivot-1],
                   candles[pivot+1],candles[pivot+2])
        high=now_high>max(x.high for x in neighbors)
        low=now_low<min(x.low for x in neighbors)
        newly_confirmed={
            "LONG":high and not low,
            "SHORT":low and not high,
        }
        # WM3: a confirmed pivot from at least the previous closed bar
        # experiences its FIRST price cross only at the now-closed bar.
        # When a replacement fractal is confirmed on the same bar, skip
        # ambiguous intrabar ordering, never imply a filled order.
        for side,sign in (("LONG",1),("SHORT",-1)):
            prev=pending[side]
            if prev is not None:
                level=prev["level"]
                touched=(bar.high>level if sign==1 else bar.low<level)
                if touched:
                    if not newly_confirmed[side] and (
                        lines["teeth"][t] is not None and
                        sign*(level-lines["teeth"][t])>0
                    ):
                        found.append((side,"WM3"))
                    pending[side]=None
            if newly_confirmed[side]:
                pending[side]={"pivot_index":pivot,
                               "confirmed_index":t,
                               "level":now_high if sign==1 else now_low}
        # WM1: visible reversal bar away from a displaced Alligator;
        # numerical distance/slope only suggests, never proves ANGULATION.
        jaw=lines["jaw"][t]
        oldjaw=lines["jaw"][t-5]
        teeth=lines["teeth"][t]
        lips=lines["lips"][t]
        for side,sign in (("LONG",1),("SHORT",-1)):
            if any(z is None for z in (jaw,oldjaw,teeth,lips)):
                continue
            midpoint=(bar.high+bar.low)/2
            if sign==1:
                pattern=bar.low<before.low and bar.close>midpoint
                beyond=bar.high<min(jaw,teeth,lips)
                edge=(bar.low-candles[t-5].low)/5
                context=candles[t-5].close>before.close
                spread=abs(bar.low-jaw)>abs(candles[t-4].low-lines["jaw"][t-4])
            else:
                pattern=bar.high>before.high and bar.close<midpoint
                beyond=bar.low>max(jaw,teeth,lips)
                edge=(bar.high-candles[t-5].high)/5
                context=candles[t-5].close<before.close
                spread=abs(bar.high-jaw)>abs(candles[t-4].high-lines["jaw"][t-4])
            angled=abs(edge)>abs((jaw-oldjaw)/5)*1.05
            if pattern and beyond and context and spread and angled:
                found.append((side,"WM1"))
            ao=lines["ao"]
            # WM2 Super AO candidate: 4 consecutive increasing (or
            # decreasing) AO values including current bar; same as reader.
            changes=all(
                sign*(ao[t-k]-ao[t-k-1])>0
                for k in (0,1,2)
            )
            previous=all(
                sign*(ao[t-k-1]-ao[t-k-2])>0
                for k in (0,1,2)
            )
            if changes and not previous:
                found.append((side,"WM2"))
        for side,kind in found:
            key=(symbol,bar.at.isoformat(),kind,side)
            if key in seen:
                raise AssertionError("Duplicate actual market event")
            seen.add(key)
            sign=1 if side=="LONG" else -1
            future=candles[t+horizon]
            delta=sign*(future.close-bar.close)/bar.close
            prior=sign*(bar.close-candles[t-horizon].close)/candles[t-horizon].close
            if not isfinite(delta) or not isfinite(prior):
                raise ValueError("Nonfinite research outcome")
            out.append({
                "event_id":":".join(key),
                "symbol":symbol,
                "edition":"TC2",
                "kind":kind,
                "side":side,
                "decision_index":t,
                "decision_at":bar.at.isoformat(),
                "resolved_index":t+horizon,
                "resolved_at":future.at.isoformat(),
                "context_features":_context_features(candles,lines,t,side),
                # Label only for RESEARCH scoring; never included in
                # a decision's features or fed to the LLM at t.
                "future_endpoint_side_return":delta,
                "past_window_side_return":prior,
                "label_direction_favorable":delta>0,
                "observed_reversal_from_prior_window":prior<0 and delta>0,
                "true_williams_entry_validated":False,
                "future_was_available_to_decision":False,
                "execution_allowed":False,
                "trade_action":None,
            })
    return out


def _date(timestamp:str)->datetime:
    return datetime.fromisoformat(timestamp.replace("Z","+00:00")).astimezone(timezone.utc)


def score_january_to_february(markets:dict[str,list[dict[str,Any]]])->dict[str,Any]:
    jan_end=datetime(2025,2,1,tzinfo=timezone.utc)
    mar_start=datetime(2025,3,1,tzinfo=timezone.utc)
    jan=[]
    feb=[]
    signals=Counter()
    by_pair=Counter()
    authentic_h1_states=0
    family_months=defaultdict(Counter)
    feb_events_for_family=[]
    for symbol,rows in sorted(markets.items()):
        authentic_h1_states+=len(rows)
        for e in replay_closed_history(rows,symbol=symbol):
            signals[e["kind"]]+=1
            by_pair[symbol]+=1
            p=ResearchSample(
                e["decision_index"],e["resolved_index"],
                e["decision_at"],e["resolved_at"],e["side"],
                tuple(e["context_features"]),
                e["future_endpoint_side_return"],False)
            moment=_date(e["decision_at"])
            mature=_date(e["resolved_at"])
            if moment<jan_end and mature<jan_end:
                jan.append(p)
                family_months[("TRAIN_JAN",e["kind"],e["side"])]["count"]+=1
                family_months[("TRAIN_JAN",e["kind"],e["side"])]["favorable"]+=bool(
                    e["label_direction_favorable"])
                family_months[("TRAIN_JAN",e["kind"],e["side"])]["reversal_after_opposite_prior_window"]+=bool(
                    e["observed_reversal_from_prior_window"])
            elif jan_end<=moment<mar_start and mature<mar_start:
                feb.append(p)
                family_months[("TEST_FEB",e["kind"],e["side"])]["count"]+=1
                family_months[("TEST_FEB",e["kind"],e["side"])]["favorable"]+=bool(
                    e["label_direction_favorable"])
                family_months[("TEST_FEB",e["kind"],e["side"])]["reversal_after_opposite_prior_window"]+=bool(
                    e["observed_reversal_from_prior_window"])
                feb_events_for_family.append(e)
    if len(markets)<20 or len(jan)<80 or len(feb)<80:
        raise ValueError("Not enough genuine independent-calendar evidence events")
    if any(_date(x.resolved_at)>=jan_end for x in jan):
        raise AssertionError("January training labels revealed in February")
    model=fit(jan,epochs=120)
    model["training_positive_rate"]=sum(x.net_return>0 for x in jan)/len(jan)
    test=_metrics(model,feb)
    skill=test["constant_prior_brier"]-test["brier_score"]
    by_family={}
    family_brier=0.
    for e in feb_events_for_family:
        key=("TRAIN_JAN",e["kind"],e["side"])
        history=family_months[key]
        # Laplace smoothing fixed before the February test.
        conditional=(history["favorable"]+1)/(history["count"]+2)
        y=float(e["label_direction_favorable"])
        family_brier+=(conditional-y)**2
    family_brier/=len(feb_events_for_family)
    for name in FAMILIES:
        for side in ("LONG","SHORT"):
            train_family=family_months[("TRAIN_JAN",name,side)]
            eval_family=family_months[("TEST_FEB",name,side)]
            by_family[name+"_"+side]={
                "january_matured_training_events":train_family["count"],
                "january_favorable_fraction":round(
                    train_family["favorable"]/train_family["count"],5)
                    if train_family["count"] else None,
                "february_frozen_test_events":eval_family["count"],
                "february_favorable_fraction":round(
                    eval_family["favorable"]/eval_family["count"],5)
                    if eval_family["count"] else None,
                "february_actual_prior_opposite_then_future_favorable_fraction":round(
                    eval_family["reversal_after_opposite_prior_window"]/eval_family["count"],5)
                    if eval_family["count"] else None,
                "no_independent_wave_truth":True,
            }
    # Never promote even if score is positive: February has been
    # previously reviewed by developers, signals only approximations,
    # no image proof, campaign or actual exchange stop receipts.
    return {
        "mode":"ACTUAL_TF_H1_COMPLETED_WILLIAMS_SIGN_EVENTS_JAN_FEB_2025",
        "symbols":len(markets),
        "original_real_sign_events":sum(signals.values()),
        "authentic_unique_h1_closed_bar_states":authentic_h1_states,
        "actual_authentic_closed_bar_states_toward_10million_target":authentic_h1_states,
        "remaining_genuine_closed_bar_states_for_10million":max(
            0,10_000_000-authentic_h1_states),
        "ten_million_unique_real_chart_states_confirmed":authentic_h1_states>=10_000_000,
        "ten_million_synthetic_counterfactuals_counted_as_real":0,
        "distinct_symbol_time_kind_side_keys":True,
        "signal_family_counts":dict(signals),
        "january_matured_train_events":len(jan),
        "february_scored_events":len(feb),
        "horizon_closed_h1_bars":HORIZON,
        "future_labels_hidden_until_after_decision":True,
        "labels_crossing_month_boundary_purged":True,
        "model_family":"EVENT_FEATURE_LOGISTIC_RESEARCH_NOT_GGUF",
        "trained_january_only":True,
        "february_previous_developer_seen":True,
        "not_10million_independent_real_events":True,
        "source_book_wm1_angulation_validated":False,
        "source_book_wm3_exchange_fill_validated":False,
        "february_brier":round(test["brier_score"],6),
        "january_constant_baseline_brier":round(test["constant_prior_brier"],6),
        "february_skill_vs_january_prior":round(skill,6),
        "january_family_prior_february_brier":round(family_brier,6),
        "january_family_prior_february_advantage_vs_constant":round(
            test["constant_prior_brier"]-family_brier,6),
        "observed_by_signal_family_not_neural_comprehension":by_family,
        "overlapping_horizons_make_samples_statistically_dependent":True,
        "fixed_probability_0_6_selected":test["selected_at_fixed_0_6"],
        "profitability_certified":False,
        "neural_GGUF_weights_changed":False,
        "research_model_promotion_allowed":False,
        "trade_action":None,
        "execution_allowed":False,
    }


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=7)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    from .many_token_learning import obtain_predeclared_training
    bundle=obtain_predeclared_training(workers=args.workers)
    result=score_january_to_february(bundle["markets"])
    result["source_archive_zip_checksums_verified"]=True
    result["sources_confirmed"]=bundle["available"]
    if args.out.exists():
        raise FileExistsError("Do not overwrite audit result")
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),
                        encoding="utf-8")
    print("REAL_WILLIAMS_CHART_EVENT_REPLAY",
          json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
