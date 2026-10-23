"""Sealed-prefix historical test of what the *deterministic* Williams AI can know.

No model weights, no Binance order, no training or parameter fitting.
For each checkpoint, only candles <= t are sent to analyze_chart; the
evaluator (separate scope) opens t+24h later only to score directional
hypotheses. This is a diagnostic association, not an executable strategy or
proof of model understanding. Against a *same-sample* simple momentum baseline.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter,defaultdict
from statistics import median
from typing import Any

from .causal_paper_lab import frozen_markets
from .chart_reader import analyze_chart
from .causal_timeframes import aggregate_closed_h1

CUTS=(321,411,501,591,681,771,861,951,1041,1131)
HORIZON_H1_BARS=24


def _signed_outcome(price_delta:float,threshold:float)->int:
    if price_delta>threshold:return 1
    if price_delta< -threshold:return -1
    return 0


def _label(result:dict[str,Any])->str:
    obs=result["principle_evidence"]
    compatible={
        x["hypothesis"] for x in obs["competing_market_scenarios"]
        if x["status"]=="EVIDENCE_COMPATIBLE_NOT_PREDICTION"
    }
    up="CONTINUATION_UP" in compatible
    down="CONTINUATION_DOWN" in compatible
    return "UP" if up and not down else "DOWN" if down and not up else "AMBIGUOUS_OR_NO_EDGE"


def historical_comprehension() -> dict[str,Any]:
    pairs=frozen_markets()
    events=[]
    per_pair={}
    higher_scan=[]
    for sym in sorted(pairs):
        market=pairs[sym];rows=market["bars"]
        pstats=Counter()
        for idx in CUTS:
            if idx+HORIZON_H1_BARS>=len(rows):
                raise ValueError("Not enough previously frozen bars for withheld evaluation")
            # This function sees NOTHING after idx, including indicator history.
            visible=rows[:idx+1]
            o=analyze_chart(
                sym,"H1",visible,as_of=visible[-1]["close_time"],
                tick_size=market["tick_size"],
            )
            if o["no_lookahead"] is not True or o["execution_allowed"] is not False:
                raise ValueError("Future contamination / execution not blocked")
            belief=_label(o)
            bullet=o["tc1_five_magic_bullets"]
            for candidate in bullet["last_n_wave_candidates"]:
                if candidate["confirmed_through"]>visible[-1]["close_time"]:
                    raise ValueError("Unconfirmed future wave pivot leaked")
                if candidate["all_five_observed"]:
                    raise ValueError("H1 public klines cannot certify tick-volume magic bullet")
            # ALL subsequent bars withheld until model/algorithm's conclusion.
            past_close=float(visible[-1]["close"])
            prev_close=float(visible[-25]["close"])
            median_range=median(
                max(0.,float(row["high"])-float(row["low"]))
                for row in visible[-24:]
            )
            threshold=0.25*median_range
            if threshold<=0:threshold=1e-12
            past_baseline="UP" if past_close>prev_close else (
                "DOWN" if past_close<prev_close else "FLAT"
            )
            # EVALUATOR ONLY: reveal the next 24h AFTER freezing the thought.
            withheld_future_close=float(rows[idx+HORIZON_H1_BARS]["close"])
            actual=_signed_outcome(withheld_future_close-past_close,threshold)
            pstats["cuts"]+=1
            if belief!="AMBIGUOUS_OR_NO_EDGE":
                pstats["unique_direction_calls"]+=1
                hit=int((belief=="UP" and actual==1) or
                        (belief=="DOWN" and actual==-1))
                loss=int((belief=="UP" and actual==-1) or
                         (belief=="DOWN" and actual==1))
                same_baseline=int((past_baseline=="UP" and actual==1) or
                                  (past_baseline=="DOWN" and actual==-1))
                pstats["correct_non_neutral"]+=hit
                pstats["wrong_non_neutral"]+=loss
                pstats["neutral_outcomes"]+=actual==0
                pstats["baseline_correct_on_identical_selections"]+=same_baseline
                events.append({
                    "symbol":sym,"bar_index":idx,"as_of":visible[-1]["close_time"],
                    "scenario_direction":belief,
                    "later_24h": "UP" if actual==1 else "DOWN" if actual==-1 else "FLAT",
                    "prior_24h_baseline_direction":past_baseline,
                    "correct":bool(hit),"baseline_correct":bool(same_baseline),
                    "trade_action":None,
                })
            else:
                pstats["ambiguous_or_no_edge"]+=1
            pstats["five_bullet_wave_candidates"]+=bullet["wave_candidate_count"]
            pstats["all_five_confirmed"]+=sum(
                bool(v["all_five_observed"])
                for v in bullet["last_n_wave_candidates"]
            )
            pstats["squat_unknown_candidates"]+=sum(
                v["bullets"]["SQUAT_MFI_VOLUME"]["status"]=="UNKNOWN"
                for v in bullet["last_n_wave_candidates"]
            )
        per_pair[sym]=dict(pstats)
        # Independent source on higher TF: no forward-filled or reconstructed
        # lower bars. Include D1 only if it has 45+ *fully CLOSED* bars.
        for idx in (1041,1151):
            visible=rows[:idx+1]
            for tf in ("H1","H4","D1"):
                agg=visible if tf=="H1" else aggregate_closed_h1(visible,tf)
                o=analyze_chart(sym,tf,agg,as_of=agg[-1]["close_time"],
                                tick_size=market["tick_size"])
                b=o.get("tc1_five_magic_bullets")
                higher_scan.append({
                    "symbol":sym,"tf":tf,"index":idx,
                    "last_visible_time":visible[-1]["close_time"],
                    "last_closed_tf_time":agg[-1]["close_time"],
                    "bars":len(agg),
                    "magic_bullet_module_present":isinstance(b,dict),
                    "untrusted_tick_count_does_not_certify_squat":
                        not b or all(
                            x["bullets"]["SQUAT_MFI_VOLUME"]["status"]=="UNKNOWN"
                            for x in b["last_n_wave_candidates"]
                        ),
                    "trade_action":None,
                })
                if agg[-1]["close_time"]>visible[-1]["close_time"]:
                    raise ValueError("Aggregated candle in the future")
    total=Counter()
    for x in per_pair.values():total.update(x)
    return {
        "mode":"SEQUENTIAL_CLOSED_PREFIX_INFERENCE_DIAGNOSTIC_NOT_TRADING",
        "pairs":len(pairs),"historical_h1_checkpoints":total["cuts"],
        "assessed_lookahead_hours":HORIZON_H1_BARS,
        "different_timeframe_passports":len(higher_scan),
        "every_h1_decision_only_saw_closed_past":True,
        "notes":[
            "Hypotheses from rules/code, not neural GGUF or vision perception.",
            "Frozen public Binance data, no M5/M15 bars or real tick counts.",
            "24h future accessed ONLY AFTER each prediction was sealed.",
            "Direction accuracy is NOT risk-adjusted profitability or an order fill.",
            "Scenario heuristic was already developed with access to markets in general; historical cohort is not independent.",
            "No optimizations or hyperparameters were fitted in this run.",
        ],
        "metrics":dict(total),
        "directional_accuracy_excluding_neutral":round(
            total["correct_non_neutral"]/
            max(1,total["correct_non_neutral"]+total["wrong_non_neutral"]),4
        ) if total["unique_direction_calls"] else None,
        "naive_24h_past_momentum_hitrate_on_same_selected_samples":round(
            total["baseline_correct_on_identical_selections"]/
            max(1,total["correct_non_neutral"]+total["wrong_non_neutral"]),4
        ) if total["unique_direction_calls"] else None,
        "per_symbol":per_pair,
        "selected_examples":events[:10],
        "unseen_after_trade_PnL":None,
        "live_trade_authorized":False,
    }


def main():
    ap=argparse.ArgumentParser(description="Sequential read-only historical market comprehension")
    ap.add_argument("--summary",action="store_true")
    args=ap.parse_args()
    report=historical_comprehension()
    if args.summary:
        report={k:v for k,v in report.items()
                if k not in ("per_symbol","selected_examples")}
    print("HISTORICAL_COMPREHENSION_RESULT",
          json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
