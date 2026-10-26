"""Deterministic blind market-forecast cases from frozen REAL Binance 15-pair H1.

This tests a later model's causal directional predictions; it does not
supply independently expert-labeled waves, Wise Men or trade-entry answers.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .chart_supervision import (
    SCHEMA, checked_chart_context, market_digest
)

CUTS = (420, 920)
LOOKBACK = 240
HORIZON = 24


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms/1000,timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00","Z")


def build_blind_market_cases(archive: dict[str,Any]) -> tuple[list[dict],list[dict],dict]:
    if (not isinstance(archive,dict)
            or archive.get("schema") != "WILLIAMS_AI_PUBLIC_BINANCE_15_H1_V1"
            or archive.get("evaluation_design") !=
               "UNSEEN_PRIOR_HISTORY_LOCKED_BENCHMARK_NO_HYPERPARAMETER_TUNING"
            or archive.get("data_is_public") is not True
            or not isinstance(archive.get("pairs"), dict)
            or len(archive["pairs"]) != 15):
        raise ValueError("EXPLICIT_REAL_15PAIR_FROZEN_HISTORY_REQUIRED")
    questions, outcomes=[],[]
    for symbol,pair in sorted(archive["pairs"].items()):
        klines=pair["klines"]
        if (not isinstance(klines,list) or len(klines)!=1200
                or any(not isinstance(bar,list) or len(bar)!=6
                       or type(bar[0]) is not int
                       or any(type(v) not in (int,float) or not math.isfinite(v)
                              for v in bar[1:])
                       for bar in klines)
                or any(y[0]-x[0]!=3600000 for x,y in zip(klines,klines[1:]))
                or any(not (0<bar[3]<=min(bar[1],bar[4])
                            and max(bar[1],bar[4])<=bar[2])
                       for bar in klines)):
            raise ValueError("REAL_H1_HISTORY_NOT_CAUSAL_OR_MALFORMED")
        for cutoff in CUTS:
            visible=klines[cutoff-LOOKBACK+1:cutoff+1]
            future=klines[cutoff+1:cutoff+HORIZON+1]
            if len(visible)!=LOOKBACK or len(future)!=HORIZON:
                raise ValueError("INSUFFICIENT_FUTURE_ONLY_FOR_SEALED_EVALUATION")
            bars=[{"close_time":_iso(b[0]),
                    "open":b[1],"high":b[2],"low":b[3],
                    "close":b[4],"volume":b[5],"closed":True} for b in visible]
            context={"schema":SCHEMA,"symbol":symbol,
                     "venue":"FUTURES","timeframe":"H1",
                     "tick_size":pair["tick_size"],
                     "cutoff":_iso(visible[-1][0]),"bars":bars}
            context["market_sha256"]=market_digest(context)
            checked_chart_context(context)
            # Published question is ONLY past candles and the as-of time.
            case_id=f"{symbol}-H1-asof-{visible[-1][0]}"
            questions.append({
                "case_id":case_id,"market_context":context,
                "task":"Predict UP/DOWN/RANGE over next 24 H1 CLOSED bars, with probabilities summing to 1. No order; may abstain.",
                "status":"UNLABELED_CANDIDATE_NEEDS_INDEPENDENT_WAVE_WM1_WM2_WM3_EXPERT_REVIEW",
                "future_data_in_prompt":False,
                "trade_action":None,"execution_allowed":False,
            })
            # Absolute ATR threshold is fixed using only VISIBLE candles.
            tr=[max(b[2]-b[3], abs(b[2]-prev[4]), abs(b[3]-prev[4]))
                for prev,b in zip(visible[-15:-1],visible[-14:])]
            atr=sum(tr)/len(tr)
            last=visible[-1][4]; forward=future[-1][4]
            threshold=atr/last
            movement=(forward-last)/last
            true_class=("UP" if movement>threshold else
                        "DOWN" if movement< -threshold else "RANGE")
            outcomes.append({"case_id":case_id,"result_available_at":_iso(future[-1][0]),
                             "true_class":true_class,
                             "future_return":movement,"atr_asof_ratio":threshold})
    if len(questions)!=30 or len({x["case_id"] for x in questions})!=30:
        raise ValueError("THIRTY_DISTINCT_PUBLIC_CHART_FORECASTS_REQUIRED")
    public=json.dumps(questions,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    private=json.dumps(outcomes,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    return questions,outcomes,{
        "mode":"REAL_15PAIR_H1_BLIND_DIRECTIONAL_CANDIDATE_ONLY",
        "pairs":15,"cases":30,"cuts":list(CUTS),"lookback_closed_bars":LOOKBACK,
        "future_horizon_bars":HORIZON,"future_market_bars_in_inference_prompt":False,
        "market_context_sha256":hashlib.sha256(public.encode()).hexdigest(),
        "result_keys_sha256":hashlib.sha256(private.encode()).hexdigest(),
        "human_expert_wave_labels":0,"qwen_before_after_evaluated":False,
        "market_structure_and_trading_quality_proven":False,
        "execution_allowed":False}


def grade_blind_predictions(outcomes: list[dict],
                            predictions: list[dict]) -> dict:
    """Grade exactly-one forecast per case. Unanswered cases count incorrect."""
    by_case={row["case_id"]:row for row in outcomes}
    if (len(by_case)!=len(outcomes) or not predictions
            or len({p.get("case_id") for p in predictions})!=len(predictions)
            or set(p["case_id"] for p in predictions)!=set(by_case)):
        raise ValueError("EXACT_COHORT_PREDICTION_PRECOMMIT_REQUIRED")
    counts=0; brier=0.
    for pred in predictions:
        forecast=pred.get("probabilities")
        if not isinstance(forecast,dict) or set(forecast)!={"UP","DOWN","RANGE"}:
            raise ValueError("COMPLETE_UP_DOWN_RANGE_REQUIRED")
        if any(type(x) not in (float,int) or not math.isfinite(x)
               or not 0<=x<=1 for x in forecast.values()) or abs(sum(forecast.values())-1)>1e-5:
            raise ValueError("FINITE_CALIBRATED_TRIPLET_REQUIRED")
        expected=by_case[pred["case_id"]]["true_class"]
        counts+=max(forecast,key=forecast.get)==expected
        brier+=sum((forecast[klass]-(1 if klass==expected else 0))**2
                   for klass in ("UP","DOWN","RANGE"))/3
    n=len(predictions)
    return {"cases":n,"correct":counts,"top1_accuracy":counts/n,
            "multiclass_brier":brier/n,
            "price_outcomes_not_expert_wave_labels":True,
            "trading_profit_not_measured":True,
            "execution_allowed":False}
