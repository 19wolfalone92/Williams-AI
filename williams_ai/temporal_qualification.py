"""Pre-registered, cross-asset, PURGED January->February 2025 validation.

The research scorer learns from REAL January only and is judged on the
already available February 2025 calendar. Nothing from February, 2026,
test PnL, BTC news, or future exchange fills can modify coefficients.
A losing scorer is REJECTED instead of being advertised as "learned AI".
Even a PASS here would remain research-only, NOT a trading authorization.

All samples within one calendar are directionally correlated. Brier scoring
has no implication that an OHLC setup is a validated Williams entry.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime,timezone
from typing import Any

from .paper_learning import ResearchSample, paper_samples, fit, _metrics


JAN_START=datetime(2025,1,1,tzinfo=timezone.utc)
FEB_START=datetime(2025,2,1,tzinfo=timezone.utc)
MAR_START=datetime(2025,3,1,tzinfo=timezone.utc)

MIN_JAN=500
MIN_FEB=500
MIN_TEST_SYMBOLS=20
# Thresholds are fixed in source, not selected to fit 2025/2026 history.
REQUIRED_BRIER_MARGIN=0.001
REQUIRED_SYMBOL_FRACTION=0.55


def _when(value:str)->datetime:
    if not isinstance(value,str):
        raise ValueError("Timestamp missing from research sample")
    parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Naive research sample timestamp not permitted")
    return parsed.astimezone(timezone.utc)


def split_calendar_samples(samples:list[ResearchSample])->tuple[list, list]:
    """PURGE samples that would resolve in the next, nominally unseen month."""
    jan, feb=[],[]
    for x in samples:
        start,end=_when(x.decision_at),_when(x.resolved_at)
        if end<=start:
            raise ValueError("Sample resolves before or at decision time")
        if JAN_START<=start<FEB_START and end<FEB_START:
            jan.append(x)
        elif FEB_START<=start<MAR_START and end<MAR_START:
            feb.append(x)
        # A Jan decision whose label resolves in Feb is discarded, not
        # assigned to January training or February validation.
    if any(_when(x.resolved_at)>=FEB_START for x in jan):
        raise AssertionError("January training outcome leaks from February")
    if any(_when(x.decision_at)<FEB_START for x in feb):
        raise AssertionError("Validation uses January decision")
    return jan,feb


def qualify_verified_training(
    markets:dict[str,list[dict[str,Any]]],
    *,
    min_train:int=MIN_JAN,
    min_validation:int=MIN_FEB,
    required_margin:float=REQUIRED_BRIER_MARGIN,
    required_symbol_fraction:float=REQUIRED_SYMBOL_FRACTION,
)->dict[str,Any]:
    if not markets or len(markets)<MIN_TEST_SYMBOLS:
        raise ValueError("Need at least 20 authentic symbol histories")
    if not 0<=required_margin<=0.1 or not .5<=required_symbol_fraction<=1:
        raise ValueError("No in-sample threshold search or unreasonable margin")
    by_symbol={}
    training=[]
    validation=[]
    for symbol,rows in sorted(markets.items()):
        if not symbol.endswith("USDT"):
            raise ValueError("Unexpected historical research symbol")
        examples=paper_samples(rows,"H1",horizon=12,fee_bps=4,
                               slippage_bps=2,stop_atr=2)
        jan,feb=split_calendar_samples(examples)
        if jan and feb:
            training.extend(jan)
            validation.extend(feb)
            by_symbol[symbol]={"jan_train":len(jan),"feb_validation":len(feb)}
    if len(training)<min_train or len(validation)<min_validation:
        raise ValueError("Insufficient source-authentic, purged calendar samples")
    if len(by_symbol)<MIN_TEST_SYMBOLS:
        raise ValueError("Insufficient eligible cross-asset validation coverage")
    model=fit(training,epochs=120)
    model["training_positive_rate"]=sum(x.net_return>0 for x in training)/len(training)
    result=_metrics(model,validation)
    by_symbol_skill={}
    for symbol in sorted(by_symbol):
        v=by_symbol[symbol]
        # Decision timestamps have no lookahead. Validation is strictly Feb.
        feb=[x for x in split_calendar_samples(
                paper_samples(markets[symbol],"H1",horizon=12,fee_bps=4,
                              slippage_bps=2,stop_atr=2))[1]]
        record=_metrics(model,feb)
        diff=record["constant_prior_brier"]-record["brier_score"]
        by_symbol_skill[symbol]={
            **v,
            "brier":round(record["brier_score"],6),
            "constant_january_prior_brier":round(record["constant_prior_brier"],6),
            "relative_skill":round(diff,6),
            "outperforms_baseline":diff>0,
        }
    wins=sum(x["outperforms_baseline"] for x in by_symbol_skill.values())
    change=result["constant_prior_brier"]-result["brier_score"]
    strong=(
        change>=required_margin
        and wins/len(by_symbol_skill)>=required_symbol_fraction
    )
    # Model coefficients NEVER get exported as a certified trading strategy.
    return {
        "mode":"STRICT_PURGED_JAN2025_TRAIN_FEB2025_VALIDATION",
        "training_samples":len(training),
        "validation_samples":len(validation),
        "training_month":"2025-01",
        "independent_validation_month":"2025-02",
        "purged_cross_month_labels":True,
        "research_model_family":"LOGISTIC_NOT_GGUF",
        "training_only_positive_prior":round(model["training_positive_rate"],6),
        "february_brier":round(result["brier_score"],6),
        "february_january_prior_brier":round(result["constant_prior_brier"],6),
        "validation_brier_skill":round(change,6),
        "precommitted_min_skill":required_margin,
        "symbols_with_better_skill":wins,
        "tested_symbols":len(by_symbol_skill),
        "fraction_beating_prior":round(wins/len(by_symbol_skill),6),
        "precommitted_min_fraction":required_symbol_fraction,
        "qualification":(
            "MODEL_PASSED_STATISTICAL_RESEARCH_GATE_NOT_TRADING"
            if strong else "REJECT_RESEARCH_MODEL_NO_PROVEN_EDGE"
        ),
        "promotion_to_execution_allowed":False,
        "auto_adopted_model":False,
        "neural_GGUF_weights_changed":False,
        "full_Williams_book_signal_validated":False,
        "portfolio_funding_real_SL_validated":False,
        "per_symbol":by_symbol_skill,
        "warnings":[
            "January and February are separate by calendar but share correlated markets.",
            "Brier calibration is not Williams campaign profitability.",
            "Validation thresholds fixed before testing and cannot be retuned on February.",
            "Any deployment requires an independent untouched horizon and Binance Demo stop receipts.",
        ],
        "trade_action":None,"execution_allowed":False,
    }
