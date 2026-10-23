"""Many real Binance symbols for predeclared temporal learning, not a trading bot.

January-February 2025 is fixed BEFORE this test; later March-April 2025,
July-October 2026 are evaluation periods, but the latter have been reviewed
previously by developers. A model score only tests research features, NOT
complete Williams entry/exit, GGUF fine-tuning or profitable live execution.

The approved universe is fixed in source; inaccessible/delisted instruments
are explicitly excluded with reasons. No substitution with synthetic prices.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
import argparse,hashlib,json
from pathlib import Path
from typing import Any

from .unseen_2025_cohort import fetch_month
from .chart_reader import validate_closed_bars
from .paper_learning import paper_samples,fit,_metrics
from .causal_paper_lab import frozen_markets,FROZEN_DATA

TRAIN_MONTHS=("2025-01","2025-02")
TRAIN_SYMBOLS=(
    "AAVEUSDT","ADAUSDT","ALGOUSDT","APTUSDT","ARBUSDT","ATOMUSDT",
    "AVAXUSDT","BCHUSDT","BNBUSDT","BTCUSDT","COMPUSDT","CRVUSDT",
    "DOGEUSDT","DOTUSDT","ENAUSDT","EOSUSDT","ETCUSDT","ETHUSDT",
    "FETUSDT","FILUSDT","GALAUSDT","HBARUSDT","ICPUSDT","INJUSDT",
    "JUPUSDT","KAVAUSDT","LINKUSDT","LDOUSDT","LTCUSDT","MANAUSDT",
    "NEARUSDT","ONDOUSDT","OPUSDT","PEPEUSDT","RUNEUSDT","SANDUSDT",
    "SEIUSDT","SHIBUSDT","SOLUSDT","STXUSDT","SUIUSDT","TIAUSDT",
    "TONUSDT","TRXUSDT","UNIUSDT","VETUSDT","WIFUSDT","WLDUSDT",
    "XLMUSDT","XRPUSDT","XTZUSDT",
)
if len(set(TRAIN_SYMBOLS))!=len(TRAIN_SYMBOLS):
    raise RuntimeError("Training symbols not unique")


def _download_symbol(symbol:str)->tuple[str,list[dict[str,Any]]|None,list[dict],str|None]:
    # An intentional unavailable archive is never backfilled with mock data.
    try:
        parts=[]
        receipts=[]
        for month in TRAIN_MONTHS:
            bars,proof=fetch_month(symbol,month,approved_symbols=TRAIN_SYMBOLS,
                                   approved_months=TRAIN_MONTHS)
            parts.extend(bars)
            receipts.append(proof)
        candles,gaps=validate_closed_bars(parts,"H1",parts[-1]["close_time"])
        if gaps or len(candles)!=len(parts):
            raise ValueError("MISSING_H1_BARS")
        if len(parts)<1200:
            raise ValueError("UNDER_1200_CLOSED_H1_BARS")
        return symbol,parts,receipts,None
    except Exception as exc:
        # Exclude unavailable histories; no secrets or source URLs in log.
        return symbol,None,[],type(exc).__name__


def obtain_predeclared_training(*,workers:int=7)->dict[str,Any]:
    if not 1<=workers<=12:
        raise ValueError("worker budget outside research limits")
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(_download_symbol,s):s for s in TRAIN_SYMBOLS}
        for future in as_completed(pending):
            results.append(future.result())
    good={s:rows for s,rows,_,err in results if err is None}
    proofs={s:receipts for s,rows,receipts,err in results if err is None}
    unavailable={s:err for s,rows,receipts,err in results if err is not None}
    if len(good)<20:
        raise ValueError("CANNOT_VERIFY_MINIMUM_20_REAL_USDM_INSTRUMENTS")
    return {"months":TRAIN_MONTHS,"requested":len(TRAIN_SYMBOLS),
            "available":len(good),"markets":good,
            "receipts":proofs,"missing":unavailable,
            "raw_closed_h1_bars":sum(len(rows) for rows in good.values()),
            "note":"Only SHA-256-verified Binance public ZIP bars, no trade execution",
            "trade_action":None}


def _safe_rates(model:dict,samples:list)->dict[str,Any]:
    r=_metrics(model,samples)
    return {
        "samples":r["samples"],"decision_times":r["decision_times"],
        "brier":round(r["brier_score"],6),
        "constant_training_prior_brier":round(r["constant_prior_brier"],6),
        "brier_advantage_vs_training_constant":round(
            r["constant_prior_brier"]-r["brier_score"],6),
        "selected_at_fixed_0_6":r["selected_at_fixed_0_6"],
        "mean_selected_net_return": (
            round(r["mean_selected_net_return"],8)
            if r["mean_selected_net_return"] is not None else None
        ),
        "safe_only":True,"trading_permission":False,
    }


def learn_and_evaluate(training:dict[str,Any], *,
                       test_2026:dict[str,Any]|None=None)->dict[str,Any]:
    alltrain=[]
    total=Counter()
    for name,rows in sorted(training["markets"].items()):
        samples=paper_samples(rows,"H1",horizon=12,fee_bps=4,
                              slippage_bps=2,stop_atr=2)
        alltrain+=samples
        total["training_samples"]+=len(samples)
    if len(alltrain)<2000:
        raise ValueError("Too few authentic training observations")
    # No 2026 data may be used for coefficient estimation, tuning, or
    # selecting fixed 0.6 threshold.
    model=fit(alltrain,epochs=120)
    model["training_positive_rate"]=sum(x.net_return>0 for x in alltrain)/len(alltrain)
    tested={}
    holdout=test_2026 if test_2026 is not None else frozen_markets(FROZEN_DATA)
    for symbol,market in sorted(holdout.items()):
        rows=market["bars"]
        samples=paper_samples(rows,"H1",horizon=12,fee_bps=4,
                              slippage_bps=2,stop_atr=2)
        tested[symbol]=_safe_rates(model,samples)
    n=sum(x["samples"] for x in tested.values())
    # Weight paired errors by the number of sample opportunities; do not
    # pretend each LONG/SHORT or neighboring altcoin is independent.
    avg=lambda field:round(sum(x[field]*x["samples"] for x in tested.values())/n,6)
    edge=avg("brier_advantage_vs_training_constant")
    wins=sum(x["brier_advantage_vs_training_constant"]>0 for x in tested.values())
    return {
        "mode":"RESEARCH_LOGISTIC_LEARNING_FROM_REAL_2025_JAN_FEB_VS_REVIEWED_2026",
        "sources_verified_with_binance_monthly_sha256":True,
        "symbols_requested":training["requested"],
        "symbols_train_available":training["available"],
        "symbols_unavailable":training["missing"],
        "raw_training_h1_bars":training["raw_closed_h1_bars"],
        "trained_paper_labels":total["training_samples"],
        "model_features":model["features"],
        "weights_sha256":hashlib.sha256(
            json.dumps(model["weights"],separators=(",",":")).encode()
        ).hexdigest(),
        "trained_labeled_outcomes_are_not_actual_book_WM_entries":True,
        "model_family":"logistic_research_scorer_not_Qwen_GGUF",
        "Qwen_GGUF_neural_weights_changed":False,
        "execution_allowed":False,
        "evaluation_2026_already_seen_by_developers":True,
        "per_2026_symbol":tested,
        "evaluation_symbol_count":len(tested),
        "evaluation_paper_samples":n,
        "aggregate_weighted_brier":avg("brier"),
        "aggregate_constant_prior_brier":avg("constant_training_prior_brier"),
        "aggregate_brier_advantage":edge,
        "symbol_brier_better_than_constant":wins,
        "profitable_trading_proven":False,
        "portfolio_risk_or_funding_or_real_sl_ack_verified":False,
        "limitations":[
            "Coefficient fitting used only authentic 2025-01/02 prefixes, not 2026.",
            "2026 research labels were inspected before; this is no untouched holdout.",
            "Horizon mark-to-market is research, NOT a Williams campaign exit rule.",
            "Related crypto pairs and overlapping directional samples are not independent.",
            "No inference or fine-tuning of the actual 1.7B GGUF occurs in this training job.",
            "No Binance orders, partial fills, funding, liquidations or actual SL confirmations.",
        ],
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--workers",type=int,default=7)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    train=obtain_predeclared_training(workers=a.workers)
    report=learn_and_evaluate(train)
    if a.out.exists():
        raise FileExistsError("Cannot overwrite evaluation report")
    with a.out.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,sort_keys=True,indent=2)
    print("MANY_TOKEN_REAL_MARKET_LEARNING",
          json.dumps({k:v for k,v in report.items() if k not in
                      ("per_2026_symbol","symbols_unavailable")},
                     ensure_ascii=False,sort_keys=True),flush=True)
    print("SYMBOL_SET",json.dumps({"available":train["available"],
                                 "requested":train["requested"],
                                 "unavailable":train["missing"]},
                                 sort_keys=True),flush=True)


if __name__=="__main__":
    main()
