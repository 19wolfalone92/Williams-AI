"""Prospective public H1 trial with REAL local GGUF, retained immutable inputs.

PREPARE: capture public *current* fully closed BTC/ETH/SOL H1, send only the
causal chart context to a pinned offline real GGUF, seal forecasts and market
prefixes, then upload the JSON files as a run artifact.

SETTLE: after 24+ fully closed new H1 bars exist, reload the PREVIOUS
commitments and re-download the matching venue's public bars from EXACTLY
the stored original first H1 timestamp; reject changed or truncated prefix.
Never ask GGUF again during settlement.

The trial supports FUTURES and SPOT as EXPLICIT DISTINCT test populations.
It cannot create a LIVE/DEMO order, and 24h direction is NOT PnL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .public_forward_capture import (
    capture_latest_closed_h1, HOUR_MS, VENUES,
)
from .prospective_qualification import (
    freeze_prospective_call, settle_prospective_call,
    summarize_sealed_results,
    _fingerprint,
)


def prepare_forward_cohort(
    *, venue: str, symbols: tuple[str,...], model_id: str, code_id: str,
    predictor: Callable[[dict[str,Any]],dict[str,Any]],
    fetch_json: Callable[[str],Any] | None = None,
    clock_ms: int | None = None,
) -> dict[str,Any]:
    """No hidden calls to unrelated history or user accounts."""
    if venue not in VENUES or not isinstance(symbols,tuple) or not symbols or (
        len(set(symbols))!=len(symbols) or len(symbols)>10
    ):
        raise ValueError("Distinct explicit public venue and symbols required")
    if not callable(predictor):
        raise ValueError("A genuine predictor callback must be supplied")
    trials=[]
    for symbol in symbols:
        kwargs={"venue":venue,"clock_ms":clock_ms}
        if fetch_json is not None:
            kwargs["fetch_json"]=fetch_json
        public=capture_latest_closed_h1(symbol,**kwargs)
        market=public["market"]
        sealed=freeze_prospective_call(
            symbol,market["bars"],tick_size=market["tick_size"],
            model_id=model_id,code_id=code_id,predictor=predictor,
        )
        if sealed["as_of"]!=public["source"]["as_of"] or (
            sealed["history_sha256"]!=_fingerprint(market["bars"])
        ):
            raise ValueError("Public market and model commitment disagree")
        trials.append({
            "symbol":symbol,"source":public["source"],
            "market_prefix":market,
            "sealed_forecast":sealed,
        })
    manifest={
        "schema":"WILLIAMS_AI_PUBLIC_FORWARD_COHORT_V1",
        "venue":venue, "model_id":model_id,"code_id":code_id,
        "symbols":list(symbols),"trials":trials,
        "created_by":"closed-market-data-plus-sealed-real-predictor",
        "note":"PRECOMMITTED_UNSCORED_24H_FUTURE_NOT_YET_KNOWN",
        "trade_action":None,"execution_allowed":False,
    }
    return {**manifest,"manifest_sha256":_fingerprint(manifest)}


def settle_forward_cohort(
    cohort:dict[str,Any], *,
    fetch_json:Callable[[str],Any] | None = None,
    clock_ms:int | None = None,
) -> dict[str,Any]:
    """Cannot mutate/refit committed output or swap SPOT vs FUTURES."""
    if not isinstance(cohort,dict):
        raise ValueError("Sealed cohort required")
    core={k:v for k,v in cohort.items() if k!="manifest_sha256"}
    if (cohort.get("manifest_sha256")!=_fingerprint(core)
            or core.get("schema")!="WILLIAMS_AI_PUBLIC_FORWARD_COHORT_V1"
            or core.get("execution_allowed") is not False
            or core.get("trade_action","INVALID") is not None
            or core.get("venue") not in VENUES):
        raise ValueError("Cohort seal mutated or execution metadata unsafe")
    trials=core.get("trials")
    symbols=core.get("symbols")
    if (not isinstance(trials,list) or not isinstance(symbols,list) or
            len(trials)!=len(symbols) or len(set(symbols))!=len(symbols)
            or not 1<=len(trials)<=10):
        raise ValueError("Malformed sealed cohort")
    results=[]
    for record,symbol in zip(trials,symbols):
        if not isinstance(record,dict) or record.get("symbol")!=symbol:
            raise ValueError("Pair ownership mutated")
        market=record.get("market_prefix")
        source=record.get("source")
        sealed=record.get("sealed_forecast")
        if (not isinstance(market,dict) or not isinstance(source,dict) or
                not isinstance(sealed,dict) or source.get("venue")!=core["venue"]
                or market.get("symbol")!=symbol or sealed.get("symbol")!=symbol
                or source.get("as_of")!=sealed.get("as_of")
                or sealed.get("model_id")!=core["model_id"]
                or sealed.get("code_id")!=core["code_id"]):
            raise ValueError("Market venue or study identity changed")
        prev=market.get("bars")
        if not isinstance(prev,list) or len(prev)<220:
            raise ValueError("Original public market history missing")
        if sealed.get("history_sha256")!=_fingerprint(prev):
            raise ValueError("Market prefix was modified after precommitment")
        stamp=prev[0]["close_time"]
        from datetime import datetime
        first_close=int(round(datetime.fromisoformat(stamp).timestamp()*1000))
        start=first_close-(HOUR_MS-1)
        kwargs={"venue":core["venue"],"clock_ms":clock_ms,
                "limit":750,"start_time_ms":start}
        if fetch_json is not None:
            kwargs["fetch_json"]=fetch_json
        updated=capture_latest_closed_h1(symbol,**kwargs)
        if updated["market"]["bars"][:len(prev)]!=prev or (
                updated["market"]["tick_size"]!=market["tick_size"]):
            raise ValueError("Original data changed or venue silently switched")
        outcome=settle_prospective_call(sealed,updated["market"]["bars"])
        results.append(outcome)
    summary=summarize_sealed_results(results)
    return {
        "mode":"PUBLIC_PROSPECTIVE_24H_SETTLED_NOT_TRADING",
        "commitment_manifest_sha256":cohort["manifest_sha256"],
        "venue":core["venue"],
        "settled":results, "summary":summary,
        "verified_real_account_PnL":False,
        "trade_action":None,"execution_allowed":False,
    }


def _json(path:Path)->dict[str,Any]:
    from .prospective_cli import _read_json
    val=_read_json(path)
    if not isinstance(val,dict):
        raise ValueError("Expected JSON object")
    return val


def main()->None:
    p=argparse.ArgumentParser(description="Precommit a fresh public-data GGUF study and settle later")
    sub=p.add_subparsers(dest="mode",required=True)
    pre=sub.add_parser("prepare",help="run real pinned GGUF against NEW closed public bars")
    post=sub.add_parser("settle",help="revisit only after next 24 H1 closes")
    pre.add_argument("--venue",choices=sorted(VENUES),required=True)
    pre.add_argument("--symbols",nargs="+",default=["BTCUSDT","ETHUSDT","SOLUSDT"])
    pre.add_argument("--model",type=Path,required=True)
    pre.add_argument("--expected-sha256",required=True)
    pre.add_argument("--code-id",required=True)
    pre.add_argument("--gpu-layers",type=int,default=0)
    post.add_argument("--cohort",type=Path,required=True)
    for command in (pre,post):
        command.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.mode=="prepare":
        from .real_gguf_acceptance import actual_gguf_sha256
        from .study_cli import _model_identity
        from .local_model import LocalWilliamsModel
        actual_gguf_sha256(args.model,args.expected_sha256)
        model=LocalWilliamsModel(args.model,n_ctx=8192,gpu_layers=args.gpu_layers)
        report=prepare_forward_cohort(
            venue=args.venue,symbols=tuple(args.symbols),
            model_id=_model_identity(args.model),code_id=args.code_id,
            predictor=model.propose_24h_research_forecast,
        )
    else:
        report=settle_forward_cohort(_json(args.cohort))
    from .prospective_cli import _write_once_private
    _write_once_private(args.out,report)
    if args.mode=="prepare":
        for trial in report["trials"]:
            print("SEALED_NEW_H1",trial["source"]["venue"],
                  trial["symbol"],trial["source"]["as_of"],
                  trial["sealed_forecast"]["direction"],
                  trial["sealed_forecast"]["commitment_sha256"],flush=True)
        print("PROSPECTIVE_COHORT_SHA",report["manifest_sha256"],"UNSCORED_NO_TRADE")
    else:
        print("SETTLED_COHORT_SHA",report["commitment_manifest_sha256"],
              "NOT_A_TRADE_BACKTEST",flush=True)
        print("FORWARD_DIRECTIONAL_SUMMARY",json.dumps(report["summary"],
               ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
