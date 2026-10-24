"""Prepare unlabelled expert-review tasks from SEALED future-blind H1 cohorts.

A prospective public market capture supplies time, venue and exact OHLC
prefix fingerprint. The preparer publishes NO model's direction/setup and
NO future market data to reviewers. It never fabricates Elliott wave truth,
never certifies reviewer independence and never issues an exchange order.

Reviewers independently use these metadata to label the same closed chart
without seeing the AI answer, then pass two separate completed JSONL reviews
and a matching sealed model prediction to expert_wave_gold.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .prospective_qualification import _fingerprint


def prepare_wave_review_tasks(cohort:dict[str,Any])->dict[str,Any]:
    if not isinstance(cohort,dict):
        raise ValueError("SEALED_PUBLIC_COHORT_REQUIRED")
    manifest={k:v for k,v in cohort.items() if k!="manifest_sha256"}
    if (cohort.get("manifest_sha256")!=_fingerprint(manifest)
            or cohort.get("schema")!="WILLIAMS_AI_PUBLIC_FORWARD_COHORT_V1"
            or cohort.get("trade_action","MISSING") is not None
            or cohort.get("execution_allowed") is not False):
        raise ValueError("UNTRUSTED_OR_TAMPERED_WAVE_REVIEW_COHORT")
    venue=cohort.get("venue")
    if venue not in ("SPOT","FUTURES"):
        raise ValueError("UNKNOWN_VENUE")
    trials=cohort.get("trials")
    if not isinstance(trials,list) or not 1<=len(trials)<=10:
        raise ValueError("NEED_SEALED_CLOSED_MARKETS")
    pages=(("TC1",53),("NTD",91),("TC2",122),("COURSE",56))
    rows=[]
    seen=set()
    for i,trial in enumerate(trials):
        if not isinstance(trial,dict):
            raise ValueError("INVALID_PROSPECTIVE_TRIAL")
        symbol=trial.get("symbol")
        src=trial.get("source")
        market=trial.get("market_prefix")
        seal=trial.get("sealed_forecast")
        if not all(isinstance(x,dict) for x in (src,market,seal)):
            raise ValueError("MISSING_FROZEN_MARKET_PREFIX")
        bars=market.get("bars")
        if (not isinstance(symbol,str)
                or symbol in seen
                or not isinstance(bars,list) or len(bars)<220
                or src.get("venue")!=venue
                or trial.get("symbol")!=market.get("symbol")
                or trial.get("symbol")!=seal.get("symbol")
                or not isinstance(seal.get("as_of"),str)
                or seal.get("as_of")!=src.get("as_of")
                or bars[-1].get("close_time")!=src.get("as_of")
                or seal.get("history_sha256")!=_fingerprint(bars)
                or any(b.get("closed") is not True
                       or b.get("close_time")>src["as_of"]
                       for b in bars)):
            raise ValueError("UNSEALED_FUTURE_OR_INCONSISTENT_CHART")
        seen.add(symbol)
        edition,page=pages[i%len(pages)]
        case_id=f"{symbol}-H1-{seal['history_sha256'][:20]}"
        rows.append({
            "case_id":case_id,
            "symbol":symbol,"timeframe":"H1","venue":venue,
            "cutoff":src["as_of"],"visible_prefix_sha256":seal["history_sha256"],
            "edition":edition,"source_page":page,
            "fully_closed_bar_count":len(bars),
            "review_status":"UNLABELLED_REQUIRES_TWO_INDEPENDENT_EXPERTS",
            "expert_instructions":(
                "Without seeing the model answer, identify observable market "
                "structure and alternative nested wave counts at this cutoff; "
                "distinguish confirmed 2-right-bar pivots from candidate peaks, "
                "and specify why degree remains unverified."
            ),
            "model_prediction_hidden_from_reviewers":True,
            "no_market_bars_after_cutoff":True,
            "trade_action":None,"execution_allowed":False,
        })
    return {
        "schema":"WILLIAMS_AI_UNLABELLED_WAVE_REVIEW_TASKS_V1",
        "source_precommit_sha256":cohort["manifest_sha256"],
        "source_venue":venue,
        "tasks":rows,
        "all_wave_labels_missing_on_purpose":True,
        "cannot_serve_as_gold_without_two_real_independent_reviewers":True,
        "model_predicted_setups_or_future_prices_not_exposed":True,
        "trade_action":None,"execution_allowed":False,
    }


def main()->None:
    ap=argparse.ArgumentParser(description="Prepare fresh private wave-review queue without leaking GGUF predictions")
    ap.add_argument("--cohort",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise ValueError("REFUSE_OVERWRITE_PREVIOUS_WAVE_REVIEW_TASKS")
    data=prepare_wave_review_tasks(json.loads(args.cohort.read_text()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    print("WAVE_GOLD_REVIEW_QUEUE_PREPARED",len(data["tasks"]),
          data["source_precommit_sha256"],flush=True)


if __name__=="__main__":
    main()
