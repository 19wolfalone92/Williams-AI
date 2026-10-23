"""Large chronological exam: real local GGUF, 15 pairs and four editions.

The metric is SOURCE/CAUSAL CLAIM CONSISTENCY ONLY, not 100% cognition,
not realized PnL, not a certification of full books or licensed advice.

python -m williams_ai.reasoning_batch --model /private/real.gguf
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from .causal_paper_lab import FROZEN_DATA, frozen_markets
from .reasoning_trial import evaluate_historical_thought

EDITION_ORDER=("TC1","NTD","TC2","COURSE")


def examine_many(
    markets:dict[str,dict[str,Any]], model:Any, *,
    editions:tuple[str,...]=EDITION_ORDER,
    checkpoints:tuple[int,...]=(299,599,899,1199),
)->dict[str,Any]:
    if not markets:
        raise ValueError("Need historical market data")
    if not isinstance(editions,tuple) or not editions or any(x not in EDITION_ORDER for x in editions):
        raise ValueError("Unknown source edition in batch")
    if (not isinstance(checkpoints,tuple) or not checkpoints
            or sorted(set(checkpoints))!=list(checkpoints)):
        raise ValueError("Checkpoints must be strictly ordered")
    findings=[]
    counts=Counter()
    failed_cases=0
    malformed_cases=0
    abstained_cases=0
    substantive_cases=0
    substantive_pass=0
    for symbol in sorted(markets):
        market=markets[symbol]
        for last_bar in checkpoints:
            if not 60<=last_bar<len(market["bars"]):
                raise ValueError("Checkpoint outside available closed history")
            for edition in editions:
                try:
                    result=evaluate_historical_thought(
                        symbol,market["bars"],tick_size=market["tick_size"],
                        bar_index=last_bar,edition=edition,model=model)
                except (ValueError,TypeError,RuntimeError,KeyError,IndexError) as exc:
                    # Do not stop the exam when the GGUF emits invalid JSON.
                    malformed_cases+=1
                    findings.append({
                        "symbol":symbol,"bar_index":last_bar,"edition":edition,
                        "verdict":"MALFORMED_MODEL_OUTPUT","codes":[type(exc).__name__],
                    })
                    continue
                audit=result["audit"]
                errors=audit["hard_failures"]
                substantive=audit["setup"] not in {"NO_SETUP","UNKNOWN"}
                if not substantive:
                    abstained_cases+=1
                else:
                    substantive_cases+=1
                    if not errors:substantive_pass+=1
                if errors:
                    failed_cases+=1
                    for code in errors:
                        counts[code]+=1
                findings.append({
                    "symbol":symbol,"bar_index":last_bar,"edition":edition,
                    "verdict":audit["verdict"],"codes":errors,
                    "selected_setup":audit["setup"],
                    "uncertainty_count":len(audit["uncertainties"]),
                    "trade_action":None,
                })
    total=len(findings)
    passing=total-failed_cases-malformed_cases
    return {
        "mode":"LOCAL_GGUF_HISTORICAL_SOURCE_CONSISTENCY_EXAM",
        "tested_cases":total,"checked_editions":list(editions),
        "checked_symbols":sorted(markets),
        "source_consistent_cases":passing,
        "unsupported_claim_cases":failed_cases,
        "malformed_model_cases":malformed_cases,
        "abstained_or_unknown_cases":abstained_cases,
        "substantive_claim_cases":substantive_cases,
        "substantive_source_consistent_cases":substantive_pass,
        "substantive_source_consistency_percent":(
            round(100*substantive_pass/substantive_cases,2)
            if substantive_cases else None
        ),
        "source_consistency_percent":round(100*passing/total,2) if total else None,
        "full_book_understanding_percent":None,
        "book_rule_coverage_percent":None,
        "gross_pnl":None,"trade_action":None,"execution_allowed":False,
        "failure_counts":dict(sorted(counts.items())),
        "case_trace":findings,
        "limitations":[
            "This examines rule consistency, NOT equivalent to complete book comprehension.",
            "A model that always answers NO_SETUP could score 100% on safety consistency; check abstention counts.",
            "Answers to chart questions are not independently labeled profitable trading actions.",
            "Image reading is evaluated by the separate private multimodal vision adapter.",
            "No Binance orders, fills, stops, funding or position accounting are verified.",
        ],
    }


def main()->None:
    p=argparse.ArgumentParser(description="Four-book, 15-market closed-chart local GGUF consistency exam")
    p.add_argument("--local-model",type=Path,required=True)
    p.add_argument("--dataset",type=Path,default=FROZEN_DATA)
    p.add_argument("--gpu-layers",type=int,default=0)
    p.add_argument("--edition",choices=[*EDITION_ORDER,"ALL"],default="ALL")
    p.add_argument("--pairs",type=int,default=15)
    p.add_argument("--checkpoints",type=int,choices=[1,2,4],default=4)
    args=p.parse_args()
    from .local_model import LocalWilliamsModel
    model=LocalWilliamsModel(args.local_model,gpu_layers=args.gpu_layers)
    all_markets=frozen_markets(args.dataset)
    if not 1<=args.pairs<=len(all_markets):
        p.error("--pairs out of range")
    select=dict(sorted(all_markets.items())[:args.pairs])
    positions={1:(899,),2:(599,999),4:(299,599,899,1199)}[args.checkpoints]
    editions=EDITION_ORDER if args.edition=="ALL" else (args.edition,)
    output=examine_many(select,model,editions=editions,checkpoints=positions)
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
