"""Run ACTUAL (not mocked) local GGUF inference on PUBLIC book decisions.

This is intentionally a small public-model smoke exam, not the user's
private-model fine-tuning. Real model checksum, model loader, number of actual
inference calls, and scores are reported. No private PDF, API key, user
device, wallet, position or order touches the model.

Usage:
  python -m williams_ai.real_gguf_acceptance \
     --local-model /tmp/Qwen3-0.6B-Q4_K_M.gguf \
     --expected-sha256 <exact 64-char expected file hash> \
     --max-questions 10 --history-cases 2

A passing Python CI using mock weights is NOT a real-inference acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any

from .competency_exam import CASES, grade_exam
from .causal_paper_lab import frozen_markets
from .reasoning_trial import evaluate_historical_thought


def actual_gguf_sha256(path: Path,expected:str)->dict[str,Any]:
    """Verify actual published bytes BEFORE constructing llama.cpp runtime."""
    p=Path(path).expanduser()
    if not p.is_file() or p.is_symlink() or p.suffix.lower()!=".gguf":
        raise FileNotFoundError("A REAL ordinary GGUF weights file is required")
    with p.open("rb") as f:
        if f.read(4)!=b"GGUF":
            raise ValueError("Not genuine GGUF file content")
    size=p.stat().st_size
    if size<50_000_000:
        raise ValueError("Research GGUF is unexpectedly tiny; mock weights prohibited")
    h=hashlib.sha256()
    with p.open("rb") as file:
        for block in iter(lambda:file.read(4*1024*1024),b""):
            h.update(block)
    digest=h.hexdigest()
    if not isinstance(expected,str) or len(expected)!=64 or digest!=expected.lower():
        raise ValueError("Model weight SHA256 mismatch with pinned public source")
    return {"model_sha256":digest,"model_bytes":size,"verified_model_magic":"GGUF"}


def public_model_trial(
    model:Any, *,weight_report:dict[str,Any],
    max_questions:int=10,history_cases:int=2,
)->dict[str,Any]:
    if not 1<=max_questions<=len(CASES):
        raise ValueError("Need 1..10 curated questions")
    if not 0<=history_cases<=5:
        raise ValueError("Need 0..5 historical scenarios")
    # No one can call this routine an actual weight-based exam unless the
    # caller has FIRST validated real bytes and passed real llama.cpp model.
    if weight_report.get("verified_model_magic")!="GGUF":
        raise ValueError("Model has not passed published-byte verification")
    start=monotonic()
    public_questions=CASES[:max_questions]
    exam=grade_exam(model.answer_exam_case,public_questions)
    print("REAL_GGUF_MC_SCORE",exam["correct_cases"],"of",exam["total_cases"],
          "not a blinded competency test",flush=True)
    names=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT")
    audits=[]
    for symbol in names[:history_cases]:
        market=frozen_markets()[symbol]
        # The model sees only 900 first H1 CLOSED candles, not future prices.
        try:
            case=evaluate_historical_thought(
                symbol,market["bars"],tick_size=market["tick_size"],
                bar_index=899,edition="TC2",model=model)
        except (ValueError,TypeError,RuntimeError,KeyError,IndexError) as exc:
            # Bad GGUF JSON is a failed actual inference, not a successful
            # trading decision. The audit must not crash before producing
            # a report or silently default to a fake NO_SETUP.
            audits.append({
                "symbol":symbol,"edition":"TC2",
                "last_visible_at":market["bars"][899]["close_time"],
                "verdict":"MALFORMED_MODEL_DECISION",
                "selected_setup":None,"hard_failures":["MALFORMED_DECISION:"+type(exc).__name__],
                "uncertainties":[],"trade_action":None,
            })
            print("REAL_GGUF_CHART_DECISION_INVALID",symbol,type(exc).__name__,flush=True)
            continue
        audit=case["audit"]
        audits.append({
            "symbol":symbol,"edition":"TC2",
            "last_visible_at":case["last_visible_at"],
            "decision_has_future_candles":case["model_saw_future_bars"],
            "verdict":audit["verdict"],
            "selected_setup":audit["setup"],
            "hard_failures":audit["hard_failures"],
            "uncertainties":audit["uncertainties"],
            "trade_action":None,
        })
        print("REAL_GGUF_CHART_DECISION",symbol,audit["verdict"],flush=True)
    return {
        "mode":"ACTUAL_PUBLIC_GGUF_LLAMACPP_CPU_INFERENCE",
        "weights_verified":weight_report,
        "actual_model_inference_calls":max_questions+history_cases,
        "curated_public_exam":{
            "questions":exam["total_cases"],
            "correct":exam["correct_cases"],
            "percent":exam["score_on_curated_cases_percent"],
            "results":[{"id":x["id"],"edition":x["edition"],
                        "chosen":x["chosen"],"correct":x["correct"]}
                       for x in exam["results"]],
            "not_blind_or_complete_book_test":True,
        },
        "historical_read_only_chart_decisions":audits,
        "elapsed_wall_seconds":round(monotonic()-start,2),
        "acceptance_verdict":(
            "PUBLIC_MODEL_UNQUALIFIED" if (
                exam["correct_cases"] < max_questions
                or any(x["verdict"]!="PROVISIONAL_REASONING_ONLY" for x in audits)
            ) else "PUBLIC_DIAGNOSTIC_ONLY_NO_TRADING_AUTHORITY"
        ),
        "user_private_book_text_used":False,
        "training_weights_modified":False,
        "actual_book_mastery_percent":None,
        "independent_trade_profitability":None,
        "vision_understanding_percent":None,
        "trade_action":None,"execution_allowed":False,
        "limitations":[
            "Public pre-trained GGUF is a SMALL research baseline; it is NOT Williams AI user's own fine-tuned weights.",
            "Public multiple-choice questions may be answered by guessing or memorization; no claims of full knowledge.",
            "Historical audit checks source/causality only; no real Binance order/SL ACK or portfolio PnL.",
            "No private licensed book contents, vision model, or mmproj projector were passed.",
        ],
    }


def main()->None:
    p=argparse.ArgumentParser(description="Actual public GGUF weight smoke exam, no simulation")
    p.add_argument("--local-model",type=Path,required=True)
    p.add_argument("--expected-sha256",required=True)
    p.add_argument("--max-questions",type=int,default=10)
    p.add_argument("--history-cases",type=int,default=2)
    p.add_argument("--ctx",type=int,default=8192)
    p.add_argument("--require-public-smoke-pass",action="store_true",
                   help="Fail job if model is unqualified; report still printed first")
    args=p.parse_args()
    checked=actual_gguf_sha256(args.local_model,args.expected_sha256)
    from .local_model import LocalWilliamsModel
    model=LocalWilliamsModel(args.local_model,n_ctx=args.ctx,gpu_layers=0,
                             use_decision_schema=True)
    report=public_model_trial(
        model,weight_report=checked,
        max_questions=args.max_questions,history_cases=args.history_cases)
    print("REAL_GGUF_EXAM",json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)
    if args.require_public_smoke_pass and report["acceptance_verdict"]=="PUBLIC_MODEL_UNQUALIFIED":
        raise SystemExit(2)


if __name__=="__main__":
    main()
