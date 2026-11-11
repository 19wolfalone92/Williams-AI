"""Two-condition actual local GGUF four-book knowledge diagnostic.

The same verified model answers the same frozen 16/32 CASES:
(A) question only, (B) question plus automatically retrieved reviewed
*public rule cards*. This is NOT an independent blind holdout or a proof
that the model's weights learned the books. Never put answer keys in prompts.
Private book text, page images, adapters and credentials are never exported.

By default GitHub Actions tests a publicly available BASE Qwen3-4B GGUF
as a CONTROL. That is NOT the user's trained Williams QLoRA checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .four_book_blind_exam import grade_four_book_model
from .real_gguf_acceptance import actual_gguf_sha256


def assess_model(model, *, model_sha256: str, adapter_sha256: str | None = None,
                 per_edition: int = 4, seed: int = 20261024) -> dict:
    if not isinstance(model_sha256, str) or len(model_sha256) != 64 or any(
            letter not in "0123456789abcdef" for letter in model_sha256):
        raise ValueError("ACTUAL_PINNED_GGUF_SHA256_REQUIRED")
    if adapter_sha256 is not None and (not isinstance(adapter_sha256, str) or
            len(adapter_sha256) != 64 or any(
                letter not in "0123456789abcdef" for letter in adapter_sha256)):
        raise ValueError("INVALID_TRAINED_ADAPTER_DIGEST")
    a=grade_four_book_model(model, seed=seed,per_edition=per_edition,
                            autonomous_source_retrieval=False)
    b=grade_four_book_model(model, seed=seed,per_edition=per_edition,
                            autonomous_source_retrieval=True)
    if a["protocol_sha256"]!=b["protocol_sha256"] or a["total"]!=b["total"]:
        raise ValueError("A_B_DIFFERENT_QUESTIONS_OR_HIDDEN_KEY")
    failures={
        label:[{"case_id":row["id"],"edition":row["edition"],
                "rule_id":row["source_rule_id"],"stage":row["stage"],
                "abstained":row["abstained"],"failure":row["failure"]}
               for row in report["results"] if not row["correct"]]
        for label,report in (("closed_book",a),("source_assisted",b))}
    solved_without={r["id"] for r in a["results"] if r["correct"]}
    solved_with={r["id"] for r in b["results"] if r["correct"]}
    regressions=sorted(solved_without - solved_with)
    new_passes=sorted(solved_with - solved_without)
    return {
        "schema":"WILLIAMS_ACTUAL_GGUF_BOOK_KNOWLEDGE_A_B_V1",
        "evaluated_at":datetime.now(timezone.utc).isoformat(),
        "gguf_sha256":model_sha256,"adapter_sha256":adapter_sha256,
        "model_class":"ACTUAL_USER_ADAPTER_ONLY_IF_VERIFIED" if adapter_sha256
                      else "PUBLIC_BASE_QWEN3_4B_CONTROL_NOT_WILLIAMS_TRAINED",
        "frozen_protocol_sha256":a["protocol_sha256"],
        "seed":seed,"balanced_cases_per_edition":per_edition,
        "conditions":{"closed_book":a,"source_assisted":b},
        "scores":{"without_sources":a["correct"],"with_sources":b["correct"],
                  "case_count":a["total"],
                  "net_delta":b["correct"]-a["correct"],
                  "improved_case_ids":new_passes,
                  "regressed_case_ids":regressions,
                  "marginal_majority_letter_baseline":a["majority_letter_baseline"]},
        "review_queue":failures,
        "same_questions_in_both_conditions":True,
        "correct_answer_never_provided_to_inference":True,
        "test_questions_source_derived_not_independent_market_gold":True,
        "real_model_inference_performed":True,
        "book_visual_comprehension_tested":False,
        "market_future_sealed_exam_completed":False,
        "user_trained_model_benchmarked":adapter_sha256 is not None,
        "model_weights_modified":False,
        "private_books_exported":False,
        "trade_action":None,"execution_allowed":False,
    }


def main():
    p=argparse.ArgumentParser(
        description="Actual pinned-GGUF source-knowledge A/B with no trading")
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--expected-sha256",required=True)
    p.add_argument("--lora-adapter",type=Path)
    p.add_argument("--lora-manifest",type=Path)
    p.add_argument("--per-edition",type=int,default=4,choices=range(1,9))
    p.add_argument("--seed",type=int,default=20261024)
    p.add_argument("--ctx",type=int,default=4096)
    p.add_argument("--reasoning-budget",type=int,default=512)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if bool(args.lora_adapter)!=bool(args.lora_manifest):
        raise ValueError("ADAPTER_AND_ATTESTED_MANIFEST_REQUIRED_TOGETHER")
    verified=actual_gguf_sha256(args.model,args.expected_sha256)
    from .local_model import LocalWilliamsModel
    from .model_bundle import runtime_identity
    identity=runtime_identity(args.model,adapter=args.lora_adapter,
                              manifest=args.lora_manifest)
    print("ACTUAL_GGUF_RUNTIME_IDENTITY",
          json.dumps({"model_sha256":verified["model_sha256"],
                      "adapter_verified":bool(args.lora_adapter),
                      "actual_book_eval_start":True,
                      "execution_allowed":False},sort_keys=True),flush=True)
    model=LocalWilliamsModel(
        args.model,n_ctx=args.ctx,gpu_layers=0,learning_mode="auto",
        reasoning_budget=args.reasoning_budget,
        lora_adapter=args.lora_adapter,lora_manifest=args.lora_manifest)
    adapter_sha=identity.get("adapter_sha256") if args.lora_adapter else None
    if args.lora_adapter and not adapter_sha:
        raise ValueError("ATTESTED_TRAINED_ADAPTER_HASH_REQUIRED")
    report=assess_model(
        model,model_sha256=verified["model_sha256"],
        adapter_sha256=adapter_sha,
        per_edition=args.per_edition,seed=args.seed)
    output=args.output.expanduser().resolve()
    from .book_index import ROOT
    if output.exists() or output==ROOT or ROOT in output.parents:
        raise ValueError("NEW_RESULTS_FILE_OUTSIDE_SOURCE_TREE_REQUIRED")
    import os
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,sort_keys=True,indent=2)
    print("ACTUAL_QWEN4B_KNOWLEDGE_AB_REPORT",
          json.dumps({"score":report["scores"],
                      "user_trained_model_benchmarked":report["user_trained_model_benchmarked"],
                      "real_model_inference_performed":True,
                      "private_books_exported":False,
                      "execution_allowed":False},ensure_ascii=False,sort_keys=True),
          flush=True)


if __name__=="__main__":
    main()
