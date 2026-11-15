"""Recover the TWO missing historical100 Qwen GPU pairs without repeating the 98.

Authentic private 80-step LoRA, original pinned book/chart cases, one T4
inference session only, no training and no broker orders.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path

from scripts.ai_v6_historical100_models import gather
from scripts.ai_v6_historical100_real_binance import run as collect_official_data
from scripts.ai_v6_hist100_kaggle_gpu_worker import (
    find_trained_lora, compact_gpu_prompt, decode_hf, need, write_json_atomic,
    GPU_VISIBLE_CLOSED_H1_MAX, SEED,
)
from williams_ai.independent_exam_v6 import FAMILIES, canonical
from williams_ai.v6_expert_model_runner import prompt_for_case, protocol_sha256
from williams_ai.v6_v5_lora_binding import MODEL_COMMIT, RAW_ADAPTER_SHA

CASE_IDS = ("HIST100-25-3", "HIST100-25-4")
SCHEMA = "WILLIAMS_V6_HIST100_NF4_REAL_GPU_TWO_MISSING_PAIRED_20261115_V1"
FALLBACK_MAX_INPUT_TOKENS = 4096
FALLBACK_MIN_H1_BARS = 16

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    need(not args.out_dir.exists(), "LAST_TWO_GPU_OUTPUT_OVERWRITE_FORBIDDEN")
    args.out_dir.mkdir(parents=True)
    import torch
    need(torch.cuda.is_available() and torch.cuda.device_count()>=1, "GPU_NOT_ALLOCATED")
    root = args.out_dir/"official_market_hist100"
    market = collect_official_data(args.plan,root)
    need(market["distinct_chart_windows"] == 100 and
         market["no_future_in_model_chart"] is True,
         "OFFICIAL_H1_SOURCE_VALIDATION_FAILED")
    selected, market_plan_sha = gather(args.plan,root)
    pairs = [row for row in selected if row[0]["id"] in CASE_IDS]
    need(len(pairs)==2 and tuple(row[0]["id"] for row in pairs)==CASE_IDS,
         "FROZEN_MISSING_CASES_CHANGED")
    adapter=find_trained_lora()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-4B",
        revision=MODEL_COMMIT,trust_remote_code=False)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    base=AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B",revision=MODEL_COMMIT,trust_remote_code=False,
        device_map={"":0},quantization_config=quant,torch_dtype=torch.float16)
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
    model.eval()
    need(next(model.parameters()).is_cuda,"MODEL_NOT_ON_GPU")
    print("REAL_KAGGLE_GPU_FINISH2_CUDA_READY",torch.cuda.get_device_name(0),flush=True)
    out=args.out_dir/"ACTUAL_HIST100_GPU_LAST_TWO_QWEN_BASE_V5_NO_GOLD.json"
    receipt={
        "schema":SCHEMA,"complete":False,
        "original_100_case_selection_sha256":market_plan_sha,
        "original_cpu_full_protocol_sha256":protocol_sha256(),
        "original_gpu_backend":"HuggingFace Transformers v4.57 + PEFT + NF4",
        "original_trained_v5_adapter_sha256":RAW_ADAPTER_SHA,
        "original_base_revision":MODEL_COMMIT,
        "original_gpu_causal_20_48_bars_max_3200_for_first_98":True,
        "missing_2_gpu_memory_fallback_used":True,
        "fallback_max_input_tokens":FALLBACK_MAX_INPUT_TOKENS,
        "fallback_min_closed_H1":FALLBACK_MIN_H1_BARS,
        "no_gold":True,"accuracy":None,"no_trading":True,
        "real_paired_cases":0,
        "cuda_allocated":True,
        "observed":{"matched_base":[],"candidate_v5":[]}}
    write_json_atomic(out,receipt)
    for case,chart,evidence,chart_sha in pairs:
        original,ev=prompt_for_case(case,chart)
        need(ev==evidence,"CHANGED_REVIEWED_RULES")
        gpu_messages,hash_prompt,tokens,visible=compact_gpu_prompt(
            tokenizer,original,max_input_tokens=FALLBACK_MAX_INPUT_TOKENS,
            min_closed_h1_bars=FALLBACK_MIN_H1_BARS)
        need(16<=visible<=48 and tokens<=FALLBACK_MAX_INPUT_TOKENS,
             "FALLBACK_PROMPT_BOUNDS")
        allowed=list(FAMILIES[case["family"]])+["ABSTAIN"]
        for name,enabled in (("matched_base",False),("candidate_v5",True)):
            if evidence["source_missing"]:
                ans=("ABSTAIN","stop",None,0.,0,0)
            else:
                ans=decode_hf(model,tokenizer,gpu_messages,allowed,
                              use_adapter=enabled,torch=torch)
            decision,finish,reason,seconds,in_tokens,out_tokens=ans
            row={
                "id":case["id"],"edition":case["edition"],"family":case["family"],
                "symbol":chart["symbol"],"cutoff_utc":case["cutoff_utc"],
                "decision":decision,"finish_reason":finish,"model_reason":reason,
                "chart_sha256":chart_sha,"prompt_sha256":hash_prompt,
                "original_cpu_full_prompt_sha256":ev["prompt_sha256"],
                "gpu_prompt_tokens":tokens,"gpu_visible_closed_h1_bars":visible,
                "retrieved_rule_ids":ev["retrieved_rule_ids"],
                "source_missing":ev["source_missing"],
                "duration_sec":seconds,"tokens_in":in_tokens,"tokens_out":out_tokens,
                "fallback_after_prior_3200_context_overflow":True,
            }
            receipt["observed"][name].append(row)
        receipt["real_paired_cases"]+=1
        write_json_atomic(out,receipt)
        print("REAL_KAGGLE_GPU_LAST_TWO_PAIRED_CASE",case["id"],
              receipt["real_paired_cases"],flush=True)
        torch.cuda.empty_cache()
    need(receipt["real_paired_cases"]==2 and
         all(len(receipt["observed"][r])==2 for r in receipt["observed"]),
         "NOT_TWO_REAL_GPU_PAIRED_PREDICTIONS")
    for a,b in zip(receipt["observed"]["matched_base"],
                   receipt["observed"]["candidate_v5"],strict=True):
        need(all(a[k]==b[k] for k in ("id","edition","symbol","family",
                    "chart_sha256","prompt_sha256","retrieved_rule_ids")),
             "GPU_FALLBACK_INPUT_NOT_MATCHED")
    receipt["complete"]=True
    write_json_atomic(out,receipt)
    print("REAL_GPU_LAST_TWO_MODEL_PAIRS_COMPLETE_NO_GOLD",flush=True)

if __name__=="__main__":
    main()
