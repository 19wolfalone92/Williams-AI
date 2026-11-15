"""Versioned real-format-only HF/NF4 v7 pilot, eight frozen H1 cases (not gold).

Identical original source/user prompt to v6, but DIFFERENT guided JSON decoding.
All real inference is Qwen3-4B pinned base + original private v5 LoRA.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from scripts.ai_v6_historical100_models import gather
from scripts.ai_v6_historical100_real_binance import run as collect_archive
from scripts.ai_v6_hist100_kaggle_gpu_worker import find_trained_lora,compact_gpu_prompt,need,write_json_atomic,SEED
from williams_ai.v6_expert_model_runner import prompt_for_case,protocol_sha256
from williams_ai.v6_v5_lora_binding import MODEL_COMMIT,RAW_ADAPTER_SHA
from williams_ai.v7_structured_decoder import infer_one_hf,SCHEMA as DECODER_SCHEMA

IDS=tuple(f"HIST100-{symbol:02d}-{date}" for symbol in (1,2) for date in (1,2,3,4))
SCHEMA="WILLIAMS_V7_JSON_GUIDED_REAL_QWEN_V5_8_HISTORICAL_H1_NO_GOLD_V1"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    need(not a.out_dir.exists(),"V7_OUTPUT_EXISTS")
    a.out_dir.mkdir(parents=True)
    import torch
    need(torch.cuda.is_available() and torch.cuda.device_count()>=1,"T4_MISSING")
    root=a.out_dir/"official_hist100"
    corpus=collect_archive(a.plan,root)
    need(corpus["distinct_chart_windows"]==100 and
         corpus["no_future_in_model_chart"] is True,"MARKET_FUTURE_OR_INCOMPLETE")
    source,plan_sha=gather(a.plan,root)
    cases=[x for x in source if x[0]["id"] in IDS]
    need(len(cases)==8 and [x[0]["id"] for x in cases]==list(IDS),
         "NOT_THE_SAME_8_CASES")
    adapter=find_trained_lora()
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-4B",
        revision=MODEL_COMMIT,trust_remote_code=False)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    base=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B",
        revision=MODEL_COMMIT,trust_remote_code=False,device_map={"":0},
        quantization_config=quant,torch_dtype=torch.float16)
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
    model.eval()
    need(next(model.parameters()).is_cuda,"MODEL_NOT_ON_CUDA")
    state={
        "schema":SCHEMA,"decoder_schema":DECODER_SCHEMA,
        "complete":False,"original_market_100_selection_sha256":plan_sha,
        "original_cpu_full_prompt_protocol_sha256":protocol_sha256(),
        "gpu_prompt_limit":3200,"source_cases":list(IDS),"paired_cases":0,
        "model_base_revision":MODEL_COMMIT,
        "raw_v5_adapter_sha256":RAW_ADAPTER_SHA,
        "research_only":True,"expert_gold":None,"trading_allowed":False,
        "weights_unchanged":True,"training_started":False,
        "observed":{"matched_base":[],"candidate_v5":[]},
    }
    outfile=a.out_dir/"ACTUAL_V7_SCHEMA_GUIDED_EIGHT_PAIRED_MODEL_ANSWERS.json"
    write_json_atomic(outfile,state)
    for case,chart,evidence,chart_sha in cases:
        original, meta=prompt_for_case(case,chart)
        need(meta==evidence,"CITATION_OR_PROMPT_CHANGED")
        messages,gpu_sha,tokens,bars=compact_gpu_prompt(tokenizer,original)
        for role,enabled in (("matched_base",False),("candidate_v5",True)):
            from contextlib import nullcontext
            ctx=nullcontext() if enabled else model.disable_adapter()
            with ctx:
                ans=infer_one_hf(model,tokenizer,messages,case["family"],torch=torch)
            state["observed"][role].append({
                "id":case["id"],"edition":case["edition"],"family":case["family"],
                "symbol":case["id"].split("-")[1],"cutoff_utc":case["cutoff_utc"],
                "original_symbol":chart["symbol"],"chart_sha256":chart_sha,
                "prompt_sha256":gpu_sha,"original_cpu_full_prompt_sha256":evidence["prompt_sha256"],
                "rules":evidence["retrieved_rule_ids"],"gpu_visible_bars":bars,
                **ans,
            })
        state["paired_cases"]+=1
        write_json_atomic(outfile,state)
        print("REAL_V7_JSON_GUIDED_PAIRED_CASE",case["id"],state["paired_cases"],flush=True)
        torch.cuda.empty_cache()
    need(state["paired_cases"]==8 and
         all(len(state["observed"][r])==8 for r in state["observed"]),
         "INCOMPLETE_8_REAL_QWEN_PAIRS")
    for a,b in zip(state["observed"]["matched_base"],
                   state["observed"]["candidate_v5"],strict=True):
        need(all(a[k]==b[k] for k in ("id","edition","family",
             "cutoff_utc","chart_sha256","prompt_sha256","rules")),
             "BASE_AND_LORA_INPUT_DIFFERENCE")
    state["complete"]=True
    write_json_atomic(outfile,state)
    print("REAL_V7_8_BASE_8_LORA_GUIDED_FORMAT_PILOT_COMPLETE_NO_GOLD",flush=True)

if __name__=="__main__":main()
