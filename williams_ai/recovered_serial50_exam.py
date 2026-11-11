"""Actual local GGUF inference with ORIGINAL 50-cycle trained Qwen3-4B LoRA.

This is intentionally separate from live trading. The artifact is recovered
and authenticated by the preceding script using the runner's existing secret,
then PEFT weights are converted in private temp and loaded by llama.cpp.
No fake model, no substitute random adapter, no unbound memory or live orders.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from .knowledge_abc_exam_v1 import assess_model
from .local_model import LocalWilliamsModel
from .real_gguf_acceptance import actual_gguf_sha256

class OriginalSerial50LoRAExam:
    # Reuse EXACT same answer prompt, Qwen3 thinking budget and JSON
    # validation as the pre-existing LocalWilliamsModel baseline.
    answer_exam_case=LocalWilliamsModel.answer_exam_case
    _learning_completion=LocalWilliamsModel._learning_completion

    def __init__(self, *, base:Path,adapter:Path,report:dict):
        from llama_cpp import Llama
        if (not adapter.is_file() or adapter.suffix.lower()!=".gguf"
            or adapter.stat().st_size<1000000
            or not isinstance(report,dict)
            or report.get("status")!="AUTHENTICATED_TRAINED_QWEN3_4B_LORA_RECOVERED"
            or report.get("research_gate_passed") is not False
            or report.get("optimizer_updates")!=1000
            or report.get("selected_checkpoint")!="checkpoint-500"
            or report.get("execution_allowed") is not False
            or len(report.get("adapter_sha256",""))!=64):
            raise ValueError("VERIFIED_GENUINE_UNPROMOTED_50_CYCLE_ADAPTER_REQUIRED")
        with adapter.open("rb") as stream:
            if stream.read(4)!=b"GGUF":
                raise ValueError("REAL_CONVERTED_LORA_GGUF_REQUIRED")
        self._model=Llama(
            model_path=str(base.resolve()),
            lora_path=str(adapter.resolve()),lora_scale=1.0,
            n_ctx=4096,n_gpu_layers=0,verbose=False)
        self.model_id="serial50-"+report["adapter_sha256"][:24]
        self._learning_reasoning=True
        self._reasoning_budget=512
        self.latest_learning_call=None

def main():
    p=argparse.ArgumentParser(description="Original recovered trained LoRA vs prior public Qwen3-4B baseline")
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--base-sha256",required=True)
    p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--original-safetensors",type=Path,required=True)
    p.add_argument("--recovery-receipt",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--per-edition",type=int,default=4,choices=range(1,9))
    args=p.parse_args()
    pinned=actual_gguf_sha256(args.base,args.base_sha256)
    recovery=json.loads(args.recovery_receipt.read_text(encoding="utf-8"))
    with args.original_safetensors.open("rb") as stream:
        sha=sha256()
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            sha.update(chunk)
    if sha.hexdigest()!=recovery.get("adapter_sha256"):
        raise ValueError("RECOVERED_TRAINED_WEIGHTS_NO_LONGER_MATCH_ORIGINAL_AUDIT")
    model=OriginalSerial50LoRAExam(base=args.base,adapter=args.adapter,report=recovery)
    result=assess_model(model,model_sha256=pinned["model_sha256"],
                        adapter_sha256=recovery["adapter_sha256"],
                        per_edition=args.per_edition,
                        model_verified_by_loader=True)
    result["trained_source"]="RECOVERED_GITHUB_ACTIONS_42882670406_CHECKPOINT_500"
    result["original_50_cycle_training_gate_passed"]=False
    result["actual_lora_runtime_loaded"]=True
    result["gguf_lora_sha256"]=sha256(args.adapter.read_bytes()).hexdigest()
    result["hf_base_revision_independently_attested_against_gguf_conversion"]=False
    result["trading_qualification"]=False
    if args.output.exists():
        raise ValueError("NEVER_OVERWRITE_PRIOR_BLIND_MODEL_EXAM_REPORT")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"w",encoding="utf-8") as out:
        json.dump(result,out,ensure_ascii=False,indent=2,sort_keys=True)
    print("REAL_TRAINED_WILLIAMS_QWEN3_4B_BOOK_EXAM",
          json.dumps({"scores":result["scores"],
                      "trained_source":result["trained_source"],
                      "actual_lora_runtime_loaded":True,
                      "private_weights_exported":False,
                      "execution_allowed":False},ensure_ascii=False,sort_keys=True),
          flush=True)

if __name__=="__main__":
    main()
