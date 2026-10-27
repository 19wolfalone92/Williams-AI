"""Actual CUDA NF4 QLoRA fit of OFFICIAL pretrained Qwen3-4B.

PUBLIC paraphrased Williams book-rule practice ONLY. Original privately
licensed PDFs and independently reviewed chart labels are never uploaded.
This is a reproducible capability pilot, NOT trading readiness.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

MODEL_REPO = "Qwen/Qwen3-4B"
# Official upstream Qwen commit. Pin prevents silent base-model drift.
MODEL_REVISION = "3fb356690bbdae7353c94efb3eecdce276e6ed1c"
ED = ("TC1","NTD","TC2","COURSE")


def digest(file: Path) -> str:
    h=sha256()
    with file.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def train_public_qlora(new_output: Path, steps: int = 32, rank: int = 8) -> dict:
    if (type(steps) is not int or not 2<=steps<=80
            or type(rank) is not int or rank not in (4,8,16)
            or new_output.exists() or new_output.is_symlink()):
        raise ValueError("BOUNDED_NEW_PUBLIC_RESEARCH_FIT_REQUIRED")
    import torch
    from huggingface_hub import HfApi
    from transformers import (
        AutoConfig,AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    )
    from peft import prepare_model_for_kbit_training
    from williams_ai.private_lora import _tokenized
    from williams_ai.lora_training import fit_lora,LoraPlan
    from scripts.ai_real_qwen_source_lora_start import build_public_exercises

    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("EXACTLY_ONE_VISIBLE_NVIDIA_CUDA_DEVICE_REQUIRED")
    torch.cuda.set_device(0)
    # Public source practice is deliberately not a human expert chart dataset.
    train,val,provenance=build_public_exercises()
    if provenance.get("independent_expert_chart_review") is not False:
        raise ValueError("FALSE_PUBLIC_EXPERT_REVIEW_CLAIM")
    if not all(val.values()) or len(train)<20:
        raise ValueError("PUBLIC_FOUR_BOOK_CURRICULUM_INCOMPLETE")
    info=HfApi().model_info(MODEL_REPO, revision=MODEL_REVISION,files_metadata=True)
    if info.sha!=MODEL_REVISION:
        raise ValueError("HF_BASE_MODEL_COMMIT_NOT_IMMUTABLY_VERIFIED")
    upstream_weights={
        f.rfilename:f.lfs.sha256
        for f in info.siblings
        if f.rfilename.endswith(".safetensors") and f.lfs
    }
    if not upstream_weights:
        raise ValueError("HF_OFFICIAL_SAFETENSORS_SHA_REQUIRED")
    config=AutoConfig.from_pretrained(MODEL_REPO,revision=MODEL_REVISION,
                                      trust_remote_code=False)
    if (config.model_type!="qwen3" or config.hidden_size!=2560
            or config.num_hidden_layers!=36):
        raise ValueError("OFFICIAL_QWEN3_4B_ARCHITECTURE_REQUIRED")
    tokenizer=AutoTokenizer.from_pretrained(
        MODEL_REPO,revision=MODEL_REVISION,trust_remote_code=False)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    trainrows=_tokenized(train,tokenizer,max_tokens=512)
    validation={ed:_tokenized(val[ed],tokenizer,max_tokens=512) for ed in ED}
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,revision=MODEL_REVISION,trust_remote_code=False,
        use_safetensors=True,quantization_config=quant,
        dtype=dtype,device_map={"":0})
    model=prepare_model_for_kbit_training(
        model,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False})
    new_output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fitted,metrics=fit_lora(
        model,trainrows,validation,
        workdir=Path("/tmp/.williams-qwen4b-public-fit"),
        pad_token_id=tokenizer.pad_token_id,max_steps=steps,
        plan=LoraPlan(rank=rank,learning_rate=8e-5,
                      gradient_accumulation=2,seed=20261027),
        use_cpu=False)
    if not 0<metrics["optimizer_steps"]<=steps or not metrics["adapter_weights_changed"]:
        raise RuntimeError("QWEN3_4B_LORA_OPTIMIZER_DID_NOT_REALLY_UPDATE_WEIGHTS")
    if not metrics["validation_gate"]["accepted"]:
        raise RuntimeError("QWEN3_4B_ADAPTER_REJECTED_BY_FOUR_BOOK_SOURCE_VALIDATION")
    new_output.mkdir(mode=0o700,exist_ok=False)
    adapter=new_output/"adapter"
    fitted.save_pretrained(adapter,safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    weight=adapter/"adapter_model.safetensors"
    if not weight.is_file():raise RuntimeError("PEFT_ADAPTER_NOT_SAVED")
    report={
        "mode":"REAL_QWEN3_4B_NVIDIA_GPU_NF4_PUBLIC_PRACTICE_RESEARCH_ONLY",
        "trained_official_pretrained_qwen3_4b":True,
        "model_repo":MODEL_REPO,"model_revision":MODEL_REVISION,
        "official_upstream_weight_sha256":upstream_weights,
        "source_commit":os.environ.get("TRAINING_SOURCE_SHA"),
        "torch_cuda_device":torch.cuda.get_device_name(0),
        "cuda_visible_devices":torch.cuda.device_count(),
        "torch_version":torch.__version__,
        "public_four_book_curriculum":provenance,
        "steps":metrics["optimizer_steps"],
        "requested_maximum_optimizer_steps":steps,
        "rank":rank,
        "trainable_parameters":metrics["trainable_parameters"],
        "adapter_weights_changed":True,
        "adapter_model_sha256":digest(weight),
        "baseline_by_book":metrics["baseline_by_book"],
        "adapted_by_book":metrics["adapted_by_book"],
        "validation_gate":metrics["validation_gate"],
        "public_small_validation_only":True,
        "full_original_four_books_trained":False,
        "independent_expert_chart_validation":False,
        "blind_unseen_trading_skill_proven":False,
        "real_market_orders_performed":False,
        "not_android_deployable_without_conversion":True,
        "execution_allowed":False,"trade_action":None,
    }
    (new_output/"research_receipt.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),
        encoding="utf-8")
    print("ACTUAL_PRETRAINED_QWEN3_4B_GPU_OPTIMIZER_STEPS_COMPLETED",steps,flush=True)
    print("PUBLIC_FOUR_BOOK_ADAPTER_WEIGHTS_SHA256",report["adapter_model_sha256"],flush=True)
    print("PUBLIC_BOOK_VALIDATION_GATE",report["validation_gate"],flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description="Qwen3-4B actual 4-bit PEFT GPU study, public knowledge cards only")
    p.add_argument("--new-output",required=True,type=Path)
    p.add_argument("--steps",type=int,default=32)
    p.add_argument("--rank",type=int,default=8)
    a=p.parse_args()
    train_public_qlora(a.new_output,steps=a.steps,rank=a.rank)


if __name__=="__main__":main()
