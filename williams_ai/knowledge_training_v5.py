"""Actual Qwen3-4B NF4 QLoRA for a strictly rule-disjoint four-book curriculum.

NOT expert-labelled wave forecasting, live trading or private copyrighted PDF
training. Trains answer-only SFT on reviewed source CARD paraphrases.
The TEST cohort is not tokenized, used for checkpoint selection or passed to
any GPU fit function. Underlying Qwen tokenizer chat template is mandatory.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path

from .knowledge_curriculum_v5 import build,digest,EDITIONS,SCHEMA

BASE="Qwen/Qwen3-4B"
REV="3fb356690bbdae7353c94efb3eecdce276e6ed1c"

SYSTEM=("Ты исследователь книг Билла Вильямса. Не смешивай правила TC1, NTD, "
        "TC2 и Course. Используй только указанную проверенную карточку и "
        "объясняй ограниченность фактов. Ответ в заданном JSON, "
        "без длинных внутренних рассуждений. Нет котировок — нет сделки. "
        "Не утверждай экспертную проверку волн, стопов или ордеров.")

def tokenize(rows, tokenizer, *, max_tokens=1536):
    if not isinstance(rows,list) or not rows or type(max_tokens) is not int or not 512<=max_tokens<=2048:
        raise ValueError("BOUNDED_TRAINING_ROWS_AND_CONTEXT_REQUIRED")
    result=[]
    for row in rows:
        if row.get("schema")!=SCHEMA or row.get("split") not in {"train","validation"}:
            raise ValueError("NEVER_PASS_TEST_OR_UNVERIFIED_CASE_TO_GPU")
        if row["split"]=="train" and row["eligible_for_sft"] is not True:
            raise ValueError("SOURCE_PARAPHRASE_ELIGIBILITY_REQUIRED")
        if (row["chart_expert_gold"] is not False or
            row["visual_semantics_independently_verified"] is not False or
            row["execution_allowed"] is not False):
            raise ValueError("FALSE_EXPERT_MARKET_OR_VISUAL_GOLD")
        prompt=row["prompt"]
        msgs=[{"role":"system","content":SYSTEM},
              {"role":"user","content":json.dumps(prompt,ensure_ascii=False,
                                                  sort_keys=True)}]
        prefix=tokenizer.apply_chat_template(
            msgs,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        full=tokenizer.apply_chat_template(
            msgs+[{"role":"assistant","content":row["answer"]}],
            tokenize=True,add_generation_prompt=False,enable_thinking=False)
        if (not isinstance(prefix,list) or not isinstance(full,list) or
                not prefix or full[:len(prefix)]!=prefix or
                not len(prefix)<len(full)<=max_tokens):
            raise ValueError("QWEN_CHAT_TEMPLATE_ANSWER_ONLY_NO_TRUNCATION")
        if all(x==-100 for x in [-100]*len(prefix)+full[len(prefix):]):
            raise ValueError("TARGET_TOKENS_MISSING")
        result.append({"input_ids":full,"attention_mask":[1]*len(full),
                       "labels":[-100]*len(prefix)+full[len(prefix):]})
    return result


def train(*, output:Path,steps:int=120,rank:int=8,max_tokens:int=1536)->dict:
    if (type(steps) is not int or not 30<=steps<=240 or steps%10!=0 or
        type(rank) is not int or rank not in (8,16) or
        output.exists() or output.is_symlink()):
        raise ValueError("BOUNDED_NEW_GPU_TRAINING_RUN_REQUIRED")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("SINGLE_CUDA_DEVICE_REQUIRED_FOR_REPRODUCIBLE_SFT")
    from huggingface_hub import HfApi
    from transformers import (
        AutoConfig,AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig)
    from peft import prepare_model_for_kbit_training
    from .lora_training import fit_lora,LoraPlan
    from .lora_integrity import validation_admission

    cards,manifest=build()
    train_rows=[x for x in cards if x["split"]=="train"]
    dev={ed:[x for x in cards if x["split"]=="validation"
            and x["edition"]==ed] for ed in EDITIONS}
    if not all(dev.values()) or len(train_rows)<120:
        raise ValueError("RULE_DISJOINT_FOUR_BOOK_TRAIN_AND_DEV_REQUIRED")
    ids={key:{x["rule_id"] for x in cards if x["split"]==key}
         for key in ("train","validation","test")}
    if ids["train"]&ids["validation"] or ids["train"]&ids["test"] or ids["validation"]&ids["test"]:
        raise ValueError("NO_HOLDOUT_TRAIN_LEAKAGE")
    upstream=HfApi().model_info(BASE,revision=REV,files_metadata=True)
    if upstream.sha!=REV:
        raise ValueError("MUTABLE_QWEN_BASE_REJECTED")
    official={x.rfilename:x.lfs.sha256 for x in upstream.siblings
              if x.rfilename.endswith(".safetensors") and x.lfs is not None}
    if not official:
        raise ValueError("OFFICIAL_QWEN_HF_WEIGHTS_NOT_PINNED")
    cfg=AutoConfig.from_pretrained(BASE,revision=REV,trust_remote_code=False)
    if cfg.model_type!="qwen3" or cfg.hidden_size!=2560 or cfg.num_hidden_layers!=36:
        raise ValueError("WRONG_QWEN3_4B_ARCHITECTURE")
    tokenizer=AutoTokenizer.from_pretrained(BASE,revision=REV,trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    tokenized_train=tokenize(train_rows,tokenizer,max_tokens=max_tokens)
    tokenized_dev={ed:tokenize(dev[ed],tokenizer,max_tokens=max_tokens)
                   for ed in EDITIONS}
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    model=AutoModelForCausalLM.from_pretrained(
        BASE,revision=REV,trust_remote_code=False,use_safetensors=True,
        quantization_config=quant,dtype=dtype,device_map={"":0})
    model=prepare_model_for_kbit_training(
        model,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False})

    # HF/TRL guidance: LoRA requires distinct learning-rate tuning.
    # A conservative 5e-5 avoids aggressive memorization on a small corpus.
    fit,metrics=fit_lora(
        model,tokenized_train,tokenized_dev,
        workdir=output.parent/".williams-v5-qlora-checkpoints",
        pad_token_id=tokenizer.pad_token_id,max_steps=steps,
        plan=LoraPlan(rank=rank,learning_rate=5e-5,
                      gradient_accumulation=4,seed=20261111),
        use_cpu=False)
    if not metrics["adapter_weights_changed"] or metrics["optimizer_steps"]<=0:
        raise ValueError("NO_AUTHENTIC_LORA_WEIGHT_UPDATES")
    output.mkdir(parents=True,mode=0o700,exist_ok=False)
    adapter=output/"adapter"
    fit.save_pretrained(adapter,safe_serialization=True)
    weight=adapter/"adapter_model.safetensors"
    if weight.stat().st_size<1000000:
        raise ValueError("EMPTY_TRAINED_QWEN_ADAPTER")
    hash_value=sha256(weight.read_bytes()).hexdigest()
    receipt={
        "schema":"WILLIAMS_V5_GENUINE_QWEN3_4B_SOURCE_ONLY_LORA",
        "base_model":BASE,"model_commit":REV,
        "official_model_weights_sha256":official,
        "rule_registry_sha256":manifest["source_registry_sha256"],
        "dataset_split_prompt_sha256":manifest["split_prompt_sha256"],
        "dataset_split_gold_sha256":manifest["split_gold_sha256"],
        "train_case_count":len(train_rows),
        "validation_case_counts":{ed:len(dev[ed]) for ed in EDITIONS},
        "test_cases_provided_to_trainer":0,
        "test_gold_examined_during_checkpoint_selection":False,
        "training_type":"SOURCE_CARD_SFT_ONLY_NOT_WAVE_EXPERT_GOLD",
        "source_pdf_or_ocr_uploaded":False,
        "expert_gold_market_examples":0,
        "selected_validation_checkpoint":metrics["selected_checkpoint_step"],
        "optimizer_steps":metrics["optimizer_steps"],
        "validation_baseline_per_edition":metrics["baseline_by_book"],
        "validation_after_per_edition":metrics["adapted_by_book"],
        "validation_gate":metrics["validation_gate"],
        "weights_changed":True,
        "adapter_sha256":hash_value,
        "execution_allowed":False,
        "trading_qualification":False,
    }
    (output/"training_receipt.json").write_text(
        json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True))
    print("WILLIAMS_V5_REAL_QWEN3_4B_QLORA_TRAINED",
        json.dumps({"steps":metrics["optimizer_steps"],
                    "selected_checkpoint":metrics["selected_checkpoint_step"],
                    "train_cases":len(train_rows),
                    "adapter_sha256":hash_value,
                    "book_validation_loss":metrics["adapted_by_book"],
                    "validation_gate":metrics["validation_gate"],
                    "trading_qualification":False},sort_keys=True),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--new-output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=120)
    p.add_argument("--rank",type=int,default=8)
    p.add_argument("--max-tokens",type=int,default=1536)
    a=p.parse_args()
    train(output=a.new_output,steps=a.steps,rank=a.rank,max_tokens=a.max_tokens)


if __name__=="__main__":
    main()
