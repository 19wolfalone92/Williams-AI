"""Real, bounded, private CUDA Qwen3-4B objective practice smoke.

No V6 unreviewed candidate, original book PDF, market chart or final-exam
label enters training. Reuse the existing resumable QLoRA Trainer. A 10+10
step interrupted run is technical QA and a narrow development experiment;
it is NOT independent qualification or the A/B corrective experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from williams_ai.v6_objective_practice import SYSTEM,curriculum
from williams_ai.knowledge_training_v5 import BASE,REV
from williams_ai.qlora_gpu_research import (
    _fit_resumable_adapter,_runtime_identity,contract,latest_checkpoint,digest_file,
)


def messages(row):
    return [{"role":"system","content":SYSTEM},
            {"role":"user","content":row["edition"]+": "+row["question"]}]


def encode(rows,tokenizer):
    if not rows or any(row["split"] not in ("train","validation") for row in rows):
        raise ValueError("NO_DEVELOPMENT_LABELS_IN_OPTIMIZER_OR_VALIDATION")
    out=[]
    for row in rows:
        msgs=messages(row)
        prefix=tokenizer.apply_chat_template(msgs,tokenize=True,
            add_generation_prompt=True,enable_thinking=False)
        complete=tokenizer.apply_chat_template(msgs+[{"role":"assistant","content":row["answer"]}],
            tokenize=True,add_generation_prompt=False,enable_thinking=False)
        if (not isinstance(prefix,list) or not isinstance(complete,list) or
            not prefix or not len(prefix)<len(complete)<=512 or complete[:len(prefix)]!=prefix):
            raise ValueError("STRICT_QWEN_ANSWER_ONLY_NO_TRUNCATION")
        out.append({"input_ids":complete,"attention_mask":[1]*len(complete),
                    "labels":[-100]*len(prefix)+complete[len(prefix):]})
    return out


def grade(raw,answer,truncated=False):
    if truncated:return "TECHNICAL_TRUNCATION"
    try:
        def unique(pairs):
            result={}
            for k,v in pairs:
                if k in result:raise ValueError("DUPLICATE_JSON_KEY")
                result[k]=v
            return result
        candidate=json.loads(raw,object_pairs_hook=unique)
        expected=json.loads(answer)
        if not isinstance(candidate,dict) or set(candidate)!={"answer"}:
            return "INVALID_FORMAT"
        a=candidate["answer"];b=expected["answer"]
        if type(a) is not type(b):return "INVALID_FORMAT"
        return "CORRECT" if a==b else "INCORRECT"
    except (TypeError,ValueError):
        return "INVALID_FORMAT"


def score(model,tok,rows):
    import torch
    model.eval()
    records=[]
    for row in rows:
        if row["split"]!="development":
            raise ValueError("ONLY_SEPARATE_DEVELOPMENT_CASES_FOR_GENERATION")
        ids=tok.apply_chat_template(messages(row),tokenize=True,add_generation_prompt=True,
            enable_thinking=False,return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            prediction=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                max_new_tokens=64,do_sample=False,use_cache=True,
                pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        suffix=prediction[0,ids.shape[1]:]
        raw=tok.decode(suffix,skip_special_tokens=True)
        truncated=len(suffix)>=64 and int(suffix[-1])!=tok.eos_token_id
        records.append({"id":row["case_id"],"family":row["family"],"edition":row["edition"],
                        "grade":grade(raw,row["answer"],truncated),"raw":raw})
    return records


def aggregate(records):
    return {"total":len(records),"counts":dict(Counter(r["grade"] for r in records)),
            "by_family":{f:dict(Counter(r["grade"] for r in records if r["family"]==f))
                         for f in sorted({r["family"] for r in records})}}


def run(output:Path,source_sha:str):
    if (output.exists() or output.is_symlink() or not re.fullmatch(r"[0-9a-f]{40}",source_sha)):
        raise ValueError("EXACT_NEW_PRIVATE_RUN_REQUIRED")
    source=Path(__file__).resolve().parents[1]
    actual=subprocess.run(["git","rev-parse","HEAD"],cwd=source,check=True,
                          capture_output=True,text=True).stdout.strip()
    if actual!=source_sha or subprocess.run(["git","diff","--quiet","HEAD","--",
            "williams_ai","scripts","knowledge"],cwd=source).returncode:
        raise ValueError("PINNED_CLEAN_SOURCE_REQUIRED")
    import torch
    from huggingface_hub import HfApi,snapshot_download
    from transformers import (AutoConfig,AutoModelForCausalLM,AutoTokenizer,
                              BitsAndBytesConfig,set_seed)
    from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training,PeftModel
    from williams_ai.lora_integrity import local_hf_identity
    from safetensors.torch import load_file
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("EXACTLY_ONE_REAL_CUDA_DEVICE_REQUIRED")
    torch.cuda.set_device(0)
    rows,manifest=curriculum()
    official=HfApi().model_info(BASE,revision=REV,files_metadata=True)
    if official.sha!=REV:raise ValueError("OFFICIAL_PINNED_QWEN_REVISION_REQUIRED")
    hashes={f.rfilename:f.lfs.sha256 for f in official.siblings
            if f.rfilename.endswith(".safetensors") and f.lfs}
    if not hashes:raise ValueError("OFFICIAL_WEIGHT_HASHES_REQUIRED")
    # Keep the 8GB public base outside the private kernel output tree.
    root=Path(snapshot_download(BASE,revision=REV,local_dir="/kaggle/temp/williams-v6-qwen3-4b",
        allow_patterns=["*.safetensors","*.safetensors.index.json","config.json",
            "generation_config.json","tokenizer*","vocab.json","merges.txt",
            "special_tokens_map.json","added_tokens.json","chat_template.jinja"]))
    identity=local_hf_identity(root)
    actual_weights={k:v for k,v in identity["files"].items() if k.endswith(".safetensors")}
    if actual_weights!=hashes:raise ValueError("ACTUAL_BASE_SHARD_HASH_MISMATCH")
    cfg=AutoConfig.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    if (cfg.model_type,cfg.hidden_size,cfg.num_hidden_layers)!=("qwen3",2560,36):
        raise ValueError("WRONG_QWEN3_4B")
    tok=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    if tok.eos_token_id is None:raise ValueError("QWEN_EOS_REQUIRED")
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    train=encode([r for r in rows if r["split"]=="train"],tok)
    validation={ed:encode([r for r in rows if r["split"]=="validation" and r["edition"]==ed],tok)
                for ed in ("TC1","NTD","TC2","COURSE")}
    development=[r for r in rows if r["split"]=="development"]
    spec=contract(dataset_sha=manifest["dataset_sha256"],base_sha=identity["sha256"],
        revision=source_sha,max_steps=20,max_tokens=512,rank=8,rate=2e-5,seed=20261124,
        runtime=_runtime_identity(),source_only_research=True)
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    def base():
        set_seed(spec["seed"])
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        model=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,
            trust_remote_code=False,use_safetensors=True,quantization_config=quant,
            dtype=torch.float16,device_map={"":0})
        return prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    def adapter():
        model=get_peft_model(base(),LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
            target_modules=spec["target_modules"],task_type=TaskType.CAUSAL_LM,bias="none"))
        model.config.use_cache=False
        return model
    model=base()
    baseline=score(model,tok,development)
    print("V6_OBJECTIVE_BASELINE",json.dumps(aggregate(baseline),sort_keys=True),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    model=adapter()
    first=_fit_resumable_adapter(model,train,validation,output=output/"first",spec=spec,
        pad_token_id=tok.pad_token_id,session_steps=10)
    checkpoint,receipt=latest_checkpoint(output/"first")
    if first["new_optimizer_steps"]!=10 or checkpoint.name!="checkpoint-10":
        raise ValueError("FIRST_REAL_TEN_STEPS_REQUIRED")
    print("V6_OBJECTIVE_FIRST_CHECKPOINT",first["new_optimizer_steps"],flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    model=adapter()
    second=_fit_resumable_adapter(model,train,validation,output=output/"second",spec=spec,
        pad_token_id=tok.pad_token_id,session_steps=10,
        resume_checkpoint=checkpoint,resume_receipt=receipt)
    if second["new_optimizer_steps"]!=10 or second["total_optimizer_steps"]!=20:
        raise ValueError("EXACT_RESUMED_TEN_STEPS_REQUIRED")
    if second["original_baseline"]!=first["original_baseline"]:
        raise ValueError("BASELINE_CHANGED_ON_RESUME")
    print("V6_OBJECTIVE_RESUMED_CHECKPOINT",second["total_optimizer_steps"],flush=True)
    target=output/"second"/"latest_adapter"
    weight=target/"adapter_model.safetensors"
    matrices=load_file(str(weight),device="cpu")
    if not matrices or any(not torch.isfinite(t).all().item() for t in matrices.values()):
        raise ValueError("NONFINITE_OR_MISSING_SAVED_LORA")
    # Verify actual reloaded adapter, not merely the in-memory fitted object.
    model.eval()
    test_ids=tok.apply_chat_template(messages(development[0]),tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        expected_logits=model(test_ids,use_cache=False).logits[:,-1,:].float().cpu()
    del model;gc.collect();torch.cuda.empty_cache()
    reloaded=PeftModel.from_pretrained(base(),target,local_files_only=True).eval()
    with torch.inference_mode():
        logits=reloaded(test_ids,use_cache=False).logits[:,-1,:].float().cpu()
    max_difference=float((expected_logits-logits).abs().max())
    if not torch.isfinite(logits).all() or not torch.allclose(logits,expected_logits,atol=.003,rtol=.003):
        raise ValueError("RELOADED_ADAPTER_LOGITS_MISMATCH")
    after=score(reloaded,tok,development)
    if local_hf_identity(root)!=identity:raise ValueError("PINNED_BASE_BYTES_CHANGED")
    result={"schema":"WILLIAMS_V6_REAL_OBJECTIVE_CUDA_SMOKE_V1","source_commit":source_sha,
        "base_model":BASE,"base_revision":REV,"base_identity_sha256":identity["sha256"],
        "official_weights_sha256":hashes,"dataset":manifest,"gpu":torch.cuda.get_device_name(0),
        "runtime":spec["runtime"],"new_optimizer_steps":20,"session_steps":[10,10],
        "optimizer_skips":first["optimizer_skips_in_this_session"]+second["optimizer_skips_in_this_session"],
        "first_checkpoint_sha256":digest_file(checkpoint/"adapter_model.safetensors"),
        "adapter_sha256":digest_file(weight),"tensor_count":len(matrices),"all_finite":True,
        "full_optimizer_state_resume":True,"reloaded_adapter":True,
        "max_reload_logit_difference":max_difference,"baseline":aggregate(baseline),
        "adapted":aggregate(after),"first_validation":first["final_validation"],
        "second_validation":second["final_validation"],"independent_final_exam":False,
        "williams_expert_qualified":False,"execution_allowed":False}
    (output/"private_predictions.json").write_text(json.dumps({"before":baseline,"after":after},ensure_ascii=False))
    (output/"aggregate.json").write_text(json.dumps(result,sort_keys=True,allow_nan=False,indent=2))
    print("V6_OBJECTIVE_CUDA_SMOKE_RESULT",json.dumps(result,sort_keys=True),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--source-sha",required=True)
    a=p.parse_args()
    run(a.output,a.source_sha)
