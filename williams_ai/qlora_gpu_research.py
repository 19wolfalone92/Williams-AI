"""Optional REAL CUDA NF4 QLoRA experiment with resumable private checkpoints.

Not an Android runtime, not auto-training, no orders or credential access.
Requires operator-supplied permitted training data and unquantized HF Qwen.
Unsafe PyTorch optimizer state must NEVER come from an untrusted third party.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

ED=("TC1","NTD","TC2","COURSE")


def digest_file(file:Path)->str:
    if file.is_symlink() or not file.is_file():
        raise ValueError("REGULAR_CHECKPOINT_FILE_REQUIRED")
    h=hashlib.sha256()
    with file.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def contract(*,dataset_sha:str,base_sha:str,revision:str,max_steps:int,
             max_tokens:int,rank:int,rate:float,seed:int)->dict[str,Any]:
    for h in (dataset_sha,base_sha):
        if not isinstance(h,str) or len(h)!=64 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError("FULL_SOURCE_AND_MODEL_SHA256_REQUIRED")
    if not isinstance(revision,str) or not revision.strip():
        raise ValueError("SOURCE_REVISION_REQUIRED")
    if type(max_steps) is not int or not 10<=max_steps<=10000 or max_steps%10:
        raise ValueError("MAX_STEPS_MUST_BE_MULTIPLE_OF_TEN")
    if type(max_tokens) is not int or not 128<=max_tokens<=4096:
        raise ValueError("BOUNDED_TOKENS_REQUIRED")
    if type(rank) is not int or rank not in (4,8,16,32):
        raise ValueError("BOUNDED_RANK_REQUIRED")
    if type(rate) is not float or not math.isfinite(rate) or not 1e-6<=rate<=5e-4:
        raise ValueError("BOUNDED_LEARNING_RATE_REQUIRED")
    if type(seed) is not int or not 0<=seed<2**31:
        raise ValueError("BOUNDED_SEED_REQUIRED")
    return {"schema":"WILLIAMS_QLORA_RESEARCH_CONTRACT_V1",
        "dataset_sha256":dataset_sha,"hf_base_sha256":base_sha,
        "source_revision":revision,"max_steps":max_steps,
        "max_tokens":max_tokens,"lora_rank":rank,"learning_rate":rate,
        "seed":seed,"quantization":"NF4_DOUBLE_FLOAT16",
        "gradient_accumulation":8,"evaluation_interval":10,
        "target_modules":["q_proj","k_proj","v_proj","o_proj"],
        "execution_allowed":False}


def _files(folder:Path)->dict[str,str]:
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("REGULAR_CHECKPOINT_DIRECTORY_REQUIRED")
    files={}
    for child in folder.iterdir():
        if child.is_symlink() or not child.is_file():
            raise ValueError("NESTED_OR_LINKED_CHECKPOINT_CONTENT")
        files[child.name]=digest_file(child)
    return files


def _step(checkpoint:Path)->int:
    if not checkpoint.name.startswith("checkpoint-") or not checkpoint.name[11:].isdigit():
        raise ValueError("INVALID_CHECKPOINT_NAME")
    step=int(checkpoint.name[11:])
    if not 0<step<=10000:
        raise ValueError("INVALID_CHECKPOINT_STEP")
    state=checkpoint/"trainer_state.json"
    if state.is_symlink() or not state.is_file():
        raise ValueError("MISSING_TRAINER_STATE")
    parsed=json.loads(state.read_text(encoding="utf-8"))
    if type(parsed.get("global_step")) is not int or parsed["global_step"]!=step:
        raise ValueError("CHECKPOINT_STEP_MISMATCH")
    return step


def seal_checkpoints(output:Path)->dict:
    """Atomic receipt after a full trainer checkpoint has been written."""
    croot=output/"checkpoints"
    if not croot.is_dir() or croot.is_symlink():
        raise ValueError("CHECKPOINT_OUTPUT_REQUIRED")
    seals={}
    for cp in sorted(croot.glob("checkpoint-*")):
        if not all((cp/name).is_file() and not (cp/name).is_symlink() for name
                   in ("optimizer.pt","scheduler.pt","rng_state.pth","trainer_state.json")):
            continue # incomplete saves cannot become approved receipts
        _step(cp)
        seals[cp.name]=_files(cp)
    if not seals:
        raise ValueError("NO_COMPLETE_RESUMABLE_CHECKPOINT")
    manifest=output/"checkpoint_sha256.json"
    if manifest.is_symlink(): raise ValueError("UNSAFE_CHECKPOINT_MANIFEST")
    tmp=output/"checkpoint_sha256.json.new"
    if tmp.exists() or tmp.is_symlink():
        raise ValueError("STALE_TEMPORARY_MANIFEST")
    try:
        with tmp.open("x",encoding="utf-8") as out:
            out.write(json.dumps(seals,sort_keys=True,indent=2))
            out.flush();os.fsync(out.fileno())
        os.replace(tmp,manifest)
    finally:
        if tmp.exists(): tmp.unlink()
    return seals


def verify_resume(checkpoint:Path,receipt:Path,wanted:dict)->int:
    """Verify bytes/contract, not sender authenticity or pickle safety."""
    root=receipt.parent
    if receipt.is_symlink() or not receipt.is_file() or root.is_symlink():
        raise ValueError("TRUSTED_RESUME_RECEIPT_REQUIRED")
    data=json.loads(receipt.read_text(encoding="utf-8"))
    if not isinstance(data,dict) or data.get("contract")!=wanted:
        raise ValueError("INCOMPATIBLE_RESUME_CONTRACT")
    manifest=root/"checkpoint_sha256.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("CHECKPOINT_SEAL_REQUIRED")
    known=json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(known,dict) or not isinstance(known.get(checkpoint.name),dict):
        raise ValueError("CHECKPOINT_NOT_IN_MANIFEST")
    files=_files(checkpoint)
    if files!=known[checkpoint.name]:
        raise ValueError("CHANGED_CHECKPOINT_BYTES")
    for name in ("optimizer.pt","scheduler.pt","rng_state.pth","trainer_state.json"):
        if name not in files: raise ValueError("FULL_OPTIMIZER_STATE_REQUIRED")
    step=_step(checkpoint)
    if step>=wanted["max_steps"]:
        raise ValueError("CHECKPOINT_ALREADY_COMPLETE")
    return step


def _dataset(rows,token_id:int):
    import torch
    class Dataset:
        def __init__(self,records):self.records=records
        def __len__(self):return len(self.records)
        def __getitem__(self,i):return self.records[i]
    def collate(batch):
        length=max(len(r["input_ids"]) for r in batch)
        def vector(k,pad):
            return torch.tensor([r[k]+[pad]*(length-len(r[k])) for r in batch],
                                dtype=torch.long)
        return {"input_ids":vector("input_ids",token_id),
                "attention_mask":vector("attention_mask",0),
                "labels":vector("labels",-100)}
    return Dataset(rows),collate


def run_gpu(*,data:Path,model_dir:Path,output:Path,source_revision:str,
            total_steps:int=200,session_steps:int=10,
            max_tokens:int=1024,rank:int=8,learning_rate:float=5e-5,
            seed:int=42,source_only_research:bool=False,
            resume_checkpoint:Path|None=None,resume_receipt:Path|None=None,
            trusted_own_resume:bool=False,cloud_data_authorized:bool=False)->dict:
    if not cloud_data_authorized:
        raise ValueError("EXPLICIT_PRIVATE_DATA_CLOUD_PERMISSION_REQUIRED")
    if (type(session_steps) is not int or not 10<=session_steps<=500 or session_steps%10):
        raise ValueError("SESSION_STEPS_MUST_BE_MULTIPLE_OF_TEN")
    if bool(resume_checkpoint)!=bool(resume_receipt):
        raise ValueError("COMPLETE_RESUME_PATHS_REQUIRED")
    if resume_checkpoint and not trusted_own_resume:
        raise ValueError("UNTRUSTED_OPTIMIZER_PICKLE_RESUME_FORBIDDEN")

    # Existing rigorous four-book review and market supervision, not generated Qwen guesses.
    from .private_lora import _private_file,checked_private_training_data,_tokenized
    from .lora_integrity import local_hf_identity,validation_admission
    from .chart_supervision import require_market_supervision
    reviewed=checked_private_training_data(_private_file(data))
    if not source_only_research:
        require_market_supervision(reviewed)
    hf=_private_file(model_dir,must_exist=False)
    dst=_private_file(output,must_exist=False)
    if not hf.is_dir() or hf.is_symlink() or dst.exists() or hf in dst.parents or dst in hf.parents:
        raise ValueError("LOCAL_HF_MODEL_AND_NEW_SEPARATE_OUTPUT_REQUIRED")
    identity=local_hf_identity(hf)
    spec=contract(dataset_sha=reviewed["source_dataset_sha256"],base_sha=identity["sha256"],
                  revision=source_revision,max_steps=total_steps,max_tokens=max_tokens,
                  rank=rank,rate=float(learning_rate),seed=seed)
    done=verify_resume(resume_checkpoint,resume_receipt,spec) if resume_checkpoint else 0
    if done+10>total_steps:
        raise ValueError("INSUFFICIENT_STEPS_REMAINING")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("REAL_NVIDIA_CUDA_GPU_REQUIRED")
    from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training
    from transformers import (AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,
                              Trainer,TrainerCallback,TrainingArguments,set_seed)
    set_seed(seed)
    tok=AutoTokenizer.from_pretrained(str(hf),local_files_only=True,trust_remote_code=False)
    if tok.pad_token is None:
        if tok.eos_token is None: raise ValueError("QWEN_EOS_REQUIRED")
        tok.pad_token=tok.eos_token
    train=_tokenized(reviewed["items_for_local_fit"]["train"],tok,max_tokens=max_tokens)
    validate={ed:_tokenized([r for r in reviewed["items_for_local_fit"]["validation"]
                               if r["edition"]==ed],tok,max_tokens=max_tokens) for ed in ED}
    if any(not v for v in validate.values()):raise ValueError("FOUR_BOOK_VALIDATION_REQUIRED")
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    base=AutoModelForCausalLM.from_pretrained(str(hf),local_files_only=True,
            trust_remote_code=False,use_safetensors=True,quantization_config=quant,
            device_map={"":0},torch_dtype=torch.float16)
    if local_hf_identity(hf)["sha256"]!=identity["sha256"]:
        raise ValueError("BASE_MODEL_CHANGED_WHILE_LOADING")
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
    model=get_peft_model(base,LoraConfig(r=rank,lora_alpha=rank*2,lora_dropout=.05,
        target_modules=spec["target_modules"],task_type=TaskType.CAUSAL_LM,bias="none"))
    learnable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not learnable or any(".lora_" not in n for n,_ in learnable):
        raise ValueError("ONLY_LORA_MATRICES_MAY_TRAIN")
    model.config.use_cache=False

    dst.mkdir(parents=True,mode=0o700,exist_ok=False)
    (dst/"contract.json").write_text(json.dumps({"contract":spec},indent=2),
                                     encoding="utf-8")
    all_valid=[row for ed in ED for row in validate[ed]]
    ds,collate=_dataset(train,tok.pad_token_id)
    ev,_=_dataset(all_valid,tok.pad_token_id)
    ceiling=min(total_steps,done+session_steps)
    class StopAtSessionBoundary(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step>=ceiling:
                control.should_save=True
                control.should_training_stop=True
            return control
        def on_save(self,args,state,control,**kwargs):
            seal_checkpoints(dst)
            return control
    opts=TrainingArguments(output_dir=str(dst/"checkpoints"),max_steps=total_steps,
        per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,learning_rate=learning_rate,
        lr_scheduler_type="cosine",warmup_ratio=.05,max_grad_norm=1.0,
        eval_strategy="steps",eval_steps=10,save_strategy="steps",save_steps=10,
        save_total_limit=2,save_only_model=False,save_safetensors=True,
        load_best_model_at_end=False,remove_unused_columns=False,label_names=["labels"],
        logging_steps=10,report_to="none",disable_tqdm=True,
        fp16=True,bf16=False,gradient_checkpointing=False,
        dataloader_pin_memory=True,seed=seed,data_seed=seed)
    coach=Trainer(model=model,args=opts,train_dataset=ds,
                  eval_dataset=ev,data_collator=collate,callbacks=[StopAtSessionBoundary()])
    def scores():
        result={"ALL":float(coach.evaluate()["eval_loss"])}
        for ed in ED:
            d,_=_dataset(validate[ed],tok.pad_token_id)
            result[ed]=float(coach.evaluate(eval_dataset=d)["eval_loss"])
        validation_admission(result,result) # Reject NaN/Inf even baseline
        return result
    # Only the initial session's before/after can be compared here.
    before=scores() if not done else None
    coach.train(resume_from_checkpoint=str(resume_checkpoint) if done else None)
    after=scores()
    steps=coach.state.global_step
    if steps<=done or steps>ceiling or any(not torch.isfinite(p).all().item()
                                            for _,p in learnable):
        raise ValueError("INVALID_ACTUAL_OPTIMIZER_STEPS_OR_WEIGHTS")
    seal_checkpoints(dst)
    result={"mode":"RESEARCH_QWEN_QLORA_NOT_TRADE_QUALIFIED",
        "new_optimizer_steps":steps-done,"total_optimizer_steps":steps,
        "source_only_research":source_only_research,
        "initial_session_baseline":before,"final_validation":after,
        "initial_session_gate":validation_admission(before,after) if before else None,
        "source_contract":spec,"full_optimizer_checkpoint_saved":True,
        "independent_test_holdout_used_for_selection":False,
        "independent_expert_trading_quality_proven":False,
        "execution_allowed":False,"trade_action":None}
    (dst/"research_result.json").write_text(json.dumps(result,ensure_ascii=False,
                                            indent=2,allow_nan=False),encoding="utf-8")
    return result


def main()->None:
    p=argparse.ArgumentParser(description="WILLIAMS-AI private NF4 CUDA research, never trading")
    p.add_argument("--reviewed-jsonl",type=Path,required=True)
    p.add_argument("--hf-base",type=Path,required=True)
    p.add_argument("--new-output",type=Path,required=True)
    p.add_argument("--source-revision",required=True)
    p.add_argument("--max-steps",type=int,default=200)
    p.add_argument("--session-steps",type=int,default=10)
    p.add_argument("--max-tokens",type=int,default=1024)
    p.add_argument("--rank",type=int,default=8)
    p.add_argument("--learning-rate",type=float,default=5e-5)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--source-only-research",action="store_true")
    p.add_argument("--resume-checkpoint",type=Path)
    p.add_argument("--resume-receipt",type=Path)
    p.add_argument("--trusted-own-resume",action="store_true")
    p.add_argument("--authorize-cloud-data-processing",action="store_true")
    a=p.parse_args()
    print(json.dumps(run_gpu(data=a.reviewed_jsonl,model_dir=a.hf_base,
        output=a.new_output,source_revision=a.source_revision,
        total_steps=a.max_steps,session_steps=a.session_steps,
        max_tokens=a.max_tokens,rank=a.rank,learning_rate=a.learning_rate,
        seed=a.seed,source_only_research=a.source_only_research,
        resume_checkpoint=a.resume_checkpoint,resume_receipt=a.resume_receipt,
        trusted_own_resume=a.trusted_own_resume,
        cloud_data_authorized=a.authorize_cloud_data_processing),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
