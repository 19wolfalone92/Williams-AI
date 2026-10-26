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
import re
import tempfile
from typing import Any

from .lora_integrity import strict_json, validation_admission

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
             max_tokens:int,rank:int,rate:float,seed:int,
             runtime:dict[str,str]|None=None,source_only_research:bool=False,
             synthetic_cpu_diagnostic:bool=False)->dict[str,Any]:
    for h in (dataset_sha,base_sha):
        if not isinstance(h,str) or len(h)!=64 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError("FULL_SOURCE_AND_MODEL_SHA256_REQUIRED")
    if not isinstance(revision,str) or not re.fullmatch(r"[0-9a-f]{40}",revision):
        raise ValueError("EXACT_SOURCE_REVISION_REQUIRED")
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
    if (type(source_only_research) is not bool or type(synthetic_cpu_diagnostic) is not bool
        or not isinstance(runtime,dict) or not runtime
        or any(not isinstance(k,str) or not isinstance(v,str) or not k or not v
               for k,v in runtime.items())):
        raise ValueError("EXPLICIT_RUNTIME_AND_SUPERVISION_MODE_REQUIRED")
    return {"schema":"WILLIAMS_QLORA_RESEARCH_CONTRACT_V2",
        "dataset_sha256":dataset_sha,"hf_base_sha256":base_sha,
        "source_revision":revision,"max_steps":max_steps,
        "max_tokens":max_tokens,"lora_rank":rank,"learning_rate":rate,
        "seed":seed,"quantization":("SYNTHETIC_CPU_FP32" if synthetic_cpu_diagnostic
                                    else "NF4_DOUBLE_FLOAT16"),
        "runtime":dict(sorted(runtime.items())),"source_only_research":source_only_research,
        "gradient_accumulation":8,"evaluation_interval":10,
        "batch_size":1,"optimizer":"adamw_torch","weight_decay":0.01,
        "adam_beta1":0.9,"adam_beta2":0.999,"adam_epsilon":1e-8,
        "lr_scheduler":"cosine","warmup_ratio":0.05,"max_grad_norm":1.0,
        "lora_alpha":2*rank,"lora_dropout":0.05,
        "gradient_checkpointing":"NON_REENTRANT",
        "shuffle":"GLOBAL_STEP_EPOCH_SEEDED_V1",
        "tokenization":"ANSWER_ONLY_NO_TRUNCATION_V1",
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


def _json_file(path:Path,limit:int=8_000_000)->Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size>limit:
        raise ValueError("BOUNDED_REGULAR_RECEIPT_REQUIRED")
    return strict_json(path.read_bytes())


def _atomic_json(path:Path,value:Any)->None:
    """A killed writer's temporary file must not block the next checkpoint."""
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("UNSAFE_CHECKPOINT_MANIFEST")
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,
                prefix="."+path.name+"-",suffix=".tmp",delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


def _receipt(path:Path)->dict:
    data=_json_file(path,1_000_000)
    if (not isinstance(data,dict) or set(data)!={"contract","original_baseline"}
        or not isinstance(data["contract"],dict)
        or data["contract"].get("schema")!="WILLIAMS_QLORA_RESEARCH_CONTRACT_V2"):
        raise ValueError("V2_BASELINE_AND_CONTRACT_RECEIPT_REQUIRED")
    validation_admission(data["original_baseline"],data["original_baseline"])
    return data


def _required_files(spec:dict)->set[str]:
    required={"adapter_model.safetensors","adapter_config.json","training_args.bin",
              "optimizer.pt","scheduler.pt","rng_state.pth","trainer_state.json"}
    if spec["quantization"]=="NF4_DOUBLE_FLOAT16":required.add("scaler.pt")
    return required


def _step(checkpoint:Path,spec:dict)->int:
    if not checkpoint.name.startswith("checkpoint-") or not checkpoint.name[11:].isdigit():
        raise ValueError("INVALID_CHECKPOINT_NAME")
    step=int(checkpoint.name[11:])
    if not 0<step<=10000:
        raise ValueError("INVALID_CHECKPOINT_STEP")
    state=checkpoint/"trainer_state.json"
    if state.is_symlink() or not state.is_file():
        raise ValueError("MISSING_TRAINER_STATE")
    parsed=_json_file(state)
    if (not isinstance(parsed,dict) or type(parsed.get("global_step")) is not int
        or parsed["global_step"]!=step or step%spec["evaluation_interval"]
        or type(parsed.get("max_steps")) is not int
        or parsed["max_steps"]!=spec["max_steps"] or step>spec["max_steps"]):
        raise ValueError("CHECKPOINT_STEP_MISMATCH")
    adapter=_json_file(checkpoint/"adapter_config.json",1_000_000)
    if (not isinstance(adapter,dict) or adapter.get("peft_type")!="LORA"
        or adapter.get("r")!=spec["lora_rank"]
        or adapter.get("lora_alpha")!=spec["lora_alpha"]
        or adapter.get("lora_dropout")!=spec["lora_dropout"]
        or not isinstance(adapter.get("target_modules"),list)
        or sorted(adapter["target_modules"])!=sorted(spec["target_modules"])):
        raise ValueError("CHECKPOINT_ADAPTER_CONTRACT_MISMATCH")
    return step


def seal_checkpoints(output:Path)->dict:
    """Atomic receipt after a full trainer checkpoint has been written."""
    croot=output/"checkpoints"
    if not croot.is_dir() or croot.is_symlink():
        raise ValueError("CHECKPOINT_OUTPUT_REQUIRED")
    receipt=_receipt(output/"contract.json")
    required=_required_files(receipt["contract"])
    seals={}
    for cp in sorted(croot.glob("checkpoint-*")):
        if cp.is_symlink() or not cp.is_dir():
            raise ValueError("REGULAR_CHECKPOINT_DIRECTORY_REQUIRED")
        if not all((cp/name).is_file() and not (cp/name).is_symlink() for name
                   in required):
            continue # incomplete saves cannot become approved receipts
        if any((cp/name).stat().st_size==0 for name in required):continue
        _step(cp,receipt["contract"])
        seals[cp.name]=_files(cp)
    if not seals:
        raise ValueError("NO_COMPLETE_RESUMABLE_CHECKPOINT")
    _atomic_json(output/"checkpoint_sha256.json",{
        "schema":"WILLIAMS_QLORA_CHECKPOINT_SEAL_V2",
        "receipt_sha256":digest_file(output/"contract.json"),"checkpoints":seals})
    return seals


def verify_resume(checkpoint:Path,receipt:Path,wanted:dict)->int:
    """Verify bytes/contract, not sender authenticity or pickle safety."""
    root=receipt.parent
    if receipt.is_symlink() or not receipt.is_file() or root.is_symlink():
        raise ValueError("TRUSTED_RESUME_RECEIPT_REQUIRED")
    data=_receipt(receipt)
    if data["contract"]!=wanted:
        raise ValueError("INCOMPATIBLE_RESUME_CONTRACT")
    if checkpoint.parent.resolve()!=root.resolve()/"checkpoints":
        raise ValueError("CHECKPOINT_MUST_BELONG_TO_RECEIPT")
    manifest=root/"checkpoint_sha256.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("CHECKPOINT_SEAL_REQUIRED")
    sealed=_json_file(manifest)
    if (not isinstance(sealed,dict)
        or sealed.get("schema")!="WILLIAMS_QLORA_CHECKPOINT_SEAL_V2"
        or sealed.get("receipt_sha256")!=digest_file(receipt)):
        raise ValueError("CHANGED_BASELINE_OR_CONTRACT_RECEIPT")
    known=sealed.get("checkpoints")
    if not isinstance(known,dict) or not isinstance(known.get(checkpoint.name),dict):
        raise ValueError("CHECKPOINT_NOT_IN_MANIFEST")
    files=_files(checkpoint)
    if files!=known[checkpoint.name]:
        raise ValueError("CHANGED_CHECKPOINT_BYTES")
    for name in _required_files(wanted):
        if name not in files: raise ValueError("FULL_OPTIMIZER_STATE_REQUIRED")
        if (checkpoint/name).stat().st_size==0:raise ValueError("EMPTY_CHECKPOINT_STATE")
    step=_step(checkpoint,wanted)
    if step>=wanted["max_steps"]:
        raise ValueError("CHECKPOINT_ALREADY_COMPLETE")
    return step


def latest_checkpoint(directory:Path)->tuple[Path,Path]:
    """Choose the latest SEALED checkpoint, ignoring unsealed partial writes.

    This verifies local integrity, never the trustworthiness of its author.
    A tampered sealed checkpoint raises instead of silently falling back.
    """
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("REGULAR_RESUME_DIRECTORY_REQUIRED")
    receipt=directory/"contract.json"
    data=_receipt(receipt)
    sealed=_json_file(directory/"checkpoint_sha256.json")
    known=sealed.get("checkpoints") if isinstance(sealed,dict) else None
    if not isinstance(known,dict) or not known or any(
        not isinstance(name,str) or not re.fullmatch(r"checkpoint-[1-9][0-9]*",name)
        for name in known):
        raise ValueError("NO_SEALED_RESUME_CHECKPOINT")
    name=max(known,key=lambda value:int(value[11:]))
    checkpoint=directory/"checkpoints"/name
    verify_resume(checkpoint,receipt,data["contract"])
    return checkpoint,receipt


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


def _runtime_identity(*,synthetic_cpu:bool=False)->dict[str,str]:
    import importlib.metadata
    import platform
    import torch
    packages=("torch","transformers","peft","accelerate","safetensors","numpy")
    if not synthetic_cpu:packages+=("bitsandbytes",)
    result={name:importlib.metadata.version(name) for name in packages}
    result["python"]=platform.python_version()
    result["device"]="CPU_SYNTHETIC_ONLY" if synthetic_cpu else torch.cuda.get_device_name(0)
    result["cuda"]=str(torch.version.cuda)
    if not synthetic_cpu:
        result["compute_capability"]=".".join(map(str,torch.cuda.get_device_capability(0)))
    return result


def _adapter_digest(parameters)->str:
    """Fingerprint trainable matrices only; no full base copy in RAM."""
    import torch
    digest=hashlib.sha256()
    for name,parameter in sorted(parameters.items()):
        digest.update(name.encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _fit_resumable_adapter(model,train,validation,*,output:Path,spec:dict,
        pad_token_id:int,session_steps:int,resume_checkpoint:Path|None=None,
        resume_receipt:Path|None=None,use_cpu:bool=False)->dict:
    """The actual shared Trainer path. CPU is only for synthetic software QA.

    Public reviewed-data entry remains run_gpu, which requires CUDA and audit.
    Last optimizer state is retained for continuation; no adapter is promoted.
    """
    import torch
    from transformers import Trainer,TrainerCallback,TrainingArguments
    if (type(use_cpu) is not bool or type(session_steps) is not int
        or not 10<=session_steps<=500 or session_steps%10
        or not train or set(validation)!=set(ED) or any(not rows for rows in validation.values())
        or output.exists() or output.is_symlink()
        or bool(resume_checkpoint)!=bool(resume_receipt)
        or (spec["quantization"]=="SYNTHETIC_CPU_FP32")!=use_cpu):
        raise ValueError("BOUNDED_NEW_RESUMABLE_SESSION_REQUIRED")
    if spec["runtime"]!=_runtime_identity(synthetic_cpu=use_cpu):
        raise ValueError("TRAINING_RUNTIME_CHANGED")
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if not parameters or any(".lora_" not in name for name in parameters):
        raise ValueError("ONLY_LORA_MATRICES_MAY_TRAIN")
    done=verify_resume(resume_checkpoint,resume_receipt,spec) if resume_checkpoint else 0
    if done and resume_receipt.parent.resolve() in output.resolve().parents:
        raise ValueError("NEW_OUTPUT_MUST_NOT_MODIFY_OLD_CHECKPOINT_TREE")
    ceiling=min(spec["max_steps"],done+session_steps)
    baseline=_receipt(resume_receipt)["original_baseline"] if done else None
    output.mkdir(parents=True,mode=0o700,exist_ok=False)
    dataset,collate=_dataset(train,pad_token_id)
    combined=[row for edition in ED for row in validation[edition]]
    evaluation,_=_dataset(combined,pad_token_id)
    measurements={}

    class SessionCheckpoint(TrainerCallback):
        def on_train_begin(self,args,state,control,**kwargs):
            # Trainer has loaded resumed adapter bytes before this event.
            measurements["before_updates"]=_adapter_digest(parameters)
            return control

        def on_pre_optimizer_step(self,args,state,control,**kwargs):
            gradients=[p.grad for p in parameters.values() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all().item() for g in gradients):
                raise ValueError("NONFINITE_OR_MISSING_LORA_GRADIENTS")
            return control

        def on_log(self,args,state,control,logs=None,**kwargs):
            if any(key in (logs or {}) and not math.isfinite(float(logs[key]))
                   for key in ("loss","grad_norm")):
                raise ValueError("NONFINITE_LORA_TRAINING_LOSS")
            return control

        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step>=ceiling:
                control.should_save=True
                control.should_training_stop=True
            return control

        def on_save(self,args,state,control,**kwargs):
            seal_checkpoints(output)
            return control

    options=TrainingArguments(output_dir=str(output/"checkpoints"),max_steps=spec["max_steps"],
        per_device_train_batch_size=spec["batch_size"],per_device_eval_batch_size=1,
        gradient_accumulation_steps=spec["gradient_accumulation"],
        learning_rate=spec["learning_rate"],optim=spec["optimizer"],
        weight_decay=spec["weight_decay"],adam_beta1=spec["adam_beta1"],
        adam_beta2=spec["adam_beta2"],adam_epsilon=spec["adam_epsilon"],
        lr_scheduler_type=spec["lr_scheduler"],warmup_ratio=spec["warmup_ratio"],
        max_grad_norm=spec["max_grad_norm"],eval_strategy="steps",eval_steps=10,
        save_strategy="steps",save_steps=10,save_total_limit=2,
        save_only_model=False,save_safetensors=True,load_best_model_at_end=False,
        remove_unused_columns=False,label_names=["labels"],logging_steps=10,
        logging_nan_inf_filter=False,report_to="none",disable_tqdm=True,
        fp16=not use_cpu,bf16=False,use_cpu=use_cpu,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
        dataloader_pin_memory=not use_cpu,seed=spec["seed"],data_seed=spec["seed"])

    class ResumeTrainer(Trainer):
        def _get_train_sampler(self,train_dataset=None):
            # Accelerate 1.10.1 skip_first_batches constructs a fresh shard
            # with iteration=0, resetting its seedable sampler mid-epoch.
            # Derive the epoch from restored optimizer progress instead; this
            # sampler is not RandomSampler and is not replaced by Accelerate.
            records=self.train_dataset if train_dataset is None else train_dataset
            trainer=self
            updates_per_epoch=math.ceil(len(records)/spec["gradient_accumulation"])
            class OptimizerEpochSampler(torch.utils.data.Sampler):
                def __len__(self):return len(records)
                def __iter__(self):
                    epoch=trainer.state.global_step//updates_per_epoch
                    generator=torch.Generator().manual_seed(spec["seed"]+epoch)
                    return iter(torch.randperm(len(records),generator=generator).tolist())
            return OptimizerEpochSampler()

    coach=ResumeTrainer(model=model,args=options,train_dataset=dataset,
        eval_dataset=evaluation,data_collator=collate,callbacks=[SessionCheckpoint()])

    def scores():
        result={"ALL":float(coach.evaluate()["eval_loss"])}
        for edition in ED:
            subset,_=_dataset(validation[edition],pad_token_id)
            result[edition]=float(coach.evaluate(eval_dataset=subset)["eval_loss"])
        validation_admission(result,result)
        return result

    if baseline is None:baseline=scores()
    # Persist BEFORE any optimizer step, so an interrupted session retains its
    # original four-book baseline, bound by the checkpoint's integrity seal.
    _atomic_json(output/"contract.json",{"contract":spec,"original_baseline":baseline})
    fitted=coach.train(resume_from_checkpoint=str(resume_checkpoint) if done else None)
    if not math.isfinite(float(fitted.training_loss)):
        raise ValueError("NONFINITE_LORA_TRAINING_LOSS")
    steps=coach.state.global_step
    if (type(steps) is not int or steps!=ceiling or steps<=done
        or any(not torch.isfinite(p).all().item() for p in parameters.values())):
        raise ValueError("INVALID_ACTUAL_OPTIMIZER_STEPS_OR_WEIGHTS")
    after_digest=_adapter_digest(parameters)
    if after_digest==measurements.get("before_updates"):
        raise ValueError("LORA_WEIGHTS_DID_NOT_CHANGE")
    after=scores()
    seals=seal_checkpoints(output)
    checkpoint_name="checkpoint-"+str(steps)
    if checkpoint_name not in seals:raise ValueError("FINAL_CHECKPOINT_NOT_SEALED")
    # Separate resumable latest state from admission. Validation loss alone
    # never activates a model or touches the frozen independent test cohort.
    model.save_pretrained(output/"latest_adapter",safe_serialization=True)
    result={"mode":("SYNTHETIC_CPU_SOFTWARE_DIAGNOSTIC" if use_cpu else
                    "RESEARCH_QWEN_QLORA_NOT_TRADE_QUALIFIED"),
        "new_optimizer_steps":steps-done,"total_optimizer_steps":steps,
        "source_only_research":spec["source_only_research"],
        "original_baseline":baseline,"final_validation":after,
        "baseline_comparison":validation_admission(baseline,after),
        "original_baseline_preserved_across_sessions":True,
        "adapter_weights_changed_in_this_session":True,"adapter_sha256":after_digest,
        "latest_adapter_files":_files(output/"latest_adapter"),
        "source_contract":spec,"full_optimizer_checkpoint_saved":True,
        "latest_sealed_checkpoint":checkpoint_name,"automatic_model_promotion":False,
        "independent_test_holdout_used_for_selection":False,
        "independent_expert_trading_quality_proven":False,
        "execution_allowed":False,"trade_action":None}
    _atomic_json(output/"research_result.json",result)
    return result


def run_gpu(*,data:Path,model_dir:Path,output:Path,source_revision:str,
            total_steps:int=200,session_steps:int=10,
            max_tokens:int=1024,rank:int=8,learning_rate:float=5e-5,
            seed:int=42,source_only_research:bool=False,
            resume_checkpoint:Path|None=None,resume_receipt:Path|None=None,
            trusted_own_resume:bool=False,cloud_data_authorized:bool=False,
            resume_directory:Path|None=None)->dict:
    if cloud_data_authorized is not True:
        raise ValueError("EXPLICIT_PRIVATE_DATA_CLOUD_PERMISSION_REQUIRED")
    if type(source_only_research) is not bool or type(trusted_own_resume) is not bool:
        raise ValueError("EXPLICIT_BOOLEAN_TRAINING_MODE_REQUIRED")
    if (type(session_steps) is not int or not 10<=session_steps<=500 or session_steps%10):
        raise ValueError("SESSION_STEPS_MUST_BE_MULTIPLE_OF_TEN")
    if resume_directory:
        if resume_checkpoint or resume_receipt:
            raise ValueError("CHOOSE_ONE_RESUME_PATH_MODE")
        if not trusted_own_resume:
            raise ValueError("UNTRUSTED_OPTIMIZER_PICKLE_RESUME_FORBIDDEN")
        resume_checkpoint,resume_receipt=latest_checkpoint(resume_directory)
    if bool(resume_checkpoint)!=bool(resume_receipt):
        raise ValueError("COMPLETE_RESUME_PATHS_REQUIRED")
    if resume_checkpoint and not trusted_own_resume:
        raise ValueError("UNTRUSTED_OPTIMIZER_PICKLE_RESUME_FORBIDDEN")

    # Existing rigorous four-book review and market supervision, not generated Qwen guesses.
    from .private_lora import _private_file,checked_private_training_data,_tokenized
    from .lora_integrity import local_hf_identity
    from .chart_supervision import require_market_supervision
    reviewed=checked_private_training_data(_private_file(data))
    if not source_only_research:
        require_market_supervision(reviewed)
    hf=_private_file(model_dir,must_exist=False)
    dst=_private_file(output,must_exist=False)
    if not hf.is_dir() or hf.is_symlink() or dst.exists() or hf in dst.parents or dst in hf.parents:
        raise ValueError("LOCAL_HF_MODEL_AND_NEW_SEPARATE_OUTPUT_REQUIRED")
    identity=local_hf_identity(hf)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("REAL_NVIDIA_CUDA_GPU_REQUIRED")
    if torch.cuda.device_count()!=1:
        raise ValueError("SINGLE_VISIBLE_GPU_REQUIRED_SET_CUDA_VISIBLE_DEVICES")
    spec=contract(dataset_sha=reviewed["source_dataset_sha256"],base_sha=identity["sha256"],
                  revision=source_revision,max_steps=total_steps,max_tokens=max_tokens,
                  rank=rank,rate=learning_rate,seed=seed,runtime=_runtime_identity(),
                  source_only_research=source_only_research)
    # Verify the supplied commit against the executed code, not a user label.
    import subprocess
    from .book_index import ROOT
    actual=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,
                          capture_output=True,text=True).stdout.strip()
    clean=subprocess.run(["git","diff","--quiet","HEAD","--","williams_ai"],cwd=ROOT)
    if actual!=source_revision or clean.returncode!=0:
        raise ValueError("SOURCE_CHECKOUT_MUST_MATCH_PINNED_REVISION")
    if resume_checkpoint:verify_resume(resume_checkpoint,resume_receipt,spec)
    from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
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
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
    model=get_peft_model(base,LoraConfig(r=rank,lora_alpha=rank*2,lora_dropout=.05,
        target_modules=spec["target_modules"],task_type=TaskType.CAUSAL_LM,bias="none"))
    model.config.use_cache=False
    return _fit_resumable_adapter(model,train,validate,output=dst,spec=spec,
        pad_token_id=tok.pad_token_id,session_steps=session_steps,
        resume_checkpoint=resume_checkpoint,resume_receipt=resume_receipt)


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
    p.add_argument("--resume-directory",type=Path)
    p.add_argument("--trusted-own-resume",action="store_true")
    p.add_argument("--authorize-cloud-data-processing",action="store_true")
    a=p.parse_args()
    print(json.dumps(run_gpu(data=a.reviewed_jsonl,model_dir=a.hf_base,
        output=a.new_output,source_revision=a.source_revision,
        total_steps=a.max_steps,session_steps=a.session_steps,
        max_tokens=a.max_tokens,rank=a.rank,learning_rate=a.learning_rate,
        seed=a.seed,source_only_research=a.source_only_research,
        resume_checkpoint=a.resume_checkpoint,resume_receipt=a.resume_receipt,
        resume_directory=a.resume_directory,
        trusted_own_resume=a.trusted_own_resume,
        cloud_data_authorized=a.authorize_cloud_data_processing),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
