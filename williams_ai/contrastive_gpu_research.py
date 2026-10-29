"""Two-T4 joint source replay and contrastive rule learning; research only.

Rank0 is the sole artifact writer. Original test pages and sealed answer keys
never enter this process. A mutable stop flag supports a user-requested stop.
"""
from __future__ import annotations
from contextlib import nullcontext
from datetime import timedelta
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import shutil
import threading
import time
from urllib.request import Request,urlopen

from .contrastive_book_practice import (SYSTEM,RULES,paired_cases,quick_development_cases,
    question_rows,questions_digest,score_responses,tokenize_typed,weighted_causal_loss,strict_answer)
from .deep_book_research import EDITIONS,MODEL,REVISION,checked_payload

CONTROL_URL="https://raw.githubusercontent.com/19wolfalone92/Williams-AI/research/private-four-book-deep-training-20261027/experiments/contrastive-training-control.json"
RUN_SLUG="williams-ai-contrastive-2t4-20261027"


def checked_stop_control(data:dict)->bool:
    if (set(data)!={"run_slug","stop"} or data["run_slug"]!=RUN_SLUG or type(data["stop"]) is not bool):
        raise ValueError("EXPLICIT_OWN_TRAINING_STOP_CONTROL_REQUIRED")
    return data["stop"]


class StopWatcher:
    def __init__(self):self.requested=threading.Event();self.finished=threading.Event()
    def start(self):
        def watch():
            failed_since=None
            while not self.finished.is_set():
                try:
                    url=CONTROL_URL+"?probe="+str(int(time.time())//20)
                    with urlopen(Request(url,headers={"Cache-Control":"no-cache"}),timeout=8) as response:
                        if checked_stop_control(json.load(response)):self.requested.set();return
                    failed_since=None
                except Exception:
                    failed_since=failed_since or time.monotonic()
                    if time.monotonic()-failed_since>300:self.requested.set();return
                self.finished.wait(20)
        threading.Thread(target=watch,daemon=True).start()


def checked_baselines(reference:dict,audit:dict)->dict:
    if (reference.get("source_commit")!="39b81e5f5090a34a22f6c2dbd00ac55add67c922"
        or reference.get("model_repo")!=MODEL or reference.get("model_revision")!=REVISION
        or reference.get("initial_adapter_sha256")!="3b3030afe4c19fdfd533013309f87e5b544f92ab276e03bfa9ce8a54d8f1c6ba"
        or reference.get("execution_allowed") is not False
        or reference.get("source_audit",{}).get("manifest_sha256")!=audit["manifest_sha256"]
        or reference.get("practice_audit",{}).get("validation_examples")!=122):
        raise ValueError("IMMUTABLE_416_SAME_VALIDATION_REFERENCE_REQUIRED")
    fields=("source_baseline_loss","practice_baseline_loss","official_base_source_loss","official_base_practice_loss")
    for field in fields:
        values=reference.get(field,{})
        if set(values)!=set(EDITIONS)|{"ALL"} or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values.values()):
            raise ValueError("FINITE_SAME_COHORT_BASELINES_REQUIRED")
    return {k:reference[k] for k in fields}


def training_lessons():
    from .structured_rule_practice import balanced_recovery_curriculum
    old,old_validation,_=balanced_recovery_curriculum()
    prose=[r for r in old if strict_answer(r["answer"]) is None]
    typed=paired_cases("train")
    for rule in ("TC1.PSY.PLAN","TC2.WM1.TRIGGER"):
        for row in [r for r in typed if r["rule_id"]==rule][::2]:
            explanation=("Бюджет риска = equity × доля риска. Количество = floor(бюджет / расстояние / шаг) × шаг. Округление только вниз. "
                if rule=="TC1.PSY.PLAN" else
                "Для бычьего WM1: entry=high+tick, stop=low-tick. Для медвежьего: entry=low-tick, stop=high+tick. ")
            prose.append({**row,"question":"Объясни расчёт и формулу: "+row["question"].split(" Исследование без ордеров.")[0],
                "answer":explanation+"В этих исходных данных итог: "+row["answer"]+" Это учебный расчёт без отправки заявки."})
    return prose,typed,old_validation


def train_joint(payload_path:Path,questions_path:Path,initial_adapter:Path,reference_path:Path,output:Path)->dict:
    import torch
    import torch.distributed as dist
    from huggingface_hub import HfApi,snapshot_download
    from transformers import (AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,Trainer,TrainingArguments,
        TrainerCallback,EarlyStoppingCallback,StoppingCriteria,StoppingCriteriaList,set_seed)
    from peft import PeftConfig,PeftModel,prepare_model_for_kbit_training
    from .private_book_source import tokenize_source_pages
    from .private_lora import _tokenized
    from .lora_integrity import validation_admission
    started=time.monotonic();rank=int(os.environ["LOCAL_RANK"]);world=int(os.environ["WORLD_SIZE"])
    if world!=2 or torch.cuda.device_count()!=2:raise ValueError("TWO_ACTUAL_CUDA_DEVICES_AND_DDP_RANKS_REQUIRED")
    torch.cuda.set_device(rank);dist.init_process_group("nccl",timeout=timedelta(minutes=12))
    device=torch.device("cuda",rank);watcher=StopWatcher()
    watcher.start()
    if rank==0:print("ACTUAL_DUAL_T4_DDP",json.dumps({"world_size":world,"gpu_names":[torch.cuda.get_device_name(i) for i in range(2)]}),flush=True)
    def gather(rows):
        parts=[None]*world;dist.all_gather_object(parts,rows)
        return [row for part in parts for row in part]
    def stopped():
        flag=torch.tensor(int(watcher.requested.is_set() if rank==0 else 0),device=device);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        return bool(flag.item())
    set_seed(20261028)
    source=checked_payload(json.loads(payload_path.read_bytes()));reference=json.loads(reference_path.read_bytes())
    baselines=checked_baselines(reference,source["audit"])
    questions=json.loads(questions_path.read_bytes())
    if len(questions)!=56 or any(set(q)!={"case_id","domain","prompt"} for q in questions):
        raise ValueError("SEALED_56_PROMPTS_WITHOUT_EXPECTED_KEYS_REQUIRED")
    cfg=PeftConfig.from_pretrained(initial_adapter,local_files_only=True)
    if cfg.r!=16 or cfg.lora_alpha!=32 or cfg.bias!="none" or set(cfg.target_modules)!={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}:
        raise ValueError("OWN_RANK16_ATTENTION_MLP_REQUIRED")
    if rank==0:
        output.mkdir(mode=0o700)
        shutil.copytree(initial_adapter,output/"source_adapter")
    dist.barrier()
    info=HfApi().model_info(MODEL,revision=REVISION,files_metadata=True)
    if info.sha!=REVISION:raise ValueError("PINNED_QWEN3_4B_REQUIRED")
    official={x.rfilename:x.lfs.sha256 for x in info.siblings if x.rfilename.endswith(".safetensors") and x.lfs}
    if rank==0:
        base=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=["*.json","*.safetensors","*.txt","*.jinja"]))
        for name,digest in official.items():
            if sha256((base/name).read_bytes()).hexdigest()!=digest:raise ValueError("OFFICIAL_QWEN_BASE_SHA_FAILED")
    dist.barrier()
    base=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    tokenizer=AutoTokenizer.from_pretrained(base,trust_remote_code=False,local_files_only=True)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    prose,typed,old_validation=training_lessons()
    source_rows=tokenize_source_pages(source["train"],tokenizer,max_tokens=384)
    source_val={e:tokenize_source_pages([r for r in source["validation"] if r["edition"]==e],tokenizer,max_tokens=384) for e in EDITIONS}
    task_val={e:_tokenized(old_validation[e],tokenizer,max_tokens=768) for e in EDITIONS}
    def ordinary(rows):return [{**r,"loss_weights":[float(t!=-100) for t in r["labels"]]} for r in rows]
    typed_tokens=tokenize_typed(typed,tokenizer)
    train_rows=ordinary(source_rows)+ordinary(_tokenized(prose,tokenizer,max_tokens=768))+typed_tokens*2
    random.Random(20261028).shuffle(train_rows)
    quick=quick_development_cases();quick_questions=question_rows(quick)
    quick_sources={e:[source_val[e][round(i*(len(source_val[e])-1)/3)] for i in range(4)] for e in EDITIONS}
    dev_rows=ordinary([r for e in EDITIONS for r in quick_sources[e]])+ordinary(tokenize_typed(quick,tokenizer))
    base_model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,use_safetensors=True,
        dtype=torch.float16,device_map={"":rank},quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16))
    if base_model.config.model_type!="qwen3" or base_model.config.hidden_size!=2560 or base_model.config.num_hidden_layers!=36:
        raise ValueError("EXACT_QWEN3_4B_ARCHITECTURE_REQUIRED")
    base_model=prepare_model_for_kbit_training(base_model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False})
    model=PeftModel.from_pretrained(base_model,initial_adapter,is_trainable=True,local_files_only=True)
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if sum(p.numel() for p in parameters.values())!=33030144 or any(".lora_" not in n for n in parameters):
        raise ValueError("ONLY_EXPECTED_LORA_PARAMETERS_TRAINABLE")
    initial={n:p.detach().cpu().clone() for n,p in parameters.items()}
    report={"mode":"TWO_T4_CONTRASTIVE_VALUE_WEIGHTED_FOUR_BOOK_REPLAY_V1","model_repo":MODEL,"model_revision":REVISION,
        "source_commit":os.environ.get("TRAINING_SOURCE_SHA"),"official_weights_sha256":official,
        "gpu_names":[torch.cuda.get_device_name(i) for i in range(2)],"world_size":world,"rank":16,"trainable_parameters":33030144,
        "initial_adapter_sha256":sha256((initial_adapter/"adapter_model.safetensors").read_bytes()).hexdigest(),
        "initialization":"OWN_416_CHECKPOINT_NEW_OPTIMIZER","optimizer_state_resumed":False,
        "source_audit":source["audit"],"source_train_windows":len(source_rows),"joint_train_rows_including_repeats":len(train_rows),
        "paired_unique_train":len(typed),"paired_train_repetitions":2,"prose_train":len(prose),
        "new_parameter_test_cases":56,"quick_development_cases":len(quick),"validation_examples":122,
        "source_baseline_loss":baselines["source_baseline_loss"],"practice_baseline_loss":baselines["practice_baseline_loss"],
        "official_base_source_loss":baselines["official_base_source_loss"],"official_base_practice_loss":baselines["official_base_practice_loss"],
        "full_baseline_reused_from_verified_same_cohort_receipt":True,
        "source_adapter_role":"READ_ONLY_416_REFERENCE","source_adapter_sha256":sha256((initial_adapter/"adapter_model.safetensors").read_bytes()).hexdigest(),
        "old24_are_development_diagnostic_not_new_test":True,"market_profitability_measured":False,"expert_market_labels":False,
        "execution_allowed":False,"trading_qualification":False,"training_wall_budget_seconds":6900}
    def write_report():
        if rank==0:(output/"research_receipt.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    write_report()
    class UserStop(StoppingCriteria):
        def __init__(self,deadline):self.deadline=deadline
        def __call__(self,input_ids,scores,**kwargs):return watcher.requested.is_set() or time.monotonic()>=self.deadline
    def predict(rows,*,stage,thinking=False,time_limit=None):
        was_training=model.training;cache=model.config.use_cache;model.eval();model.config.use_cache=True
        local=[]
        with torch.inference_mode():
            for idx,row in enumerate(rows):
                if idx%world!=rank:continue
                prefix=tokenizer.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":row["prompt"]}],
                    tokenize=True,add_generation_prompt=True,enable_thinking=thinking)
                ids=torch.tensor([prefix],device=device)
                options=({"do_sample":True,"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"max_new_tokens":768}
                    if thinking else {"do_sample":False,"max_new_tokens":128})
                torch.manual_seed(20261028+idx)
                deadline=time.monotonic()+(time_limit if time_limit is not None else (60 if thinking else 45))
                with (model.disable_adapter() if stage=="base" else nullcontext()):
                    generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),pad_token_id=tokenizer.eos_token_id,
                        stopping_criteria=StoppingCriteriaList([UserStop(deadline)]),**options)[0,len(prefix):].tolist()
                complete=True
                if thinking:
                    close=tokenizer.convert_tokens_to_ids("</think>")
                    if close not in generated:complete=False;generated=[]
                    else:generated=generated[len(generated)-generated[::-1].index(close):]
                response=tokenizer.decode(generated,skip_special_tokens=True).strip()
                local.append({"case_id":row["case_id"],"response":response,"thinking_complete":complete,
                    "response_time_limit_reached":time.monotonic()>=deadline})
        values=gather(local);by_id={r["case_id"]:r for r in values}
        if len(by_id)!=len(rows):raise ValueError("NO_MISSING_OR_REPLAYED_GENERATIONS")
        model.config.use_cache=cache;model.train(was_training)
        return [by_id[r["case_id"]] for r in rows]
    def source_scores(groups):
        was_training=model.training;model.eval();local=[]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
            for ed,rows in groups.items():
                for idx,row in enumerate(rows):
                    if idx%world!=rank:continue
                    batch={k:torch.tensor([row[k]],device=device) for k in ("input_ids","attention_mask","labels")}
                    value=float(model(**batch,use_cache=False).loss)
                    if not math.isfinite(value):raise ValueError("FINITE_UNWEIGHTED_SOURCE_LOSS_REQUIRED")
                    local.append((ed,idx,value))
        values=gather(local);result={e:sum(x[2] for x in values if x[0]==e)/len(groups[e]) for e in EDITIONS}
        result["ALL"]=sum(result[e]*len(groups[e]) for e in EDITIONS)/sum(map(len,groups.values()))
        model.train(was_training);return result
    initial_dev_source=source_scores(quick_sources)
    report["initial_dev_source_loss"]=initial_dev_source
    report["initial_direct_development"]=score_responses(quick,predict(quick_questions,stage="initial"))
    write_report()
    class Rows:
        def __init__(self,rows):self.rows=rows
        def __len__(self):return len(self.rows)
        def __getitem__(self,index):return self.rows[index]
    def collate(batch):
        width=max(len(r["input_ids"]) for r in batch)
        return {key:torch.tensor([r[key]+[fill]*(width-len(r[key])) for r in batch],dtype=torch.float32 if key=="loss_weights" else torch.long)
            for key,fill in (("input_ids",tokenizer.pad_token_id),("attention_mask",0),("labels",-100),("loss_weights",0.0))}
    class Deadline(TrainerCallback):
        def __init__(self):self.saw_nonfinite=False;self.actual_updates=0;self.skipped=0;self.consecutive_skips=0;self.updates_by_checkpoint={}
        def on_pre_optimizer_step(self,args,state,control,**kwargs):
            gradients=[p.grad for p in parameters.values() if p.grad is not None]
            if not gradients:raise ValueError("MISSING_REAL_LORA_GRADIENTS")
            self.saw_nonfinite=not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
            if self.saw_nonfinite and trainer.accelerator.scaler is None:
                raise ValueError("NONFINITE_GRADIENT_WITHOUT_AMP_SKIP_GUARD")
        def on_optimizer_step(self,args,state,control,**kwargs):
            skipped=bool(trainer.accelerator.optimizer_step_was_skipped)
            flags=torch.tensor([int(skipped),int(skipped)],device=device)
            dist.all_reduce(flags[:1],op=dist.ReduceOp.MIN);dist.all_reduce(flags[1:],op=dist.ReduceOp.MAX)
            if flags[0]!=flags[1]:raise ValueError("DDP_OPTIMIZER_SKIP_MUST_MATCH_BOTH_RANKS")
            if self.saw_nonfinite and not skipped:raise ValueError("AMP_MUST_SKIP_NONFINITE_GRADIENT_UPDATE")
            if skipped:
                self.skipped+=1;self.consecutive_skips+=1
                if self.consecutive_skips>=8:raise ValueError("PERSISTENT_FP16_OVERFLOW_NO_VALID_UPDATE")
                if rank==0:print("SAFE_FP16_OVERFLOW_STEP_SKIPPED",self.skipped,trainer.accelerator.scaler.get_scale(),flush=True)
            else:
                self.actual_updates+=1;self.consecutive_skips=0
                if not bool(torch.stack([torch.isfinite(p).all() for p in parameters.values()]).all()):
                    raise ValueError("NONFINITE_ADAPTER_AFTER_OPTIMIZER_UPDATE")
        def on_step_end(self,args,state,control,**kwargs):
            flag=torch.tensor([int(time.monotonic()-started>=6900),int(watcher.requested.is_set())],device=device)
            dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            if bool(flag.any()):
                control.should_training_stop=True;control.should_save=True
                control.should_evaluate=not bool(flag[1])
                if flag[1]:args.load_best_model_at_end=False
        def on_epoch_end(self,args,state,control,**kwargs):
            control.should_evaluate=not watcher.requested.is_set();control.should_save=True
        def on_save(self,args,state,control,**kwargs):self.updates_by_checkpoint[state.global_step]=self.actual_updates
    class SemanticTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            inputs=dict(inputs);labels=inputs.pop("labels");weights=inputs.pop("loss_weights")
            outputs=model(**inputs,use_cache=False);loss=weighted_causal_loss(outputs.logits,labels,weights)
            if not bool(torch.isfinite(loss)):raise ValueError("NONFINITE_WEIGHTED_LOSS_REJECTED")
            return (loss,outputs) if return_outputs else loss
        def evaluation_loop(self,*args,**kwargs):
            result=super().evaluation_loop(*args,**kwargs)
            if stopped():return result._replace(metrics={**result.metrics,"eval_semantic_replay_score":-2.0})
            generated=predict(quick_questions,stage="candidate");accuracy=score_responses(quick,generated)["accuracy"]
            values=source_scores(quick_sources)
            regression=max(values[e]/initial_dev_source[e]-1 for e in EDITIONS)
            score=accuracy if regression<=0.02 else -1-regression
            metrics={**result.metrics,"eval_semantic_accuracy":accuracy,"eval_source_dev_max_regression":regression,
                "eval_semantic_replay_score":score}
            if rank==0:print("CONTRASTIVE_DEV_SELECTION",json.dumps({"accuracy":accuracy,"max_source_regression":regression,
                "selection_score":score,"optimizer_step":self.state.global_step}),flush=True)
            return result._replace(metrics=metrics)
    args=TrainingArguments(output_dir=str(output.parent/"contrastive_private_checkpoints"),num_train_epochs=1.0,
        per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=1,
        learning_rate=1.5e-5,warmup_ratio=0.03,weight_decay=0.01,lr_scheduler_type="cosine",max_grad_norm=1.0,
        optim="adamw_torch",eval_strategy="steps",eval_steps=128,save_strategy="steps",save_steps=128,
        save_total_limit=2,save_safetensors=True,load_best_model_at_end=True,metric_for_best_model="semantic_replay_score",
        greater_is_better=True,prediction_loss_only=True,logging_steps=16,logging_nan_inf_filter=False,report_to="none",
        disable_tqdm=True,remove_unused_columns=False,label_names=["labels"],seed=20261028,data_seed=20261028,
        fp16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
        ddp_backend="nccl",ddp_find_unused_parameters=False,ddp_broadcast_buffers=False,ddp_timeout=720)
    deadline=Deadline()
    trainer=SemanticTrainer(model=model,args=args,train_dataset=Rows(train_rows),eval_dataset=Rows(dev_rows),
        data_collator=collate,callbacks=[deadline,EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.model_accepts_loss_kwargs=False
    if trainer.accelerator.scaler is None:raise ValueError("FP16_DYNAMIC_SCALER_REQUIRED")
    trainer.accelerator.scaler=torch.amp.GradScaler("cuda",init_scale=1024.0,growth_interval=2000)
    report["initial_fp16_loss_scale"]=1024.0
    write_report()
    if not stopped():trainer.train()
    report["joint_optimizer_steps"]=deadline.actual_updates
    report["optimizer_attempted_steps"]=trainer.state.global_step
    report["fp16_overflow_steps_skipped"]=deadline.skipped
    report["actual_epoch"]=trainer.state.epoch
    report["selected_checkpoint"]=Path(trainer.state.best_model_checkpoint).name if trainer.state.best_model_checkpoint else None
    report["selected_metric"]=trainer.state.best_metric
    report["actual_updates_at_selected_checkpoint"]=(deadline.updates_by_checkpoint.get(int(Path(trainer.state.best_model_checkpoint).name.split("-")[-1]))
        if trainer.state.best_model_checkpoint else None)
    report["training_stopped_by_user_or_control_unavailable"]=stopped()
    if report["training_stopped_by_user_or_control_unavailable"]:
        report["selected_checkpoint"]=f"interrupted-step-{trainer.state.global_step}"
        report["selected_metric"]=None
    changed=any(not torch.equal(p.detach().cpu(),initial[n]) for n,p in parameters.items())
    if any(not bool(torch.isfinite(p).all()) for p in parameters.values()):raise ValueError("FINITE_ADAPTER_WEIGHTS_REQUIRED")
    report["adapter_weights_changed"]=changed
    if rank==0:
        model.save_pretrained(output/"candidate_adapter",safe_serialization=True);tokenizer.save_pretrained(output/"candidate_adapter")
        report["candidate_adapter_sha256"]=sha256((output/"candidate_adapter/adapter_model.safetensors").read_bytes()).hexdigest()
    digests=[None]*world;dist.all_gather_object(digests,report.get("candidate_adapter_sha256"));report["candidate_adapter_sha256"]=digests[0]
    write_report();dist.barrier()
    if report["training_stopped_by_user_or_control_unavailable"]:
        watcher.finished.set();dist.destroy_process_group();return report
    report["source_final_loss"]=source_scores(source_val);report["practice_final_loss"]=source_scores(task_val)
    report["final_source_validation_gate"]=validation_admission(report["source_baseline_loss"],report["source_final_loss"])
    report["final_practice_validation_gate"]=validation_admission(report["practice_baseline_loss"],report["practice_final_loss"])
    report["text_validation_gates_passed"]=report["final_source_validation_gate"]["accepted"] and report["final_practice_validation_gate"]["accepted"]
    report["candidate_accepted_for_further_research"]=False
    report["admission_status"]="PENDING_CPU_SEALED_GENERATED_TEST"
    model.gradient_checkpointing_disable();model.eval()
    report["final_direct_development"]=score_responses(quick,predict(quick_questions,stage="candidate"))
    report["final_thinking_development"]=score_responses(quick,predict(quick_questions,stage="candidate",thinking=True))
    mode=report["final_thinking_development"]["accuracy"]>report["final_direct_development"]["accuracy"]
    report["test_inference_mode_selected_on_development_only"]="thinking" if mode else "direct"
    write_report()
    model.load_adapter(initial_adapter,adapter_name="own416_reference",is_trainable=False,local_files_only=True)
    precommit={"questions_sha256":questions_digest(questions),"stages":{},"inference_mode":report["test_inference_mode_selected_on_development_only"],
        "expected_keys_loaded_on_gpu":False,"new_unseen_numeric_parameters":True,"execution_allowed":False}
    response_seconds=min(45.0,max(1.0,(10800-(time.monotonic()-started)-420)/(3*math.ceil(len(questions)/world))))
    precommit["same_response_time_budget_seconds_each_model"]=response_seconds
    for stage,adapter in (("base",None),("initial416","own416_reference"),("candidate","default")):
        if stopped():break
        model.set_adapter(adapter or "default")
        precommit["stages"][stage]=predict(questions,stage=stage,thinking=mode,time_limit=response_seconds)
        if rank==0:(output/"contrastive_test_precommit.json").write_text(json.dumps(precommit,ensure_ascii=False,indent=2))
    report["generated_test_stages_completed"]=len(precommit["stages"])
    report["complete_training_and_test"]=len(precommit["stages"])==3 and changed
    report["training_stopped_by_user_or_control_unavailable"]=stopped()
    write_report();watcher.finished.set();dist.barrier();dist.destroy_process_group();return report
