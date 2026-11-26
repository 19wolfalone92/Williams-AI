"""Private bounded Qwen3-4B GPU reload experiment and V6 corrective block.
The original adapter is mounted read-only from the owner's existing notebook.
No final exam, market execution, or credentials are used by this worker.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,math,random
from pathlib import Path
import subprocess
from contextlib import nullcontext
from williams_ai.knowledge_training_v5 import BASE,REV
from williams_ai.v6_corrective_curriculum import corrective_curriculum,SYSTEM_PROMPT,validate_curriculum
from williams_ai.qlora_gpu_research import _fit_resumable_adapter,_runtime_identity,contract,digest_file
from scripts.ai_v6_corrective_probe_cpu import grade_response

SOURCE_SHA="f4979cf0bebbf33b7c8e5746326c1e43762f31fe730e89cf5e39c7a8a155c309"
CONFIG_SHA="4c78116814bdc938b21ee70e69e4bd5c54a039794e642b17696878d91a244e2d"
DATA_SHA="be9dca4f5cc3131f778ea3c746a3fb53fcb58d9e607acafacc8e686b37c17c57"
ATOL=RTOL=.003

def locate_source(root):
    matches=[]
    for p in root.rglob("adapter_model.safetensors"):
        if p.is_file() and not p.is_symlink() and p.parent.name=="latest_adapter" and digest_file(p)==SOURCE_SHA:
            cfg=p.parent/"adapter_config.json"
            if digest_file(cfg)!=CONFIG_SHA:raise ValueError("V6_CONFIG_SHA_MISMATCH")
            matches.append(p.parent)
    if not matches:raise ValueError("VERIFIED_V6_SOURCE_NOT_MOUNTED")
    return sorted(matches)[0]

def messages(row):
    return [{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":row["prompt"]}]

def encode(rows,tok):
    out=[]
    for row in rows:
        if row["split"] not in ("train","validation"):raise ValueError("EVALUATION_LABEL_IN_TRAINING")
        prefix=tok.apply_chat_template(messages(row),tokenize=True,add_generation_prompt=True,enable_thinking=False)
        full=tok.apply_chat_template(messages(row)+[{"role":"assistant","content":row["answer"]}],tokenize=True,add_generation_prompt=False,enable_thinking=False)
        if not isinstance(prefix,list) or not isinstance(full,list) or not len(prefix)<len(full)<=768 or full[:len(prefix)]!=prefix:
            raise ValueError("ANSWER_ONLY_NO_TRUNCATION")
        out.append({"input_ids":full,"attention_mask":[1]*len(full),"labels":[-100]*len(prefix)+full[len(prefix):]})
    return out

def publish(path,data):
    path.write_text(json.dumps(data,sort_keys=True,allow_nan=False,indent=2))
    print("WILLIAMS_A_GPU_RECEIPT",path.name,json.dumps(data,sort_keys=True,allow_nan=False),flush=True)

def compare(a,b):
    import torch
    return {"pass":bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.allclose(a,b,atol=ATOL,rtol=RTOL)),
            "max_abs":float((a-b).abs().max()),"atol":ATOL,"rtol":RTOL}

def logits(model,ids,amp=False):
    import torch
    model.eval()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16,enabled=amp):
        return model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[:,-1,:].float().cpu()

def assert_saved(model,folder):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    expected=load_file(str(folder/"adapter_model.safetensors"),device="cpu")
    actual=get_peft_model_state_dict(model)
    if len(expected)!=288 or set(expected)!=set(actual):raise ValueError("ADAPTER_TENSOR_INVENTORY_MISMATCH")
    for k,v in expected.items():
        a=actual[k].detach().cpu()
        if v.dtype!=a.dtype or v.shape!=a.shape or not torch.isfinite(v).all() or not torch.equal(v,a):
            raise ValueError("ADAPTER_TENSOR_BYTES_MISMATCH")
    return len(expected)

def trainable(model):
    ps=[p for n,p in model.named_parameters() if p.requires_grad and ".lora_" in n]
    if len(ps)!=288 or len(ps)!=sum(p.requires_grad for p in model.parameters()):
        raise ValueError("ONLY_EXPECTED_LORA_PARAMETERS_TRAINABLE")
    return ps

def frozen_digest(model):
    import torch
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode()+b"\0")
            h.update(str(p.dtype).encode()+str(tuple(p.shape)).encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def score(model,tok,rows):
    import torch
    model.eval()
    records=[]
    for row in rows:
        if row["split"]!="development":raise ValueError("DEVELOPMENT_ONLY")
        ids=tok.apply_chat_template(messages(row),tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            pred=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=384,
                do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        suffix=pred[0,ids.shape[1]:]
        raw=tok.decode(suffix,skip_special_tokens=True)
        finish="length" if len(suffix)>=384 and int(suffix[-1])!=tok.eos_token_id else "stop"
        grade=grade_response(raw,row["answer"],finish)
        objective=all(grade[k] for k in ("format_valid","answer_correct","entry_stop_correct","rule_id_correct"))
        records.append({"id":row["case_id"],"family":row["family"],"edition":row["edition"],
            "answer_correct":grade["answer_correct"],"format_valid":grade["format_valid"],"objective_correct":objective,"raw":raw})
    return records

def summary(records):
    def stats(rs):
        return {"n":len(rs),"answer_correct":sum(r["answer_correct"] for r in rs),
                "format_valid":sum(r["format_valid"] for r in rs),"objective_correct":sum(r["objective_correct"] for r in rs)}
    return {"overall":stats(records),"by_family":{f:stats([r for r in records if r["family"]==f]) for f in sorted({r["family"] for r in records})}}

def paired(a,b):
    if [r["id"] for r in a]!=[r["id"] for r in b]:raise ValueError("PAIR_ID_MISMATCH")
    x=sum(i["objective_correct"] and not j["objective_correct"] for i,j in zip(a,b))
    y=sum(j["objective_correct"] and not i["objective_correct"] for i,j in zip(a,b))
    n=x+y
    return {"gains":x,"losses":y,"mcnemar_exact_p":min(1.,2*sum(math.comb(n,k) for k in range(min(x,y)+1))/2**n) if n else 1.}

def run(output,source_sha):
    import torch
    from accelerate import Accelerator
    from accelerate.utils import extract_model_from_parallel
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
    from peft import PeftModel,prepare_model_for_kbit_training
    from huggingface_hub import HfApi,snapshot_download
    from williams_ai.lora_integrity import local_hf_identity
    from safetensors.torch import load_file
    if subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()!=source_sha:raise ValueError("SOURCE_SHA_MISMATCH")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("ONE_CUDA_GPU_REQUIRED")
    torch.cuda.set_device(0)
    source=locate_source(Path("/kaggle/input"))
    rows,manifest=corrective_curriculum()
    if manifest["dataset_sha256"]!=DATA_SHA:raise ValueError("CORRECTIVE_DATASET_SHA_MISMATCH")
    validate_curriculum(rows)
    # Isolated rule practice; no qualification or semantic independence claim.
    train_rows=[r for r in rows if r["split"]=="train"]
    validation_rows=[r for r in rows if r["split"]=="validation"]
    dev=[r for r in rows if r["split"]=="development"]
    if (len(train_rows),len(validation_rows),len(dev))!=(128,32,32):raise ValueError("SPLIT_COUNT_MISMATCH")
    info=HfApi().model_info(BASE,revision=REV,files_metadata=True)
    official={f.rfilename:f.lfs.sha256 for f in info.siblings if f.rfilename.endswith(".safetensors") and f.lfs}
    if info.sha!=REV or not official:raise ValueError("BASE_REVISION_UNVERIFIED")
    root=Path(snapshot_download(BASE,revision=REV,local_dir="/kaggle/temp/williams-corrective-qwen",
        allow_patterns=["*.safetensors","*.safetensors.index.json","config.json","generation_config.json","tokenizer*","vocab.json","merges.txt","special_tokens_map.json","added_tokens.json","chat_template.jinja"]))
    identity=local_hf_identity(root)
    if {k:v for k,v in identity["files"].items() if k.endswith(".safetensors")}!=official:raise ValueError("BASE_HASH_MISMATCH")
    cfg=AutoConfig.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    if (cfg.model_type,cfg.hidden_size,cfg.num_hidden_layers)!=("qwen3",2560,36):raise ValueError("WRONG_BASE")
    tok=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    def load(folder):
        set_seed(20261126)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        base=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,trust_remote_code=False,use_safetensors=True,
            quantization_config=quant,dtype=torch.float16,device_map={"":0})
        base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
        model=PeftModel.from_pretrained(base,folder,is_trainable=True,local_files_only=True)
        model.config.use_cache=False
        assert_saved(model,folder)
        trainable(model)
        return model
    def clean():
        gc.collect();torch.cuda.empty_cache()
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    train=encode(train_rows,tok)
    validation={ed:encode([r for r in validation_rows if r["edition"]==ed],tok) for ed in ("TC1","NTD","TC2","COURSE")}
    ids=tok.apply_chat_template(messages(dev[0]),tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt").to("cuda:0")
    diagnostic=output/"diagnostic"
    diagnostic.mkdir()
    acc=Accelerator(mixed_precision="fp16")
    def update(model,opt,example):
        model.train()
        opt.zero_grad()
        batch={k:torch.tensor([v],device="cuda:0") for k,v in example.items()}
        loss=model(**batch,use_cache=False).loss
        if not torch.isfinite(loss):raise ValueError("NONFINITE_DIAGNOSTIC_LOSS")
        acc.backward(loss)
        acc.clip_grad_norm_(trainable(model),1.0)
        grads=[p.grad for p in trainable(model) if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads):
            raise ValueError("INVALID_DIAGNOSTIC_GRADIENTS")
        opt.step()
        if acc.optimizer_step_was_skipped:raise ValueError("DIAGNOSTIC_OPTIMIZER_SKIPPED")
        return float(loss.detach())
    model=load(source)
    baseline_logits=logits(model,ids)
    opt=torch.optim.AdamW(trainable(model),lr=1e-5)
    model,opt=acc.prepare(model,opt)
    before=[p.detach().cpu().clone() for p in trainable(model)]
    loss1=update(model,opt,train[0])
    if not any(not torch.equal(a,p.detach().cpu()) for a,p in zip(before,trainable(model))):raise ValueError("DIAGNOSTIC_NO_WEIGHT_UPDATE")
    wrapped=logits(model,ids)
    bare=acc.unwrap_model(model,keep_fp32_wrapper=False)
    bare_fp=logits(bare,ids)
    bare_amp=logits(bare,ids,amp=True)
    bare.save_pretrained(diagnostic/"adapter",safe_serialization=True)
    assert_saved(bare,diagnostic/"adapter")
    torch.save({"optimizer":opt.state_dict(),"scaler":acc.scaler.state_dict() if acc.scaler else None,
        "cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state()},diagnostic/"optimizer.pt")
    acc.free_memory()
    del model,bare,opt,before
    clean()
    restored=load(diagnostic/"adapter")
    new_fp=logits(restored,ids)
    new_amp=logits(restored,ids,amp=True)
    same_fp=compare(bare_fp,new_fp);same_amp=compare(bare_amp,new_amp)
    cross=compare(wrapped,new_fp);same_wrapper=compare(wrapped,new_amp)
    receipt={"source_adapter_sha256":SOURCE_SHA,"diagnostic_adapter_sha256":digest_file(diagnostic/"adapter"/"adapter_model.safetensors"),
       "gpu":torch.cuda.get_device_name(0),"diagnostic_optimizer_steps":1,"loss":loss1,"saved_tensor_count":288,
       "same_mode_no_autocast":same_fp,"same_mode_fp16_autocast":same_amp,"old_wrapped_vs_bare_reload":cross,
       "wrapped_vs_explicit_amp_reload":same_wrapper,"amp_hypothesis_reproduced":bool(not cross["pass"] and same_fp["pass"] and same_amp["pass"] and same_wrapper["pass"]),
       "reload_pass":bool(same_fp["pass"] and same_amp["pass"] and same_wrapper["pass"]),"execution_allowed":False}
    publish(output/"diagnostic.json",receipt)
    if not receipt["reload_pass"]:raise ValueError("CONTROLLED_RELOAD_LOGITS_MISMATCH")
    opt2=torch.optim.AdamW(trainable(restored),lr=1e-5)
    restored,opt2=acc.prepare(restored,opt2)
    state=torch.load(diagnostic/"optimizer.pt",weights_only=True,map_location="cpu")
    opt2.load_state_dict(state["optimizer"])
    if not opt2.state_dict()["state"]:raise ValueError("OPTIMIZER_STATE_NOT_RESTORED")
    for k,old_state in state["optimizer"]["state"].items():
        new_state=opt2.state_dict()["state"][k]
        for field,value in old_state.items():
            if isinstance(value,torch.Tensor) and not torch.equal(value.cpu(),new_state[field].cpu()):raise ValueError("OPTIMIZER_STATE_TENSOR_MISMATCH")
    if acc.scaler and state["scaler"] is not None:acc.scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["cpu_rng"]);torch.cuda.set_rng_state(state["cuda_rng"])
    loss2=update(restored,opt2,train[1])
    receipt.update(diagnostic_optimizer_steps=2,optimizer_restore_pass=True,continued_loss=loss2)
    publish(output/"diagnostic.json",receipt)
    acc.free_memory()
    del restored,opt2,state,acc
    clean()
    # Fresh optimizer from the untouched V6, not the diagnostic branch.
    model=load(source)
    with model.disable_adapter():base_results=score(model,tok,dev)
    v6_results=score(model,tok,dev)
    (output/"baseline_predictions.json").write_text(json.dumps({"BASE":base_results,"V6":v6_results}))
    publish(output/"baseline.json",{"BASE":summary(base_results),"V6":summary(v6_results)})
    frozen_before=frozen_digest(model)
    spec=contract(dataset_sha=manifest["dataset_sha256"],base_sha=identity["sha256"],revision=source_sha,
        max_steps=40,max_tokens=768,rank=8,rate=1e-5,seed=20261126,runtime=_runtime_identity(),source_only_research=True)
    result=_fit_resumable_adapter(model,train,validation,output=output/"candidate-a",spec=spec,pad_token_id=tok.pad_token_id,session_steps=10)
    if result["new_optimizer_steps"]!=10 or not result["adapter_weights_changed_in_this_session"]:raise ValueError("NO_REAL_CORRECTIVE_UPDATE")
    if frozen_digest(model)!=frozen_before:raise ValueError("FROZEN_BASE_PARAMETER_CHANGED")
    folder=output/"candidate-a"/"latest_adapter"
    canonical=extract_model_from_parallel(model,keep_fp32_wrapper=False)
    expected=logits(canonical,ids)
    assert_saved(canonical,folder)
    del canonical,model
    clean()
    model=load(folder)
    reload=compare(expected,logits(model,ids))
    if not reload["pass"]:raise ValueError("CANDIDATE_RELOAD_MISMATCH")
    candidate_results=score(model,tok,dev)
    (output/"candidate_predictions.json").write_text(json.dumps(candidate_results))
    if local_hf_identity(root)!=identity or digest_file(source/"adapter_model.safetensors")!=SOURCE_SHA:raise ValueError("SOURCE_BYTES_CHANGED")
    publish(output/"aggregate.json",{"schema":"WILLIAMS_A_CORRECTIVE_GPU_V1","source_commit":source_sha,
        "source_adapter_sha256":SOURCE_SHA,"adapter_sha256":digest_file(folder/"adapter_model.safetensors"),
        "diagnostic_steps":2,"corrective_steps":10,"optimizer_skips":0,"frozen_base_unchanged":True,
        "reload":reload,"dataset_sha256":DATA_SHA,"new_optimizer":True,"training":result,
        "BASE":summary(base_results),"V6":summary(v6_results),"candidate_A":summary(candidate_results),
        "paired_A_vs_BASE":paired(candidate_results,base_results),"paired_A_vs_V6":paired(candidate_results,v6_results),
        "explanation_semantics_verified":False,"independent_final_exam":False,"qualified":False,"execution_allowed":False})

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True);p.add_argument("--source-sha",required=True)
    a=p.parse_args();run(a.output,a.source_sha)
