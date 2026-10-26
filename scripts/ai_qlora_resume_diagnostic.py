"""Real three-process optimizer/resume QA on RANDOM tiny Qwen3, never a trader.

Only disposable synthetic token IDs and randomly initialized local weights.
This executes the same Trainer/checkpoint callbacks as the CUDA QLoRA entry,
but CPU FP32 cannot certify bitsandbytes NF4, CUDA, FP16 scaler or Kaggle.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def rows(count:int=32):
    def example(i):
        prefix=[1,20+i%4,30+i%16,8]
        answer=[5+(i+j)%12 for j in range(12)]+[2]
        return {"input_ids":prefix+answer,"attention_mask":[1]*(len(prefix)+len(answer)),
                "labels":[-100]*len(prefix)+answer}
    return [example(i) for i in range(count)],{
        edition:[example(i)] for i,edition in enumerate(("TC1","NTD","TC2","COURSE"))}


def worker(root:Path,mode:str):
    import torch
    from peft import LoraConfig,TaskType,get_peft_model
    from transformers import Qwen3ForCausalLM,set_seed
    from williams_ai.qlora_gpu_research import (
        _fit_resumable_adapter,_runtime_identity,contract,digest_file,latest_checkpoint,
    )
    torch.set_num_threads(2)
    set_seed(314)
    base=Qwen3ForCausalLM.from_pretrained(root/"base",local_files_only=True,use_safetensors=True)
    frozen=[(p,p.detach().clone()) for p in base.parameters()]
    model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        task_type=TaskType.CAUSAL_LM,bias="none"))
    model.config.use_cache=False
    train,validation=rows(json.loads((root/"source.json").read_text())["training_examples"])
    spec=contract(dataset_sha=hashlib.sha256(json.dumps(train,sort_keys=True).encode()).hexdigest(),
        base_sha=digest_file(root/"base"/"model.safetensors"),
        revision=json.loads((root/"source.json").read_text())["source_commit"],
        max_steps=20,max_tokens=128,rank=8,rate=5e-4,seed=314,
        runtime=_runtime_identity(synthetic_cpu=True),source_only_research=True,
        synthetic_cpu_diagnostic=True)
    checkpoint,receipt=latest_checkpoint(root/"split_first") if mode=="split_resume" else (None,None)
    report=_fit_resumable_adapter(model,train,validation,output=root/mode,spec=spec,
        pad_token_id=0,session_steps=20 if mode=="continuous" else 10,
        resume_checkpoint=checkpoint,resume_receipt=receipt,use_cpu=True)
    assert all(not p.requires_grad and torch.equal(p.detach(),before) for p,before in frozen)
    assert report["adapter_weights_changed_in_this_session"]
    assert report["new_optimizer_steps"]==(20 if mode=="continuous" else 10)
    if mode=="continuous":
        # A nonfinite baseline must fail BEFORE a real optimizer step.
        from unittest.mock import patch
        broken=Qwen3ForCausalLM.from_pretrained(root/"base",local_files_only=True)
        with torch.no_grad():broken.model.embed_tokens.weight.fill_(float("nan"))
        bad=get_peft_model(broken,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,
            target_modules=spec["target_modules"],task_type=TaskType.CAUSAL_LM,bias="none"))
        with patch.object(torch.optim.AdamW,"step",side_effect=AssertionError("MUST_NOT_STEP")):
            try:
                _fit_resumable_adapter(bad,train,validation,output=root/"bad-baseline",
                    spec=spec,pad_token_id=0,session_steps=10,use_cpu=True)
            except ValueError as exc:
                assert "FINITE_FOUR_BOOK_VALIDATION" in str(exc)
            else:raise AssertionError("NONFINITE_BASELINE_ACCEPTED")


def main():
    if len(sys.argv)==4 and sys.argv[1]=="--worker":
        worker(Path(sys.argv[2]),sys.argv[3])
        return
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import Qwen3Config,Qwen3ForCausalLM,set_seed
    torch.set_num_threads(2)
    set_seed(314)
    source=os.environ.get("SOURCE_HEAD") or subprocess.run(
        ["git","rev-parse","HEAD"],check=True,capture_output=True,text=True,
        cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    cases=[]
    with TemporaryDirectory(prefix="williams-qlora-resume-synthetic-") as directory:
        config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
            max_position_embeddings=128,bos_token_id=1,eos_token_id=2,pad_token_id=0)
        environment={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
                     "HF_HUB_DISABLE_TELEMETRY":"1","TOKENIZERS_PARALLELISM":"false"}
        for count in (32,35):
            root=Path(directory)/str(count)
            root.mkdir()
            (root/"source.json").write_text(json.dumps({"source_commit":source,"training_examples":count}))
            set_seed(314)
            Qwen3ForCausalLM(config).save_pretrained(root/"base",safe_serialization=True)
            for mode in ("continuous","split_first","split_resume"):
                subprocess.run([sys.executable,str(Path(__file__).resolve()),"--worker",str(root),mode],
                               env=environment,check=True)
            reports={mode:json.loads((root/mode/"research_result.json").read_text())
                     for mode in ("continuous","split_first","split_resume")}
            assert reports["continuous"]["total_optimizer_steps"]==20
            assert reports["split_resume"]["total_optimizer_steps"]==20
            assert reports["continuous"]["original_baseline"]==reports["split_resume"]["original_baseline"]
            first=load_file(root/"continuous"/"latest_adapter"/"adapter_model.safetensors")
            second=load_file(root/"split_resume"/"latest_adapter"/"adapter_model.safetensors")
            assert first.keys()==second.keys()
            difference=max(float((first[name]-second[name]).abs().max()) for name in first)
            assert difference==0.0, {"training_examples":count,"resume_weights_difference":difference}
            models=[PeftModel.from_pretrained(Qwen3ForCausalLM.from_pretrained(root/"base",
                    local_files_only=True,use_safetensors=True),root/mode/"latest_adapter",
                    local_files_only=True).eval() for mode in ("continuous","split_resume")]
            ids=torch.tensor([rows()[0][0]["input_ids"]])
            with torch.no_grad():logits=float((models[0](ids).logits-models[1](ids).logits).abs().max())
            assert logits==0.0
            cases.append({"training_examples":count,"continuous_optimizer_steps":20,
                "split_session_optimizer_steps":[10,10],"fresh_worker_processes":3,
                "resume_boundary_mid_epoch":count==32,"partial_accumulation_at_epoch_end":count==35,
                "maximum_adapter_weight_difference":difference,"maximum_reloaded_logit_difference":logits})
        result={"mode":"RANDOM_TINY_QWEN3_THREE_PROCESS_RESUME_SOFTWARE_DIAGNOSTIC",
            "source_commit":source,"cases":cases,"fresh_worker_processes":6,
            "gradient_accumulation":8,
            "original_four_book_baseline_preserved":True,"base_weights_frozen":True,
            "nonfinite_baseline_rejected_before_optimizer":True,
            "runtime":reports["continuous"]["source_contract"]["runtime"],
            "pretrained_qwen_used":False,"private_books_used":False,
            "cuda_nf4_fp16_scaler_tested":False,"kaggle_session_tested":False,
            "williams_knowledge_trained":False,"independent_trading_quality_proven":False,
            "execution_allowed":False,"trade_action":None}
        Path("/tmp/williams_qlora_resume_diagnostic.json").write_text(
            json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
        print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))


if __name__=="__main__":main()
