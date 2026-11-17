"""v7.1 paired diagnostic inference on the same eight frozen H1 cases."""
from __future__ import annotations
import argparse,inspect,json,torch
from pathlib import Path
from scripts.ai_v6_historical100_models import gather
from scripts.ai_v6_historical100_real_binance import run as collect_archive
from scripts.ai_v6_hist100_kaggle_gpu_worker import find_trained_lora,compact_gpu_prompt,need,write_json_atomic,SEED
from williams_ai.v6_expert_model_runner import prompt_for_case,protocol_sha256
from williams_ai.v6_v5_lora_binding import MODEL_COMMIT,RAW_ADAPTER_SHA
from williams_ai.v72_eos_guard import generate,SCHEMA,ENFORCER_COMMIT

IDS=tuple(f"HIST100-{n:02d}-{k}" for n in (1,2) for k in (1,2,3,4))
def main():
 p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--out-dir",type=Path,required=True);a=p.parse_args()
 need(not a.out_dir.exists(),"OUTPUT_EXISTS");a.out_dir.mkdir(parents=True)
 need(torch.cuda.is_available() and torch.cuda.device_count()>0,"NO_T4")
 root=a.out_dir/"official_hist100";receipt=collect_archive(a.plan,root)
 need(receipt["distinct_chart_windows"]==100 and receipt["no_future_in_model_chart"] is True,"BAD_CORPUS")
 source,plan_sha=gather(a.plan,root);cases=[x for x in source if x[0]["id"] in IDS]
 need(len(cases)==8 and [x[0]["id"] for x in cases]==list(IDS),"CASE_SELECTION")
 adapter=find_trained_lora()
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
 from peft import PeftModel
 import lmformatenforcer.integrations.transformers as lmf
 source_code=inspect.getsource(lmf._decode_function)
 need("clean_up_tokenization_spaces=False" in source_code,"PATCH_MISSING")
 tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-4B",revision=MODEL_COMMIT,trust_remote_code=False)
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
 base=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B",revision=MODEL_COMMIT,trust_remote_code=False,device_map={"":0},quantization_config=quant,torch_dtype=torch.float16)
 model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False);model.eval();need(next(model.parameters()).is_cuda,"MODEL_NOT_CUDA")
 state={"schema":SCHEMA,"complete":False,"enforcer_commit":ENFORCER_COMMIT,
  "enforcer_decode_patch_present":True,"eos_early_termination_guard":True,"selection_sha256":plan_sha,
  "original_v6_protocol_sha256":protocol_sha256(),"gpu_prompt_limit":3200,
  "case_ids":list(IDS),"paired_cases":0,"base_revision":MODEL_COMMIT,
  "v5_adapter_sha256":RAW_ADAPTER_SHA,"expert_gold":None,"training_started":False,
  "weights_changed":False,"trading_allowed":False,"observed":{"matched_base":[],"candidate_v5":[]}}
 dest=a.out_dir/"ACTUAL_V72_EOSGUARD_8_PAIRED_DIAGNOSTICS_NO_GOLD.json";write_json_atomic(dest,state)
 for case,chart,evidence,chart_sha in cases:
  original,meta=prompt_for_case(case,chart);need(meta==evidence,"SOURCE_PROMPT_CHANGED")
  messages,gpu_sha,_,bars=compact_gpu_prompt(tokenizer,original)
  for role,active in (("matched_base",False),("candidate_v5",True)):
   from contextlib import nullcontext
   with (nullcontext() if active else model.disable_adapter()):
    ans=generate(model,tokenizer,messages,case["family"],torch=torch)
   state["observed"][role].append({"id":case["id"],"edition":case["edition"],
    "family":case["family"],"symbol":chart["symbol"],"cutoff_utc":case["cutoff_utc"],
    "chart_sha256":chart_sha,"prompt_sha256":gpu_sha,
    "original_cpu_full_prompt_sha256":evidence["prompt_sha256"],
    "retrieved_rule_ids":evidence["retrieved_rule_ids"],"gpu_visible_bars":bars,**ans})
  state["paired_cases"]+=1;write_json_atomic(dest,state)
  print("V72_EOSGUARD_PAIR",case["id"],state["paired_cases"],flush=True);torch.cuda.empty_cache()
 for b,l in zip(state["observed"]["matched_base"],state["observed"]["candidate_v5"],strict=True):
  need(all(b[k]==l[k] for k in ("id","edition","family","cutoff_utc","chart_sha256","prompt_sha256","retrieved_rule_ids")),"PAIR_INPUT_MISMATCH")
 need(state["paired_cases"]==8,"INCOMPLETE");state["complete"]=True;write_json_atomic(dest,state)
 print("V72_EOSGUARD_8X2_COMPLETE_NO_GOLD",flush=True)
if __name__=="__main__":main()
