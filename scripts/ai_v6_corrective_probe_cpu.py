"""CPU GGUF diagnostic probe for BASE/V5/V6 on a disjoint corrective development split."""
from __future__ import annotations
import argparse,gc,hashlib,json,os
from decimal import Decimal,InvalidOperation
from pathlib import Path
from typing import Any
from williams_ai.v6_corrective_curriculum import corrective_curriculum,SYSTEM_PROMPT

REQUIRED_OUTPUT_KEYS={"answer","entry_stop","rule_id","calculation","reason","counterexample","uncertainty","cancel_if"}

def _same_value(actual:Any,expected:Any)->bool:
 if actual is None or expected is None:return actual is expected
 if isinstance(actual,(str,int,float)) and isinstance(expected,(str,int,float)):
  try:return Decimal(str(actual))==Decimal(str(expected))
  except (InvalidOperation,ValueError):return str(actual)==str(expected)
 return actual==expected

def grade_response(raw:str,expected_json:str,finish_reason:str)->dict[str,Any]:
 expected=json.loads(expected_json)
 try:actual=json.loads(raw.strip())
 except Exception:
  return {"format_valid":False,"answer_correct":False,"entry_stop_correct":False,"rule_id_correct":False,
   "explanation_complete":False,"fully_correct":False,"actual_answer":None,"actual_entry_stop":None,
   "error":"TRUNCATED" if finish_reason=="length" else "INVALID_JSON"}
 format_valid=isinstance(actual,dict) and set(actual)==REQUIRED_OUTPUT_KEYS and finish_reason!="length"
 if not isinstance(actual,dict):actual={}
 answer_correct=_same_value(actual.get("answer"),expected["answer"])
 entry_stop_correct=_same_value(actual.get("entry_stop"),expected["entry_stop"])
 rule_id_correct=actual.get("rule_id")==expected["rule_id"]
 explanation_complete=all(isinstance(actual.get(k),str) and actual[k].strip() for k in
  ("calculation","reason","counterexample","uncertainty","cancel_if"))
 fully_correct=format_valid and answer_correct and entry_stop_correct and rule_id_correct and explanation_complete
 return {"format_valid":format_valid,"answer_correct":answer_correct,"entry_stop_correct":entry_stop_correct,
  "rule_id_correct":rule_id_correct,"explanation_complete":explanation_complete,"fully_correct":fully_correct,
  "actual_answer":actual.get("answer"),"actual_entry_stop":actual.get("entry_stop"),"actual_rule_id":actual.get("rule_id"),
  "finish_reason":finish_reason}

def _aggregate(records):
 n=len(records)
 def count(k):return sum(bool(x["grade"][k]) for x in records)
 return {"n":n,"answer_correct":count("answer_correct"),"answer_accuracy":count("answer_correct")/n if n else 0,
  "entry_stop_correct":count("entry_stop_correct"),"rule_id_correct":count("rule_id_correct"),
  "explanation_complete":count("explanation_complete"),"format_valid":count("format_valid"),
  "fully_correct":count("fully_correct"),"fully_correct_accuracy":count("fully_correct")/n if n else 0}

def run(args):
 from llama_cpp import Llama
 from transformers import AutoTokenizer
 from scripts.ai_submit_v5_qlora_kaggle import _seal_data
 from williams_ai.real_gguf_acceptance import actual_gguf_sha256
 root=Path(args.output_dir)
 if root.exists() or root.is_symlink():raise ValueError("NEW_OUTPUT_DIRECTORY_REQUIRED")
 root.mkdir(parents=True,exist_ok=False)
 rows,manifest=corrective_curriculum()
 if manifest["dataset_sha256"]!=args.expected_dataset_sha256:raise ValueError("CORRECTIVE_DATASET_SHA_MISMATCH")
 dev=[r for r in rows if r["split"]=="development"]
 if len(dev)!=32:raise ValueError("DEVELOPMENT_SPLIT_INVALID")
 base=Path(args.base_gguf);v5=Path(args.v5_lora_gguf);v6=Path(args.v6_lora_gguf)
 actual_gguf_sha256(base,args.base_sha256)
 for path in (v5,v6):
  if not path.is_file() or path.open("rb").read(4)!=b"GGUF":raise ValueError("LORA_GGUF_INVALID")
 tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_dir,local_files_only=True,trust_remote_code=False)
 secret=os.environ.pop(args.secret_env,"")
 if len(secret)<20:raise ValueError("KAGGLE_SECRET_MISSING")
 summaries={}
 for name,lora in (("BASE",None),("V5",v5),("V6",v6)):
  kwargs={"model_path":str(base),"n_ctx":2048,"n_gpu_layers":0,"verbose":False}
  if lora is not None:kwargs.update(lora_path=str(lora),lora_scale=1.0)
  model=Llama(**kwargs);records=[]
  for row in dev:
   messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":row["prompt"]}]
   prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
   token_ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
   if model.tokenize(prompt.encode(),add_bos=False,special=True)!=token_ids:raise ValueError("TOKENIZER_GGUF_PARITY_MISMATCH")
   completion=model.create_completion(prompt=token_ids,max_tokens=256,temperature=0.0,top_p=1.0,seed=20261125)
   choice=completion["choices"][0];raw=choice["text"]
   records.append({"case_id":row["case_id"],"family":row["family"],"edition":row["edition"],"rule_id":row["rule_id"],
    "prompt_sha256":hashlib.sha256(row["prompt"].encode()).hexdigest(),"expected":json.loads(row["answer"]),
    "grade":grade_response(raw,row["answer"],choice.get("finish_reason","")),"raw":raw})
  summaries[name]={"overall":_aggregate(records),
   "by_family":{fam:_aggregate([r for r in records if r["family"]==fam]) for fam in sorted({r["family"] for r in records})},
   "by_edition":{ed:_aggregate([r for r in records if r["edition"]==ed]) for ed in ("TC1","NTD","TC2","COURSE")},
   "adapter_gguf_sha256":hashlib.sha256(lora.read_bytes()).hexdigest() if lora else None}
  sealed=_seal_data(json.dumps(records,ensure_ascii=False,separators=(",",":")).encode(),secret)
  (root/(name.lower()+"-responses.json.sealed")).write_bytes(sealed)
  print("WILLIAMS_AI_CORRECTIVE_CPU_PROBE",name,json.dumps(summaries[name],sort_keys=True),flush=True)
  model.close();del model;gc.collect()
 report={"schema":"WILLIAMS_AI_CORRECTIVE_CPU_PROBE_V1","dataset_manifest":manifest,"models":summaries,
  "base_gguf_sha256":args.base_sha256,"development_cases":len(dev),"decoder":"greedy_llama_cpp_equal_conditions",
  "max_new_tokens":256,"n_ctx":2048,"closed_book":True,"independent_final_exam":False,
  "historical_chart_exam":False,"new_optimizer_steps":0,"execution_allowed":False,
  "qualification_claim":False,"responses_encrypted":True}
 (root/"aggregate.json").write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2))
 print("WILLIAMS_AI_CORRECTIVE_CPU_PROBE_COMPLETE",json.dumps({"development_cases":len(dev),
  "dataset_sha256":manifest["dataset_sha256"],"responses_plaintext_emitted":False},sort_keys=True),flush=True)

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--base-gguf",required=True);p.add_argument("--base-sha256",required=True)
 p.add_argument("--tokenizer-dir",required=True);p.add_argument("--v5-lora-gguf",required=True)
 p.add_argument("--v6-lora-gguf",required=True);p.add_argument("--expected-dataset-sha256",required=True)
 p.add_argument("--output-dir",required=True);p.add_argument("--secret-env",default="KAGGLE_API_TOKEN")
 run(p.parse_args())

if __name__=="__main__":main()
