"""Two-rank inference only. Reads no books, future bars or outcome labels."""
from __future__ import annotations
from contextlib import nullcontext
from datetime import timedelta
from hashlib import sha256
import json
import os
from pathlib import Path


def predict(questions:Path,adapters:Path,manifest:Path,output:Path):
    import torch
    import torch.distributed as dist
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel
    from .deep_book_research import MODEL,REVISION
    from .qwen4b_blind_evaluation import verified_blind_questions,market_prompt,candidate_first_token_ids,CLASSES
    from .restorative_gpu_research import StopWatcher
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2 or torch.cuda.device_count()!=2:raise ValueError('TWO_ACTUAL_T4_RANKS_REQUIRED')
    torch.cuda.set_device(rank);dist.init_process_group('nccl',timeout=timedelta(minutes=5));watch=StopWatcher();watch.start()
    cases,digest=verified_blind_questions(questions);identity=json.loads(manifest.read_bytes())
    if identity['questions_sha256']!=digest or identity['model_repo']!=MODEL or identity['model_revision']!=REVISION:
        raise ValueError('FROZEN_INFERENCE_INPUTS_REQUIRED')
    if rank==0:
        base=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=['*.json','*.safetensors','*.txt','*.jinja']))
        for name,expected in identity['official_weights_sha256'].items():
            if sha256((base/name).read_bytes()).hexdigest()!=expected:raise ValueError('OFFICIAL_BASE_SHA_CHANGED')
    dist.barrier();base=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    model_base=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,use_safetensors=True,
        dtype=torch.float16,device_map={'':rank},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16))
    model=PeftModel.from_pretrained(model_base,adapters/'candidate',adapter_name='candidate',is_trainable=False,local_files_only=True)
    for name in ('previous40','focused69'):model.load_adapter(adapters/name,adapter_name=name,is_trainable=False,local_files_only=True)
    model.eval();model.config.use_cache=False
    local=[];max_prefix=0
    with torch.inference_mode():
        for index,case in enumerate(cases):
            if index%world!=rank:continue
            if watch.requested.is_set():raise RuntimeError('USER_STOP_OR_CONTROL_UNAVAILABLE')
            messages=[{'role':'system','content':'Исследуй только предоставленные закрытые данные. Не отправляй ордера.'},
                {'role':'user','content':market_prompt(case)}]
            prefix,targets=candidate_first_token_ids(tokenizer,messages)
            if len(prefix)>8192:raise ValueError('FULL_PROMPT_EXCEEDS_CONTEXT_NO_TRUNCATION')
            max_prefix=max(max_prefix,len(prefix));ids=torch.tensor([prefix],device=torch.device('cuda',rank))
            for name in ('base','previous40','focused69','candidate'):
                model.set_adapter('candidate' if name=='base' else name)
                with (model.disable_adapter() if name=='base' else nullcontext()):
                    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1,targets].float()
                if not bool(torch.isfinite(logits).all()):raise ValueError('FINITE_CLASS_LOGITS_REQUIRED')
                local.append({'stage':name,'case_id':case['case_id'],'probabilities':dict(zip(CLASSES,torch.softmax(logits,-1).cpu().tolist()))})
    parts=[None]*world;dist.all_gather_object(parts,local)
    max_t=torch.tensor(max_prefix,device=torch.device('cuda',rank));dist.all_reduce(max_t,op=dist.ReduceOp.MAX)
    if rank==0:
        combined=[x for p in parts for x in p];stages={}
        for name in ('base','previous40','focused69','candidate'):
            lookup={x['case_id']:{k:v for k,v in x.items() if k!='stage'} for x in combined if x['stage']==name}
            if len(lookup)!=30:raise ValueError('EXACT_THIRTY_PREDICTIONS_PER_MODEL_REQUIRED')
            stages[name]=[lookup[c['case_id']] for c in cases]
        receipt={**identity,'mode':'REAL_H1_FOUR_MODEL_CLASS_TOKEN_COMPARISON','world_size':world,
            'gpu_names':[torch.cuda.get_device_name(i) for i in range(2)],'maximum_full_prompt_tokens':int(max_t.item()),
            'scores_are_uncalibrated_normalized_class_token_probabilities':True,'future_outcomes_loaded_in_inference':False,
            'inference_source_commit':os.environ['TRAINING_SOURCE_SHA'],'trading_qualification':False,'execution_allowed':False}
        output.write_text(json.dumps({'receipt':receipt,'stages':stages},ensure_ascii=False,indent=2,allow_nan=False))
        print('FOUR_MODEL_MARKET_PREDICTIONS_SEALED',json.dumps({'models':4,'cases_each':30,'maximum_prompt_tokens':int(max_t.item())}),flush=True)
    dist.barrier();watch.finished.set();dist.destroy_process_group()
