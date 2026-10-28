"""Two-rank inference only. Reads no books, future bars or outcome labels."""
from __future__ import annotations
from contextlib import nullcontext
from datetime import timedelta
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time


def complete_class_suffixes(tokenizer,messages):
    from .qwen4b_blind_evaluation import candidate_first_token_ids,CLASSES
    prefix,_=candidate_first_token_ids(tokenizer,messages);suffixes=[]
    for label in CLASSES:
        full=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':label}],tokenize=True,
            add_generation_prompt=False,enable_thinking=False)
        tail=full[len(prefix):]
        if full[:len(prefix)]!=prefix or tokenizer.eos_token_id not in tail:raise ValueError('COMPLETE_CLASS_TEMPLATE_REQUIRED')
        tail=tail[:tail.index(tokenizer.eos_token_id)+1]
        if tokenizer.decode(tail[:-1],skip_special_tokens=True)!=label:raise ValueError('EXACT_FULL_CLASS_VERBALIZER_REQUIRED')
        suffixes.append(tail)
    return prefix,suffixes


def predict(questions:Path,adapters:Path,manifest:Path,output:Path,source_test_file:Path):
    from importlib.metadata import version
    import torch
    import torch.distributed as dist
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,StoppingCriteria,StoppingCriteriaList
    from peft import PeftModel
    from .deep_book_research import MODEL,REVISION
    from .qwen4b_blind_evaluation import verified_blind_questions,market_prompt,CLASSES
    from .restorative_gpu_research import StopWatcher
    from .frozen_source_test import checked_test_payload,recompute_source_scores
    from .private_book_source import tokenize_source_pages
    from .deep_book_research import EDITIONS
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2 or torch.cuda.device_count()!=2:raise ValueError('TWO_ACTUAL_T4_RANKS_REQUIRED')
    torch.cuda.set_device(rank);dist.init_process_group('nccl',timeout=timedelta(minutes=5));watch=StopWatcher();watch.start()
    cases,digest=verified_blind_questions(questions);identity=json.loads(manifest.read_bytes())
    if sha256(source_test_file.read_bytes()).hexdigest()!=identity['source_test_payload_sha256']:raise ValueError('FROZEN_SOURCE_TEST_PAYLOAD_CHANGED')
    test_pages,test_audit=checked_test_payload(json.loads(source_test_file.read_bytes()),expected_manifest_sha256=identity['source_manifest_sha256'])
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
    class Deadline(StoppingCriteria):
        def __init__(self,deadline):self.deadline=deadline
        def __call__(self,input_ids,scores,**kwargs):return watch.requested.is_set() or time.monotonic()>=self.deadline
    with torch.inference_mode():
        for index,case in enumerate(cases):
            if index%world!=rank:continue
            if watch.requested.is_set():raise RuntimeError('USER_STOP_OR_CONTROL_UNAVAILABLE')
            messages=[{'role':'system','content':'Исследуй только предоставленные закрытые данные. Не отправляй ордера.'},
                {'role':'user','content':market_prompt(case)}]
            prefix,suffixes=complete_class_suffixes(tokenizer,messages)
            if len(prefix)>8192:raise ValueError('FULL_PROMPT_EXCEEDS_CONTEXT_NO_TRUNCATION')
            max_prefix=max(max_prefix,len(prefix));ids=torch.tensor([prefix],device=torch.device('cuda',rank))
            for name in ('base','previous40','focused69','candidate'):
                model.set_adapter('candidate' if name=='base' else name)
                with (model.disable_adapter() if name=='base' else nullcontext()):
                    label_scores=[]
                    for suffix in suffixes:
                        full=torch.tensor([prefix+suffix],device=ids.device)
                        result=model(input_ids=full,attention_mask=torch.ones_like(full),use_cache=False)
                        selected=result.logits[0,len(prefix)-1:len(prefix)-1+len(suffix)].float()
                        targets=torch.tensor(suffix,device=ids.device)
                        score=torch.log_softmax(selected,-1).gather(-1,targets[:,None]).sum()
                        if not bool(torch.isfinite(score)):raise ValueError('FINITE_COMPLETE_CLASS_LIKELIHOOD_REQUIRED')
                        label_scores.append(float(score));del result,selected,score
                    deadline=time.monotonic()+15
                    generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=64,
                        pad_token_id=tokenizer.eos_token_id,use_cache=True,stopping_criteria=StoppingCriteriaList([Deadline(deadline)]))[0,len(prefix):].tolist()
                local.append({'stage':name,'case_id':case['case_id'],'probabilities':dict(zip(CLASSES,torch.softmax(torch.tensor(label_scores,dtype=torch.float64),-1).tolist())),
                    'generated_response':tokenizer.decode(generated,skip_special_tokens=True).strip(),
                    'generated_complete':bool(generated and generated[-1]==tokenizer.eos_token_id),
                    'generation_time_limit_reached':time.monotonic()>=deadline})
    parts=[None]*world;dist.all_gather_object(parts,local)
    # All market predictions already exist before any original test page is
    # evaluated. There is no optimizer, backward call or source retrieval in
    # the market prompt. Only preselected frozen weights are used.
    test_windows={e:tokenize_source_pages([r for r in test_pages if r['edition']==e],tokenizer,max_tokens=384) for e in EDITIONS}
    counts={e:len(v) for e,v in test_windows.items()};local_test=[]
    for stage in ('base','previous40','focused69','candidate'):
        model.set_adapter('candidate' if stage=='base' else stage)
        with (model.disable_adapter() if stage=='base' else nullcontext()),torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            for edition,rows in test_windows.items():
                for index,row in enumerate(rows):
                    if index%world!=rank:continue
                    if watch.requested.is_set():raise RuntimeError('USER_STOP_DURING_SOURCE_TEST')
                    batch={k:torch.tensor([row[k]],device=torch.device('cuda',rank)) for k in ('input_ids','attention_mask','labels')}
                    loss=float(model(**batch,use_cache=False).loss)
                    if not math.isfinite(loss):raise ValueError('FINITE_ORIGINAL_SOURCE_TEST_LOSS_REQUIRED')
                    local_test.append((stage,edition,index,loss))
    test_parts=[None]*world;dist.all_gather_object(test_parts,local_test)
    window_losses=[r for p in test_parts for r in p];source_scores=recompute_source_scores(window_losses,counts)
    max_t=torch.tensor(max_prefix,device=torch.device('cuda',rank));dist.all_reduce(max_t,op=dist.ReduceOp.MAX)
    if rank==0:
        combined=[x for p in parts for x in p];stages={}
        for name in ('base','previous40','focused69','candidate'):
            lookup={x['case_id']:{k:v for k,v in x.items() if k!='stage'} for x in combined if x['stage']==name}
            if len(lookup)!=30:raise ValueError('EXACT_THIRTY_PREDICTIONS_PER_MODEL_REQUIRED')
            stages[name]=[lookup[c['case_id']] for c in cases]
        receipt={**identity,'mode':'REAL_H1_FOUR_MODEL_COMPLETE_LABEL_AND_FREE_GENERATION_COMPARISON','world_size':world,
            'gpu_names':[torch.cuda.get_device_name(i) for i in range(2)],'maximum_full_prompt_tokens':int(max_t.item()),
            'inference_runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,
                **{name:version(name) for name in ('transformers','peft','accelerate','bitsandbytes','safetensors')}},
            'class_score_method':'normalized_full_label_plus_eos_likelihood','class_probabilities_are_not_empirically_calibrated':True,
            'future_outcomes_loaded_in_inference':False,
            'free_generation_protocol':{'mode':'greedy_non_thinking','maximum_new_tokens':64,'seconds_per_response':15,'strict_one_word_class':True},
            'source_test_audit':{**test_audit,'windows_per_book':counts,'original_test_pages_sent_only_for_final_inference':True},
            'inference_source_commit':os.environ['TRAINING_SOURCE_SHA'],'trading_qualification':False,'execution_allowed':False}
        output.write_text(json.dumps({'receipt':receipt,'stages':stages,'source_test_scores':source_scores,
            'source_test_window_losses':window_losses},ensure_ascii=False,indent=2,allow_nan=False))
        print('FOUR_MODEL_MARKET_PREDICTIONS_SEALED',json.dumps({'models':4,'cases_each':30,'maximum_prompt_tokens':int(max_t.item())}),flush=True)
    dist.barrier();watch.finished.set();dist.destroy_process_group()
