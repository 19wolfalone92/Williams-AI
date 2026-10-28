"""Evaluate our completed adapter on frozen historical cases; no trading."""
from __future__ import annotations
import json
from hashlib import sha256
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ai_run_restorative_kaggle import packed,safe_submission_error
from williams_ai.restorative_book_practice import INITIAL_SHA,PREVIOUS40_SHA
from williams_ai.deep_book_research import MODEL,REVISION
from williams_ai.spread_market_exam import frozen_market_cohort,grade_market_comparison
from williams_ai.frozen_source_test import checked_test_payload,recompute_source_scores,grade_source_test
from scripts.ai_sealed_source_transport import open_envelope
from williams_ai.direct_mode_transfer import direct_test_cases,direct_design,select_development_mode,grade_direct_test,transfer_gate
from williams_ai.restorative_book_practice import restorative_cases
from williams_ai.contrastive_book_practice import question_rows,questions_digest,score_responses
from williams_ai.lora_integrity import validation_admission

RUN_SLUG='williams-ai-spread-market30-20261028'
TRAINING_SOURCE='4b5590dde578e97adbb964b35dd4d8ac43f94fdd'


def worker_source(revision,qbytes,config,identity,source_test_payload,direct_dev_bytes,direct_test_bytes):
    if not re.fullmatch('[a-f0-9]{40}',revision):raise ValueError('PINNED_CODE_REQUIRED')
    template='''import base64,hashlib,json,lzma,os,shutil,subprocess,sys
from pathlib import Path
from urllib.request import urlopen
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
os.environ['NCCL_P2P_DISABLE']='1';os.environ['NCCL_IB_DISABLE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['TRAINING_SOURCE_SHA']=__REVISION__
repo=Path('/tmp/frozen-market-inference-source')
subprocess.run(['git','clone','-q','https://github.com/19wolfalone92/Williams-AI.git',str(repo)],check=True)
subprocess.run(['git','checkout','-q',__REVISION__],cwd=repo,check=True)
shutil.rmtree(repo/'examples')
subprocess.run([sys.executable,'-m','pip','install','--quiet','--disable-pip-version-check','transformers==4.57.1','peft==0.17.1','accelerate==1.10.1','bitsandbytes==0.48.1','safetensors==0.6.2'],check=True)
questions=Path('/tmp/market_questions_only.jsonl');questions.write_bytes(lzma.decompress(base64.b64decode(__QUESTIONS__)))
source_test=Path('/tmp/frozen_original_source_test_only.json');source_test.write_bytes(lzma.decompress(base64.b64decode(__SOURCE_TEST__)));source_test.chmod(0o600)
direct_dev=Path('/tmp/direct_development_questions_only.json');direct_dev.write_bytes(lzma.decompress(base64.b64decode(__DIRECT_DEV__)))
direct_test=Path('/tmp/direct_test_questions_only.json');direct_test.write_bytes(lzma.decompress(base64.b64decode(__DIRECT_TEST__)))
identity=__IDENTITY__;manifest=Path('/tmp/inference_identity.json');manifest.write_text(json.dumps(identity))
if hashlib.sha256(questions.read_bytes()).hexdigest()!=identity['questions_sha256']:raise RuntimeError('FROZEN_QUESTIONS_CHANGED')
adapters=Path('/tmp/verified_own_adapters');adapters.mkdir(mode=0o700)
for stage,files in __CONFIG__.items():
    if stage not in ('candidate','focused69','previous40','restorative768') or set(files)!={'adapter_model.safetensors','adapter_config.json'}:raise RuntimeError('SAFE_KNOWN_ADAPTER_FILES_REQUIRED')
    dest=adapters/stage;dest.mkdir(mode=0o700)
    for name,meta in files.items():
        raw=urlopen(meta['url'],timeout=240).read()
        if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise RuntimeError('OWN_ADAPTER_SHA_CHANGED')
        (dest/name).write_bytes(raw)
out=Path('/kaggle/working/spread_market_predictions.json')
entry=Path('/tmp/run_spread_inference.py')
entry.write_text('from pathlib import Path\\nimport sys\\nsys.path.insert(0,'+repr(str(repo))+')\\nfrom williams_ai.spread_market_gpu import predict\\npredict(Path('+repr(str(questions))+'),Path('+repr(str(adapters))+'),Path('+repr(str(manifest))+'),Path('+repr(str(out))+'),Path('+repr(str(source_test))+'),Path('+repr(str(direct_dev))+'),Path('+repr(str(direct_test))+'))\\n')
try:subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(entry)],check=True)
except Exception as e:
    Path('/kaggle/working/spread_market_failure.json').write_text(json.dumps({'error_type':type(e).__name__,'complete_comparison':False,'execution_allowed':False}))
    raise
finally:source_test.unlink(missing_ok=True)
'''
    return (template.replace('__REVISION__',repr(revision)).replace('__QUESTIONS__',repr(packed(qbytes))).replace('__CONFIG__',repr(config))
        .replace('__IDENTITY__',repr(identity)).replace('__SOURCE_TEST__',repr(packed(source_test_payload)))
        .replace('__DIRECT_DEV__',repr(packed(direct_dev_bytes))).replace('__DIRECT_TEST__',repr(packed(direct_test_bytes))))


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    token=os.environ['KAGGLE_API_TOKEN'];owner=get_username();api=KaggleApi();api.authenticate()
    revision=os.environ['GITHUB_SHA']
    if 'complete' not in str(api.kernels_status(owner+'/williams-ai-restorative-2t4-20261028').status).lower():
        raise ValueError('COMPLETED_OWN_TRAINING_REQUIRED_BEFORE_MARKET_TEST')
    def files(slug):
        found={};page=None;seen=set()
        with api.build_kaggle_client() as client:
            while True:
                q=ApiListKernelSessionOutputRequest();q.user_name=owner;q.kernel_slug=slug
                if page:q.page_token=page
                result=client.kernels.kernels_api_client.list_kernel_session_output(q)
                found.update({f.file_name:f for f in result.files});page=getattr(result,'next_page_token',None)
                if not page:break
                if page in seen:raise ValueError('OUTPUT_PAGINATION_LOOP')
                seen.add(page)
        return found
    recent=files('williams-ai-restorative-2t4-20261028');old=files('williams-ai-focused-2t4-20261027')
    raw=urlopen(recent['restorative_qwen4b/research_receipt.json'].url,timeout=60).read();receipt=json.loads(raw)
    if (receipt['source_commit']!=TRAINING_SOURCE or receipt['initial_adapter_sha256']!=INITIAL_SHA or receipt['model_repo']!=MODEL
        or receipt['model_revision']!=REVISION or receipt['complete_training_and_test'] is not True or receipt['execution_allowed'] is not False):
        raise ValueError('VERIFIED_COMPLETED_RESTORATIVE_RESEARCH_REQUIRED')
    marker=json.loads(Path('experiments/restorative-market-start.json').read_bytes())
    if (marker.get('training_source_commit')!=TRAINING_SOURCE or marker.get('candidate_adapter_sha256')!=receipt['candidate_adapter_sha256']
        or marker.get('start_frozen_inference') is not True or marker.get('execution_allowed') is not False):
        raise ValueError('FROZEN_SELECTED_CANDIDATE_MARKER_REQUIRED')
    test_payload=open_envelope(json.loads(Path('experiments/private-frozen-source-test-20261028.sealed.json').read_bytes()),token)
    _,test_audit=checked_test_payload(json.loads(test_payload),expected_manifest_sha256=receipt['source_audit']['manifest_sha256'])
    test_design=json.loads(Path('experiments/frozen-source-test-20261028-design.json').read_bytes())
    if sha256(test_payload).hexdigest()!=test_design['payload_sha256'] or marker.get('source_test_payload_sha256')!=test_design['payload_sha256']:
        raise ValueError('PRECOMMITTED_PRIVATE_SOURCE_TEST_CHANGED')
    questions,keys,qbytes,seal=frozen_market_cohort(Path('examples/ai_binance_15pair_h1_prior_holdout_20260902.json'))
    pinned=json.loads(Path('experiments/spread-market-20261028-design.json').read_bytes())
    if seal!=pinned:raise ValueError('PRECOMMITTED_MARKET_DESIGN_CHANGED')
    if direct_design()!=json.loads(Path('experiments/direct-transfer-20261028-design.json').read_bytes()):raise ValueError('PRECOMMITTED_DIRECT_DESIGN_CHANGED')
    direct_cases=direct_test_cases()
    direct_dev_bytes=json.dumps(question_rows(restorative_cases('validation')),ensure_ascii=False).encode()
    direct_test_bytes=json.dumps(question_rows(direct_cases),ensure_ascii=False).encode()
    config={};expected={'candidate':receipt['candidate_adapter_sha256'],'focused69':INITIAL_SHA,'previous40':PREVIOUS40_SHA}
    for stage,mapping,prefix in (('candidate',recent,'restorative_qwen4b/candidate_adapter/'),
        ('focused69',recent,'restorative_qwen4b/source_adapter/'),('previous40',old,'focused_qwen4b/source_adapter/')):
        config[stage]={}
        for name in ('adapter_config.json','adapter_model.safetensors'):
            item=mapping[prefix+name]
            if not item.url.startswith('https://'):raise ValueError('OWN_HTTPS_REFERENCE_REQUIRED')
            content=urlopen(item.url,timeout=240).read();digest=sha256(content).hexdigest()
            if name=='adapter_model.safetensors' and digest!=expected[stage]:raise ValueError('KNOWN_OWN_ADAPTER_SHA_REQUIRED')
            config[stage][name]={'url':item.url,'sha256':digest}
    identity={'model_repo':MODEL,'model_revision':REVISION,'official_weights_sha256':receipt['official_weights_sha256'],
        'training_source_commit':TRAINING_SOURCE,'training_receipt_sha256':sha256(raw).hexdigest(),
        'adapter_sha256':expected,'questions_sha256':seal['questions_file_sha256'],
        'source_manifest_sha256':test_audit['manifest_sha256'],'source_test_payload_sha256':sha256(test_payload).hexdigest(),
        'direct_development_file_sha256':sha256(direct_dev_bytes).hexdigest(),'direct_test_file_sha256':sha256(direct_test_bytes).hexdigest()}
    worker=worker_source(revision,qbytes,config,identity,test_payload,direct_dev_bytes,direct_test_bytes)
    if token in worker or 'WITHHELD_OUTCOME_KEYS' in worker or len(worker)>3_000_000:raise ValueError('NO_SECRETS_OR_OUTCOME_KEYS_IN_WORKER')
    compile(worker,'frozen_market_worker','exec')
    metadata={'id':owner+'/'+RUN_SLUG,'title':RUN_SLUG,'code_file':'worker.py','language':'python','kernel_type':'script',
        'is_private':'true','enable_gpu':'true','enable_internet':'true','machine_shape':'NvidiaTeslaT4','dataset_sources':[],
        'model_sources':[],'competition_sources':[],'kernel_sources':[]}
    with tempfile.TemporaryDirectory() as td:
        p=Path(td);(p/'worker.py').write_text(worker);(p/'worker.py').chmod(0o600);(p/'kernel-metadata.json').write_text(json.dumps(metadata))
        submitted=subprocess.run(['kaggle','kernels','push','-p',td,'--accelerator','NvidiaTeslaT4','--timeout','6000'],capture_output=True,text=True,timeout=180)
        if submitted.returncode:
            print(safe_submission_error(submitted.stdout+submitted.stderr,token),flush=True);raise RuntimeError('PRIVATE_MARKET_INFERENCE_REJECTED')
    state='UNKNOWN'
    for i in range(360):
        state=str(api.kernels_status(owner+'/'+RUN_SLUG).status).upper()
        if i%10==0:print('OWN_MARKET_INFERENCE_STATE',state,flush=True)
        if any(s in state for s in ('COMPLETE','ERROR','CANCEL')):break
        time.sleep(20)
    dest=Path(os.environ['RUNNER_TEMP'])/'spread_market_results';dest.mkdir(mode=0o700)
    result=subprocess.run(['kaggle','kernels','output',owner+'/'+RUN_SLUG,'-p',str(dest),'--file-pattern',r'.*spread_market_(predictions|failure)[.]json$'],capture_output=True,text=True,timeout=180)
    if result.returncode or 'COMPLETE' not in state:raise RuntimeError('MARKET_EXAM_INCOMPLETE_NO_SCORE_CLAIM')
    prediction_file=dest/'spread_market_predictions.json';predictions=json.loads(prediction_file.read_bytes())
    report=predictions['receipt']
    if any(report[k]!=v for k,v in identity.items()) or report['inference_source_commit']!=revision or report['future_outcomes_loaded_in_inference'] is not False:
        raise ValueError('SEALED_MARKET_PREDICTION_IDENTITY_MISMATCH')
    if report['free_generation_protocol']!=seal['free_generation_protocol']:raise ValueError('FROZEN_FREE_GENERATION_PROTOCOL_CHANGED')
    if report['class_score_method']!=seal['class_score_method']:raise ValueError('FROZEN_COMPLETE_CLASS_SCORING_CHANGED')
    if report['direct_rule_protocol']!={'mode':'direct','maximum_new_tokens':128,'seconds_per_response':10,'expected_keys_on_gpu':False}:
        raise ValueError('FROZEN_DIRECT_RULE_PROTOCOL_CHANGED')
    if set(predictions['stages'])!={'base','previous40','focused69','candidate'}:raise ValueError('ALL_FOUR_MODELS_REQUIRED')
    actual_audit=report['source_test_audit']
    if any(actual_audit[k]!=v for k,v in test_audit.items()):raise ValueError('SOURCE_TEST_AUDIT_MISMATCH')
    scores=recompute_source_scores(predictions['source_test_window_losses'],actual_audit['windows_per_book'])
    if scores!=predictions['source_test_scores']:raise ValueError('CPU_RECOMPUTED_SOURCE_TEST_LOSS_MISMATCH')
    direct_dev_grade=score_responses(restorative_cases('validation'),predictions['direct_rule_development'])
    mode_selection=select_development_mode(receipt['final_thinking_development'],direct_dev_grade)
    (dest/'development_mode_selection.json').write_text(json.dumps(mode_selection,ensure_ascii=False,indent=2))
    # Mode has now been fixed using development only, before grading any new
    # direct holdout answers or market outcomes. Frozen weights never change.
    grade=grade_market_comparison(keys,predictions['stages']);grade['seal']=seal;grade['predictions_file_sha256']=sha256(prediction_file.read_bytes()).hexdigest()
    grade['direct_rule_transfer']=grade_direct_test(direct_cases,predictions['direct_rule_test'])
    grade['direct_development_grade']=direct_dev_grade;grade['generation_mode_selection']=mode_selection
    grade['source_test']=grade_source_test(scores)
    thinking=json.loads(urlopen(recent['restorative_qwen4b/restorative_test_precommit.json'].url,timeout=60).read())
    old_cases=restorative_cases('test')
    if thinking['inference_mode']!='thinking' or thinking['questions_sha256']!=questions_digest(question_rows(old_cases)):
        raise ValueError('ORIGINAL_THINKING_EXAM_PRECOMMIT_REQUIRED')
    thinking_grades={s:score_responses(old_cases,rows) for s,rows in thinking['stages'].items()}
    grade['original_thinking_exam']={'stages':thinking_grades,'transfer_gate_passed':transfer_gate(thinking_grades),
        'preserved_original_exam_not_new_retest':True,'same_response_seconds_each_model':thinking['same_response_time_budget_seconds_each_model']}
    text_gates={}
    for name,before in (('final_source_validation_gate','source_baseline_loss'),('final_practice_validation_gate','practice_baseline_loss'),
        ('initial_focused_source_gate','initial_focused_source_loss'),('initial_focused_practice_gate','initial_focused_practice_loss')):
        after='source_final_loss' if 'source' in name else 'practice_final_loss'
        text_gates[name]=validation_admission(receipt[before],receipt[after])
        if text_gates[name]!=receipt[name]:raise ValueError('RECOMPUTED_TRAIN_VALIDATION_GATE_MISMATCH')
    selected_transfer=grade['direct_rule_transfer']['transfer_gate_passed'] if mode_selection['selected_mode']=='direct' else grade['original_thinking_exam']['transfer_gate_passed']
    grade['combined_research_admission']={'selected_mode':mode_selection['selected_mode'],
        'accepted_for_further_research':selected_transfer and all(g['accepted'] for g in text_gates.values()) and grade['source_test']['source_test_gate_passed'],
        'selected_mode_transfer_gate_passed':selected_transfer,'source_and_practice_validation_gates':text_gates,
        'original_source_test_gate_passed':grade['source_test']['source_test_gate_passed'],
        'market_results_did_not_select_mode_or_weights':True,'trading_qualification':False,'execution_allowed':False}
    (dest/'spread_market_grade.json').write_text(json.dumps(grade,ensure_ascii=False,indent=2,allow_nan=False))
    (dest/'withheld_market_outcomes_cpu_only.json').write_text(json.dumps(keys,ensure_ascii=False,indent=2))
    (dest/'spread_market_seal.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2))
    print('ACTUAL_FOUR_MODEL_MARKET_GRADE',json.dumps(grade,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
