"""Evaluate five frozen variants; no training or trading in this worker."""
from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ai_run_spread_market_kaggle import worker_source
from scripts.ai_run_restorative_kaggle import safe_submission_error
from scripts.ai_sealed_source_transport import open_envelope
from williams_ai.source_consolidation import START_SHA,RUN_SLUG as TRAIN_SLUG,fresh_test_cases
from williams_ai.restorative_book_practice import INITIAL_SHA,PREVIOUS40_SHA,restorative_cases
from williams_ai.deep_book_research import MODEL,REVISION
from williams_ai.contrastive_book_practice import question_rows,questions_digest,score_responses
from williams_ai.frozen_source_test import checked_test_payload,recompute_source_scores,grade_source_test
from williams_ai.spread_market_exam import frozen_market_cohort,grade_market_comparison
from williams_ai.direct_mode_transfer import grade_direct_test
from williams_ai.lora_integrity import validation_admission

TRAINING_SOURCE='29ef792c258d1faf5891106e114f322bdb7acc11'
RUN_SLUG='williams-ai-source-final-5models-20261028'
STAGES=('base','previous40','focused69','restorative768','candidate')


def training_text_gates(r):
    gates={}
    for name,before,after in (
        ('final_source_validation_gate','source_baseline_loss','source_final_loss'),('final_practice_validation_gate','practice_baseline_loss','practice_final_loss'),
        ('initial_focused_source_gate','initial_focused_source_loss','source_final_loss'),('initial_focused_practice_gate','initial_focused_practice_loss','practice_final_loss'),
        ('parent_source_validation_gate','parent_source_loss','source_final_loss'),('parent_practice_validation_gate','parent_practice_loss','practice_final_loss')):
        gates[name]=validation_admission(r[before],r[after])
        if gates[name]!=r[name]:raise ValueError('SIX_TRAINING_TEXT_GATES_MUST_RECOMPUTE')
    return gates


def combined_admission(direct,source,gates):
    return {'selected_mode':'direct','accepted_for_further_research':direct['transfer_gate_passed'] and source['source_test_gate_passed'] and all(g['accepted'] for g in gates.values()),
        'selected_mode_transfer_gate_passed':direct['transfer_gate_passed'],'source_and_practice_validation_gates':gates,
        'original_source_test_gate_passed':source['source_test_gate_passed'],'market_results_did_not_select_mode_or_weights':True,
        'trading_qualification':False,'execution_allowed':False}


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    token=os.environ['KAGGLE_API_TOKEN'];owner=get_username();api=KaggleApi();api.authenticate();revision=os.environ['GITHUB_SHA']
    if 'complete' not in str(api.kernels_status(owner+'/'+TRAIN_SLUG).status).lower():raise ValueError('COMPLETED_SOURCE_CONSOLIDATION_REQUIRED')
    def files(slug):
        found={};page=None;seen=set()
        with api.build_kaggle_client() as client:
            while True:
                q=ApiListKernelSessionOutputRequest();q.user_name=owner;q.kernel_slug=slug
                if page:q.page_token=page
                result=client.kernels.kernels_api_client.list_kernel_session_output(q)
                found.update({f.file_name:f for f in result.files});page=getattr(result,'next_page_token',None)
                if not page:break
                if page in seen:raise ValueError('OWN_OUTPUT_PAGINATION_LOOP')
                seen.add(page)
        return found
    current=files(TRAIN_SLUG);restorative=files('williams-ai-restorative-2t4-20261028');old=files('williams-ai-focused-2t4-20261027')
    raw=urlopen(current['restorative_qwen4b/research_receipt.json'].url,timeout=60).read();receipt=json.loads(raw)
    if (receipt['source_commit']!=TRAINING_SOURCE or receipt['initial_adapter_sha256']!=START_SHA or receipt['model_repo']!=MODEL
        or receipt['model_revision']!=REVISION or receipt['complete_source_consolidation'] is not True
        or receipt['complete_training_and_test'] is not False or receipt['execution_allowed'] is not False):raise ValueError('VERIFIED_TRAIN_ONLY_CANDIDATE_REQUIRED')
    marker=json.loads(Path('experiments/source-consolidation-final-start.json').read_bytes())
    if (marker.get('training_source_commit')!=TRAINING_SOURCE or marker.get('candidate_adapter_sha256')!=receipt['candidate_adapter_sha256']
        or marker.get('start_frozen_inference') is not True or marker.get('execution_allowed') is not False):raise ValueError('FROZEN_FIVE_MODEL_MARKER_REQUIRED')
    design=json.loads(Path('experiments/source-consolidation-final-20261028-design.json').read_bytes());cases=fresh_test_cases()
    if (design['fresh_questions_sha256']!=questions_digest(question_rows(cases)) or design['models']!=list(STAGES)
        or design['training_source_commit']!=TRAINING_SOURCE):raise ValueError('PRECOMMITTED_FIVE_MODEL_DESIGN_REQUIRED')
    payload=open_envelope(json.loads(Path('experiments/private-frozen-source-test-20261028.sealed.json').read_bytes()),token)
    _,test_audit=checked_test_payload(json.loads(payload),expected_manifest_sha256=receipt['source_audit']['manifest_sha256'])
    if sha256(payload).hexdigest()!=design['source_test_payload_sha256'] or marker.get('source_test_payload_sha256')!=design['source_test_payload_sha256']:
        raise ValueError('FROZEN_ORIGINAL_SOURCE_TEST_CHANGED')
    questions,keys,qbytes,seal=frozen_market_cohort(Path('examples/ai_binance_15pair_h1_prior_holdout_20260902.json'))
    if seal!=json.loads(Path('experiments/spread-market-20261028-design.json').read_bytes()):raise ValueError('FROZEN_MARKET_COHORT_CHANGED')
    dev_bytes=json.dumps(question_rows(restorative_cases('validation')),ensure_ascii=False).encode();test_bytes=json.dumps(question_rows(cases),ensure_ascii=False).encode()
    expected={'candidate':receipt['candidate_adapter_sha256'],'restorative768':START_SHA,'focused69':INITIAL_SHA,'previous40':PREVIOUS40_SHA};config={}
    for stage,lookup,prefix in (('candidate',current,'restorative_qwen4b/candidate_adapter/'),('restorative768',current,'restorative_qwen4b/source_adapter/'),
        ('focused69',restorative,'restorative_qwen4b/source_adapter/'),('previous40',old,'focused_qwen4b/source_adapter/')):
        config[stage]={}
        for name in ('adapter_config.json','adapter_model.safetensors'):
            item=lookup[prefix+name]
            if not item.url.startswith('https://'):raise ValueError('OWN_HTTPS_ADAPTER_REFERENCE_REQUIRED')
            content=urlopen(item.url,timeout=240).read();digest=sha256(content).hexdigest()
            if name=='adapter_model.safetensors' and digest!=expected[stage]:raise ValueError('EXACT_FIVE_MODEL_ADAPTER_SHA_REQUIRED')
            config[stage][name]={'url':item.url,'sha256':digest}
    identity={'model_repo':MODEL,'model_revision':REVISION,'official_weights_sha256':receipt['official_weights_sha256'],
        'training_source_commit':TRAINING_SOURCE,'training_receipt_sha256':sha256(raw).hexdigest(),'adapter_sha256':expected,
        'questions_sha256':seal['questions_file_sha256'],'source_manifest_sha256':test_audit['manifest_sha256'],
        'source_test_payload_sha256':sha256(payload).hexdigest(),'direct_development_file_sha256':sha256(dev_bytes).hexdigest(),
        'direct_test_file_sha256':sha256(test_bytes).hexdigest(),'source_consolidation_final':True}
    worker=worker_source(revision,qbytes,config,identity,payload,dev_bytes,test_bytes)
    if token in worker or len(worker)>3_000_000:raise ValueError('NO_SECRET_IN_PRIVATE_FINAL_WORKER')
    compile(worker,'five_model_final_worker','exec')
    metadata={'id':owner+'/'+RUN_SLUG,'title':RUN_SLUG,'code_file':'worker.py','language':'python','kernel_type':'script','is_private':'true',
        'enable_gpu':'true','enable_internet':'true','machine_shape':'NvidiaTeslaT4','dataset_sources':[],'model_sources':[],'competition_sources':[],'kernel_sources':[]}
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/'worker.py').write_text(worker);(root/'worker.py').chmod(0o600);(root/'kernel-metadata.json').write_text(json.dumps(metadata))
        result=subprocess.run(['kaggle','kernels','push','-p',td,'--accelerator','NvidiaTeslaT4','--timeout','6000'],capture_output=True,text=True,timeout=180)
        if result.returncode:print(safe_submission_error(result.stdout+result.stderr,token),flush=True);raise RuntimeError('PRIVATE_FIVE_MODEL_SUBMISSION_REJECTED')
    state='UNKNOWN'
    for i in range(360):
        state=str(api.kernels_status(owner+'/'+RUN_SLUG).status).upper()
        if i%15==0:print('OWN_FROZEN_FIVE_MODEL_STATE',state,flush=True)
        if any(s in state for s in ('COMPLETE','ERROR','CANCEL')):break
        time.sleep(20)
    dest=Path(os.environ['RUNNER_TEMP'])/'source_consolidation_final_results';dest.mkdir(mode=0o700)
    result=subprocess.run(['kaggle','kernels','output',owner+'/'+RUN_SLUG,'-p',str(dest),'--file-pattern',r'.*spread_market_(predictions|failure)[.]json$'],capture_output=True,text=True,timeout=180)
    if result.returncode or 'COMPLETE' not in state:raise RuntimeError('FIVE_MODEL_EXAM_INCOMPLETE_NO_QUALIFICATION')
    prediction_file=dest/'spread_market_predictions.json';predictions=json.loads(prediction_file.read_bytes());report=predictions['receipt']
    if (any(report[k]!=v for k,v in identity.items()) or report['inference_source_commit']!=revision
        or report['future_outcomes_loaded_in_inference'] is not False or set(predictions['stages'])!=set(STAGES)):
        raise ValueError('IMMUTABLE_FIVE_MODEL_PREDICTIONS_REQUIRED')
    if (report['direct_rule_protocol']!={'mode':'direct','maximum_new_tokens':128,'seconds_per_response':10,'expected_keys_on_gpu':False}
        or report['free_generation_protocol']!=seal['free_generation_protocol'] or report['class_score_method']!=seal['class_score_method']):
        raise ValueError('FROZEN_FIVE_MODEL_RESPONSE_PROTOCOL_CHANGED')
    if any(report['source_test_audit'][k]!=v for k,v in test_audit.items()):raise ValueError('FROZEN_SOURCE_TEST_AUDIT_CHANGED')
    scores=recompute_source_scores(predictions['source_test_window_losses'],report['source_test_audit']['windows_per_book'],include_restorative_parent=True)
    if scores!=predictions['source_test_scores']:raise ValueError('FIVE_MODEL_SOURCE_WINDOW_RECOMPUTATION_FAILED')
    grade=grade_market_comparison(keys,predictions['stages']);grade['seal']=seal;grade['predictions_file_sha256']=sha256(prediction_file.read_bytes()).hexdigest()
    grade['direct_rule_transfer']=grade_direct_test(cases,predictions['direct_rule_test'],include_restorative_parent=True)
    grade['direct_development_grade']=score_responses(restorative_cases('validation'),predictions['direct_rule_development'])
    grade['source_test']=grade_source_test(scores,include_restorative_parent=True)
    grade['combined_research_admission']=combined_admission(grade['direct_rule_transfer'],grade['source_test'],training_text_gates(receipt))
    grade['mode_was_precommitted_direct_before_final_test']=True
    (dest/'spread_market_grade.json').write_text(json.dumps(grade,ensure_ascii=False,indent=2,allow_nan=False))
    (dest/'withheld_market_outcomes_cpu_only.json').write_text(json.dumps(keys,ensure_ascii=False,indent=2))
    (dest/'spread_market_seal.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2))
    (dest/'development_mode_selection.json').write_text(json.dumps({'selected_mode':'direct','precommitted_before_training':True,'test_scores_used_for_selection':False,'execution_allowed':False}))
    print('ACTUAL_FROZEN_FIVE_MODEL_FINAL_GRADE',json.dumps(grade,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
