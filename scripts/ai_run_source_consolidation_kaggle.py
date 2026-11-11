"""Train only on frozen train/validation data; do not fetch any final exam."""
from __future__ import annotations
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ai_run_restorative_kaggle import worker_source,safe_submission_error
from scripts.ai_sealed_source_transport import open_envelope
from williams_ai.deep_book_research import checked_payload,MODEL,REVISION
from williams_ai.restorative_gpu_research import checked_baselines,checked_stop_control,CONTROL_URL
from williams_ai.restorative_book_practice import PREVIOUS40_SHA,PREVIOUS_RECEIPT_SHA
from williams_ai.source_consolidation import START_SHA,PARENT_RECEIPT_SHA,RUN_SLUG,design,selection
from williams_ai.lora_integrity import validation_admission
from williams_ai.restorative_book_practice import restorative_cases
from williams_ai.contrastive_book_practice import score_responses


def checked_archive(path:Path,destination:Path,*,source_commit:str,parent_receipt:dict):
    import numpy as np
    from safetensors.numpy import load
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):raise ValueError('SAFE_TRAINED_ARCHIVE_REQUIRED')
        r=json.loads(z.read('research_receipt.json'))
        if (r['mode']!='TWO_T4_SOURCE_CONSOLIDATION_DIRECT_REPLAY_V1' or r['source_commit']!=source_commit
            or r['model_repo']!=MODEL or r['model_revision']!=REVISION or r['initial_adapter_sha256']!=START_SHA
            or r['complete_source_consolidation'] is not True or r['complete_training_and_test'] is not False
            or r['source_consolidation_protocol'] is not True or r['final_test_scores_used_for_training_or_selection'] is not False
            or r['world_size']!=2 or r['execution_allowed'] is not False or r['source_audit']!=parent_receipt['source_audit']
            or r['official_weights_sha256']!=parent_receipt['official_weights_sha256'] or r['new_parameter_test_cases']!=0
            or r['parent_source_loss']!=parent_receipt['source_final_loss'] or r['parent_practice_loss']!=parent_receipt['practice_final_loss']
            or r['parent_restorative_receipt_sha256']!=PARENT_RECEIPT_SHA
            or 'restorative_test_precommit.json' in names):raise ValueError('TRAIN_ONLY_SOURCE_CONSOLIDATION_PROVENANCE_REQUIRED')
        data={s:z.read(s+'_adapter/adapter_model.safetensors') for s in ('source','candidate')}
        tensors={s:load(b) for s,b in data.items()}
        if sha256(data['source']).hexdigest()!=START_SHA:raise ValueError('EXACT_RESTORATIVE768_INITIAL_WEIGHTS_REQUIRED')
        if any(sha256(data[s]).hexdigest()!=r[s+'_adapter_sha256'] for s in data):raise ValueError('SOURCE_STAGE_WEIGHT_SHA_CHANGED')
        if set(tensors['source'])!=set(tensors['candidate']):raise ValueError('UNCHANGED_LORA_MATRIX_SET_REQUIRED')
        for s,values in tensors.items():
            if len(values)!=504 or sum(v.size for v in values.values())!=33030144 or any(
                v.dtype!=np.float32 or not np.isfinite(v).all() or '.lora_' not in n for n,v in values.items()):raise ValueError('ALL_FINITE_RANK16_LORA_MATRICES_REQUIRED')
        if any(tensors['source'][n].shape!=v.shape for n,v in tensors['candidate'].items()):raise ValueError('LORA_SHAPES_CHANGED')
        changed=sum(not np.array_equal(tensors['source'][n],v) for n,v in tensors['candidate'].items())
        changed_values=sum(int(np.count_nonzero(tensors['source'][n]!=v)) for n,v in tensors['candidate'].items())
        if not changed or not changed_values:raise ValueError('REAL_SOURCE_CONSOLIDATION_CHANGES_REQUIRED')
        if (any(type(r[k]) is not int for k in ('optimizer_attempted_steps','joint_optimizer_steps','fp16_overflow_steps_skipped','actual_updates_at_selected_checkpoint'))
            or r['optimizer_attempted_steps']!=r['joint_optimizer_steps']+r['fp16_overflow_steps_skipped']
            or not 0<r['actual_updates_at_selected_checkpoint']<=r['joint_optimizer_steps']):raise ValueError('ACTUAL_SELECTED_OPTIMIZER_UPDATES_REQUIRED')
        baselines={k:r[k] for k in ('source_baseline_loss','initial_focused_source_loss','practice_baseline_loss','initial_focused_practice_loss','parent_source_loss','parent_practice_loss')}
        history=r['development_checkpoints']
        for row in history:
            g={'accuracy':row['accuracy'],'per_rule':row['per_rule'],'constant_fields_despite_varied_expected_values':row['constant_fields']}
            actual=selection(g,row['source_validation_loss'],row['practice_validation_loss'],baselines)
            if any(not math.isclose(actual[a],row[b],rel_tol=1e-10,abs_tol=1e-10) for a,b in (
                ('score','selection_score'),('minimum_rule_accuracy','minimum_rule_accuracy'),('max_source_regression','max_source_regression'),('max_practice_regression','max_practice_regression'))):
                raise ValueError('CPU_SOURCE_CHECKPOINT_SELECTION_RECOMPUTATION_FAILED')
        best=max(history,key=lambda row:row['selection_score'])
        if r['selected_checkpoint']!='checkpoint-'+str(best['optimizer_step']) or not math.isclose(r['selected_metric'],best['selection_score'],abs_tol=1e-10):
            raise ValueError('SOURCE_CHECKPOINT_NOT_FROZEN_DEVELOPMENT_BEST')
        gates={}
        for filename,field,total in (('source_validation_window_losses.json','source_final_loss',178),('practice_validation_window_losses.json','practice_final_loss',122)):
            windows=json.loads(z.read(filename));counts=windows['window_counts'];seen=set();values={e:[] for e in counts}
            for edition,index,loss in windows['window_losses']:
                if (edition not in counts or type(index) is not int or not 0<=index<counts[edition] or (edition,index) in seen
                    or not math.isfinite(loss) or loss<0):raise ValueError('EXACT_UNIQUE_FINITE_VALIDATION_WINDOWS_REQUIRED')
                seen.add((edition,index));values[edition].append((index,loss))
            if len(seen)!=sum(counts.values()) or len(seen)!=total:raise ValueError('ALL_FROZEN_VALIDATION_WINDOWS_REQUIRED')
            recomputed={e:sum(v for _,v in sorted(rows))/len(rows) for e,rows in values.items()}
            recomputed['ALL']=sum(recomputed[e]*counts[e] for e in counts)/total
            if any(not math.isclose(v,r[field][e],rel_tol=1e-12,abs_tol=1e-12) for e,v in recomputed.items()):raise ValueError('VALIDATION_WINDOW_LOSS_RECOMPUTATION_FAILED')
        direct=score_responses(restorative_cases('validation'),json.loads(z.read('final_direct_development_predictions.json')))
        if direct!=r['final_direct_development']:raise ValueError('FINAL_DIRECT_DEVELOPMENT_RECOMPUTATION_FAILED')
        for name,before in (('final_source_validation_gate','source_baseline_loss'),('final_practice_validation_gate','practice_baseline_loss'),
            ('initial_focused_source_gate','initial_focused_source_loss'),('initial_focused_practice_gate','initial_focused_practice_loss'),
            ('parent_source_validation_gate','parent_source_loss'),('parent_practice_validation_gate','parent_practice_loss')):
            gates[name]=validation_admission(r[before],r['source_final_loss' if 'source' in name else 'practice_final_loss'])
            if gates[name]!=r[name]:raise ValueError('SOURCE_STAGE_TEXT_GATE_RECOMPUTATION_FAILED')
        proof={'source_commit':source_commit,'archive_sha256':sha256(path.read_bytes()).hexdigest(),'initial_adapter_sha256':START_SHA,
            'candidate_adapter_sha256':r['candidate_adapter_sha256'],'parent_receipt_sha256':PARENT_RECEIPT_SHA,
            'matrices':504,'parameters':33030144,'all_finite':True,'changed_matrices':changed,'changed_values':changed_values,
            'actual_updates':r['joint_optimizer_steps'],'attempted_steps':r['optimizer_attempted_steps'],'skipped_steps':r['fp16_overflow_steps_skipped'],
            'selected_checkpoint':r['selected_checkpoint'],'selected_actual_updates':r['actual_updates_at_selected_checkpoint'],
            'selected_on_development_only':True,'final_test_performed_here':False,'text_gates':gates,
            'direct_development':r['final_direct_development'],'trading_qualification':False,'execution_allowed':False}
        (destination/'research_receipt.json').write_bytes(z.read('research_receipt.json'))
        (destination/'source_consolidation_training_audit.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2))
        return r,proof


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglehub.auth import get_username
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest,ApiGetAcceleratorQuotaStatisticsRequest
    token=os.environ['KAGGLE_API_TOKEN'];revision=os.environ['GITHUB_SHA'];owner=get_username();api=KaggleApi();api.authenticate()
    if design()!=json.loads(Path('experiments/source-consolidation-20261028-design.json').read_bytes()):raise ValueError('FROZEN_SOURCE_CONSOLIDATION_DESIGN_CHANGED')
    with api.build_kaggle_client() as client:
        q=client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
        if q and (q.total_time_allowed-q.time_used).total_seconds()<10800:raise ValueError('THREE_FREE_GPU_HOURS_REQUIRED_NO_PAID_FALLBACK')
    with urlopen(CONTROL_URL+'?preflight='+str(int(time.time())),timeout=20) as response:
        if checked_stop_control(json.load(response)):raise ValueError('USER_STOP_NO_SOURCE_TRAINING_SUBMITTED')
    if 'complete' not in str(api.kernels_status(owner+'/williams-ai-restorative-2t4-20261028').status).lower():raise ValueError('COMPLETED_PARENT_GPU_REQUIRED')
    def files(slug):
        found={};page=None;seen=set()
        with api.build_kaggle_client() as client:
            while True:
                request=ApiListKernelSessionOutputRequest();request.user_name=owner;request.kernel_slug=slug
                if page:request.page_token=page
                result=client.kernels.kernels_api_client.list_kernel_session_output(request)
                found.update({f.file_name:f for f in result.files});page=getattr(result,'next_page_token',None)
                if not page:break
                if page in seen:raise ValueError('OWN_OUTPUT_PAGINATION_LOOP')
                seen.add(page)
        return found
    current=files('williams-ai-restorative-2t4-20261028');old=files('williams-ai-focused-2t4-20261027')
    parent_bytes=urlopen(current['restorative_qwen4b/research_receipt.json'].url,timeout=60).read()
    if sha256(parent_bytes).hexdigest()!=PARENT_RECEIPT_SHA:raise ValueError('INDEPENDENTLY_VERIFIED_PARENT_RECEIPT_REQUIRED')
    parent=json.loads(parent_bytes)
    if parent['candidate_adapter_sha256']!=START_SHA or parent['complete_training_and_test'] is not True:raise ValueError('EXACT_FROZEN_RESTORATIVE768_PARENT_REQUIRED')
    reference_bytes=urlopen(old['focused_qwen4b/research_receipt.json'].url,timeout=60).read()
    if sha256(reference_bytes).hexdigest()!=PREVIOUS_RECEIPT_SHA:raise ValueError('KNOWN_FOCUSED69_BASELINES_REQUIRED')
    reference=json.loads(reference_bytes)
    reference.update({'parent_source_loss':parent['source_final_loss'],'parent_practice_loss':parent['practice_final_loss']})
    initial={'files':{}};reference40={'files':{}}
    for target,lookup,prefix,expected in ((initial,current,'restorative_qwen4b/candidate_adapter/',START_SHA),
        (reference40,old,'focused_qwen4b/source_adapter/',PREVIOUS40_SHA)):
        for name in ('adapter_config.json','adapter_model.safetensors'):
            item=lookup[prefix+name]
            if not item.url.startswith('https://'):raise ValueError('OWN_HTTPS_OUTPUT_REFERENCE_REQUIRED')
            raw=urlopen(item.url,timeout=240).read();digest=sha256(raw).hexdigest()
            if name=='adapter_model.safetensors' and digest!=expected:raise ValueError('EXACT_OWN_PARENT_ADAPTER_SHA_REQUIRED')
            target['files'][name]={'url':item.url,'sha256':digest}
    payload=open_envelope(json.loads(Path('experiments/private-source-20261027-course-full.sealed.json').read_bytes()),token)
    audited=checked_payload(json.loads(payload));checked_baselines(reference,audited['audit'])
    if audited['audit']!=parent['source_audit']:raise ValueError('FROZEN_ORIGINAL_TRAIN_VALIDATION_PARTITION_REQUIRED')
    worker=worker_source(revision,payload,b'[]',initial,reference,reference40,source_consolidation=True)
    if token in worker or len(worker)>3_000_000:raise ValueError('NO_CREDENTIAL_IN_PRIVATE_GPU_WORKER')
    compile(worker,'source_consolidation_worker','exec')
    metadata={'id':owner+'/'+RUN_SLUG,'title':RUN_SLUG,'code_file':'worker.py','language':'python','kernel_type':'script',
        'is_private':'true','enable_gpu':'true','enable_internet':'true','machine_shape':'NvidiaTeslaT4','dataset_sources':[],
        'model_sources':[],'competition_sources':[],'kernel_sources':[]}
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/'worker.py').write_text(worker);(root/'worker.py').chmod(0o600);(root/'kernel-metadata.json').write_text(json.dumps(metadata))
        p=subprocess.run(['kaggle','kernels','push','-p',td,'--accelerator','NvidiaTeslaT4','--timeout','10800'],capture_output=True,text=True,timeout=180)
        if p.returncode:print(safe_submission_error(p.stdout+p.stderr,token),flush=True);raise RuntimeError('PRIVATE_SOURCE_CONSOLIDATION_REJECTED')
    state='UNKNOWN'
    for i in range(585):
        state=str(api.kernels_status(owner+'/'+RUN_SLUG).status).upper()
        if i%15==0:print('OWN_SOURCE_CONSOLIDATION_STATE',state,flush=True)
        if any(s in state for s in ('COMPLETE','ERROR','CANCEL')):break
        time.sleep(20)
    dest=Path(os.environ['RUNNER_TEMP'])/'source_consolidation_results';dest.mkdir(mode=0o700)
    p=subprocess.run(['kaggle','kernels','output',owner+'/'+RUN_SLUG,'-p',str(dest),'--file-pattern',r'.*(SOURCE_CONSOLIDATION_DUAL_T4[.]zip|restorative_failure[.]json)$'],capture_output=True,text=True,timeout=420)
    archive=dest/'Williams_Qwen3_4B_SOURCE_CONSOLIDATION_DUAL_T4.zip'
    if p.returncode or not archive.is_file():raise RuntimeError('SOURCE_STAGE_ARCHIVE_MISSING_NO_COMPLETION_CLAIM')
    report,proof=checked_archive(archive,dest,source_commit=revision,parent_receipt=parent)
    print('VERIFIED_SOURCE_CONSOLIDATION',json.dumps({k:v for k,v in proof.items() if k!='direct_development'}),flush=True)
    if 'COMPLETE' not in state or report['complete_source_consolidation'] is not True:raise RuntimeError('SOURCE_CONSOLIDATION_INCOMPLETE')


if __name__=='__main__':main()
