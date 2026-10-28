"""Independently audit saved adapters, development selection and sealed keys.

No CUDA, secrets, text from books, or execution permission is required.
"""
from __future__ import annotations
import argparse
from decimal import Decimal, ROUND_FLOOR
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from williams_ai.restorative_book_practice import INITIAL_SHA, PREVIOUS40_SHA, PREVIOUS_RECEIPT_SHA, design_manifest
from williams_ai.contrastive_book_practice import score_responses, questions_digest
from williams_ai.deep_book_research import MODEL, REVISION, EDITIONS
from williams_ai.lora_integrity import validation_admission, strict_json


def independent_key(q):
    p={k:Decimal(n) for k,n in re.findall(r'\b([A-Z][A-Z0-9_]*)=(-?\d+(?:\.\d+)?)',q['prompt'])}
    r=q['domain']
    if r=='TC1.PSY.PLAN':
        budget=p['E']*p['F'];units=budget/(p['D']*p['S'])
        answer={'risk_budget':float(budget),'quantity':float(units.to_integral_value(rounding=ROUND_FLOOR)*p['S'])}
    elif r=='COURSE.HUMP':answer={'hump_valid':int(p['B'])>int(p['A'])}
    elif r=='NTD.AC':
        needed=3 if p['AC'].is_signed() else 2
        answer={'required_green':needed,'signal_ready':int(p['G'])>=needed}
    elif r=='TC2.WM1.TRIGGER':
        if p['SIDE']==1:answer={'entry':float(p['H']+p['T']),'stop':float(p['L']-p['T'])}
        else:answer={'entry':float(p['L']-p['T']),'stop':float(p['H']+p['T'])}
    elif r=='COURSE.W4.ZERO':answer={'minimum_w4_met':p['P'].is_signed()!=p['N'].is_signed(),'w4_finished':False}
    elif r=='COURSE.AO5345':answer={'momentum':'LONG' if p['HIST']>p['SMA'] else 'SHORT'}
    elif r=='TC2.WM3':answer={'confirmed':int(p['RIGHT'])==2}
    else:raise ValueError('UNKNOWN_RULE')
    return {**answer,'execution_allowed':False}


def audit(archive:Path,previous:Path,*,source_commit:str):
    import numpy as np
    from safetensors.numpy import load
    with zipfile.ZipFile(previous) as old,zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(set(z.namelist())) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in z.namelist()):
            raise ValueError('UNSAFE_OR_DUPLICATE_ARCHIVE_ENTRY')
        r=strict_json(z.read('research_receipt.json'));prior=strict_json(old.read('research_receipt.json'))
        if (r['source_commit']!=source_commit or r['model_repo']!=MODEL or r['model_revision']!=REVISION
            or r['initial_adapter_sha256']!=INITIAL_SHA or r['reference40_adapter_sha256']!=PREVIOUS40_SHA
            or r['previous_receipt_sha256']!=PREVIOUS_RECEIPT_SHA or r['world_size']!=2
            or r['complete_training_and_test'] is not True or r['execution_allowed'] is not False
            or r['source_audit']!=prior['source_audit'] or r['official_weights_sha256']!=prior['official_weights_sha256']):
            raise ValueError('EXACT_MODEL_SOURCE_AND_INITIALIZATION_REQUIRED')
        if sha256(old.read('research_receipt.json')).hexdigest()!=PREVIOUS_RECEIPT_SHA:raise ValueError('PRIOR_RECEIPT_CHANGED')
        weights={s:z.read(s+'_adapter/adapter_model.safetensors') for s in ('source','candidate')}
        if weights['source']!=old.read('candidate_adapter/adapter_model.safetensors'):raise ValueError('INITIAL_WEIGHTS_DIFFER')
        tensors={s:load(v) for s,v in weights.items()}
        if set(tensors['source'])!=set(tensors['candidate']):raise ValueError('LORA_MATRIX_SET_CHANGED')
        if any(tensors['source'][n].shape!=v.shape for n,v in tensors['candidate'].items()):raise ValueError('LORA_MATRIX_SHAPE_CHANGED')
        proof={}
        for stage,values in tensors.items():
            digest=sha256(weights[stage]).hexdigest()
            if digest!=r[stage+'_adapter_sha256'] or len(values)!=504 or sum(v.size for v in values.values())!=33030144:
                raise ValueError('COMPLETE_LORA_SHA_AND_PARAMETER_COUNT_REQUIRED')
            if any('.lora_' not in n or a.dtype!=np.float32 or not np.isfinite(a).all() for n,a in values.items()):
                raise ValueError('FINITE_FLOAT32_LORA_ONLY_REQUIRED')
            proof[stage]={'sha256':digest,'matrices':len(values),'parameters':sum(v.size for v in values.values()),'all_finite':True}
        changed=sum(not np.array_equal(tensors['source'][n],v) for n,v in tensors['candidate'].items())
        changed_values=sum(int(np.count_nonzero(tensors['source'][n]!=v)) for n,v in tensors['candidate'].items())
        if changed==0 or changed_values==0:raise ValueError('ACTUAL_TRAINED_CHANGES_REQUIRED')
        if r['optimizer_attempted_steps']!=r['joint_optimizer_steps']+r['fp16_overflow_steps_skipped']:raise ValueError('OPTIMIZER_COUNTS_INCONSISTENT')
        history=r['development_checkpoints']
        if not history:raise ValueError('DEVELOPMENT_SELECTION_HISTORY_REQUIRED')
        for row in history:
            minimum=min(v['correct']/v['total'] for v in row['per_rule'].values())
            accuracy=sum(v['correct'] for v in row['per_rule'].values())/sum(v['total'] for v in row['per_rule'].values())
            src=max(row['source_validation_loss'][e]/prior['source_baseline_loss'][e]-1 for e in EDITIONS)
            practice=max(row['practice_validation_loss'][e]/prior['practice_baseline_loss'][e]-1 for e in EDITIONS)
            score=minimum+.1*accuracy-.1*row['collapsed_rules']
            if max(src,practice)>.02:score=-3-max(src,practice)
            if any(not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10) for a,b in ((minimum,row['minimum_rule_accuracy']),(accuracy,row['accuracy']),
                (src,row['max_source_regression']),(practice,row['max_practice_regression']),(score,row['selection_score']))):
                raise ValueError('INDEPENDENT_DEVELOPMENT_SELECTION_MISMATCH')
        best=max(history,key=lambda x:x['selection_score'])
        if r['selected_checkpoint']!='checkpoint-'+str(best['optimizer_step']):raise ValueError('SELECTED_CHECKPOINT_NOT_PRECOMMITTED_BEST')
        pre=strict_json(z.read('restorative_test_precommit.json'))
        from williams_ai.restorative_book_practice import restorative_cases
        from williams_ai.contrastive_book_practice import question_rows
        questions=question_rows(restorative_cases('test'))
        if questions_digest(questions)!=pre['questions_sha256'] or pre['questions_sha256']!=design_manifest()['questions_sha256']['test']:
            raise ValueError('FROZEN_TEST_QUESTIONS_CHANGED')
        cases=[{'case_id':q['case_id'],'rule_id':q['domain'],'answer':json.dumps(independent_key(q),ensure_ascii=False)} for q in questions]
        if set(pre['stages'])!={'base','previous40','focused69','candidate'} or pre['inference_mode']!='thinking' or pre['expected_keys_loaded_on_gpu'] is not False:
            raise ValueError('SAME_FOUR_MODEL_THINKING_PROTOCOL_REQUIRED')
        grades={s:score_responses(cases,v) for s,v in pre['stages'].items()}
        gates={}
        for name,before,after in (('source_previous40','source_baseline_loss','source_final_loss'),('practice_previous40','practice_baseline_loss','practice_final_loss'),
                ('source_initial_focused','initial_focused_source_loss','source_final_loss'),('practice_initial_focused','initial_focused_practice_loss','practice_final_loss')):
            gates[name]=validation_admission(r[before],r[after])
        c=grades['candidate']
        transfer=(c['accuracy']>=.8 and c['accuracy']>=max(grades[s]['accuracy'] for s in ('base','previous40','focused69'))+.05
            and all(v['correct']/v['total']>=.75 for v in c['per_rule'].values()) and not c['constant_fields_despite_varied_expected_values'])
        return {'source_commit':source_commit,'model_revision':REVISION,'archive_sha256':sha256(archive.read_bytes()).hexdigest(),'weights':proof,
            'changed_matrices':changed,'changed_values':changed_values,'actual_updates':r['joint_optimizer_steps'],'attempted_steps':r['optimizer_attempted_steps'],
            'skipped_steps':r['fp16_overflow_steps_skipped'],'selected_checkpoint':r['selected_checkpoint'],'selected_actual_updates':r['actual_updates_at_selected_checkpoint'],
            'development_selection_independently_recomputed':True,'selected_checkpoint_guard_passed':max(best['max_source_regression'],best['max_practice_regression'])<=.02,
            'questions_sha256':pre['questions_sha256'],'keys_independently_recalculated_from_prompts':True,'generated_grades':grades,'text_gates':gates,
            'transfer_gate_accepted':transfer,'all_research_gates_accepted':transfer and all(v['accepted'] for v in gates.values()),'execution_allowed':False,'trading_qualification':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--new-output',type=Path,required=True);a=p.parse_args()
    if a.new_output.exists():raise ValueError('NEW_AUDIT_OUTPUT_REQUIRED')
    proof=audit(a.archive,a.previous,source_commit=a.source_commit)
    a.new_output.write_text(json.dumps(proof,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('weights','generated_grades')},ensure_ascii=False))
