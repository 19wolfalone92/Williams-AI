"""Independently recalculate frozen five-model final results on CPU."""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ai_verify_spread_market_result import independent_market_keys,independent_rule_cases,numerical_equal
from scripts.ai_run_source_consolidation_final_kaggle import STAGES,TRAINING_SOURCE,training_text_gates,combined_admission
from williams_ai.source_consolidation import fresh_test_cases,START_SHA
from williams_ai.restorative_book_practice import INITIAL_SHA,PREVIOUS40_SHA,restorative_cases
from williams_ai.contrastive_book_practice import score_responses,question_rows
from williams_ai.spread_market_exam import frozen_market_cohort,grade_market_comparison
from williams_ai.direct_mode_transfer import grade_direct_test
from williams_ai.frozen_source_test import recompute_source_scores,grade_source_test
from williams_ai.deep_book_research import MODEL,REVISION


def verify(directory:Path,source_archive:Path,market_archive:Path,*,inference_commit:str):
    read=lambda n:json.loads((directory/n).read_bytes())
    p=read('spread_market_predictions.json');g=read('spread_market_grade.json');r=p['receipt']
    with zipfile.ZipFile(source_archive) as z:
        raw=z.read('research_receipt.json');training=json.loads(raw)
        actual_weight_sha=sha256(z.read('candidate_adapter/adapter_model.safetensors')).hexdigest()
    expected={'candidate':actual_weight_sha,'restorative768':START_SHA,'focused69':INITIAL_SHA,'previous40':PREVIOUS40_SHA}
    if (training['candidate_adapter_sha256']!=actual_weight_sha or training['source_commit']!=TRAINING_SOURCE
        or r['training_source_commit']!=TRAINING_SOURCE or r['training_receipt_sha256']!=sha256(raw).hexdigest()
        or r['adapter_sha256']!=expected or r['inference_source_commit']!=inference_commit or r['source_consolidation_final'] is not True
        or r['model_repo']!=MODEL or r['model_revision']!=REVISION or r['official_weights_sha256']!=training['official_weights_sha256']
        or r['world_size']!=2 or r['gpu_names']!=['Tesla T4','Tesla T4'] or r['future_outcomes_loaded_in_inference'] is not False
        or r['execution_allowed'] is not False or set(p['stages'])!=set(STAGES)):
        raise ValueError('IMMUTABLE_FIVE_MODEL_TRAINING_AND_INFERENCE_IDENTITY_REQUIRED')
    _,keys,_,seal=frozen_market_cohort(market_archive);independent=independent_market_keys(market_archive,seal)
    for observed in (keys,read('withheld_market_outcomes_cpu_only.json')):
        if len(observed)!=30:raise ValueError('THIRTY_OUTCOMES_REQUIRED')
        for a,b in zip(independent,observed):
            if any(a[k]!=b[k] for k in ('case_id','true_class','result_available_at')) or any(
                not math.isclose(a[k],b[k],rel_tol=1e-10,abs_tol=1e-12) for k in ('future_return','atr_asof_ratio')):
                raise ValueError('INDEPENDENT_MARKET_PRICE_AND_ATR_KEY_MISMATCH')
    if g['seal']!=seal or read('spread_market_seal.json')!=seal or r['questions_sha256']!=seal['questions_file_sha256']:
        raise ValueError('FROZEN_MARKET_SEAL_CHANGED')
    if (r['free_generation_protocol']!=seal['free_generation_protocol'] or r['class_score_method']!=seal['class_score_method']
        or r['direct_rule_protocol']!={'mode':'direct','maximum_new_tokens':128,'seconds_per_response':10,'expected_keys_on_gpu':False}):
        raise ValueError('EXACT_PRECOMMITTED_RESPONSE_PROTOCOL_REQUIRED')
    for name,cases in (('development',restorative_cases('validation')),('test',fresh_test_cases())):
        raw_questions=json.dumps(question_rows(cases),ensure_ascii=False).encode()
        if r['direct_'+name+'_file_sha256']!=sha256(raw_questions).hexdigest():raise ValueError('EXACT_FROZEN_NUMERIC_QUESTIONS_REQUIRED')
    if any(type(row.get(flag)) is not bool for rows in p['direct_rule_test'].values() for row in rows for flag in ('thinking_complete','response_time_limit_reached')):
        raise ValueError('EXPLICIT_DIRECT_COMPLETION_FLAGS_REQUIRED')
    direct=grade_direct_test(independent_rule_cases(fresh_test_cases()),p['direct_rule_test'],include_restorative_parent=True)
    development=score_responses(independent_rule_cases(restorative_cases('validation')),p['direct_rule_development'])
    if not numerical_equal(g['direct_rule_transfer'],direct) or not numerical_equal(g['direct_development_grade'],development):
        raise ValueError('INDEPENDENT_DIRECT_KEY_RECOMPUTATION_FAILED')
    counts=r['source_test_audit']['windows_per_book']
    if counts!={'TC1':63,'NTD':93,'TC2':62,'COURSE':33}:raise ValueError('EXACT_251_FROZEN_SOURCE_WINDOWS_REQUIRED')
    source_design=json.loads(Path('experiments/frozen-source-test-20261028-design.json').read_bytes())
    if r['source_test_payload_sha256']!=source_design['payload_sha256'] or any(r['source_test_audit'][k]!=source_design[k] for k in ('manifest_sha256','test_file_sha256','test_pages','total_pages')):
        raise ValueError('EXACT_FROZEN_96_SOURCE_PAGE_IDENTITIES_REQUIRED')
    scores=recompute_source_scores(p['source_test_window_losses'],counts,include_restorative_parent=True)
    source=grade_source_test(scores,include_restorative_parent=True)
    if not numerical_equal(scores,p['source_test_scores']) or not numerical_equal(source,g['source_test']):raise ValueError('ALL_FIVE_MODEL_SOURCE_WINDOWS_MUST_RECOMPUTE')
    market=grade_market_comparison(keys,p['stages'])
    if any(not numerical_equal(g[k],v) for k,v in market.items()):raise ValueError('ALL_MARKET_RESULTS_MUST_RECOMPUTE')
    combined=combined_admission(direct,source,training_text_gates(training))
    if not numerical_equal(combined,g['combined_research_admission']):raise ValueError('COMBINED_RESEARCH_ADMISSION_MISMATCH')
    actual_prediction_sha=sha256((directory/'spread_market_predictions.json').read_bytes()).hexdigest()
    if g['predictions_file_sha256']!=actual_prediction_sha:raise ValueError('PREDICTION_BYTES_CHANGED_AFTER_GRADING')
    return {'candidate_adapter_sha256':actual_weight_sha,'all_cpu_recomputations_agree':True,'inference_source_commit':inference_commit,
        'keys_independently_recalculated_from_operands_and_prices':True,'source_pages':96,'source_windows_per_model':251,
        'class_distribution':dict(Counter(k['true_class'] for k in independent)),'direct_rule_transfer':direct,'source_test':source,
        'market_stages':market['stages'],'combined_research_admission':combined,'predictions_sha256':actual_prediction_sha,
        'floating_recomputation_relative_and_absolute_tolerance':1e-12,'execution_allowed':False,'trading_qualification':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('result-directory','source-archive','market-archive','new-output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--inference-source-commit',required=True);a=parser.parse_args()
    if a.new_output.exists():raise ValueError('NEW_INDEPENDENT_AUDIT_OUTPUT_REQUIRED')
    proof=verify(a.result_directory,a.source_archive,a.market_archive,inference_commit=a.inference_source_commit)
    a.new_output.write_text(json.dumps(proof,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('direct_rule_transfer','source_test','market_stages')},ensure_ascii=False))
