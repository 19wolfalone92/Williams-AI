"""Recompute the final four-model exam on CPU, including independent keys."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
from decimal import Decimal
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.ai_verify_restorative_result import independent_key
from williams_ai.spread_market_exam import frozen_market_cohort,grade_market_comparison,ARCHIVE_SHA
from williams_ai.direct_mode_transfer import direct_test_cases,select_development_mode,grade_direct_test,transfer_gate
from williams_ai.restorative_book_practice import restorative_cases,INITIAL_SHA,PREVIOUS40_SHA
from williams_ai.contrastive_book_practice import question_rows,questions_digest,score_responses
from williams_ai.frozen_source_test import recompute_source_scores,grade_source_test
from williams_ai.lora_integrity import validation_admission


def numerical_equal(a,b):
    """Allow only floating summation roundoff across Python 3.11/3.12.

    Counts, flags, strings, cohort sizes and dictionary keys remain exact.
    This does not change any admission threshold or repair model answers.
    """
    if type(a) is float and type(b) is float:
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(numerical_equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(numerical_equal(x,y) for x,y in zip(a,b))
    return a==b


def independent_market_keys(archive:Path,seal):
    raw=archive.read_bytes()
    if sha256(raw).hexdigest()!=ARCHIVE_SHA:raise ValueError('EXACT_FROZEN_PUBLIC_MARKET_HISTORY_REQUIRED')
    pairs=json.loads(raw)['pairs'];keys=[]
    for symbol,cuts in sorted(seal['cuts_by_symbol'].items()):
        bars=pairs[symbol]['klines']
        for cut in cuts:
            last=Decimal(str(bars[cut][4]));forward=Decimal(str(bars[cut+24][4]))
            ranges=[]
            for i in range(cut-13,cut+1):
                high,low,previous=(Decimal(str(v)) for v in (bars[i][2],bars[i][3],bars[i-1][4]))
                ranges.append(max(high-low,abs(high-previous),abs(low-previous)))
            threshold=sum(ranges)/Decimal(14)/last;movement=(forward-last)/last
            label='UP' if movement>threshold else 'DOWN' if movement < -threshold else 'RANGE'
            available=datetime.fromtimestamp(bars[cut+24][0]/1000,timezone.utc).isoformat(timespec='milliseconds').replace('+00:00','Z')
            keys.append({'case_id':f'{symbol}-H1-asof-{bars[cut][0]}','result_available_at':available,
                'true_class':label,'future_return':float(movement),'atr_asof_ratio':float(threshold)})
    return keys


def independent_rule_cases(cases):
    rows=question_rows(cases)
    return [{'case_id':q['case_id'],'rule_id':q['domain'],'answer':json.dumps(independent_key(q),ensure_ascii=False)} for q in rows]


def verify(directory:Path,gpu_archive:Path,market_archive:Path,*,expected_inference_source_commit:str):
    read=lambda n:json.loads((directory/n).read_bytes())
    predictions=read('spread_market_predictions.json');grade=read('spread_market_grade.json');report=predictions['receipt']
    with zipfile.ZipFile(gpu_archive) as z:
        raw_receipt=z.read('research_receipt.json');receipt=json.loads(raw_receipt)
        thinking=json.loads(z.read('restorative_test_precommit.json'))
    expected={'candidate':receipt['candidate_adapter_sha256'],'focused69':INITIAL_SHA,'previous40':PREVIOUS40_SHA}
    if (set(predictions['stages'])!={'base','previous40','focused69','candidate'}
        or report['inference_source_commit']!=expected_inference_source_commit
        or report['adapter_sha256']!=expected or report['training_source_commit']!=receipt['source_commit']
        or report['training_receipt_sha256']!=sha256(raw_receipt).hexdigest()
        or any(report[k]!=receipt[k] for k in ('model_repo','model_revision','official_weights_sha256'))
        or report['world_size']!=2 or report['gpu_names']!=['Tesla T4','Tesla T4']
        or report['future_outcomes_loaded_in_inference'] is not False or report['execution_allowed'] is not False):
        raise ValueError('FROZEN_TRAINING_IDENTITY_REQUIRED')
    _,keys,_,seal=frozen_market_cohort(market_archive)
    independent=independent_market_keys(market_archive,seal);saved=read('withheld_market_outcomes_cpu_only.json')
    for expected_keys in (keys,saved):
        if len(expected_keys)!=30:raise ValueError('ALL_THIRTY_MARKET_OUTCOMES_REQUIRED')
        for a,b in zip(independent,expected_keys):
            if any(a[k]!=b[k] for k in ('case_id','result_available_at','true_class')) or any(
                not math.isclose(a[k],b[k],rel_tol=1e-10,abs_tol=1e-12) for k in ('future_return','atr_asof_ratio')):
                raise ValueError('INDEPENDENT_PRICE_OUTCOME_OR_ATR_KEY_MISMATCH')
    market=grade_market_comparison(keys,predictions['stages'])
    for k,v in market.items():
        if not numerical_equal(grade[k],v):raise ValueError('CPU_MARKET_GRADE_RECOMPUTATION_MISMATCH:'+k)
    if grade['seal']!=seal or read('spread_market_seal.json')!=seal:raise ValueError('FROZEN_MARKET_SEAL_CHANGED')
    if (report['questions_sha256']!=seal['questions_file_sha256'] or report['class_score_method']!=seal['class_score_method']
        or report['free_generation_protocol']!=seal['free_generation_protocol']
        or report['direct_rule_protocol']!={'mode':'direct','maximum_new_tokens':128,'seconds_per_response':10,'expected_keys_on_gpu':False}):
        raise ValueError('FROZEN_FOUR_MODEL_INFERENCE_PROTOCOL_REQUIRED')
    source_design=json.loads(Path('experiments/frozen-source-test-20261028-design.json').read_bytes())
    if (report['source_test_payload_sha256']!=source_design['payload_sha256'] or any(
        report['source_test_audit'][k]!=source_design[k] for k in ('manifest_sha256','test_file_sha256','test_pages','total_pages'))):
        raise ValueError('FROZEN_ORIGINAL_SOURCE_TEST_IDENTITY_REQUIRED')
    if grade['predictions_file_sha256']!=sha256((directory/'spread_market_predictions.json').read_bytes()).hexdigest():
        raise ValueError('GPU_PREDICTIONS_CHANGED_AFTER_CPU_GRADING')
    development=score_responses(independent_rule_cases(restorative_cases('validation')),predictions['direct_rule_development'])
    selection=select_development_mode(receipt['final_thinking_development'],development)
    direct=grade_direct_test(independent_rule_cases(direct_test_cases()),predictions['direct_rule_test'])
    if grade['direct_development_grade']!=development or grade['direct_rule_transfer']!=direct or grade['generation_mode_selection']!=selection or read('development_mode_selection.json')!=selection:
        raise ValueError('INDEPENDENT_DIRECT_KEYS_OR_DEVELOPMENT_MODE_SELECTION_MISMATCH')
    old_cases=independent_rule_cases(restorative_cases('test'))
    if thinking['questions_sha256']!=questions_digest(question_rows(restorative_cases('test'))):raise ValueError('ORIGINAL_THINKING_TEST_CHANGED')
    original={s:score_responses(old_cases,rows) for s,rows in thinking['stages'].items()}
    if original!=grade['original_thinking_exam']['stages'] or transfer_gate(original)!=grade['original_thinking_exam']['transfer_gate_passed']:
        raise ValueError('ORIGINAL_THINKING_EXAM_CHANGED')
    counts=report['source_test_audit']['windows_per_book']
    if counts!={'TC1':63,'NTD':93,'TC2':62,'COURSE':33}:raise ValueError('EXACT_TOKENIZER_PREFLIGHT_WINDOWS_REQUIRED')
    source_scores=recompute_source_scores(predictions['source_test_window_losses'],counts);source=grade_source_test(source_scores)
    if not numerical_equal(source_scores,predictions['source_test_scores']) or not numerical_equal(grade['source_test'],source):raise ValueError('SOURCE_TEST_WINDOW_RECOMPUTATION_MISMATCH')
    text_gates={}
    for name,before in (('final_source_validation_gate','source_baseline_loss'),('final_practice_validation_gate','practice_baseline_loss'),
        ('initial_focused_source_gate','initial_focused_source_loss'),('initial_focused_practice_gate','initial_focused_practice_loss')):
        text_gates[name]=validation_admission(receipt[before],receipt['source_final_loss' if 'source' in name else 'practice_final_loss'])
        if receipt[name]!=text_gates[name]:raise ValueError('ORIGINAL_TEXT_VALIDATION_GATE_MISMATCH')
    transfer=direct['transfer_gate_passed'] if selection['selected_mode']=='direct' else transfer_gate(original)
    combined={'selected_mode':selection['selected_mode'],'accepted_for_further_research':transfer and all(g['accepted'] for g in text_gates.values()) and source['source_test_gate_passed'],
        'selected_mode_transfer_gate_passed':transfer,'source_and_practice_validation_gates':text_gates,
        'original_source_test_gate_passed':source['source_test_gate_passed'],'market_results_did_not_select_mode_or_weights':True,
        'trading_qualification':False,'execution_allowed':False}
    if not numerical_equal(combined,grade['combined_research_admission']):raise ValueError('COMBINED_RESEARCH_ADMISSION_MISMATCH')
    return {'candidate_adapter_sha256':receipt['candidate_adapter_sha256'],'all_cpu_recomputations_agree':True,
        'inference_source_commit':expected_inference_source_commit,
        'market_outcomes_and_atr_independently_recalculated':True,'direct_rule_keys_independently_recalculated':True,
        'original_thinking_exam_preserved':True,'source_test_windows':sum(counts.values()),'source_test_pages':96,
        'class_distribution':dict(Counter(k['true_class'] for k in independent)),
        'generation_mode_selection':selection,'combined_research_admission':combined,
        'direct_test_summary':{s:{k:v for k,v in g.items() if k not in ('cases',)} for s,g in direct['stages'].items()},
        'market_stages':market['stages'],'original_source_test':source,
        'predictions_sha256':grade['predictions_file_sha256'],'floating_recomputation_relative_and_absolute_tolerance':1e-12,
        'execution_allowed':False,'trading_qualification':False}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('result-directory','gpu-archive','market-archive','new-output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--inference-source-commit',required=True)
    a=p.parse_args()
    if a.new_output.exists():raise ValueError('NEW_AUDIT_FILE_REQUIRED')
    proof=verify(a.result_directory,a.gpu_archive,a.market_archive,expected_inference_source_commit=a.inference_source_commit)
    a.new_output.write_text(json.dumps(proof,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('market_stages','original_source_test','direct_test_summary')},ensure_ascii=False))
