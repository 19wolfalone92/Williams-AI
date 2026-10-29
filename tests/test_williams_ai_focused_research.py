import ast
from contextlib import redirect_stdout
from collections import Counter
from decimal import Decimal, ROUND_FLOOR
from hashlib import sha256
import json
import io
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from williams_ai.contrastive_book_practice import RULES,paired_cases,question_rows,questions_digest,score_responses
from williams_ai.focused_book_practice import (focused_cases,focused_training_lessons,checkpoint_selection,
    INITIAL_SHA,REFERENCE416_SHA,RUN_SLUG,design_manifest)
from williams_ai.focused_gpu_research import checked_baselines,checked_stop_control,MODEL,REVISION
from williams_ai.lora_integrity import validation_admission
from scripts.ai_run_focused_kaggle import worker_source,checked_result_archive,FOCUSED_OUTPUT_PATTERN
from scripts.ai_observe_own_training import numerical_progress


class FocusedResearchTests(unittest.TestCase):
    def test_new_splits_and_revealed_old_exam_are_separate(self):
        rows={s:focused_cases(s) for s in ('train','validation','test')}
        self.assertEqual([len(rows[s]) for s in rows],[1664,84,84])
        for a,b in (('train','validation'),('train','test'),('validation','test')):
            self.assertFalse({r['question'] for r in rows[a]} & {r['question'] for r in rows[b]})
        old={r['question'] for r in paired_cases('test')}
        self.assertFalse(old & {r['question'] for v in rows.values() for r in v})
        self.assertEqual(Counter(r['rule_id'] for r in rows['test']),dict.fromkeys(RULES,12))

    def test_focused_numeric_keys_recalculate_from_question_operands(self):
        numeric=r'-?\d+(?:\.\d+)?'
        for split in ('train','validation','test'):
            for row in focused_cases(split):
                text=row['question'].split(' Только один JSON:')[0];ans=json.loads(row['answer']);rule=row['rule_id']
                self.assertIs(ans['execution_allowed'],False)
                if rule=='TC1.PSY.PLAN':
                    e,f,d,s=map(Decimal,re.findall(numeric,text.split(' Стоимость пункта')[0]))
                    budget=e*f;quantity=(budget/d/s).to_integral_value(rounding=ROUND_FLOOR)*s
                    self.assertEqual(ans['risk_budget'],float(budget));self.assertEqual(ans['quantity'],float(quantity))
                    self.assertEqual(quantity%s,0);self.assertLessEqual(quantity*d,budget);self.assertGreater((quantity+s)*d,budget)
                elif rule=='COURSE.HUMP':
                    end,begin=map(int,re.findall(numeric,text)[-2:]);self.assertIs(ans['hump_valid'],begin>end)
                elif rule=='NTD.AC':
                    values=list(map(Decimal,re.findall(numeric,text.split(' AO=')[0])))
                    ac=next(x for x in values if abs(x)<1);green=next(x for x in values if x>=1)
                    required=3 if ac<0 else 2
                    self.assertEqual(ans['required_green'],required);self.assertIs(type(ans['required_green']),int)
                    self.assertIs(ans['signal_ready'],green>=required)
                elif rule=='TC2.WM1.TRIGGER':
                    low,high,tick=map(Decimal,re.search(r'Low=([0-9.]+), high=([0-9.]+), tick=([0-9.]+)\.',text).groups())
                    long='бычий WM1' in text
                    self.assertEqual(ans['entry'],float(high+tick if long else low-tick))
                    self.assertEqual(ans['stop'],float(low-tick if long else high+tick))
                    for value in (low,high,Decimal(str(ans['entry'])),Decimal(str(ans['stop']))):self.assertEqual(value%tick,0)
                elif rule=='COURSE.W4.ZERO':
                    a,b=map(Decimal,re.search('изменился с ('+numeric+') на ('+numeric+')',text).groups())
                    self.assertIs(ans['minimum_w4_met'],a*b<0);self.assertIs(ans['w4_finished'],False)
                elif rule=='COURSE.AO5345':
                    a,b=map(Decimal,re.search('histogram=('+numeric+'), собственная SMA\\(5\\) этой гистограммы=('+numeric+')',text).groups())
                    self.assertEqual(ans['momentum'],'LONG' if a>b else 'SHORT')
                elif rule=='TC2.WM3':
                    right=int(re.search(r'справа закрыто (\d+)',text).group(1));self.assertIs(ans['confirmed'],right==2)

    def test_hump_equality_overlap_and_nonoverlap_and_ac_thresholds_are_covered(self):
        for split in ('train','validation','test'):
            rows=focused_cases(split)
            h=[r for r in rows if r['rule_id']=='COURSE.HUMP']
            self.assertEqual(Counter(json.loads(r['answer'])['hump_valid'] for r in h),{True:len(h)//2,False:len(h)//2})
            ac=[json.loads(r['answer']) for r in rows if r['rule_id']=='NTD.AC']
            self.assertEqual(set(r['required_green'] for r in ac),{2,3})
            self.assertEqual(set(r['signal_ready'] for r in ac),{True,False})

    def test_replay_prose_uses_only_training_and_keeps_all_legacy_validation(self):
        prose,typed,val=focused_training_lessons()
        self.assertEqual((len(prose),len(typed),sum(map(len,val.values()))),(742,2112,122))
        packed=json.dumps([prose,typed])
        self.assertNotIn('focused-test-',packed);self.assertNotIn('focused-validation-',packed)
        self.assertEqual({r['edition'] for r in prose},{'TC1','NTD','TC2','COURSE'})

    def test_checkpoint_selection_does_not_hide_weak_or_constant_rule(self):
        g={'accuracy':78/84,'per_rule':{r:{'correct':12,'total':12} for r in RULES},
           'constant_fields_despite_varied_expected_values':{'COURSE.HUMP':['hump_valid']}}
        g['per_rule']['COURSE.HUMP']['correct']=6
        balanced={'accuracy':8/12,'per_rule':{r:{'correct':8,'total':12} for r in RULES},
                  'constant_fields_despite_varied_expected_values':{}}
        self.assertGreater(checkpoint_selection(balanced,0)['score'],checkpoint_selection(g,0)['score'])
        self.assertLess(checkpoint_selection(balanced,0.021)['score'],-3)
        broken=json.loads(json.dumps(balanced));broken['per_rule'][RULES[0]]['total']=0
        with self.assertRaises(ValueError):checkpoint_selection(broken,0)

    def test_baseline_rejects_another_model_checkpoint_or_source(self):
        audit={'manifest_sha256':'fixed'};scores=dict.fromkeys(('TC1','NTD','TC2','COURSE','ALL'),2.0)
        ref={'source_commit':'ec0396e8f23469ffce0ebe25b41bbc325ce58dc0','model_repo':MODEL,'model_revision':REVISION,
             'candidate_adapter_sha256':INITIAL_SHA,'source_adapter_sha256':REFERENCE416_SHA,'world_size':2,
             'complete_training_and_test':True,'execution_allowed':False,'source_audit':audit,'validation_examples':122}
        for field in ('source_baseline_loss','practice_baseline_loss','source_final_loss','practice_final_loss','official_base_source_loss','official_base_practice_loss'):ref[field]=dict(scores)
        self.assertEqual(checked_baselines(ref,audit)['source_baseline_loss'],scores)
        for field in ('model_revision','candidate_adapter_sha256','source_adapter_sha256'):
            bad={**ref,field:'other'}
            with self.assertRaises(ValueError):checked_baselines(bad,audit)
        with self.assertRaises(ValueError):checked_baselines(ref,{'manifest_sha256':'changed'})

    def test_actual_worker_entry_is_two_gpu_prompt_only_and_uses_two_verified_initializations(self):
        files={'files':{n:{'url':'https://example.invalid/own-expiring','sha256':'a'*64} for n in ('adapter_config.json','adapter_model.safetensors')}}
        questions=question_rows(focused_cases('test'))
        worker=worker_source('a'*40,b'private source train+validation only',json.dumps(questions).encode(),files,{},files)
        tree=ast.parse(worker);self.assertNotIn('KAGGLE_API_TOKEN',worker);self.assertNotIn('"expected":',worker)
        self.assertIn('--nproc_per_node=2',worker);self.assertIn(INITIAL_SHA,worker);self.assertIn(REFERENCE416_SHA,worker)
        with tempfile.TemporaryDirectory() as td:
            entry=Path(td)/'entry.py'
            node=next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='entry' and n.value.func.attr=='write_text')
            exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_entry_writer','exec'),{'entry':entry,'repo':Path(td),'private':Path(td)/'p','questions':Path(td)/'q','adapter':Path(td)/'a','reference':Path(td)/'r'})
            entry_tree=ast.parse(entry.read_text());call=next(n.value for n in entry_tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='train_joint')
            self.assertEqual(len(call.args),6)
        source=Path('williams_ai/focused_gpu_research.py').read_text()
        self.assertNotIn('focused_cases("test")',source)

    def test_focused_stop_and_frozen_direct_design(self):
        self.assertTrue(checked_stop_control({'run_slug':RUN_SLUG,'stop':True}))
        with self.assertRaises(ValueError):checked_stop_control({'run_slug':'wrong','stop':True})
        self.assertLessEqual(len(RUN_SLUG),50)
        d=design_manifest();self.assertEqual(d['generated_development_cases_actually_used'],84)
        self.assertEqual(d['inference_mode_precommitted'],'direct')
        self.assertNotEqual(d['questions_sha256']['validation'],d['questions_sha256']['test'])

    def test_readonly_observer_emits_only_explicit_finite_numeric_fields(self):
        row={'accuracy':0.5,'max_source_regression':0.0,'selection_score':0.1,'optimizer_step':512,
             'minimum_rule_accuracy':0.25,'collapsed_rules':1}
        self.assertEqual(numerical_progress('FOCUSED_DEV_SELECTION '+json.dumps(row)),[{'generated_dev':True,**row}])
        self.assertEqual(numerical_progress('FOCUSED_DEV_SELECTION '+json.dumps({**row,'private_answer':'secret'})),[])
        self.assertEqual(numerical_progress('FOCUSED_DEV_SELECTION '+json.dumps({**row,'accuracy':float('nan')})),[])
        self.assertEqual(numerical_progress('FOCUSED_TEST_STAGE_COMPLETED {"completed_stages":2,"responses":84}'),
            [{'generated_test_progress':True,'completed_stages':2,'responses':84}])

    def test_both_complete_weights_archive_and_failure_evidence_are_retrievable(self):
        self.assertIsNotNone(re.fullmatch(FOCUSED_OUTPUT_PATTERN,'Williams_Qwen3_4B_FOCUSED_DUAL_T4.zip'))
        self.assertIsNotNone(re.fullmatch(FOCUSED_OUTPUT_PATTERN,'focused_failure.json'))
        self.assertIsNone(re.fullmatch(FOCUSED_OUTPUT_PATTERN,'private_source.json'))

    def test_cpu_grade_rejects_constant_family_even_when_total_exceeds_eighty_percent(self):
        cases=focused_cases('test');questions=question_rows(cases)
        source=b'reference';candidate=b'changed'
        before=dict.fromkeys(('TC1','NTD','TC2','COURSE','ALL'),2.0);after=dict.fromkeys(before,1.8)
        report={'initial_adapter_sha256':sha256(source).hexdigest(),'source_adapter_sha256':sha256(source).hexdigest(),
            'candidate_adapter_sha256':sha256(candidate).hexdigest(),'world_size':2,'execution_allowed':False,'model_repo':MODEL,'model_revision':REVISION,
            'source_commit':'b'*40,'complete_training_and_test':True,'adapter_weights_changed':True,'joint_optimizer_steps':1,'text_validation_gates_passed':True}
        for field in ('source_baseline_loss','practice_baseline_loss','reference416_source_loss','reference416_practice_loss'):report[field]=before
        for field in ('source_final_loss','practice_final_loss'):report[field]=after
        for field in ('final_source_validation_gate','final_practice_validation_gate','reference416_source_gate','reference416_practice_gate'):report[field]=validation_admission(before,after)
        def answer(c,text):return {'case_id':c['case_id'],'response':text,'thinking_complete':True,'response_time_limit_reached':False}
        good=[answer(c,c['answer']) for c in cases]
        pre={'questions_sha256':questions_digest(questions),'inference_mode':'direct','expected_keys_loaded_on_gpu':False,
             'same_response_time_budget_seconds_each_model':45.0,'stages':{s:[answer(c,'{}') for c in cases] for s in ('base','initial416','previous40')}}
        pre['stages']['candidate']=good
        for row,case in zip(good,cases):
            if case['rule_id']=='COURSE.HUMP':row['response']='{"hump_valid":true,"execution_allowed":false}'
        with tempfile.TemporaryDirectory() as td,patch('scripts.ai_run_focused_kaggle.INITIAL_SHA',sha256(source).hexdigest()):
            root=Path(td);archive=root/'result.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('research_receipt.json',json.dumps(report));z.writestr('focused_test_precommit.json',json.dumps(pre))
                z.writestr('source_adapter/adapter_model.safetensors',source);z.writestr('candidate_adapter/adapter_model.safetensors',candidate)
            with redirect_stdout(io.StringIO()):
                checked_result_archive(archive,root,questions,cases,expected_source_commit='b'*40)
            grade=json.loads((root/'focused_test_grade.json').read_text())
            self.assertGreater(grade['stages']['candidate']['accuracy'],0.8)
            self.assertFalse(grade['all_research_gates_accepted'])
            self.assertFalse(json.loads((root/'final_research_admission.json').read_text())['accepted_for_further_research'])


if __name__=='__main__':unittest.main()
