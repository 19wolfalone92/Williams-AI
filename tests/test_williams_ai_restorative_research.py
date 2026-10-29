import ast
from collections import Counter
from contextlib import redirect_stdout
from decimal import Decimal, ROUND_FLOOR
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from williams_ai.contrastive_book_practice import RULES, question_rows, questions_digest
from williams_ai.restorative_book_practice import (restorative_cases, restorative_selection,
    INITIAL_SHA, PREVIOUS40_SHA, RUN_SLUG, design_manifest)
from williams_ai.restorative_gpu_research import checked_baselines, checked_stop_control, MODEL, REVISION
from williams_ai.lora_integrity import validation_admission
from scripts.ai_run_restorative_kaggle import worker_source, checked_result_archive


class RestorativeResearchTests(unittest.TestCase):
    def test_unique_new_splits_and_revealed_prior_exam_remain_separate(self):
        from williams_ai.focused_book_practice import focused_cases
        rows = {s: restorative_cases(s) for s in ('train','validation','test')}
        self.assertEqual([len(v) for v in rows.values()], [1152,112,112])
        for a,b in (('train','validation'),('train','test'),('validation','test')):
            self.assertFalse({r['question'] for r in rows[a]} & {r['question'] for r in rows[b]})
        self.assertFalse({r['question'] for r in focused_cases('test')} & {r['question'] for r in rows['test']})
        self.assertEqual(Counter(r['rule_id'] for r in rows['test']), dict.fromkeys(RULES,16))

    def test_all_keys_recalculate_from_visible_operands_without_using_generator_answers(self):
        for split in ('train','validation','test'):
            for row in restorative_cases(split):
                v = {k:Decimal(n) for k,n in re.findall(r'\b([A-Z][A-Z0-9_]*)=(-?\d+(?:\.\d+)?)',row['question'])}
                a=json.loads(row['answer']);rule=row['rule_id']
                self.assertIs(a['execution_allowed'],False)
                if rule=='TC1.PSY.PLAN':
                    budget=v['E']*v['F'];quantity=(budget/v['D']/v['S']).to_integral_value(rounding=ROUND_FLOOR)*v['S']
                    self.assertEqual(a['risk_budget'],float(budget));self.assertEqual(a['quantity'],float(quantity))
                    self.assertLessEqual(quantity*v['D'],budget);self.assertGreater((quantity+v['S'])*v['D'],budget)
                elif rule=='COURSE.HUMP':self.assertIs(a['hump_valid'],v['B']>v['A'])
                elif rule=='NTD.AC':
                    needed=3 if str(v['AC']).startswith('-') else 2
                    self.assertEqual(a['required_green'],needed);self.assertIs(a['signal_ready'],int(v['G'])>=needed)
                elif rule=='TC2.WM1.TRIGGER':
                    buy=v['SIDE']==1
                    self.assertEqual(a['entry'],float(v['H']+v['T'] if buy else v['L']-v['T']))
                    self.assertEqual(a['stop'],float(v['L']-v['T'] if buy else v['H']+v['T']))
                elif rule=='COURSE.W4.ZERO':
                    self.assertIs(a['minimum_w4_met'],(v['P']<0)!=(v['N']<0));self.assertIs(a['w4_finished'],False)
                elif rule=='COURSE.AO5345':self.assertEqual(a['momentum'],'LONG' if v['HIST']>v['SMA'] else 'SHORT')
                elif rule=='TC2.WM3':self.assertIs(a['confirmed'],int(v['RIGHT'])==2)

    def test_hump_equality_and_both_ac_thresholds_are_not_omitted(self):
        for split in ('train','validation','test'):
            rows=restorative_cases(split);h=[r for r in rows if r['rule_id']=='COURSE.HUMP']
            relations=Counter()
            for r in h:
                a,b=(int(re.search(k+r'=(\d+)',r['question']).group(1)) for k in ('A','B'))
                relations['equal' if a==b else 'overlap' if b<a else 'separate']+=1
            self.assertEqual(set(relations),{'equal','overlap','separate'})
            ac=[json.loads(r['answer']) for r in rows if r['rule_id']=='NTD.AC']
            self.assertEqual({r['required_green'] for r in ac},{2,3})
            self.assertEqual({r['signal_ready'] for r in ac},{True,False})

    def test_ntd_regression_vetoes_an_otherwise_perfect_checkpoint(self):
        g={'accuracy':1.0,'per_rule':{r:{'correct':16,'total':16} for r in RULES},'constant_fields_despite_varied_expected_values':{}}
        before=dict.fromkeys(('TC1','NTD','TC2','COURSE','ALL'),1.0)
        good={k:0.98 for k in before};bad={**good,'NTD':1.03,'ALL':0.8}
        self.assertGreater(restorative_selection(g,good,good,before,before)['score'],1)
        self.assertLess(restorative_selection(g,good,bad,before,before)['score'],-3)
        with self.assertRaises(ValueError):restorative_selection(g,{**good,'NTD':float('nan')},good,before,before)

    def test_frozen_thinking_protocol_and_stop_are_explicit(self):
        d=design_manifest()
        self.assertEqual(d['inference_mode_precommitted'],'thinking')
        self.assertEqual(d['new_unique_cases']['test'],112)
        self.assertTrue(d['all_122_practice_and_178_source_windows_guard_checkpoints'])
        self.assertTrue(checked_stop_control({'run_slug':RUN_SLUG,'stop':True}))
        with self.assertRaises(ValueError):checked_stop_control({'run_slug':'another','stop':True})

    def test_cpu_reference_matches_initial_focused_weights_and_same_cohort(self):
        scores=dict.fromkeys(('TC1','NTD','TC2','COURSE','ALL'),1.0);audit={'manifest_sha256':'fixed'}
        r={'source_commit':'8711a2a72393382fc617c286e7fd9c7e38940caa','model_repo':MODEL,'model_revision':REVISION,
            'candidate_adapter_sha256':INITIAL_SHA,'source_adapter_sha256':PREVIOUS40_SHA,'world_size':2,
            'complete_training_and_test':True,'execution_allowed':False,'source_audit':audit,'validation_examples':122}
        for f in ('source_baseline_loss','practice_baseline_loss','source_final_loss','practice_final_loss','official_base_source_loss','official_base_practice_loss'):r[f]=scores
        self.assertEqual(checked_baselines(r,audit)['initial_focused_source_loss'],scores)
        with self.assertRaises(ValueError):checked_baselines({**r,'candidate_adapter_sha256':'other'},audit)

    def test_worker_has_exact_previous40_digest_no_keys_or_unsubstituted_tokens(self):
        f={'files':{n:{'url':'https://example.invalid/own-expiring','sha256':'a'*64} for n in ('adapter_config.json','adapter_model.safetensors')}}
        w=worker_source('a'*40,b'private training/validation only',json.dumps(question_rows(restorative_cases('test'))).encode(),f,{},f)
        tree=ast.parse(w)
        self.assertIn(INITIAL_SHA,w);self.assertIn(PREVIOUS40_SHA,w)
        self.assertNotIn('KAGGLE_API_TOKEN',w);self.assertNotIn('"answer":',w)
        self.assertFalse(re.search(r'__[A-Z][A-Z0-9_]+__',w))
        self.assertIn('--nproc_per_node=2',w)
        with tempfile.TemporaryDirectory() as td:
            entry=Path(td)/'entry.py'
            node=next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='entry' and n.value.func.attr=='write_text')
            exec(compile(ast.Module(body=[node],type_ignores=[]),'entry_writer','exec'),{'entry':entry,'repo':Path(td),'private':Path(td)/'p','questions':Path(td)/'q','adapter':Path(td)/'a','reference':Path(td)/'r'})
            ast.parse(entry.read_text())
        source=Path('williams_ai/restorative_gpu_research.py').read_text()
        self.assertNotIn('restorative_cases("test")',source)
        self.assertIn('reasoned_tokens=tokenize_reasoned(typed,tokenizer)',source)
        self.assertIn('practice_values=source_scores(task_val)',source)

    def test_cpu_admission_rejects_text_regression_despite_perfect_new_exam(self):
        cases=restorative_cases('test');questions=question_rows(cases);source=b'initial';candidate=b'changed'
        before=dict.fromkeys(('TC1','NTD','TC2','COURSE','ALL'),1.0);after={**before,'ALL':0.9,'NTD':1.03}
        report={'initial_adapter_sha256':sha256(source).hexdigest(),'source_adapter_sha256':sha256(source).hexdigest(),
            'candidate_adapter_sha256':sha256(candidate).hexdigest(),'world_size':2,'execution_allowed':False,'model_repo':MODEL,'model_revision':REVISION,
            'source_commit':'b'*40,'complete_training_and_test':True,'adapter_weights_changed':True,'joint_optimizer_steps':1,'text_validation_gates_passed':False}
        for field in ('source_baseline_loss','practice_baseline_loss','initial_focused_source_loss','initial_focused_practice_loss'):report[field]=before
        for field in ('source_final_loss','practice_final_loss'):report[field]=after
        for field in ('final_source_validation_gate','final_practice_validation_gate','initial_focused_source_gate','initial_focused_practice_gate'):report[field]=validation_admission(before,after)
        def answer(c,text):return {'case_id':c['case_id'],'response':text,'thinking_complete':True,'response_time_limit_reached':False}
        pre={'questions_sha256':questions_digest(questions),'inference_mode':'thinking','expected_keys_loaded_on_gpu':False,
            'same_response_time_budget_seconds_each_model':40.0,'stages':{s:[answer(c,'{}') for c in cases] for s in ('base','previous40','focused69')}}
        pre['stages']['candidate']=[answer(c,c['answer']) for c in cases]
        with tempfile.TemporaryDirectory() as td,patch('scripts.ai_run_restorative_kaggle.INITIAL_SHA',sha256(source).hexdigest()):
            root=Path(td);archive=root/'result.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('research_receipt.json',json.dumps(report));z.writestr('restorative_test_precommit.json',json.dumps(pre))
                z.writestr('source_adapter/adapter_model.safetensors',source);z.writestr('candidate_adapter/adapter_model.safetensors',candidate)
            with redirect_stdout(io.StringIO()):checked_result_archive(archive,root,questions,cases,expected_source_commit='b'*40)
            grade=json.loads((root/'restorative_test_grade.json').read_text())
            self.assertTrue(grade['further_research_transfer_gate']['accepted'])
            self.assertFalse(grade['all_research_gates_accepted'])
            self.assertFalse(json.loads((root/'final_research_admission.json').read_text())['execution_allowed'])


if __name__=='__main__':unittest.main()
