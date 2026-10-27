import ast
from collections import Counter,defaultdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from williams_ai.contrastive_book_practice import (paired_cases,quick_development_cases,correct_answer,
    score_responses,value_spans,questions_digest,question_rows,weighted_causal_loss)
from williams_ai.contrastive_gpu_research import checked_stop_control,RUN_SLUG,checked_baselines,training_lessons
from scripts.ai_run_contrastive_kaggle import worker_source,GPU_SECONDS,POLL_SECONDS,safe_submission_error


class ContrastiveResearchTests(unittest.TestCase):
    def test_splits_are_disjoint_and_contrast_pairs_change_the_answer(self):
        cases={s:paired_cases(s) for s in ('train','validation','test')}
        self.assertEqual([len(cases[s]) for s in cases],[448,112,56])
        for a,b in (('train','validation'),('train','test'),('validation','test')):
            self.assertFalse({r['question'] for r in cases[a]}&{r['question'] for r in cases[b]})
        pairs=defaultdict(list)
        for r in cases['train']:pairs[r['pair_id']].append(r)
        for rows in pairs.values():
            self.assertEqual(len(rows),2)
            if rows[0]['rule_id']!='NTD.AC':self.assertNotEqual(rows[0]['answer'],rows[1]['answer'])
        self.assertEqual(len(quick_development_cases()),14)

    def test_counterexamples_cover_both_sides_and_equality_is_not_nonoverlap(self):
        rows=paired_cases('train')
        for rule,key in (('COURSE.HUMP','hump_valid'),('COURSE.W4.ZERO','minimum_w4_met'),('TC2.WM3','confirmed')):
            self.assertEqual(Counter(json.loads(r['answer'])[key] for r in rows if r['rule_id']==rule),{False:32,True:32})
        equal=[r for r in rows if r['rule_id']=='COURSE.HUMP' and 'до 236, вторая — с 236;' in r['question']]
        self.assertTrue(equal);self.assertFalse(json.loads(equal[0]['answer'])['hump_valid'])

    def test_numeric_ground_truth_and_type_sensitive_grading(self):
        self.assertTrue(correct_answer('{"x":2,"execution_allowed":false}','{"x":2,"execution_allowed":false}'))
        for bad in ('{"x":true,"execution_allowed":false}','{"x":1,"x":2,"execution_allowed":false}',
                    '{"x":NaN,"execution_allowed":false}','{"x":2,"execution_allowed":true}'):
            self.assertFalse(correct_answer(bad,'{"x":2,"execution_allowed":false}'))
        rows=paired_cases('test');answers=[{'case_id':r['case_id'],'response':r['answer']} for r in rows]
        self.assertEqual(score_responses(rows,answers)['correct'],56)
        self.assertFalse(score_responses(rows,answers)['constant_fields_despite_varied_expected_values'])
        for row in answers:
            case=next(r for r in rows if r['case_id']==row['case_id'])
            if case['rule_id']=='COURSE.HUMP':row['response']='{"hump_valid":true,"execution_allowed":false}'
        scored=score_responses(rows,answers)
        self.assertIn('hump_valid',scored['constant_fields_despite_varied_expected_values']['COURSE.HUMP'])

    def test_value_spans_weight_semantics_above_fixed_format(self):
        answer='{"quantity":11.55,"confirmed":false,"execution_allowed":false}'
        spans=value_spans(answer)
        self.assertEqual([(answer[a:b],w) for a,b,w in spans],[('11.55',8.0),('false',8.0),('false',2.0)])
        with self.assertRaises(ValueError):value_spans('{"nested":{"x":2}}')

    @unittest.skipUnless(importlib.util.find_spec('torch'),'optional CPU torch validation')
    def test_value_gradient_is_weighted_and_prompt_padding_has_no_gradient(self):
        import torch
        logits=torch.zeros(1,4,2,requires_grad=True);labels=torch.tensor([[-100,0,1,-100]])
        weights=torch.tensor([[0.,1.,8.,0.]])
        loss=weighted_causal_loss(logits,labels,weights);loss.backward()
        self.assertAlmostEqual(float(loss.detach()),0.69314718,places=6)
        self.assertAlmostEqual(float(abs(logits.grad[0,1,1]/logits.grad[0,0,0])),8.0,places=6)
        self.assertTrue(bool((logits.grad[0,2:]==0).all()))
        with self.assertRaises(ValueError):weighted_causal_loss(logits,labels,weights*0)

    def test_prompt_only_worker_and_actual_torchrun_entry_compile(self):
        initial={'files':{n:{'url':'https://example.invalid/expiring','sha256':'a'*64} for n in ('adapter_config.json','adapter_model.safetensors')}}
        q=question_rows(paired_cases('test'))
        worker=worker_source('a'*40,b'private train and validation only',json.dumps(q).encode(),initial,{'execution_allowed':False})
        tree=ast.parse(worker)
        self.assertNotIn('KAGGLE_API_TOKEN',worker);self.assertNotIn('WITHHELD_OUTCOME_KEYS',worker)
        self.assertNotIn('"expected":',worker);self.assertIn('--nproc_per_node=2',worker)
        with tempfile.TemporaryDirectory() as td:
            entry=Path(td)/'entry.py'
            node=next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
                and n.value.func.value.id=='entry' and n.value.func.attr=='write_text')
            exec(compile(ast.Module(body=[node],type_ignores=[]),'entry_writer','exec'),
                {'entry':entry,'repo':Path(td),'private':Path(td)/'private','questions':Path(td)/'q','adapter':Path(td)/'adapter','reference':Path(td)/'ref'})
            compile(entry.read_text(),'actual_distributed_entry','exec')
        self.assertGreaterEqual(GPU_SECONDS,6900+1800)

    def test_stop_flag_is_explicit_and_does_not_start_training(self):
        self.assertFalse(checked_stop_control({'run_slug':RUN_SLUG,'stop':False}))
        self.assertTrue(checked_stop_control({'run_slug':RUN_SLUG,'stop':True}))
        for bad in ({'run_slug':'other','stop':True},{'run_slug':RUN_SLUG,'stop':'false'}):
            with self.assertRaises(ValueError):checked_stop_control(bad)

    def test_balanced_replay_keeps_prior_lessons_without_old_typed_prefix_conflict(self):
        prose,typed,val=training_lessons()
        self.assertEqual(len(typed),448);self.assertEqual(len(prose),230);self.assertEqual(sum(map(len,val.values())),122)
        self.assertEqual({r['edition'] for r in prose},{'TC1','NTD','TC2','COURSE'})

    def test_cached_baseline_rejects_another_checkpoint_or_source_corpus(self):
        with self.assertRaises(ValueError):checked_baselines({'model_repo':'other'},{'manifest_sha256':'x'})

    def test_submission_name_and_provider_diagnostic_protect_private_inputs(self):
        self.assertLessEqual(len(RUN_SLUG),50)
        raw='validation error token=private-credential https://example.invalid/signed?token=abc '+('A'*200)
        safe=safe_submission_error(raw,'private-credential')
        self.assertNotIn('private-credential',safe);self.assertNotIn('example.invalid',safe);self.assertNotIn('A'*40,safe)
        self.assertIn('validation error',safe)


if __name__=='__main__':unittest.main()
