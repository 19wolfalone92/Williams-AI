import json
from pathlib import Path
import unittest
from williams_ai.source_consolidation import fresh_test_cases
from williams_ai.direct_mode_transfer import grade_direct_test
from williams_ai.frozen_source_test import recompute_source_scores,grade_source_test
from williams_ai.deep_book_research import EDITIONS
from scripts.ai_run_source_consolidation_final_kaggle import STAGES
from scripts.ai_run_spread_market_kaggle import worker_source
from scripts.ai_verify_spread_market_result import numerical_equal


class SourceConsolidationFinalTests(unittest.TestCase):
    def test_cross_python_float_roundoff_does_not_hide_material_or_boolean_changes(self):
        self.assertTrue(numerical_equal({'brier':.21247531710472092},{'brier':.21247531710472087}))
        self.assertFalse(numerical_equal({'correct':15},{'correct':14}))
        self.assertFalse(numerical_equal({'accepted':False},{'accepted':True}))
        self.assertFalse(numerical_equal({'brier':.21247531710472092},{'brier':.213}))

    def test_parent_reference_cannot_be_missing_or_ignored(self):
        cases=fresh_test_cases();good=[{'case_id':c['case_id'],'response':c['answer'],'thinking_complete':True,'response_time_limit_reached':False} for c in cases]
        bad=[{**r,'response':'{}'} for r in good];stages={s:bad for s in STAGES};stages['candidate']=good
        self.assertTrue(grade_direct_test(cases,stages,include_restorative_parent=True)['transfer_gate_passed'])
        stages['restorative768']=good;self.assertFalse(grade_direct_test(cases,stages,include_restorative_parent=True)['transfer_gate_passed'])
        del stages['restorative768']
        with self.assertRaises(ValueError):grade_direct_test(cases,stages,include_restorative_parent=True)

    def test_constant_meaningful_field_and_incomplete_answer_remain_errors(self):
        cases=fresh_test_cases();good=[{'case_id':c['case_id'],'response':c['answer'],'thinking_complete':True,'response_time_limit_reached':False} for c in cases]
        stages={s:[{**r,'response':'{}'} for r in good] for s in STAGES}
        stages['candidate']=[{**r,'response':'{"hump_valid":true,"execution_allowed":false}'} if c['rule_id']=='COURSE.HUMP' else r for c,r in zip(cases,good)]
        g=grade_direct_test(cases,stages,include_restorative_parent=True);self.assertGreater(g['stages']['candidate']['accuracy'],.8);self.assertFalse(g['transfer_gate_passed'])
        stages['candidate']=[{**r,'thinking_complete':False} for r in good]
        self.assertEqual(grade_direct_test(cases,stages,include_restorative_parent=True)['stages']['candidate']['correct'],0)

    def test_every_five_model_source_window_and_parent_preservation_are_required(self):
        counts=dict.fromkeys(EDITIONS,2);rows=[(s,e,i,.97 if s=='candidate' else 1.) for s in STAGES for e in EDITIONS for i in range(2)]
        scores=recompute_source_scores(rows,counts,include_restorative_parent=True)
        self.assertTrue(grade_source_test(scores,include_restorative_parent=True)['source_test_gate_passed'])
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):recompute_source_scores(bad,counts,include_restorative_parent=True)
        scores['restorative768']=dict.fromkeys((*EDITIONS,'ALL'),.5)
        self.assertFalse(grade_source_test(scores,include_restorative_parent=True)['source_test_gate_passed'])

    def test_precommit_and_worker_keep_all_five_model_identifiers(self):
        d=json.loads(Path('experiments/source-consolidation-final-20261028-design.json').read_bytes())
        self.assertEqual(d['models'],list(STAGES));self.assertEqual(d['direct_seconds_per_response'],10);self.assertFalse(d['execution_allowed'])
        config={s:{n:{'url':'https://example.invalid/own-output','sha256':'a'*64} for n in ('adapter_config.json','adapter_model.safetensors')} for s in STAGES if s!='base'}
        w=worker_source('a'*40,b'past market only',config,{'source_consolidation_final':True},b'private test-only source',b'[]',b'[]')
        compile(w,'worker','exec');self.assertIn('restorative768',w);self.assertNotIn('KAGGLE_API_TOKEN',w)


if __name__=='__main__':unittest.main()
