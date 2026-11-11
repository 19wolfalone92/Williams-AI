import ast
import json
from pathlib import Path
import re
import tempfile
import unittest

from scripts.ai_run_restorative_kaggle import worker_source
from scripts.ai_verify_restorative_result import independent_key
from williams_ai.source_consolidation import START_SHA,selection,fresh_test_cases,design
from williams_ai.restorative_book_practice import restorative_cases,design_manifest
from williams_ai.direct_mode_transfer import direct_test_cases
from williams_ai.contrastive_book_practice import RULES,question_rows,correct_answer


class SourceConsolidationTests(unittest.TestCase):
    def test_text_gates_precede_aggregate_rule_accuracy_and_no_reference_book_regression_is_hidden(self):
        scores=dict.fromkeys(('TC1','NTD','TC2','COURSE','ALL'),1.)
        baselines={k:scores for k in ('source_baseline_loss','initial_focused_source_loss','practice_baseline_loss','initial_focused_practice_loss','parent_source_loss','parent_practice_loss')}
        grade=lambda n:{'accuracy':n/16,'per_rule':{r:{'correct':n,'total':16} for r in RULES},'constant_fields_despite_varied_expected_values':{}}
        not_preserved=selection(grade(16),scores,scores,baselines)
        improved=dict.fromkeys(scores,.97);preserved=selection(grade(12),improved,improved,baselines)
        self.assertGreater(preserved['score'],not_preserved['score']);self.assertTrue(preserved['all_text_gates_accepted'])
        weak=selection(grade(16),{**improved,'NTD':1.03},improved,baselines)
        self.assertLess(weak['score'],-100.)
        harder={**baselines,'initial_focused_source_loss':dict.fromkeys(scores,.9)}
        self.assertLess(selection(grade(16),improved,improved,harder)['score'],-100.)

    def test_fresh_numeric_cohort_and_independent_keys_and_old_test_unchanged(self):
        self.assertEqual(design_manifest(),json.loads(Path('experiments/restorative-20261028-design.json').read_bytes()))
        old={r['question'] for split in ('train','validation','test') for r in restorative_cases(split)}
        old|={r['question'] for r in direct_test_cases()}
        cases=fresh_test_cases();self.assertEqual(len(cases),112);self.assertFalse(old&{r['question'] for r in cases})
        for c,q in zip(cases,question_rows(cases)):self.assertTrue(correct_answer(json.dumps(independent_key(q)),c['answer']))
        self.assertEqual(design()['source_test_pages_used_for_training_or_selection'],0)

    def test_actual_two_gpu_entry_selects_source_only_protocol_without_sending_test_questions(self):
        files={'files':{n:{'url':'https://example.invalid/own-output','sha256':'a'*64} for n in ('adapter_config.json','adapter_model.safetensors')}}
        worker=worker_source('a'*40,b'private train/validation only',b'[]',files,{},files,source_consolidation=True)
        tree=ast.parse(worker);self.assertIn(START_SHA,worker);self.assertNotIn('KAGGLE_API_TOKEN',worker)
        self.assertFalse(re.search(r'__[A-Z][A-Z0-9_]+__',worker));self.assertIn('--nproc_per_node=2',worker)
        with tempfile.TemporaryDirectory() as td:
            entry=Path(td)/'entry.py'
            node=next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='entry' and n.value.func.attr=='write_text')
            exec(compile(ast.Module(body=[node],type_ignores=[]),'entry_writer','exec'),{'entry':entry,'repo':Path(td),'private':Path(td)/'p','questions':Path(td)/'q','adapter':Path(td)/'a','reference':Path(td)/'r'})
            parsed=ast.parse(entry.read_text());call=parsed.body[-1].value
            self.assertEqual(len(call.args),6);self.assertEqual([(k.arg,ast.literal_eval(k.value)) for k in call.keywords],[('source_consolidation',True)])


if __name__=='__main__':unittest.main()
