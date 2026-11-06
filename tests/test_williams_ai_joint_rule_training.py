import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ai_run_joint_rule_kaggle import groups_and_design, worker_source, require_new_kernel
from scripts.ai_verify_joint_rules import independent_boundary_key, audit_groups
from williams_ai.joint_rule_curriculum import boundary_cases, selection, PROTOCOL
from williams_ai.historical_rule_curriculum import score


class JointCorrectiveTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, cls.design = groups_and_design()

    def test_every_training_and_held_out_key_is_independently_recalculated(self):
        for split, groups in self.groups.items():
            result = audit_groups(groups)
            self.assertEqual({k:v['count'] for k,v in result.items()}, self.design['counts'][split])
        for row in self.groups['test']['boundary']:
            altered = {**row, 'answer':'{"fabricated":true}', 'teaching_explanation':'wrong'}
            self.assertEqual(independent_boundary_key(altered), json.loads(row['answer']))

    def test_own_kernel_inventory_refuses_duplicates_and_preserves_auth_errors(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        api=Mock();api.kernels_list.return_value=[]
        require_new_kernel(api,'owner')
        api.kernels_status.assert_not_called()
        api.kernels_list.return_value=[SimpleNamespace(ref='owner/williams-ai-joint-rules-20261106')]
        with self.assertRaises(RuntimeError):require_new_kernel(api,'owner')
        api.kernels_list.side_effect=PermissionError('403')
        with self.assertRaises(PermissionError):require_new_kernel(api,'owner')

    def test_ac_contains_both_sides_both_signs_ready_not_ready_flat_and_zero(self):
        rows = [r for r in boundary_cases('train') if r['rule_id']=='NTD.AC']
        strata = {(r['operands']['side'], json.loads(r['answer'])['required_bars'], json.loads(r['answer'])['signal_ready']) for r in rows}
        for side in ('LONG','SHORT'):
            for count in (2,3):
                for ready in (True,False):self.assertIn((side,count,ready),strata)
        self.assertTrue(any(json.loads(r['answer'])['required_bars'] is None for r in rows))
        self.assertTrue(any(len(set(r['operands']['ac_oldest_to_latest']))==1 for r in rows))

    def test_no_exact_boundary_operands_leak_across_the_frozen_splits(self):
        hashes = {s:{json.dumps([r['rule_id'],r['operands']],sort_keys=True) for r in gs['boundary']} for s,gs in self.groups.items()}
        self.assertFalse(hashes['train'] & hashes['validation'])
        self.assertFalse(hashes['train'] & hashes['test'])
        self.assertFalse(hashes['validation'] & hashes['test'])
        train_ids = {r['case_id'] for rows in self.groups['train'].values() for r in rows}
        test_ids = {r['case_id'] for rows in self.groups['test'].values() for r in rows}
        self.assertFalse(train_ids & test_ids)

    def test_missing_right_fractal_bars_and_equal_neighbors_are_not_confirmed(self):
        row = deepcopy(next(r for r in self.groups['train']['boundary'] if r['rule_id']=='TC2.WM3'))
        row['operands'].update(kind='HIGH',closed_extrema_oldest_first=['1','2','3','2','1'],right_closed=2)
        self.assertTrue(independent_boundary_key(row)['confirmed'])
        row['operands']['closed_extrema_oldest_first'][3]='3'
        self.assertFalse(independent_boundary_key(row)['confirmed'])
        row['operands'].update(closed_extrema_oldest_first=['1','2','3','2'],right_closed=1)
        self.assertFalse(independent_boundary_key(row)['confirmed'])

    def test_risk_rounds_down_to_nondecimal_lots_and_can_be_zero(self):
        row = deepcopy(next(r for r in self.groups['train']['boundary'] if r['rule_id']=='TC1.PSY.PLAN'))
        row['operands']={'equity':'1000','risk_fraction':'0.01','distance':'3','quantity_step':'0.03'}
        self.assertEqual(independent_boundary_key(row)['quantity'],3.33)
        row['operands']['distance']='1000'
        self.assertEqual(independent_boundary_key(row)['quantity'],0)

    def test_book_regression_and_collapse_veto_high_average_development_accuracy(self):
        grades = {}
        for name,rows in self.groups['validation'].items():
            responses=[{'case_id':r['case_id'],'response':r['answer'],'thinking_complete':True,'response_time_limit_reached':False} for r in rows]
            grades[name]=score(rows,responses)
        before=dict.fromkeys(('TC1','NTD','TC2','COURSE','ALL'),1.)
        refs={name:before for name in ('focused69_source','previous40_source','focused69_practice','previous40_practice')}
        good=selection(grades,before,before,refs)
        self.assertGreater(good['score'],1)
        self.assertLess(selection(grades,{**before,'TC2':1.021},before,refs)['score'],-3)
        broken=deepcopy(grades);broken['numeric']['per_rule']['TC1.PSY.PLAN']['correct']=0
        self.assertLess(selection(broken,before,before,refs)['score'],good['score'])
        with self.assertRaises(ValueError):selection(grades,{**before,'NTD':float('nan')},before,refs)

    def test_worker_contains_no_keys_and_pins_dual_gpu_training_entry(self):
        files={name:{'url':'https://example.invalid/own-adapter','sha256':'a'*64} for name in ('adapter_config.json','adapter_model.safetensors')}
        worker=worker_source('a'*40,b'PRIVATE_TRAIN_SOURCE',b'NO_EXAM_KEYS',b'PROMPTS_ONLY',files,files,{})
        tree=ast.parse(worker)
        self.assertIn('--nproc_per_node=2',worker)
        self.assertNotIn('KAGGLE_API_TOKEN',worker)
        self.assertIn("shutil.rmtree(repo/'examples')",worker)
        with tempfile.TemporaryDirectory() as td:
            entry=Path(td)/'entry.py'
            node=next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
                and n.value.func.value.id=='entry' and n.value.func.attr=='write_text')
            exec(compile(ast.Module(body=[node],type_ignores=[]),'entry_writer','exec'),{'entry':entry,'repo':Path(td),'private':Path(td)/'private','out':Path(td)/'out'})
            ast.parse(entry.read_text())
            self.assertIn('joint_rule_gpu import train',entry.read_text())
        self.assertFalse(PROTOCOL['do_sample'])

    def test_truncated_or_repaired_answers_never_count_as_model_success(self):
        rows=self.groups['test']['boundary']
        answers=[{'case_id':r['case_id'],'response':r['answer'],'thinking_complete':True,'response_time_limit_reached':False} for r in rows]
        self.assertEqual(score(rows,answers)['correct'],144)
        answers[0]['thinking_complete']=False
        answers[1]['response']='```json\n'+answers[1]['response']+'\n```'
        self.assertEqual(score(rows,answers)['correct'],142)


if __name__=='__main__':unittest.main()
