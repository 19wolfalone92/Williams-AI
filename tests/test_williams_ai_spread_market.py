import ast
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

from williams_ai.independent_chart_holdout import build_blind_market_cases
from williams_ai.spread_market_exam import frozen_market_cohort,grade_market_comparison,free_generation_grade,FEE_PER_SIDE,SLIPPAGE_PER_SIDE
from scripts.ai_run_spread_market_kaggle import worker_source

DATA=Path('examples/ai_binance_15pair_h1_prior_holdout_20260902.json')


class SpreadMarketTests(unittest.TestCase):
    def test_fifteen_assets_twenty_dates_and_no_reuse_of_old_thirty_cutoffs(self):
        q,k,raw,seal=frozen_market_cohort(DATA)
        old,_,_=build_blind_market_cases(json.loads(DATA.read_bytes()))
        self.assertEqual((len(q),len(k),len(seal['utc_dates'])),(30,30,20))
        self.assertFalse({r['case_id'] for r in q}&{r['case_id'] for r in old})
        self.assertTrue(all(len(r['market_context']['bars'])==240 for r in q))
        self.assertNotIn(b'true_class',raw);self.assertNotIn(b'future_return',raw)
        self.assertFalse(seal['candidate_selection_on_this_exam'])

    def test_future_change_cannot_change_visible_input_in_either_case_of_asset(self):
        archive=json.loads(DATA.read_bytes());_,_,_,seal=frozen_market_cohort(DATA)
        cuts=seal['cuts_by_symbol'];q,k,_=build_blind_market_cases(archive,cuts_by_symbol=cuts)
        changed=copy.deepcopy(archive);index=cuts['BTCUSDT'][0]+24
        for i in (1,2,3,4):changed['pairs']['BTCUSDT']['klines'][index][i]*=1.2
        q2,k2,_=build_blind_market_cases(changed,cuts_by_symbol=cuts)
        self.assertEqual(q,q2);self.assertNotEqual(k,k2)
        with self.assertRaises(ValueError):build_blind_market_cases(archive,cuts_by_symbol={s:(300,400) for s in cuts})

    def test_fixed_friction_and_abstentions_do_not_create_a_trading_qualification(self):
        q,k,_,_=frozen_market_cohort(DATA)
        rows=[{'case_id':r['case_id'],'probabilities':{'UP':0.,'DOWN':0.,'RANGE':1.},
            'generated_response':'RANGE','generated_complete':True,'generation_time_limit_reached':False} for r in q]
        report=grade_market_comparison(k,{'candidate':rows})
        cost=report['stages']['candidate']['friction_diagnostic']
        self.assertEqual(cost['range_abstentions'],30);self.assertEqual(cost['conditional_24h_directional_positions'],0)
        self.assertIsNone(cost['mean_net_return_per_conditional_position']);self.assertEqual(cost['mean_net_return_per_case'],0)
        self.assertEqual(cost['round_trip_friction_assumption'],2*(FEE_PER_SIDE+SLIPPAGE_PER_SIDE))
        self.assertFalse(report['trading_qualification']);self.assertFalse(report['execution_allowed'])
        self.assertAlmostEqual(report['uniform_probability_baseline']['multiclass_brier'],2/9)

    def test_generated_json_extra_words_timeout_or_missing_eos_are_not_repaired_to_pass(self):
        _,keys,_,_=frozen_market_cohort(DATA)
        good=[{'case_id':k['case_id'],'generated_response':k['true_class'],'generated_complete':True,'generation_time_limit_reached':False} for k in keys]
        self.assertEqual(free_generation_grade(keys,good)['correct'],30)
        for change in ({'generated_response':'{"direction":"UP"}'},{'generated_response':'UP because trend'},
                {'generated_complete':False},{'generation_time_limit_reached':True}):
            bad=[{**r,**change} for r in good]
            result=free_generation_grade(keys,bad)
            self.assertEqual(result['correct'],0);self.assertEqual(result['invalid_or_incomplete_answers'],30)
            self.assertFalse(result['execution_allowed'])

    def test_real_two_rank_entry_has_no_future_payload_or_unsubstituted_tokens(self):
        _,_,qbytes,seal=frozen_market_cohort(DATA)
        w=worker_source('a'*40,qbytes,{}, {'questions_sha256':seal['questions_file_sha256']},b'private original source TEST only; never train')
        tree=ast.parse(w);self.assertNotIn('KAGGLE_API_TOKEN',w);self.assertNotIn('WITHHELD_OUTCOME_KEYS',w)
        self.assertFalse(re.search(r'__[A-Z][A-Z0-9_]+__',w));self.assertIn('--nproc_per_node=2',w)
        with tempfile.TemporaryDirectory() as td:
            entry=Path(td)/'entry.py'
            node=next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='entry' and n.value.func.attr=='write_text')
            exec(compile(ast.Module(body=[node],type_ignores=[]),'entry_writer','exec'),
                {'entry':entry,'repo':Path(td),'questions':Path(td)/'q','adapters':Path(td)/'a','manifest':Path(td)/'m','out':Path(td)/'o','source_test':Path(td)/'t'})
            ast.parse(entry.read_text())
        self.assertIn("shutil.rmtree(repo/'examples')",w)


if __name__=='__main__':unittest.main()
