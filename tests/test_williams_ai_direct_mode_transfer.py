import json
from pathlib import Path
import unittest

from williams_ai.direct_mode_transfer import direct_test_cases,direct_design,select_development_mode,grade_direct_test
from williams_ai.restorative_book_practice import restorative_cases,design_manifest
from williams_ai.contrastive_book_practice import RULES,score_responses


class DirectModeTransferTests(unittest.TestCase):
    def test_original_thinking_design_unchanged_and_new_numbers_disjoint(self):
        self.assertEqual(design_manifest(),json.loads(Path('experiments/restorative-20261028-design.json').read_text()))
        old={r['question'] for s in ('train','validation','test') for r in restorative_cases(s)}
        new=direct_test_cases();self.assertEqual(len(new),112);self.assertFalse(old&{r['question'] for r in new})
        self.assertFalse(direct_design()['expected_test_keys_sent_to_gpu'])

    def test_mode_selection_prioritizes_weakest_rule_and_does_not_receive_test_scores(self):
        cases=restorative_cases('validation')
        def generated_grade(limits):
            seen=dict.fromkeys(RULES,0);rows=[]
            for c in cases:
                r=c['rule_id'];seen[r]+=1
                rows.append({'case_id':c['case_id'],'response':c['answer'] if seen[r]<=limits[r] else '{}',
                    'thinking_complete':True,'response_time_limit_reached':False})
            return score_responses(cases,rows)
        a=generated_grade(dict.fromkeys(RULES,14))
        b=generated_grade({**dict.fromkeys(RULES,15),RULES[0]:8})
        self.assertEqual(select_development_mode(a,b)['selected_mode'],'thinking')
        b=generated_grade({**dict.fromkeys(RULES,15),RULES[0]:14})
        result=select_development_mode(a,b);self.assertEqual(result['selected_mode'],'direct');self.assertFalse(result['test_scores_used_for_selection'])
        self.assertEqual(select_development_mode(a,a)['selected_mode'],'thinking')
        with self.assertRaises(ValueError):select_development_mode(a,{**b,'accuracy':1.0})
        different={**b,'cases':[{**r,'case_id':'different-'+r['case_id']} for r in b['cases']]}
        with self.assertRaises(ValueError):select_development_mode(a,different)

    def test_direct_admission_requires_every_family_even_if_total_is_high(self):
        cases=direct_test_cases()
        good=[{'case_id':c['case_id'],'response':c['answer'],'thinking_complete':True,'response_time_limit_reached':False} for c in cases]
        empty=[{**r,'response':'{}'} for r in good]
        stages={s:empty for s in ('base','previous40','focused69')};stages['candidate']=good
        self.assertTrue(grade_direct_test(cases,stages)['transfer_gate_passed'])
        broken=[]
        for c,r in zip(cases,good):
            broken.append({**r,'response':'{"hump_valid":true,"execution_allowed":false}'} if c['rule_id']=='COURSE.HUMP' else r)
        stages['candidate']=broken;result=grade_direct_test(cases,stages)
        self.assertGreater(result['stages']['candidate']['accuracy'],.8)
        self.assertFalse(result['transfer_gate_passed']);self.assertFalse(result['execution_allowed'])


if __name__=='__main__':unittest.main()
