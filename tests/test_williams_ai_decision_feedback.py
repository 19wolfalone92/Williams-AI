"""Retry receives its real error, not a generated answer or weakened critic."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.algorithm_guard import audit_reasoning
from williams_ai.decision_contract import previous_decision_review
from williams_ai.local_model import LocalWilliamsModel
from williams_ai.deliberate_learning import DeliberateBookModel
import test_williams_ai_deliberate_learning as fixtures
from test_williams_ai_deliberate_learning import Student, proposal


class DecisionFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DeliberateLearningTests.setUpClass()
        cls.obs=deepcopy(fixtures.DeliberateLearningTests.obs)

    def test_feedback_reruns_critic_and_does_not_supply_a_corrected_answer(self):
        old=proposal(self.obs,setup='WM2')
        old['forged_audit']={'hard_failures':[],'correct_setup':'WM3'}
        before=deepcopy((self.obs,old))
        feedback=previous_decision_review(self.obs,old,'TC2')
        self.assertEqual(feedback['independent_critic_failures'],audit_reasoning(self.obs,old)['hard_failures'])
        self.assertIn('WM2_SUPER_AO_NOT_OBSERVED',feedback['independent_critic_failures'])
        self.assertNotIn('forged_audit',feedback['previous_model_answer_unverified'])
        self.assertEqual(feedback['previous_model_answer_unverified']['setup'],'WM2')
        self.assertFalse(feedback['correct_answer_supplied'])
        self.assertEqual((self.obs,old),before)

    def test_other_cutoff_symbol_or_edition_cannot_enter_same_chart_retry(self):
        old=proposal(self.obs,setup='WM2')
        for key,value in (('symbol','BTCUSDT'),('as_of','2027-01-01T00:00:00Z'),('edition','NTD')):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'MUST_MATCH'):
                previous_decision_review(self.obs,{**old,key:value},'TC2')

    def test_unbounded_or_nonfinite_prior_model_text_is_rejected(self):
        old=proposal(self.obs)
        for changed in ({**old,'reason':'x'*601},{**old,'unknown':float('nan')},
                        {**old,'learning_question':'x'*401}):
            with self.assertRaisesRegex(ValueError,'BOUNDED_PREVIOUS'):
                previous_decision_review(self.obs,changed,'TC2')

    def test_deliberate_retry_receives_exact_failed_answer_but_first_call_does_not(self):
        with TemporaryDirectory() as directory:
            student=Student(first={'setup':'WM2'},second={'setup':'WM2'})
            wrapper=DeliberateBookModel(student,None,Path(directory)/'memory.sqlite3',
                model_id='gguf-feedback-fixture',public_rules_only=True)
            result=wrapper.propose_chart_reasoning(self.obs)
            self.assertNotIn('previous_decision',student.calls[0][1])
            self.assertEqual(student.calls[1][1]['previous_decision'],proposal(self.obs,setup='WM2'))
            self.assertEqual(student.calls[0][0],student.calls[1][0])
            self.assertTrue(wrapper.latest['reconsideration_feedback_supplied'])
            self.assertEqual(wrapper.latest['status'],'UNSUPPORTED_AFTER_STUDY')
            self.assertTrue(audit_reasoning(self.obs,result)['hard_failures'])

    def test_actual_prompt_builder_includes_feedback_without_saving_its_claim_as_a_source(self):
        seen=[]
        old=proposal(self.obs,setup='WM2')
        class Runtime:
            def __init__(self,**kwargs):pass
            def create_chat_completion(self,**kwargs):
                seen.append(kwargs)
                return {'choices':[{'message':{'content':json.dumps(proposal(DecisionFeedbackTests.obs))}}]}
        with TemporaryDirectory() as directory:
            path=Path(directory)/'fake.gguf';path.write_bytes(b'GGUF NOT REAL WEIGHTS')
            model=LocalWilliamsModel(path,loader=Runtime)
            returned=model.propose_chart_reasoning(self.obs,previous_decision=old)
        payload=json.loads(seen[0]['messages'][1]['content'])
        review=payload['previous_decision_review']
        self.assertEqual(review['previous_model_answer_unverified']['setup'],'WM2')
        self.assertIn('WM2_SUPER_AO_NOT_OBSERVED',review['independent_critic_failures'])
        self.assertNotIn('previous_decision_review',returned)
        self.assertFalse(audit_reasoning(self.obs,returned)['execution_allowed'])

    def test_auto_qwen_chart_uses_reasoning_and_never_exports_private_draft(self):
        seen=[]
        class Runtime:
            metadata={'general.architecture':'qwen3'}
            def __init__(self,**kwargs):pass
            def create_chat_completion(self,**kwargs):
                seen.append(kwargs)
                text='<think>PRIVATE_CHART_DRAFT</think>'+json.dumps(proposal(DecisionFeedbackTests.obs))
                return {'choices':[{'finish_reason':'stop','message':{'content':text}}]}
        with TemporaryDirectory() as directory:
            path=Path(directory)/'fake.gguf';path.write_bytes(b'GGUF NOT REAL WEIGHTS')
            model=LocalWilliamsModel(path,loader=Runtime,learning_mode='auto',use_decision_schema=True)
            returned=model.propose_chart_reasoning(self.obs)
        self.assertNotIn('response_format',seen[0])
        self.assertEqual(seen[0]['seed'],1942)
        self.assertNotIn('PRIVATE_CHART_DRAFT',json.dumps(returned))
        self.assertNotIn('PRIVATE_CHART_DRAFT',json.dumps(model.latest_learning_call))


if __name__=='__main__':unittest.main()
