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
from williams_ai.guided_retry import safe_error_code
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

    def test_chart_reports_with_same_runtime_input_and_only_unverified_final_answer(self):
        seen=[]
        class Runtime:
            metadata={'general.architecture':'qwen3'}
            def __init__(self,**kwargs):pass
            def create_chat_completion(self,**kwargs):
                seen.append(kwargs)
                # The analysis contains the actual failure seen with Qwen:
                # several nested scenarios instead of a single setup string.
                if 'response_format' not in kwargs:
                    text='<think>PRIVATE_CHART_DRAFT</think>'+json.dumps({'setup':{'WM2':'unverified'}})
                else:
                    # An unsupported typed answer MUST remain unsupported.
                    text=json.dumps(proposal(DecisionFeedbackTests.obs,setup='WM2'))
                return {'choices':[{'finish_reason':'stop','message':{'content':text}}]}
        with TemporaryDirectory() as directory:
            path=Path(directory)/'fake.gguf';path.write_bytes(b'GGUF NOT REAL WEIGHTS')
            model=LocalWilliamsModel(path,loader=Runtime,learning_mode='auto',use_decision_schema=True)
            returned=model.propose_chart_reasoning(self.obs)
        self.assertNotIn('response_format',seen[0])
        self.assertEqual(seen[0]['seed'],1942)
        self.assertEqual(len(seen),2)
        self.assertEqual(seen[1]['messages'][1],seen[0]['messages'][1])
        self.assertTrue(seen[0]['messages'][0]['content'].startswith(seen[1]['messages'][0]['content']))
        self.assertIn('/no_think',seen[1]['messages'][-1]['content'])
        self.assertEqual(json.loads(seen[1]['messages'][2]['content']),{'setup':{'WM2':'unverified'}})
        self.assertEqual(seen[1]['response_format']['schema']['properties']['setup']['type'],'string')
        self.assertEqual(returned['setup'],'WM2')
        self.assertIn('WM2_SUPER_AO_NOT_OBSERVED',audit_reasoning(self.obs,returned)['hard_failures'])
        self.assertNotIn('PRIVATE_CHART_DRAFT',json.dumps(returned))
        self.assertNotIn('PRIVATE_CHART_DRAFT',json.dumps(model.latest_learning_call))
        self.assertNotIn('PRIVATE_CHART_DRAFT',json.dumps(model.latest_chart_call))
        self.assertNotIn('PRIVATE_CHART_DRAFT',json.dumps(seen[1]))
        self.assertEqual(model.latest_chart_call['model_calls'],2)

    def test_unfinished_analysis_never_gets_a_fabricated_structured_answer(self):
        calls=[]
        class Runtime:
            metadata={'general.architecture':'qwen3'}
            def __init__(self,**kwargs):pass
            def create_chat_completion(self,**kwargs):
                calls.append(kwargs)
                return {'choices':[{'finish_reason':'length','message':{'content':json.dumps(proposal(DecisionFeedbackTests.obs))}}]}
        with TemporaryDirectory() as directory:
            path=Path(directory)/'fake.gguf';path.write_bytes(b'NOT REAL WEIGHTS')
            model=LocalWilliamsModel(path,loader=Runtime,learning_mode='auto')
            with self.assertRaisesRegex(ValueError,'^GGUF_CHART_REASONING_OUTPUT_TRUNCATED$'):
                model.propose_chart_reasoning(self.obs)
        self.assertEqual(len(calls),1)

    def test_truncated_reporting_is_rejected_even_if_text_is_complete_json(self):
        calls=[]
        class Runtime:
            metadata={'general.architecture':'qwen3'}
            def __init__(self,**kwargs):pass
            def create_chat_completion(self,**kwargs):
                calls.append(kwargs)
                return {'choices':[{'finish_reason':'stop' if len(calls)==1 else 'length',
                    'message':{'content':json.dumps(proposal(DecisionFeedbackTests.obs))}}]}
        with TemporaryDirectory() as directory:
            path=Path(directory)/'fake.gguf';path.write_bytes(b'NOT REAL WEIGHTS')
            model=LocalWilliamsModel(path,loader=Runtime,learning_mode='auto')
            with self.assertRaisesRegex(ValueError,'^GGUF_CHART_STRUCTURED_OUTPUT_TRUNCATED$'):
                model.propose_chart_reasoning(self.obs)
        self.assertEqual(len(calls),2)

    def test_legacy_chart_also_rejects_truncated_complete_json(self):
        class Runtime:
            def __init__(self,**kwargs):pass
            def create_chat_completion(self,**kwargs):
                return {'choices':[{'finish_reason':'length',
                    'message':{'content':json.dumps(proposal(DecisionFeedbackTests.obs))}}]}
        with TemporaryDirectory() as directory:
            path=Path(directory)/'fake.gguf';path.write_bytes(b'NOT REAL WEIGHTS')
            model=LocalWilliamsModel(path,loader=Runtime)
            with self.assertRaisesRegex(ValueError,'^GGUF_CHART_STRUCTURED_OUTPUT_TRUNCATED$'):
                model.propose_chart_reasoning(self.obs)

    def test_invalid_analysis_is_not_replaced_with_a_safe_sounding_default(self):
        calls=[]
        class Runtime:
            metadata={'general.architecture':'qwen3'}
            def __init__(self,**kwargs):pass
            def create_chat_completion(self,**kwargs):
                calls.append(kwargs)
                return {'choices':[{'finish_reason':'stop','message':{'content':'<think>unfinished'}}]}
        with TemporaryDirectory() as directory:
            path=Path(directory)/'fake.gguf';path.write_bytes(b'NOT REAL WEIGHTS')
            model=LocalWilliamsModel(path,loader=Runtime,learning_mode='auto')
            with self.assertRaises(ValueError):model.propose_chart_reasoning(self.obs)
        self.assertEqual(len(calls),1)

    def test_chart_stage_errors_remain_specific_and_do_not_expose_runtime_text(self):
        for code in ('GGUF_CHART_REASONING_OUTPUT_TRUNCATED',
                     'GGUF_CHART_STRUCTURED_OUTPUT_TRUNCATED',
                     'GGUF_CHART_FINAL_SUMMARY_INVALID'):
            self.assertEqual(safe_error_code(ValueError(code)),code)
        self.assertEqual(safe_error_code(ValueError('private page /home/user/book.pdf')),
                         'UNCLASSIFIED_MODEL_ERROR')


if __name__=='__main__':unittest.main()
