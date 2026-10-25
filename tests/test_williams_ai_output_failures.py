"""Interface failures are not conceptual lessons or repaired trade answers."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from williams_ai.algorithm_guard import audit_reasoning
from williams_ai.deliberate_learning import DeliberateBookModel
from williams_ai.learning_agenda import prior_reading_focus
from williams_ai.local_model import LocalWilliamsModel
import test_williams_ai_deliberate_learning as fixtures
from test_williams_ai_deliberate_learning import Student, proposal


class OutputFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DeliberateLearningTests.setUpClass()
        cls.obs=deepcopy(fixtures.DeliberateLearningTests.obs)

    def ask(self, text, *, finish="stop"):
        class Runtime:
            def __init__(self, **kwargs): pass
            def create_chat_completion(self, **kwargs):
                return {"choices":[{"finish_reason":finish,"message":{"content":text}}]}
        with TemporaryDirectory() as directory:
            path=Path(directory)/"fixture.gguf"
            path.write_bytes(b"NOT REAL MODEL WEIGHTS")
            model=LocalWilliamsModel(path,loader=Runtime,use_decision_schema=True)
            return model.propose_chart_reasoning(self.obs)

    def test_complete_json_with_length_finish_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError,"^GGUF_CHART_STRUCTURED_OUTPUT_TRUNCATED$"):
            self.ask(json.dumps(proposal(self.obs)),finish="length")

    def test_duplicate_fields_are_rejected_at_all_depths(self):
        original=json.dumps(proposal(self.obs))
        cases=(original.replace('"setup": "NO_SETUP"','"setup":"WM2","setup":"NO_SETUP"'),
               original[:-1]+',"debug":{"key":1,"key":1}}')
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError,"^GGUF_MALFORMED_JSON_RESPONSE$"):
                self.ask(raw)

    def test_nonfinite_constants_and_numeric_overflow_are_rejected(self):
        original=json.dumps(proposal(self.obs))
        for value in ("NaN","Infinity","-Infinity","1e999"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,"^GGUF_MALFORMED_JSON_RESPONSE$"):
                self.ask(original[:-1]+',"debug":['+value+']}')

    def test_nested_setups_and_wrong_flag_or_reference_types_are_rejected(self):
        for field,value in (("setup",{"WM3":{}}),("side",[]),("symbol",1),
                            ("execution_allowed","false"),("exchange_fill_confirmed",0),
                            ("rule_ids","TC2.WM3"),("rule_ids",[{}]),("reason",None)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,"^GGUF_INVALID_DECISION_FIELD_TYPES$"):
                self.ask(json.dumps({**proposal(self.obs),field:value}))

    def test_long_text_is_rejected_instead_of_silently_rewritten(self):
        for field,size in (("reason",601),("learning_question",401)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,"^GGUF_DECISION_TEXT_TOO_LONG$"):
                self.ask(json.dumps({**proposal(self.obs),field:"x"*size}))

    def test_well_formed_wrong_trade_claim_is_unchanged_and_rejected_by_critic(self):
        wrong=proposal(self.obs,setup="WM2")
        returned=self.ask(json.dumps(wrong))
        self.assertEqual(returned,wrong)
        self.assertIn("WM2_SUPER_AO_NOT_OBSERVED",audit_reasoning(self.obs,returned)["hard_failures"])

    def test_runtime_failure_does_not_read_or_study_books_or_pollute_reading_agenda(self):
        class Broken:
            calls=0
            def propose_chart_reasoning(self,*args,**kwargs):
                self.calls+=1
                raise ValueError("GGUF_RUNTIME_CONTEXT_WINDOW_EXCEEDED")
        with TemporaryDirectory() as directory:
            student=Broken();memory=Path(directory)/"memory.sqlite3"
            wrapper=DeliberateBookModel(student,None,memory,model_id="gguf-interface-fixture",public_rules_only=True)
            with patch("williams_ai.deliberate_learning.verified_private_book_pages") as pages, \
                 patch("williams_ai.deliberate_learning.study_batch") as study:
                with self.assertRaisesRegex(ValueError,"^GGUF_RUNTIME_CONTEXT_WINDOW_EXCEEDED$"):
                    wrapper.propose_chart_reasoning(self.obs)
                pages.assert_not_called();study.assert_not_called()
            self.assertEqual(student.calls,1)
            self.assertEqual(wrapper.latest["status"],"MODEL_INTERFACE_FAILURE_NO_BOOK_STUDY")
            self.assertFalse(wrapper.latest["reconsideration_performed"])
            self.assertEqual(wrapper.latest["source_rules"],[])
            agenda=prior_reading_focus(memory,model_id="gguf-interface-fixture",edition="TC2",
                as_of="2026-10-25T00:00:00Z",source_kind="PUBLIC_REVIEWED_RULE_PARAPHRASE")
            self.assertEqual(agenda["past_requests"],0)
            self.assertEqual(agenda["topic"],"NONE")

    def test_non_object_model_output_does_not_trigger_another_inference(self):
        class Broken:
            calls=0
            def propose_chart_reasoning(self,*args,**kwargs):
                self.calls+=1
                return []
        with TemporaryDirectory() as directory:
            student=Broken()
            wrapper=DeliberateBookModel(student,None,Path(directory)/"memory.sqlite3",
                model_id="gguf-interface-fixture",public_rules_only=True)
            with self.assertRaisesRegex(ValueError,"^INVALID_MODEL_DECISION_OBJECT$"):
                wrapper.propose_chart_reasoning(self.obs)
            self.assertEqual(student.calls,1)

    def test_private_runtime_exception_is_not_returned_to_public_diagnostics(self):
        class Broken:
            def propose_chart_reasoning(self,*args,**kwargs):
                raise ValueError("private source /home/user/book.pdf")
        with TemporaryDirectory() as directory:
            wrapper=DeliberateBookModel(Broken(),None,Path(directory)/"memory.sqlite3",
                model_id="gguf-interface-fixture",public_rules_only=True)
            with self.assertRaisesRegex(ValueError,"^UNCLASSIFIED_MODEL_ERROR$") as caught:
                wrapper.propose_chart_reasoning(self.obs)
            self.assertTrue(caught.exception.__suppress_context__)
            self.assertNotIn("private source",json.dumps(wrapper.latest))

    def test_conceptual_failure_still_gets_one_source_review_and_same_critic(self):
        with TemporaryDirectory() as directory:
            student=Student(first={"setup":"WM2"},second={"setup":"NO_SETUP"})
            wrapper=DeliberateBookModel(student,None,Path(directory)/"memory.sqlite3",
                model_id="gguf-interface-fixture",public_rules_only=True)
            result=wrapper.propose_chart_reasoning(self.obs)
            self.assertEqual(len(student.calls),2)
            self.assertEqual(student.calls[0][0],student.calls[1][0])
            self.assertTrue(wrapper.latest["reconsideration_performed"])
            self.assertTrue(wrapper.latest["public_rule_references"])
            self.assertEqual(audit_reasoning(self.obs,result)["hard_failures"],[])


if __name__=="__main__": unittest.main()
