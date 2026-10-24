"""Protocol tests with a mock; these are never reported as real Qwen scores."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.book_index import index_pages
from williams_ai.book_rules import registry
from williams_ai.local_model import LocalWilliamsModel
from williams_ai.source_transfer_exam import zone_cases, run_zone_transfer_exam
from williams_ai.study import study_batch, _lesson_error_code


class SourceApplicationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.book=self.root/"books.db"
        book=next(x for x in registry()["books"] if x["id"]=="NTD")
        pages=[""]*book["pages"]
        pages[90]="Зона формируется по изменениям AO и AC. Private fixture sentinel. В Серой Зоне направления различаются."
        index_pages(self.book,"NTD",book["sha256"],pages,
                    declared_pages=book["pages"])

    def test_indicator_sign_and_candle_body_cannot_replace_delta_rule(self):
        cases=zone_cases()
        self.assertEqual(len(cases),12)
        observed=set()
        for case in cases:
            ao,ac=case["inputs"]["ao"],case["inputs"]["ac"]
            changes=(ao[1]>ao[0],ac[1]>ac[0])
            observed.add(changes)
            truth=("GREEN" if changes==(True,True) else
                   "RED" if changes==(False,False) else "GRAY")
            self.assertEqual(case["truth"],truth)
            self.assertEqual(set(case["public"]),{"id","edition","question","choices"})
        self.assertEqual(len(observed),4)
        self.assertEqual(zone_cases(17),zone_cases(17))
        self.assertNotEqual(zone_cases(17),zone_cases(18))

    def test_hidden_answers_and_private_explanations_never_exported_to_public(self):
        answers={c["public"]["id"]:c["expected"] for c in zone_cases()}
        calls=[]
        class Mock:
            def answer_exam_case(self,case,**kwargs):
                calls.append((case,kwargs))
                if not kwargs:
                    return {"choice":"", "reason":"malformed must not count as abstention success"}
                return {"choice":answers[case["id"]],"reason":"Private fixture sentinel."}
        result=run_zone_transfer_exam(Mock(),self.book)
        self.assertEqual(len(calls),24)
        self.assertTrue(all(set(c)=={"id","edition","question","choices"}
                            for c,_ in calls))
        self.assertEqual(result["phases"]["without_original_passage"]["correct"],0)
        self.assertEqual(result["phases"]["without_original_passage"]["malformed_or_failed"],12)
        self.assertEqual(result["phases"]["with_original_passage"]["correct"],12)
        self.assertNotIn("Private fixture sentinel",json.dumps(result))
        self.assertFalse(result["model_weights_modified"])
        self.assertFalse(result["execution_allowed"])

    def test_missing_original_definition_does_not_fall_back_to_model_generated_rule(self):
        book=next(x for x in registry()["books"] if x["id"]=="NTD")
        index_pages(self.book,"NTD",book["sha256"],[""]*book["pages"],
                    declared_pages=book["pages"])
        with self.assertRaisesRegex(ValueError,"REQUIRES_SOURCE_REVIEW"):
            run_zone_transfer_exam(object(),self.book)

    def test_truncated_generated_lesson_never_enters_private_memory(self):
        class Runtime:
            def __init__(self,**kwargs):pass
            def create_chat_completion(self,**kwargs):
                # Even syntactically complete text is rejected if the real
                # runtime reports it hit the hard generation limit.
                return {"choices":[{"finish_reason":"length","message":{"content":
                    json.dumps({"summary":"Incomplete","claims":[],"questions":[]})}}]}
        weights=self.root/"test.gguf";weights.write_bytes(b"TEST_ONLY")
        student=LocalWilliamsModel(weights,loader=Runtime)
        result=study_batch(self.book,self.root/"lessons.db","NTD",student,
                           model_id="test-runtime",budget=1,pdf_pages=(91,))
        self.assertEqual(result["studied_chunks"],0)
        self.assertEqual(result["failure_codes"],{"LOCAL_STUDY_OUTPUT_TRUNCATED":1})

    def test_retry_diagnostic_never_serializes_unknown_private_exception_text(self):
        self.assertEqual(_lesson_error_code(ValueError("Private fixture sentinel")),"ValueError")
        self.assertEqual(_lesson_error_code(ValueError("Invented supporting span")),
                         "LESSON_SUPPORT_NOT_IN_SOURCE")

    def test_exam_generation_limit_cannot_turn_partial_correct_choice_into_success(self):
        class Runtime:
            def __init__(self,**kwargs):self.last=None
            def create_chat_completion(self,**kwargs):
                self.last=kwargs
                return {"choices":[{"finish_reason":"length","message":{"content":
                    json.dumps({"choice":"A","reason":"Cut off before completing the argument"})}}]}
        weights=self.root/"exam-test.gguf";weights.write_bytes(b"TEST_ONLY")
        student=LocalWilliamsModel(weights,loader=Runtime)
        answer=student.answer_exam_case(zone_cases()[0]["public"])
        self.assertEqual(answer["choice"],"")
        self.assertEqual(answer["reason"],"GGUF_EXAM_OUTPUT_TRUNCATED")
        self.assertFalse(student._model.last["response_format"]["schema"]["additionalProperties"])
        self.assertNotIn("tools",student._model.last)


if __name__=="__main__":
    unittest.main()
