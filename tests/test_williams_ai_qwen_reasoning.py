"""A parser cannot grade a tentative choice inside thinking as the answer."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.local_model import LocalWilliamsModel, _final_learning_json

CASE={"id":"reasoning-test","edition":"NTD","question":"Учебный вопрос о зоне",
      "choices":{"A":"Green","B":"Red","C":"Gray"}}


class Runtime:
    metadata={"general.architecture":"qwen3"}
    def __init__(self,**kwargs):
        self.calls=[]
        self.content='<think>PRIVATE_DRAFT {"choice":"A","reason":"tentative"}</think>\n{"choice":"C","reason":"AO and AC change in opposite directions."}'
        self.finish="stop"
    def create_chat_completion(self,**kwargs):
        self.calls.append(kwargs)
        return {"choices":[{"finish_reason":self.finish,"message":{"content":self.content}}],
                "usage":{"completion_tokens":100}}


class QwenLearningReasoningTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.weights=Path(self.tmp.name)/"mock.gguf"
        self.weights.write_bytes(b"NOT_REAL_MODEL")

    def model(self,**kwargs):
        kwargs.setdefault("learning_mode","auto")
        return LocalWilliamsModel(self.weights,loader=Runtime,**kwargs)

    def test_auto_qwen3_reasons_then_grades_only_final_json(self):
        model=self.model()
        answer=model.answer_exam_case(CASE)
        self.assertEqual(answer["choice"],"C")
        self.assertNotIn("PRIVATE_DRAFT",json.dumps(answer))
        self.assertNotIn("PRIVATE_DRAFT",json.dumps(model.latest_learning_call))
        call=model._model.calls[0]
        self.assertNotIn("response_format",call)
        self.assertNotIn("tools",call)
        self.assertEqual(call["temperature"],0.6)
        self.assertEqual(call["seed"],1942)
        self.assertEqual(call["max_tokens"],2048)
        self.assertEqual(json.loads(call["messages"][1]["content"]),CASE)

    def test_incomplete_thinking_cannot_be_mistaken_for_a_final_answer(self):
        for raw in ('<think>{"choice":"C","reason":"draft"}',
                    '<think>one<think>two</think>{"choice":"C"}',
                    'unexpected prefix<think>draft</think>{"choice":"C"}'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                _final_learning_json(raw)

    def test_limit_rejects_a_correct_but_incomplete_response(self):
        model=self.model();model._model.finish="length"
        self.assertEqual(model.answer_exam_case(CASE)["choice"],"")
        self.assertEqual(model.latest_learning_call["finish_reason"],"length")

    def test_extra_fields_or_non_json_final_answer_are_not_silently_repaired(self):
        model=self.model()
        model._model.content='<think>draft</think>{"choice":"C","reason":"ok","trade_action":"BUY"}'
        self.assertEqual(model.answer_exam_case(CASE)["choice"],"")
        model._model.content='<think>draft</think>My answer is {"choice":"C","reason":"ok"}'
        with self.assertRaisesRegex(ValueError,"GGUF_INVALID_FINAL_JSON"):
            model.answer_exam_case(CASE)

    def test_explicit_legacy_mode_still_allows_controlled_baseline(self):
        model=self.model(learning_mode="legacy_json")
        model._model.content='{"choice":"B","reason":"baseline"}'
        self.assertEqual(model.answer_exam_case(CASE)["choice"],"B")
        self.assertIn("response_format",model._model.calls[0])
        self.assertEqual(model._model.calls[0]["temperature"],0)

    def test_reasoning_candidate_is_not_promoted_to_default(self):
        model=LocalWilliamsModel(self.weights,loader=Runtime)
        self.assertFalse(model._learning_reasoning)

    def test_final_fence_and_plain_json_both_parse_without_draft(self):
        for raw in ('<think>draft</think>```json\n{"choice":"B","reason":"ok"}\n```',
                    '{"choice":"B","reason":"ok"}'):
            self.assertEqual(json.loads(_final_learning_json(raw))["choice"],"B")


if __name__=="__main__":
    unittest.main()
