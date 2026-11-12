"""Regression suite for actual v5 exam decoding failures (no GPU / no keys).

The prior v5 model scored 1/16 with sources and abstained on 15/16. This
suite tests the recovery mechanism, not neural knowledge or exam accuracy.
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

from williams_ai.local_model import LocalWilliamsModel
from williams_ai.unprompted_source_application import answer_with_self_retrieval
from williams_ai.four_book_blind_exam import four_book_blind_cases


def result(content: str, finish: str="stop") -> dict:
    return {"choices":[{"finish_reason":finish,"message":{"content":content}}]}


def student(*outputs: dict, thinking: bool=True) -> LocalWilliamsModel:
    m=object.__new__(LocalWilliamsModel)
    m._model=Mock()
    m._model.create_chat_completion.side_effect=list(outputs)
    m._learning_reasoning=thinking
    m._reasoning_budget=512
    m.latest_learning_call=None
    return m


class V5ExamDecodeAudit(unittest.TestCase):
    def setUp(self):
        self.case={"id":"research-test-only","edition":"TC2",
                   "question":"Какая проверка закрытых фракталов обязательна?",
                   "choices":{"A":"Два закрытых правых бара",
                              "B":"Достаточно текущего бара",
                              "C":"Достаточно AO"}}

    def test_reasoning_length_gets_one_bounded_json_recovery(self):
        model=student(result("<think>not completed","length"),
                      result('{"choice":"A","reason":"Проверяются правые закрытые бары"}'))
        answer=model.answer_exam_case(self.case)
        self.assertEqual(answer["choice"],"A")
        self.assertEqual(model._model.create_chat_completion.call_count,2)
        self.assertEqual(model.latest_learning_call["mode"],"QWEN3_EXAM_JSON_RECOVERY")
        self.assertEqual(model.latest_learning_call["fallback_reason"],
                         "GGUF_EXAM_REASONING_TRUNCATED")
        call=model._model.create_chat_completion.call_args.kwargs
        self.assertEqual(call["temperature"],0.0)
        self.assertEqual(call["max_tokens"],240)
        self.assertFalse(call["response_format"]["schema"]["additionalProperties"])
        self.assertNotIn("tools",call)
        self.assertNotIn("expected",str(call))
        self.assertNotIn("trade_action",str(call["response_format"]["schema"]))

    def test_malformed_reasoning_recovered_without_key_or_tool(self):
        model=student(result('garbled output'),
                      result('{"choice":"C","reason":"Проверка исходных правил"}'))
        answer=model.answer_exam_case(self.case)
        self.assertEqual(answer["choice"],"C")
        self.assertEqual(model.latest_learning_call["fallback_reason"],
                         "GGUF_EXAM_REASONING_MALFORMED")
        self.assertEqual(model._model.create_chat_completion.call_count,2)

    def test_normal_reasoning_never_retries(self):
        model=student(result('<think>private thought</think>{"choice":"B","reason":"Цитата"}'))
        answer=model.answer_exam_case(self.case)
        self.assertEqual(answer["choice"],"B")
        self.assertFalse(model.latest_learning_call["fallback_used"])
        self.assertEqual(model._model.create_chat_completion.call_count,1)

    def test_retry_truncation_and_malformed_fail_closed(self):
        for reply in (result("","length"),
                      result('{"choice":"A","reason":"valid"}', "length"),
                      result('{"choice":"A","choice":"B","reason":"duplicate"}'),
                      result('{"choice":"A","reason":"ok","trade_action":"BUY"}')):
            with self.subTest(reply=reply):
                model=student(result("","length"),reply)
                got=model.answer_exam_case(self.case)
                self.assertEqual(got["choice"],"")
                self.assertEqual(model._model.create_chat_completion.call_count,2)

    def test_legacy_schema_partial_is_not_silently_retried(self):
        model=student(result('{"choice":"A","reason":"cut"}',"length"),thinking=False)
        got=model.answer_exam_case(self.case)
        self.assertEqual(got["choice"],"")
        self.assertEqual(model._model.create_chat_completion.call_count,1)

    def test_read_only_source_recovery_is_edition_scoped(self):
        case=next(t["public"] for t in four_book_blind_cases()
                  if t["public"]["edition"]=="TC2")
        model=student(result("","length"),
                      result('{"choice":"A","reason":"Исходные правила требуют проверки"}'))
        response=answer_with_self_retrieval(model,case)
        self.assertFalse(response["execution_allowed"])
        self.assertIsNone(response["trade_action"])
        self.assertEqual(response["edition"],"TC2")
        self.assertTrue(all(x.startswith("TC2.") for x in response["source_ids_used"]))
        messages=model._model.create_chat_completion.call_args.kwargs["messages"]
        shown=json.loads(messages[1]["content"])
        self.assertNotIn("expected",str(shown))
        self.assertNotIn("source_rule_id",str(shown))
        self.assertEqual(shown["case"]["edition"],"TC2")
        self.assertEqual(len(model._model.create_chat_completion.call_args.kwargs),5)


if __name__=="__main__":
    unittest.main()
