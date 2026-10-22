import json
import tempfile
import unittest
from pathlib import Path

from williams_ai.competency_exam import CASES, grade_exam, public_question
from williams_ai.local_model import LocalWilliamsModel


class FakeExamEngine:
    def __init__(self, **kwargs):
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices":[{"message":{"content":'{"choice":"B","reason":"Цена ещё не пересекла уровень; нужна проверка."}'}}]}


class BookKnowledgeExamTests(unittest.TestCase):
    def test_all_four_source_profiles_and_answer_key_never_in_prompt(self):
        self.assertEqual(len(CASES),10)
        self.assertEqual({c["edition"] for c in CASES},{"TC1","NTD","TC2","COURSE"})
        for item in CASES:
            safe=public_question(item)
            self.assertEqual(set(safe),{"id","edition","question","choices"})
            self.assertNotIn("correct",safe)
            self.assertIn(item["correct"],safe["choices"])

    def test_grade_is_not_book_comprehension_or_trading_authority(self):
        key = {c["id"]:c["correct"] for c in CASES}
        result=grade_exam(lambda x:{"choice":key[x["id"]], "reason":"synthetic oracle"})
        self.assertEqual(result["correct_cases"],10)
        self.assertEqual(result["score_on_curated_cases_percent"],100)
        self.assertIsNone(result["book_comprehension_percent"])
        self.assertFalse(result["books_retrained"])
        self.assertFalse(result["execution_allowed"])
        self.assertIsNone(result["trade_action"])

    def test_bad_and_unparseable_response_cannot_pass(self):
        result=grade_exam(lambda x:{"choice":"D","reason":"unknown"})
        self.assertEqual(result["correct_cases"],0)
        self.assertTrue(all(item["chosen"] is None for item in result["results"]))

    def test_local_gguf_inference_is_actual_choice_not_hidden_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"fake.gguf"
            path.write_bytes(b"test data only")
            model=LocalWilliamsModel(path,loader=FakeExamEngine)
            safe=public_question(CASES[0])
            answer=model.answer_exam_case(safe)
            self.assertEqual(answer["choice"],"B")
            call=model._model.calls[0]
            self.assertEqual(call["response_format"],{"type":"json_object"})
            self.assertEqual(call["temperature"],0)
            self.assertNotIn("tools",call)
            prompt=json.loads(call["messages"][1]["content"])
            self.assertEqual(set(prompt),{"id","edition","question","choices"})
            self.assertNotIn("correct",prompt)
            with self.assertRaises(ValueError):
                model.answer_exam_case({**safe,"correct":"B"})


if __name__=="__main__":
    unittest.main()
