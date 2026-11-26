import json
import unittest
from scripts.ai_v6_objective_gpu_smoke import grade,encode,messages
from williams_ai.v6_objective_practice import curriculum


class GpuSmokeContracts(unittest.TestCase):
    def test_answer_is_exact_type_and_json_not_format_guess(self):
        for raw in ('{"answer":true}','{"answer":"1"}','{"answer":1,"answer":1}',
                    '{"answer":1,"extra":0}','prefix {"answer":1}'):
            self.assertEqual(grade(raw,'{"answer":1}'),"INVALID_FORMAT")
        self.assertEqual(grade('{"answer":1}','{"answer":1}'),"CORRECT")
        self.assertEqual(grade('{"answer":2}','{"answer":1}'),"INCORRECT")
        self.assertEqual(grade('{"answer":1}','{"answer":1}',True),"TECHNICAL_TRUNCATION")

    def test_development_labels_never_admitted_to_tokenizer(self):
        row=next(r for r in curriculum()[0] if r["split"]=="development")
        with self.assertRaisesRegex(ValueError,"NO_DEVELOPMENT_LABELS"):
            encode([row],None)
        self.assertEqual(set(messages(row)[1]),{"role","content"})
        self.assertNotIn(row["answer"],messages(row)[1]["content"])
