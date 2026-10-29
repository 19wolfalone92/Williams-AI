import copy
import json
from hashlib import sha256
import unittest
from williams_ai.structured_transfer_exam import sealed_cases, grade


class StructuredTransferTests(unittest.TestCase):
    def fixture(self):
        questions,keys=sealed_cases()
        rows=[{"case_id":k["case_id"],"response":json.dumps(k["expected"])} for k in keys]
        pre={"questions_sha256":sha256(json.dumps(questions,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            "stages":{s:copy.deepcopy(rows) for s in ("base","first_private","corrected")}}
        return questions,keys,pre

    def test_questions_do_not_carry_expected_values_or_keys(self):
        q,k=sealed_cases()
        self.assertEqual(len(q),24)
        self.assertTrue(all(set(x)=={"case_id","domain","prompt"} for x in q))
        self.assertEqual(len({x["domain"] for x in q}),7)

    def test_correct_json_grades_and_boolean_cannot_masquerade_as_number(self):
        q,k,p=self.fixture()
        self.assertEqual(grade(q,k,p)["stages"]["corrected"]["correct"],24)
        p["stages"]["corrected"][0]["response"]='{"risk_budget": true, "quantity": 28, "execution_allowed": false}'
        self.assertEqual(grade(q,k,p)["stages"]["corrected"]["correct"],23)

    def test_repeated_one_answer_and_wrong_format_are_visible_failures(self):
        q,k,p=self.fixture()
        for r in p["stages"]["first_private"]:r["response"]="same repeated answer"
        r=grade(q,k,p)["stages"]["first_private"]
        self.assertEqual(r["correct"],0)
        self.assertEqual(r["largest_identical_response_count"],24)
        self.assertEqual(r["valid_json"],0)

    def test_missing_or_replayed_answers_cannot_pass(self):
        q,k,p=self.fixture()
        p["stages"]["base"][1]=p["stages"]["base"][0]
        with self.assertRaises(ValueError):grade(q,k,p)
