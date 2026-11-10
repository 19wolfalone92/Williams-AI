"""Research-only regression: detect malformed STOP/risk answers; never qualify a trader."""
import copy
import json
from hashlib import sha256
import unittest

from williams_ai.structured_transfer_exam import sealed_cases, grade


class OfflineAdmissionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.questions, self.keys = sealed_cases()
        rows = [
            {"case_id": key["case_id"], "response": json.dumps(key["expected"])}
            for key in self.keys
        ]
        digest = sha256(json.dumps(
            self.questions, sort_keys=True, ensure_ascii=False
        ).encode()).hexdigest()
        self.precommit = {
            "questions_sha256": digest,
            "stages": {
                stage: copy.deepcopy(rows)
                for stage in ("base", "first_private", "corrected")
            },
        }

    def _corrupt(self, case_prefix, mutate):
        for row in self.precommit["stages"]["corrected"]:
            if row["case_id"].startswith(case_prefix):
                response = json.loads(row["response"])
                mutate(response)
                row["response"] = json.dumps(response)
                return row["case_id"]
        self.fail("Expected research exam case is absent")

    def _assert_rejected(self, case_id):
        result = grade(self.questions, self.keys, self.precommit)
        self.assertEqual(result["stages"]["corrected"]["correct"], 23)
        self.assertFalse(next(
            row["correct"] for row in result["stages"]["corrected"]["case_results"]
            if row["case_id"] == case_id
        ))
        self.assertFalse(result["trading_qualification"])
        self.assertFalse(result["execution_allowed"])

    def test_missing_protective_stop_is_rejected(self):
        case_id = self._corrupt("wm1-", lambda value: value.pop("stop"))
        self._assert_rejected(case_id)

    def test_oversized_risk_quantity_is_rejected(self):
        case_id = self._corrupt("risk-", lambda value: value.__setitem__(
            "quantity", value["quantity"] * 1000
        ))
        self._assert_rejected(case_id)

    def test_execution_permission_true_is_rejected(self):
        case_id = self._corrupt("wm1-", lambda value: value.__setitem__(
            "execution_allowed", True
        ))
        self._assert_rejected(case_id)

    def test_perfect_rule_exam_never_claims_trading_qualification(self):
        result = grade(self.questions, self.keys, self.precommit)
        self.assertEqual(result["stages"]["corrected"]["correct"], 24)
        self.assertFalse(result["trading_qualification"])
        self.assertFalse(result["profitability_measured"])
        self.assertFalse(result["execution_allowed"])
