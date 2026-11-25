import json,unittest
from scripts.ai_v6_corrective_probe_cpu import grade_response

EXPECTED={"answer":"-3.5","entry_stop":None,"rule_id":"NTD.AO","calculation":"AO=-3.5",
 "reason":"Signed difference.","counterexample":"Absolute value is wrong.","uncertainty":"None.","cancel_if":"If averages differ in bar."}

class CorrectiveProbeGradeTests(unittest.TestCase):
 def test_strict_complete_response_passes(self):
  result=grade_response(json.dumps(EXPECTED),json.dumps(EXPECTED),"stop")
  self.assertTrue(result["fully_correct"])
 def test_numeric_json_representation_is_accepted(self):
  actual=dict(EXPECTED);actual["answer"]=-3.5
  self.assertTrue(grade_response(json.dumps(actual),json.dumps(EXPECTED),"stop")["answer_correct"])
 def test_wrong_sign_fails(self):
  actual=dict(EXPECTED);actual["answer"]="3.5"
  self.assertFalse(grade_response(json.dumps(actual),json.dumps(EXPECTED),"stop")["answer_correct"])
 def test_missing_explanation_fails_full_score(self):
  actual=dict(EXPECTED);actual["counterexample"]=""
  result=grade_response(json.dumps(actual),json.dumps(EXPECTED),"stop")
  self.assertTrue(result["answer_correct"]);self.assertFalse(result["explanation_complete"]);self.assertFalse(result["fully_correct"])
 def test_extra_keys_and_truncation_fail_format(self):
  actual=dict(EXPECTED);actual["extra"]="not allowed"
  self.assertFalse(grade_response(json.dumps(actual),json.dumps(EXPECTED),"length")["format_valid"])
 def test_invalid_json_is_not_guessed(self):
  result=grade_response("not json",json.dumps(EXPECTED),"stop")
  self.assertFalse(result["format_valid"]);self.assertFalse(result["answer_correct"])

if __name__=="__main__":unittest.main()
