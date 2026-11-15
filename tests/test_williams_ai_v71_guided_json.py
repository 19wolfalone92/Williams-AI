import unittest
from williams_ai.v71_guided_json_diagnostics import ENFORCER_COMMIT,decode,schema
class V71Tests(unittest.TestCase):
 def test_pin_and_strict_schema(self):
  self.assertEqual(ENFORCER_COMMIT,"d4d9b7cc20937bcd3a643be7a8dafb7b1ca20dfb")
  for family in ("ENTRY","EXIT","WAVE"):self.assertFalse(schema(family)["additionalProperties"])
 def test_valid_abstain(self):
  x=decode('{"decision":"ABSTAIN","reason":"Insufficient evidence."}',"ENTRY")
  self.assertTrue(x["valid_model_decision"]);self.assertEqual(x["finish_reason"],"stop")
 def test_malformed_stays_technical(self):
  for raw in ('{"decision":"NO_SETUP","reason":"x","reason":"y"}','','garbage','{"decision":"EXECUTE_LONG","reason":"x"}'):
   x=decode(raw,"ENTRY");self.assertIsNone(x["decision"]);self.assertFalse(x["valid_model_decision"])
 def test_truncation_is_not_abstain(self):
  x=decode("{}","ENTRY",at_limit=True);self.assertIsNone(x["decision"]);self.assertEqual(x["failure_code"],"TRUNCATED")
if __name__=="__main__":unittest.main()
