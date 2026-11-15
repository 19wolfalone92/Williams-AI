"""No real model weights required; fail-closed decoder unit checks."""
import unittest
from williams_ai.v7_structured_decoder import schema,decode,choices

class GuidedJSONV7(unittest.TestCase):
    def test_rules_for_all_task_families(self):
        for family,wanted in (("ENTRY","LONG_STOP_ENTRY"),("EXIT","HOLD"),
                              ("WAVE","WAVE_HYPOTHESIS")):
            x=schema(family)
            self.assertFalse(x["additionalProperties"])
            self.assertIn(wanted,x["properties"]["decision"]["enum"])
            self.assertEqual(set(x["required"]),{"decision","reason"})
    def test_valid_refusal_is_not_error(self):
        x=decode('{"decision":"ABSTAIN","reason":"Need independent review."}',"ENTRY")
        self.assertTrue(x["valid_model_decision"])
        self.assertEqual(x["finish_reason"],"stop")
        self.assertIsNone(x["trade_action"])
    def test_malformed_json_never_laundered_into_abstain(self):
        errors=[
         ('{"decision":"EXECUTE_LONG","reason":"x"}',"INVALID_CATEGORY"),
         ('{"decision":"NO_SETUP","reason":"x","order":true}',"INVALID_FIELDS"),
         ('{"decision":"NO_SETUP","reason":"x","reason":"y"}',"INVALID_JSON"),
         ('nonsense',"INVALID_JSON"),
         ('{"decision":"NO_SETUP","reason":""}',"INVALID_REASON")]
        for raw,code in errors:
            with self.subTest(raw=raw):
                x=decode(raw,"ENTRY")
                self.assertIsNone(x["decision"])
                self.assertEqual(x["failure_code"],code)
                self.assertFalse(x["valid_model_decision"])
    def test_timeout_and_no_eos_distinct(self):
        raw='{"decision":"NO_SETUP","reason":"x"}'
        self.assertEqual(decode(raw,"ENTRY",at_limit=True)["failure_code"],"TRUNCATED")
        self.assertEqual(decode(raw,"ENTRY",eos_seen=False)["failure_code"],"MISSING_EOS")
    def test_reason_is_bounded(self):
        raw='{"decision":"NO_SETUP","reason":"'+"a"*221+'"}'
        self.assertEqual(decode(raw,"ENTRY")["failure_code"],"INVALID_REASON")

if __name__=="__main__":unittest.main()
