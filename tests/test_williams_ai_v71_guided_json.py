import unittest
from williams_ai.v71_guided_json_diagnostics import ENFORCER_COMMIT,decode,schema
from scripts.ai_v71_guided_json_8_h1_submit import _is_missing_private_slug

class V71Tests(unittest.TestCase):
 def test_kaggle_wrapped_403_valueerror_detected_without_literal_status(self):
  e=ValueError("Cannot access kernel 'owner/slug' (Permission 'kernels.get' was denied).")
  self.assertTrue(_is_missing_private_slug(e,"owner","owner/slug"))
  self.assertFalse(_is_missing_private_slug(e,"other","owner/slug"))
  self.assertFalse(_is_missing_private_slug(ValueError("Permission denied"),"owner","owner/slug"))
 def test_new_kernel_title_slug_is_conservative_length_and_matches_id(self):
  from scripts.ai_v71_guided_json_8_h1_submit import SLUG
  self.assertLessEqual(len(SLUG),50)
  self.assertEqual(SLUG,"williams-v71-lmfe8-deps-20261116")
 def test_previous_kernel_missing_interegular_must_be_preinstalled(self):
  import inspect
  from scripts.ai_v71_guided_json_8_h1_submit import main
  implementation=inspect.getsource(main)
  for required in ("interegular==0.3.3","PyYAML==6.0.2",
                   "packaging==25.0","pydantic==2.11.7",
                   "import interegular, yaml, packaging, pydantic, lmformatenforcer"):
   self.assertIn(required,implementation)
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
