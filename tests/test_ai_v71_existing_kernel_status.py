import unittest
from scripts.ai_v71_existing_kernel_status import EXPECTED,_summary
class V71StatusContract(unittest.TestCase):
 def test_exactly_eight_frozen_ids(self):
  self.assertEqual(len(EXPECTED),8)
  self.assertEqual(EXPECTED[0],"HIST100-01-1")
  self.assertEqual(EXPECTED[-1],"HIST100-02-4")
 def test_invalid_json_is_counted_as_technical_not_model_abstain(self):
  rows=[{"finish_reason":"error","valid_model_decision":False,"failure_code":"INVALID_JSON",
         "decision":None,"diagnostic":{"eos_seen":True,"raw_without_special_tokens":"{"},"tokens_out":12}]
  s=_summary(rows)
  self.assertEqual(s["technical_failures"],{"INVALID_JSON":1})
  self.assertEqual(s["valid_decisions"],0)
if __name__=="__main__":unittest.main()
