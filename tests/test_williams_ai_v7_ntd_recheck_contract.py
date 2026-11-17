import json,unittest,copy
from pathlib import Path
from williams_ai.v7_ntd_recheck_contract import validate
P=Path(__file__).resolve().parents[1]/"research"/"williams_v7_ntd_four_dimensions_recheck_20261115.json"
class NTDRecheckTests(unittest.TestCase):
 def setUp(self):self.data=json.loads(P.read_text())
 def test_six_ntd_source_rechecks_and_no_gold(self):
  x=validate(self.data);self.assertEqual(x["reviews"],6);self.assertFalse(x["training_authorized"])
  self.assertEqual({r["id"] for r in self.data["reviews"]},{f"RULE_CHECK_{i:03d}" for i in range(11,17)})
 def test_reject_unverified_source_or_gold(self):
  d=copy.deepcopy(self.data);d["source_pdf_sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"SOURCE_HASH"):validate(d)
  d=copy.deepcopy(self.data);d["model_labels"]=1
  with self.assertRaisesRegex(ValueError,"GOLD_LEAK"):validate(d)
 def test_reject_wrong_rule_card(self):
  d=copy.deepcopy(self.data);d["reviews"][0]["rule_ids"]=["TC2.WM2"]
  with self.assertRaisesRegex(ValueError,"WRONG_BOOK_RULE"):validate(d)
if __name__=="__main__":unittest.main()
