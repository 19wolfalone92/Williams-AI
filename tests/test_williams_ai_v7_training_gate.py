import copy,json,unittest
from pathlib import Path
from williams_ai.v7_training_gate import validate

FILE=Path(__file__).resolve().parents[1]/"research"/"williams_v7_review_targets_20261115.json"
class PreparedButNotGold(unittest.TestCase):
 def setUp(self):self.payload=json.loads(FILE.read_bytes())
 def test_exact_pinned_originals_no_gold(self):
  x=validate(self.payload)
  self.assertEqual(x["source_review_targets"],10)
  self.assertEqual(x["independent_human_labels"],0)
  self.assertFalse(x["training_authorized"])
 def test_no_fake_label_on_any_case(self):
  obj=copy.deepcopy(self.payload);obj["cases"][0]["decision"]="LONG_STOP_ENTRY"
  with self.assertRaisesRegex(ValueError,"FORBIDDEN_LABEL"):
   validate(obj)
 def test_source_sha_must_not_be_guessed(self):
  obj=copy.deepcopy(self.payload);obj["cases"][0]["source_pdf_sha256"]="a"*64
  with self.assertRaisesRegex(ValueError,"UNPINNED_BOOK"):
   validate(obj)
 def test_cannot_trick_model_training_flag(self):
  obj=copy.deepcopy(self.payload);obj["training_approved"]=True
  with self.assertRaisesRegex(ValueError,"UNSUPPORTED_EXPERT_APPROVAL"):
   validate(obj)
if __name__=="__main__":unittest.main()
